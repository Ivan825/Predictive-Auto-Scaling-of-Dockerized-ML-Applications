# scripts/run_experiment.py

import os
import sys
import pandas as pd
from pathlib import Path

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project root imports work when running as a module: python -m scripts.run_experiment
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.simulate_data import simulate_demand
from src.simulator import simulate
from src.metrics import mae, rmse, mape, smape, scaling_scores

# Optional imports (Prophet / PyTorch LSTM)
HAVE_PROPHET = True
try:
    from src.models.prophet_model import train_prophet, forecast_prophet
except Exception as e:
    print("[WARN] Prophet unavailable or failed to import:", repr(e))
    HAVE_PROPHET = False

HAVE_LSTM = True
try:
    from src.models.lstm_model import train_lstm, predict_lstm
except Exception as e:
    print("[WARN] LSTM (PyTorch) unavailable or failed to import:", repr(e))
    HAVE_LSTM = False


def ensure_dirs():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)


def static_simulation(df_like, capacity=25, fixed_containers=2):
    sim_df = simulate(
        df=df_like,
        forecasts=df_like[[c for c in ["timestamp", "yhat", "yhat_upper"] if c in df_like.columns]],
        policy_name="policy",
        capacity=capacity,
        init_containers=fixed_containers
    )
    sim_df["containers"] = fixed_containers
    sim_df["over"] = (sim_df["containers"] * capacity > sim_df["demand"] * 1.15).astype(int)
    sim_df["under"] = (sim_df["demand"] > sim_df["containers"] * capacity).astype(int)
    return sim_df


# ---------- plotting helpers ----------

def plot_prophet_forecast(df_fc, outpath):
    """Plot ground-truth demand vs Prophet yhat and yhat_upper."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df_fc["timestamp"], df_fc["req_per_min"], label="Demand (truth)", linewidth=1.2)
    ax.plot(df_fc["timestamp"], df_fc["yhat"], label="Prophet yhat", linewidth=1.0)
    if "yhat_upper" in df_fc.columns:
        ax.plot(df_fc["timestamp"], df_fc["yhat_upper"], label="Prophet yhat_upper", alpha=0.7, linewidth=0.8)
    ax.set_title("Demand vs Prophet Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Requests per minute")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_lstm_forecast(df_eval_with_pred, outpath):
    """Plot ground-truth demand vs LSTM yhat (aligned df with columns: timestamp, req_per_min, yhat)."""
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df_eval_with_pred["timestamp"], df_eval_with_pred["req_per_min"], label="Demand (truth)", linewidth=1.2)
    ax.plot(df_eval_with_pred["timestamp"], df_eval_with_pred["yhat"], label="LSTM yhat (next-horizon avg)", linewidth=1.0)
    ax.set_title("Demand vs LSTM Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Requests per minute")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_series_by_policy(sim_map, col, title, ylabel, outpath):
    """Plot a time series (e.g., containers/latency) for each policy on one chart."""
    fig, ax = plt.subplots(figsize=(11, 4))
    for policy, sdf in sim_map.items():
        ax.plot(sdf["timestamp"], sdf[col], label=policy, linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_policy_bars(scores_map, title, outpath):
    """Bar charts for a few metrics across policies."""
    rows = []
    for policy, scores in scores_map.items():
        rows.append({
            "policy": policy,
            "under_events": scores["under_events"],
            "over_%": scores["over_%"],
            "cost_index": scores["cost_index"],
            "switches": scores["switches"],
        })
    dfb = pd.DataFrame(rows).set_index("policy")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    dfb["under_events"].plot(kind="bar", ax=axes[0,0], title="Under-provisioning Events")
    axes[0,0].set_ylabel("# events")

    dfb["over_%"].plot(kind="bar", ax=axes[0,1], title="Over-provision %")
    axes[0,1].set_ylabel("% minutes")

    dfb["cost_index"].plot(kind="bar", ax=axes[1,0], title="Cost Index (container-mins)")
    axes[1,0].set_ylabel("index")

    dfb["switches"].plot(kind="bar", ax=axes[1,1], title="Scaling Switches")
    axes[1,1].set_ylabel("# changes")

    plt.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(outpath)
    plt.close(fig)


# ---------- pipelines ----------

def run_prophet_pipeline(df):
    """
    Train Prophet WITH regressors, get in-sample predictions, simulate policies,
    and produce plots.
    """
    results = []

    print("\n=== [Prophet] Training with regressors… ===")
    m = train_prophet(df, regressor_cols=["is_event"])

    fc = forecast_prophet(m, df=df, horizon_minutes=0)  # in-sample (regressor-safe)
    fc = fc.rename(columns={"ds": "timestamp"})
    df_fc = df.merge(fc, on="timestamp", how="left")

    # Forecast metrics
    mask = ~df_fc["yhat"].isna()
    y = df_fc.loc[mask, "req_per_min"].values
    yhat = df_fc.loc[mask, "yhat"].values
    print("Prophet forecast metrics:",
          {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "SMAPE": smape(y, yhat)})

    # Plot forecast vs demand
    plot_prophet_forecast(df_fc, ROOT / "reports/prophet_demand_vs_forecast.png")

    # Simulate policies
    sim_map = {}    # policy -> sim_df
    score_map = {}  # policy -> scores
    policies = ["static2", "simple", "buffered", "conf", "policy"]

    for policy in policies:
        if policy == "static2":
            sim_df = static_simulation(df_fc, capacity=25, fixed_containers=2)
        else:
            sim_df = simulate(
                df=df_fc,
                forecasts=df_fc[["timestamp", "yhat", "yhat_upper"]],
                policy_name=policy,
                capacity=25,
                init_containers=2
            )

        scores = scaling_scores(sim_df)
        out_path = ROOT / f"reports/sim_prophet_{policy}.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[Prophet] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("prophet", policy, str(out_path), scores))

    # Plots: containers & latency across policies
    plot_series_by_policy(sim_map, "containers",
                          "Containers Over Time (Prophet policies)",
                          "containers",
                          ROOT / "reports/prophet_containers_over_time.png")

    plot_series_by_policy(sim_map, "lat_ms",
                          "Latency Over Time (Prophet policies)",
                          "latency (ms)",
                          ROOT / "reports/prophet_latency_over_time.png")

    # Bar charts
    plot_policy_bars(score_map, "Policy Comparison (Prophet)", ROOT / "reports/prophet_policy_bars.png")

    return results


def run_lstm_pipeline(df):
    """
    Train PyTorch LSTM, create rolling per-minute forecasts (avg next horizon),
    simulate policies, and produce plots.
    """
    results = []

    print("\n=== [LSTM] Training… ===")
    model, cfg = train_lstm(df, past=60, ahead=15, epochs=5, batch_size=128)

    print("[LSTM] Generating rolling forecasts…")
    PAST = cfg.get("past", 60)
    rows = []
    # start at least after PAST points (predict next-horizon avg)
    for i in range(PAST, len(df)):
        sub = df.iloc[:i].copy()
        yhat = predict_lstm(model, sub, cfg)
        rows.append({"timestamp": df["timestamp"].iloc[i], "yhat": yhat})
    lstm_fc = pd.DataFrame(rows)

    # align truth with forecast start
    df_eval = df.iloc[PAST:].reset_index(drop=True)
    df_eval = df_eval.merge(lstm_fc, on="timestamp", how="left")

    # Plot LSTM forecast vs demand
    plot_lstm_forecast(df_eval, ROOT / "reports/lstm_demand_vs_forecast.png")

    sim_map = {}
    score_map = {}
    policies = ["static2", "simple", "buffered", "policy"]

    for policy in policies:
        if policy == "static2":
            sim_df = static_simulation(df_eval, capacity=25, fixed_containers=2)
        else:
            sim_df = simulate(
                df=df_eval,
                forecasts=df_eval[["timestamp", "yhat"]],
                policy_name=policy,
                capacity=25,
                init_containers=2
            )

        scores = scaling_scores(sim_df)
        out_path = ROOT / f"reports/sim_lstm_{policy}.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[LSTM] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("lstm", policy, str(out_path), scores))

    # Plots for LSTM policies
    plot_series_by_policy(sim_map, "containers",
                          "Containers Over Time (LSTM policies)",
                          "containers",
                          ROOT / "reports/lstm_containers_over_time.png")

    plot_series_by_policy(sim_map, "lat_ms",
                          "Latency Over Time (LSTM policies)",
                          "latency (ms)",
                          ROOT / "reports/lstm_latency_over_time.png")

    plot_policy_bars(score_map, "Policy Comparison (LSTM)", ROOT / "reports/lstm_policy_bars.png")

    return results


def main():
    ensure_dirs()

    # 1) Data (create or load)
    data_path = ROOT / "data/run.csv"
    if data_path.exists():
        print(f"[Data] Loading existing: {data_path}")
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        print("[Data] Generating synthetic data (14 days, 1-min freq)…")
        df = simulate_demand(days=14)   # includes is_event
        df.to_csv(data_path, index=False)
        print(f"[Data] Saved: {data_path}")

    # 2) Run pipelines
    all_results = []
    if HAVE_PROPHET:
        all_results.extend(run_prophet_pipeline(df))
    else:
        print("[SKIP] Prophet pipeline (library unavailable).")

    if HAVE_LSTM:
        all_results.extend(run_lstm_pipeline(df))
    else:
        print("[SKIP] LSTM pipeline (library unavailable).")

    # 3) Metrics summary
    if all_results:
        rows = []
        for model_name, policy, path, scores in all_results:
            row = {"model": model_name, "policy": policy, "log_path": path}
            row.update(scores)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = ROOT / "reports/metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n[Summary] Wrote: {summary_path}")
        print(summary)
    else:
        print("\n[ERROR] No results generated. Ensure at least one model pipeline is available.")


if __name__ == "__main__":
    main()
