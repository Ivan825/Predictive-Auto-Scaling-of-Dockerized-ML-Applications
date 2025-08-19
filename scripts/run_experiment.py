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


def time_split(df: pd.DataFrame, train_days: int = 7):
    """
    Split a continuous 1-min dataframe into train (first N days) and test (rest).
    Assumes a 'timestamp' column of dtype datetime64[ns].
    """
    t0 = df["timestamp"].min()
    split_time = t0 + pd.Timedelta(days=train_days)
    df_train = df[df["timestamp"] < split_time].reset_index(drop=True)
    df_test  = df[df["timestamp"] >= split_time].reset_index(drop=True)
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Split produced empty train or test. Increase total days or adjust train_days.")
    return df_train, df_test

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

def run_prophet_pipeline(df, train_days=7):
    """
    Train Prophet WITH regressors on the TRAIN window, forecast the TEST window,
    simulate policies ONLY on TEST, and save plots/CSVs.
    """
    results = []

    # ---- split
    df_train, df_test = time_split(df, train_days=train_days)

    print("\n=== [Prophet] Training with regressors (train window)… ===")
    m = train_prophet(df_train, regressor_cols=["is_event"])

    # Forecast exactly the test horizon, using test regressors (e.g., is_event)
    horizon_minutes = len(df_test)  # 1-minute frequency
    fut_regs = df_test[["timestamp", "is_event"]] if "is_event" in df_test.columns else df_test[["timestamp"]]
    fc_test = forecast_prophet(
        m,
        df=df_train,                     # df required for schema; training df
        horizon_minutes=horizon_minutes, # forecast ahead
        future_regressors=fut_regs       # supply test regressor values
    ).rename(columns={"ds": "timestamp"})

    # Keep only the forecast rows that align with test timestamps
    fc_test = fc_test.iloc[-len(df_test):].reset_index(drop=True)
    df_fc_test = df_test.merge(fc_test, on="timestamp", how="left")

    # Forecast metrics on TEST window
    mask = ~df_fc_test["yhat"].isna()
    y = df_fc_test.loc[mask, "req_per_min"].values
    yhat = df_fc_test.loc[mask, "yhat"].values
    print("Prophet TEST forecast metrics:",
          {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "SMAPE": smape(y, yhat)})

    # Plot (test window) demand vs forecast
    plot_prophet_forecast(df_fc_test, ROOT / "reports/prophet_demand_vs_forecast_TEST.png")

    # Simulate policies on TEST only
    sim_map, score_map = {}, {}
    policies = ["static2", "simple", "buffered", "conf", "policy"]

    for policy in policies:
        if policy == "static2":
            sim_df = static_simulation(df_fc_test, capacity=25, fixed_containers=2)
        else:
            sim_df = simulate(
                df=df_fc_test,
                forecasts=df_fc_test[["timestamp", "yhat", "yhat_upper"]],
                policy_name=policy,
                capacity=25,
                init_containers=2
            )

        scores = scaling_scores(sim_df)
        out_path = ROOT / f"reports/sim_prophet_TEST_{policy}.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[Prophet|TEST] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("prophet_TEST", policy, str(out_path), scores))

    # Plots: containers & latency across policies (TEST)
    plot_series_by_policy(sim_map, "containers",
                          "Containers Over Time (Prophet policies, TEST)",
                          "containers",
                          ROOT / "reports/prophet_containers_over_time_TEST.png")

    plot_series_by_policy(sim_map, "lat_ms",
                          "Latency Over Time (Prophet policies, TEST)",
                          "latency (ms)",
                          ROOT / "reports/prophet_latency_over_time_TEST.png")

    # Bar charts (TEST)
    plot_policy_bars(score_map, "Policy Comparison (Prophet, TEST)", ROOT / "reports/prophet_policy_bars_TEST.png")

    return results



def run_lstm_pipeline(df, train_days=7):
    """
    Train LSTM on TRAIN window, generate rolling forecasts across TEST window (using only past),
    simulate policies ONLY on TEST, and save plots/CSVs.
    """
    results = []

    # ---- split
    df_train, df_test = time_split(df, train_days=train_days)

    print("\n=== [LSTM] Training (train window)… ===")
    # You can tweak past/ahead here; keep aligned with your model file defaults
    model, cfg = train_lstm(df_train, past=60, ahead=15, epochs=5, batch_size=128)

    # Rolling forecasting on TEST: at each t in test, use history up to t (train + test[0..t-1])
    print("[LSTM] Generating rolling forecasts on TEST…")
    history = df_train.copy()
    rows = []
    for i in range(len(df_test)):
        # predict next-horizon avg using current history
        yhat = predict_lstm(model, history, cfg)
        rows.append({"timestamp": df_test["timestamp"].iloc[i], "yhat": yhat})
        # append the actual observed point at time i to history (to simulate real-time arrival)
        history = pd.concat([history, df_test.iloc[i:i+1]], ignore_index=True)

    lstm_fc_test = pd.DataFrame(rows)

    # Align truth with forecast start (it's exactly df_test)
    df_eval = df_test.merge(lstm_fc_test, on="timestamp", how="left")

    # Plot LSTM forecast vs demand (TEST)
    plot_lstm_forecast(df_eval, ROOT / "reports/lstm_demand_vs_forecast_TEST.png")

    # Simulate policies on TEST
    sim_map, score_map = {}, {}
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
        out_path = ROOT / f"reports/sim_lstm_TEST_{policy}.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[LSTM|TEST] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("lstm_TEST", policy, str(out_path), scores))

    # Plots for LSTM policies (TEST)
    plot_series_by_policy(sim_map, "containers",
                          "Containers Over Time (LSTM policies, TEST)",
                          "containers",
                          ROOT / "reports/lstm_containers_over_time_TEST.png")

    plot_series_by_policy(sim_map, "lat_ms",
                          "Latency Over Time (LSTM policies, TEST)",
                          "latency (ms)",
                          ROOT / "reports/lstm_latency_over_time_TEST.png")

    plot_policy_bars(score_map, "Policy Comparison (LSTM, TEST)", ROOT / "reports/lstm_policy_bars_TEST.png")

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

    # 2) Run pipelines on train/test split
    all_results = []
    TRAIN_DAYS = 7  # first 7 days train, rest test (adjust if you generated different length)

    if HAVE_PROPHET:
        all_results.extend(run_prophet_pipeline(df, train_days=TRAIN_DAYS))
    else:
        print("[SKIP] Prophet pipeline (library unavailable).")

    if HAVE_LSTM:
        all_results.extend(run_lstm_pipeline(df, train_days=TRAIN_DAYS))
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
