# scripts/run_experiment.py

import os
import sys
import pandas as pd
from pathlib import Path
import joblib
import torch

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project root imports work when running as a module: python -m scripts.run_experiment
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

from src.simulator import simulate
from src.metrics import mae, rmse, mape, smape, scaling_scores

# Optional imports (Prophet / PyTorch LSTM)
HAVE_PROPHET = True
try:
    from src.models.prophet_model import forecast_prophet
except Exception as e:
    print(f"[WARN] Prophet unavailable or failed to import: {e!r}")
    HAVE_PROPHET = False

HAVE_LSTM = True
try:
    # We need the class definition to load the model
    from src.models.lstm_model import LSTM, predict_lstm
except Exception as e:
    print(f"[WARN] LSTM (PyTorch) unavailable or failed to import: {e!r}")
    HAVE_LSTM = False


def ensure_dirs():
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)


def static_simulation(df_like, capacity=25, fixed_containers=2):
    # Add demand column if missing for static simulation
    if 'demand' not in df_like.columns and 'req_per_min' in df_like.columns:
        df_like['demand'] = df_like['req_per_min']

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
    ax.set_title("Demand vs Prophet Forecast (Test Set)")
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
    ax.set_title("Demand vs LSTM Forecast (Test Set)")
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

def run_prophet_pipeline(df_test):
    """
    Load pre-trained Prophet model, forecast for df_test, simulate policies,
    and produce plots.
    """
    results = []

    print("=== [Prophet] Loading model and forecasting for test set… ===")
    try:
        m = joblib.load(MODELS_DIR / 'prophet_model.joblib')
    except FileNotFoundError:
        print("[ERROR] Prophet model not found. Please run `python -m scripts.train_models` first.")
        return []

    # Create future dataframe for the test period
    future = m.make_future_dataframe(periods=len(df_test), freq='min', include_history=False)
    
    # Add regressors to future dataframe if they exist in the model
    if hasattr(m, 'train_component_cols') and 'extra_regressors_additive' in m.train_component_cols:
        for regressor in m.train_component_cols['extra_regressors_additive']:
            if regressor in df_test.columns:
                future[regressor] = df_test[regressor].values
            else:
                print(f"[WARN] Regressor '{regressor}' not found in test set. Prophet may produce poor forecasts.")
                future[regressor] = 0 # Or some other default

    fc = forecast_prophet(m, future_df=future)
    fc = fc.rename(columns={"ds": "timestamp"})
    
    # Align forecast with test data
    df_fc = df_test.copy()
    df_fc['yhat'] = fc['yhat'].values
    df_fc['yhat_upper'] = fc['yhat_upper'].values
    df_fc['yhat_lower'] = fc['yhat_lower'].values

    # Forecast metrics
    y = df_fc["req_per_min"].values
    yhat = df_fc["yhat"].values
    print("Prophet forecast metrics (test set):", {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "SMAPE": smape(y, yhat)})

    # Plot forecast vs demand
    plot_prophet_forecast(df_fc, ROOT / "reports/prophet_demand_vs_forecast_TEST.png")

    # Simulate policies
    sim_map = {}
    score_map = {}
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
        out_path = ROOT / f"reports/sim_prophet_{policy}_TEST.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[Prophet] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("prophet", policy, str(out_path), scores))

    # Plots
    plot_series_by_policy(sim_map, "containers", "Containers Over Time (Prophet policies, Test Set)", "containers", ROOT / "reports/prophet_containers_over_time_TEST.png")
    plot_series_by_policy(sim_map, "lat_ms", "Latency Over Time (Prophet policies, Test Set)", "latency (ms)", ROOT / "reports/prophet_latency_over_time_TEST.png")
    plot_policy_bars(score_map, "Policy Comparison (Prophet, Test Set)", ROOT / "reports/prophet_policy_bars_TEST.png")

    return results


def run_lstm_pipeline(df_train, df_test):
    """
    Load pre-trained LSTM, create rolling forecasts for df_test,
    simulate policies, and produce plots.
    """
    results = []

    print("=== [LSTM] Loading model and forecasting for test set… ===")
    try:
        lstm_config = joblib.load(MODELS_DIR / 'lstm_config.joblib')
        model = LSTM(
            input_dim=lstm_config['input_dim'],
            hidden_dim=lstm_config['hidden_dim'],
            layer_dim=lstm_config['layer_dim'],
            output_dim=lstm_config['output_dim']
        )
        model.load_state_dict(torch.load(MODELS_DIR / 'lstm_model.pt'))
        model.eval()
    except FileNotFoundError:
        print("[ERROR] LSTM model or config not found. Please run `python -m scripts.train_models` first.")
        return []

    print("[LSTM] Generating rolling forecasts for test set…")
    PAST = lstm_config.get("past", 60)
    
    # Combine train and test for rolling window history
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined_lstm = df_combined.rename(columns={'req_per_min': 'y'})

    rows = []
    start_index = len(df_train)
    for i in range(start_index, len(df_combined_lstm)):
        sub = df_combined_lstm.iloc[i-PAST:i].copy()
        yhat = predict_lstm(model, sub, lstm_config)
        rows.append({"timestamp": df_combined["timestamp"].iloc[i], "yhat": yhat})
    
    lstm_fc = pd.DataFrame(rows)

    # Align ground truth (test set) with forecasts
    df_eval = df_test.merge(lstm_fc, on="timestamp", how="left").dropna()

    # Forecast metrics
    y = df_eval["req_per_min"].values
    yhat = df_eval["yhat"].values
    print("LSTM forecast metrics (test set):",
          {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "SMAPE": smape(y, yhat)})

    # Plot LSTM forecast vs demand
    plot_lstm_forecast(df_eval, ROOT / "reports/lstm_demand_vs_forecast_TEST.png")

    # Simulate policies
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
        out_path = ROOT / f"reports/sim_lstm_{policy}_TEST.csv"
        sim_df.to_csv(out_path, index=False)
        print(f"[LSTM] {policy}:", scores)

        sim_map[policy] = sim_df
        score_map[policy] = scores
        results.append(("lstm", policy, str(out_path), scores))

    # Plots
    plot_series_by_policy(sim_map, "containers", "Containers Over Time (LSTM policies, Test Set)", "containers", ROOT / "reports/lstm_containers_over_time_TEST.png")
    plot_series_by_policy(sim_map, "lat_ms", "Latency Over Time (LSTM policies, Test Set)", "latency (ms)", ROOT / "reports/lstm_latency_over_time_TEST.png")
    plot_policy_bars(score_map, "Policy Comparison (LSTM, Test Set)", ROOT / "reports/lstm_policy_bars_TEST.png")

    return results


def main():
    """
    Main function to run the experiment pipelines.
    This script expects that `scripts/train_models.py` has been run first.
    """
    ensure_dirs()

    print("Loading train and test sets...")
    try:
        df_train = pd.read_csv(DATA_DIR / "google_trace_train.csv", parse_dates=['timestamp'])
        df_test = pd.read_csv(DATA_DIR / "google_trace_test.csv", parse_dates=['timestamp'])
    except FileNotFoundError:
        print("[ERROR] Train/test data not found. Please run `python -m scripts.train_models` first.")
        return

    print(f"Train set size: {len(df_train)}")
    print(f"Test set size: {len(df_test)}")

    all_results = []

    # Run Prophet pipeline
    if HAVE_PROPHET:
        prophet_results = run_prophet_pipeline(df_test.copy())
        all_results.extend(prophet_results)
    else:
        print("Prophet not available. Skipping Prophet pipeline.")

    # Run LSTM pipeline
    if HAVE_LSTM:
        # LSTM needs train set for historical context of rolling forecast
        lstm_results = run_lstm_pipeline(df_train.copy(), df_test.copy())
        all_results.extend(lstm_results)
    else:
        print("LSTM not available. Skipping LSTM pipeline.")

    # Summarize and save results
    if all_results:
        rows = []
        for model_name, policy, path, scores in all_results:
            row = {"model": model_name, "policy": policy, "log_path": path}
            row.update(scores)
            rows.append(row)
        summary = pd.DataFrame(rows)
        summary_path = ROOT / "reports/metrics_summary_TEST.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[Summary] Wrote: {summary_path}")
        print(summary)
    else:
        print("[ERROR] No results generated. Ensure models and data are available.")

    print("Experiment complete.")


if __name__ == "__main__":
    main()