# scripts/run_experiment.py

import os
import sys
import pandas as pd
from pathlib import Path

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make project root imports work when running as a module: python -m scripts.run_experiment
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.simulator import simulate
from src.metrics import mae, rmse, mape, smape, scaling_scores


# FINAL CORRECTED FUNCTION
def generate_trading_workload(csv_path: str):
    """
    Loads real 1-minute stock bar data, filters it for the most recent data,
    and converts it into a system demand workload.
    """
    print(f"[Data] Loading real market data from: {csv_path}")
    try:
        market_df = pd.read_csv(csv_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"[ERROR] The data file was not found at '{csv_path}'.")
        print("Please make sure the file exists and the path is correct.")
        sys.exit(1)

    market_df = market_df.rename(columns={'date': 'timestamp'})
    
    # --- CHANGE 1: AGGRESSIVELY FILTER TO THE LAST 90 DAYS ONLY ---
    if not market_df.empty:
        max_date = market_df['timestamp'].max()
        cutoff_date = max_date - pd.Timedelta(days=90) # Keep only the last 3 months
        market_df = market_df[market_df['timestamp'] > cutoff_date].copy()
        print(f"[Data] Filtered data to keep only records after {cutoff_date.date()}")

    # Standardize the timeline
    market_df = market_df.set_index('timestamp').resample('1min').sum().reset_index()

    print("[Data] Generating demand from market volume...")
    market_open_hour = 9.5
    market_close_hour = 16.0
    market_df['hour'] = market_df['timestamp'].dt.hour + market_df['timestamp'].dt.minute / 60
    market_df['weekday'] = market_df['timestamp'].dt.weekday

    is_market_day = market_df['weekday'].between(0, 4)
    is_market_hours = market_df['hour'].between(market_open_hour, market_close_hour)
    market_is_open = is_market_day & is_market_hours

    volume_to_demand_factor = 0.001
    base_demand = 10
    
    demand_from_volume = (market_df['volume'] * volume_to_demand_factor)
    market_df['req_per_min'] = np.where(market_is_open, base_demand + demand_from_volume, np.random.uniform(1, 5))
    market_df['req_per_min'] = market_df['req_per_min'].clip(lower=1).astype(int)
    market_df['is_event'] = 0

    final_df = market_df[['timestamp', 'req_per_min', 'is_event']]
    print("[Data] Workload generation complete.")
    
    return final_df


def time_split(df: pd.DataFrame, train_days: int = 7):
    t0 = df["timestamp"].min()
    split_time = t0 + pd.Timedelta(days=train_days)
    df_train = df[df["timestamp"] < split_time].reset_index(drop=True)
    df_test  = df[df["timestamp"] >= split_time].reset_index(drop=True)
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError("Split produced empty train or test. Increase total days or adjust train_days.")
    return df_train, df_test

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

def plot_prophet_forecast(df_fc, outpath):
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

def run_prophet_pipeline(df, train_days=7):
    results = []
    df_train, df_test = time_split(df, train_days=train_days)

    print("\n=== [Prophet] Training with regressors (train window)… ===")
    m = train_prophet(df_train, regressor_cols=["is_event"])

    horizon_minutes = len(df_test)
    fut_regs = df_test[["timestamp", "is_event"]]
    fc_test = forecast_prophet(
        m,
        df=df_train,
        horizon_minutes=horizon_minutes,
        future_regressors=fut_regs
    ).rename(columns={"ds": "timestamp"})

    fc_test = fc_test.iloc[-len(df_test):].reset_index(drop=True)
    df_fc_test = df_test.merge(fc_test, on="timestamp", how="left")

    mask = ~df_fc_test["yhat"].isna()
    y = df_fc_test.loc[mask, "req_per_min"].values
    yhat = df_fc_test.loc[mask, "yhat"].values
    print("Prophet TEST forecast metrics:",
          {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE": mape(y, yhat), "SMAPE": smape(y, yhat)})

    plot_prophet_forecast(df_fc_test, ROOT / "reports/prophet_demand_vs_forecast_TEST.png")

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

    plot_series_by_policy(sim_map, "containers", "Containers Over Time (Prophet policies, TEST)", "containers", ROOT / "reports/prophet_containers_over_time_TEST.png")
    plot_series_by_policy(sim_map, "lat_ms", "Latency Over Time (Prophet policies, TEST)", "latency (ms)", ROOT / "reports/prophet_latency_over_time_TEST.png")
    plot_policy_bars(score_map, "Policy Comparison (Prophet, TEST)", ROOT / "reports/prophet_policy_bars_TEST.png")

    return results

def run_lstm_pipeline(df, train_days=7):
    results = []
    df_train, df_test = time_split(df, train_days=train_days)

    print("\n=== [LSTM] Training (train window)… ===")
    model, cfg = train_lstm(df_train, past=60, ahead=15, epochs=5, batch_size=128)

    print("[LSTM] Generating rolling forecasts on TEST…")
    history = df_train.copy()
    rows = []
    for i in range(len(df_test)):
        yhat = predict_lstm(model, history, cfg)
        rows.append({"timestamp": df_test["timestamp"].iloc[i], "yhat": yhat})
        history = pd.concat([history, df_test.iloc[i:i+1]], ignore_index=True)

    lstm_fc_test = pd.DataFrame(rows)
    df_eval = df_test.merge(lstm_fc_test, on="timestamp", how="left")
    plot_lstm_forecast(df_eval, ROOT / "reports/lstm_demand_vs_forecast_TEST.png")

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

    plot_series_by_policy(sim_map, "containers", "Containers Over Time (LSTM policies, TEST)", "containers", ROOT / "reports/lstm_containers_over_time_TEST.png")
    plot_series_by_policy(sim_map, "lat_ms", "Latency Over Time (LSTM policies, TEST)", "latency (ms)", ROOT / "reports/lstm_latency_over_time_TEST.png")
    plot_policy_bars(score_map, "Policy Comparison (LSTM, TEST)", ROOT / "reports/lstm_policy_bars_TEST.png")

    return results

def main():
    ensure_dirs()

    # 1) Data (generate from real market data)
    market_data_filepath = ROOT / "data" / "1_min_SPY_2008-2021.csv"
    df = generate_trading_workload(csv_path=str(market_data_filepath))
    
    # 2) Run pipelines on train/test split
    all_results = []
    
    # --- CHANGE 2: USE A SMALLER, GUARANTEED-TO-WORK TRAINING PERIOD ---
    TRAIN_DAYS = 14

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
