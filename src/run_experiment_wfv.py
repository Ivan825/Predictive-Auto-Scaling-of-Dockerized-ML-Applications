# src/run_experiment_wfv.py  (patched: simulate both models if both paths provided)
import os
import argparse
import pandas as pd
from data_loader_kaggle import load_spy_csv, save_load_series
from wfv_prophet import run_wfv as run_prophet_wfv
from wfv_lstm import run_wfv_lstm as run_lstm_wfv
from forecast_generators import prophet_forecast_from_model, lstm_forecast_from_files
from simulator_wrapper import recommend_capacity, simulate_policy, reactive_policy, buffered_policy
from simulator_fast import simulate_policy_fast_numba

def run_sim_for_model(df, fc, model_tag, args, cap):
    """
    Run simulation for forecast series fc and save results with model_tag suffix.
    model_tag: string like 'prophet' or 'lstm'
    """
    sim_reactive = simulate_policy(df['load'], fc, reactive_policy, capacity_per_container=cap)
    sim_buffered = simulate_policy(df['load'], fc, lambda c,p,n,f,l: buffered_policy(c,p,n,f,l,buffer=args.buffer_size), capacity_per_container=cap)
    out_re = f"results/sim_reactive_{model_tag}.csv"
    out_buf = f"results/sim_buffered_{model_tag}.csv"
    sim_reactive.to_csv(out_re)
    sim_buffered.to_csv(out_buf)
    print(f"[run] Saved {out_re} and {out_buf}")

def main(args):
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    if args.build_load:
        s = load_spy_csv(args.csv_path)
        save_load_series(s, args.output_csv)

    if args.run_prophet_wfv:
        run_prophet_wfv(args.output_csv, args.prophet_results_dir, initial_window_days=args.initial_days, period_days=args.period_days, horizon_minutes=args.horizon_minutes, max_folds=args.max_folds, downsample=args.downsample)

    if args.run_lstm_wfv:
        run_lstm_wfv(args.output_csv, args.lstm_results_dir, initial_window_days=args.initial_days, period_days=args.period_days, horizon_minutes=args.horizon_minutes, window=args.lstm_window, epochs=args.lstm_epochs, batch_size=args.lstm_batch, max_folds=args.max_folds, downsample=args.downsample)

    # Load historical load series
    df = pd.read_csv(args.output_csv, parse_dates=['datetime']).set_index('datetime')

    # Capacity selection
    cap = args.capacity if args.capacity and args.capacity > 0 else recommend_capacity(df['load'], desired_avg_containers=args.desired_avg_containers)

    # If both model paths provided, run both; otherwise run the provided one; otherwise fallback to last-val forecast
    ran_any = False

    # Prophet
    if args.prophet_model_path:
        print("[run] Generating forecast from Prophet model:", args.prophet_model_path)
        fc_prophet = prophet_forecast_from_model(args.prophet_model_path, horizon_minutes=args.horizon_minutes)
        # align/resample handled in simulate call (simulate_policy is robust)
        run_sim_for_model(df, fc_prophet, model_tag="prophet", args=args, cap=cap)
        ran_any = True

    # LSTM
    if args.lstm_model_path and args.lstm_scaler_path:
        print("[run] Generating forecast from LSTM model:", args.lstm_model_path)
        fc_lstm = lstm_forecast_from_files(args.lstm_model_path, args.lstm_scaler_path, args.output_csv, horizon_minutes=args.horizon_minutes, window=args.lstm_window)
        run_sim_for_model(df, fc_lstm, model_tag="lstm", args=args, cap=cap)
        ran_any = True

    # If neither model provided, fallback to simple last-value forecast and simulate once
    if not ran_any:
        print("[run] No model path provided — using last-value persistent forecast fallback.")
        last_val = df['load'].iloc[-1]
        idx = pd.date_range(start=df.index.max() + pd.Timedelta(minutes=1), periods=args.horizon_minutes, freq='T')
        fc = pd.Series([last_val]*len(idx), index=idx)
        run_sim_for_model(df, fc, model_tag="fallback", args=args, cap=cap)

    print("[run] All requested simulations completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/kaggle_intraday/intraday_spy_1min.csv")
    parser.add_argument("--build_load", action='store_true')
    parser.add_argument("--output_csv", default="data/load_series.csv")
    parser.add_argument("--run_prophet_wfv", action='store_true')
    parser.add_argument("--prophet_results_dir", default="results/prophet_wfv")
    parser.add_argument("--run_lstm_wfv", action='store_true')
    parser.add_argument("--lstm_results_dir", default="results/lstm_wfv")
    parser.add_argument("--initial_days", type=int, default=14)
    parser.add_argument("--period_days", type=int, default=7)
    parser.add_argument("--horizon_minutes", type=int, default=60)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--downsample", default=None)
    parser.add_argument("--lstm_window", type=int, default=60)
    parser.add_argument("--lstm_epochs", type=int, default=5)
    parser.add_argument("--lstm_batch", type=int, default=128)
    parser.add_argument("--prophet_model_path", default=None)
    parser.add_argument("--lstm_model_path", default=None)
    parser.add_argument("--lstm_scaler_path", default=None)
    parser.add_argument("--capacity", type=float, default=-1.0)
    parser.add_argument("--desired_avg_containers", type=int, default=5)
    parser.add_argument("--buffer_size", type=int, default=1)
    args = parser.parse_args()
    main(args)
