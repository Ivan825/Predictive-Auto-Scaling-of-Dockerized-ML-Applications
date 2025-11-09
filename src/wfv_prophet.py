# src/wfv_prophet.py
import argparse
import os
import pandas as pd
from prophet import Prophet

# If your project has a helper to run prophet per fold, import that; else implement below.
def run_wfv(load_csv, results_dir, initial_window_days=14, period_days=7, horizon_minutes=60, max_folds=None, downsample=None,
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, seasonality_mode='additive',
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_fourier_order=10):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(load_csv, parse_dates=['datetime']).set_index('datetime')
    if downsample:
        df = df.resample(downsample).sum()
    series = df['load']
    # create Prophet with provided hyperparams (example)
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality)
    # If you use fourier_order, you'd add custom seasonalities; omitted for brevity.
    # implement per-fold training/generation similar to LSTM wrapper.
    # Save a placeholder metrics file so tuner can read it if required
    pd.DataFrame([{"mape":0.0,"rmse":0.0}]).to_csv(os.path.join(results_dir,"prophet_wfv_metrics.csv"), index=False)
    print("[prophet_wfv] placeholder run complete.")

def parse_args_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_csv", default="data/load_series.csv")
    parser.add_argument("--results_dir", default="results/prophet_wfv")
    parser.add_argument("--initial_days", type=int, default=14)
    parser.add_argument("--period_days", type=int, default=7)
    parser.add_argument("--horizon_minutes", type=int, default=60)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--downsample", default=None)

    # Prophet hyperparams (exposed for tuner)
    parser.add_argument("--changepoint_prior_scale", type=float, default=0.05)
    parser.add_argument("--seasonality_prior_scale", type=float, default=10.0)
    parser.add_argument("--seasonality_mode", type=str, default="additive")
    parser.add_argument("--yearly_seasonality", type=int, default=1)
    parser.add_argument("--weekly_seasonality", type=int, default=1)
    parser.add_argument("--daily_seasonality", type=int, default=0)
    parser.add_argument("--seasonality_fourier_order", type=int, default=10)

    args = parser.parse_args()
    run_wfv(load_csv=args.load_csv,
            results_dir=args.results_dir,
            initial_window_days=args.initial_days,
            period_days=args.period_days,
            horizon_minutes=args.horizon_minutes,
            max_folds=args.max_folds,
            downsample=args.downsample,
            changepoint_prior_scale=args.changepoint_prior_scale,
            seasonality_prior_scale=args.seasonality_prior_scale,
            seasonality_mode=args.seasonality_mode,
            yearly_seasonality=bool(args.yearly_seasonality),
            weekly_seasonality=bool(args.weekly_seasonality),
            daily_seasonality=bool(args.daily_seasonality),
            seasonality_fourier_order=args.seasonality_fourier_order)

if __name__ == "__main__":
    parse_args_and_run()
