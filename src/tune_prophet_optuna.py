# src/tune_prophet_optuna.py
"""
Optuna tuning for Prophet hyperparameters using the WFV pipeline.

Requirements:
- Your wfv_prophet.py must accept the following CLI args and use them when creating the Prophet model:
    --changepoint_prior_scale (float)
    --seasonality_prior_scale (float)
    --seasonality_mode (additive|multiplicative)
    --yearly_seasonality (0|1)
    --weekly_seasonality (0|1)
    --daily_seasonality (0|1)
    --seasonality_fourier_order (int)

- The wfv_prophet script should save metrics to: <results_dir>/prophet_wfv_metrics.csv
  (This is the default of the WFV scripts in this project.)

Run:
  source .venv/bin/activate
  python src/tune_prophet_optuna.py

Tuning settings are conservative for speed (uses downsample '5T' and max_folds 10).
Adjust n_trials and WFV args for final runs.
"""

import optuna
import subprocess
import os
import pandas as pd
import json
from pathlib import Path

# Config: tweak these before running
PROJECT_ROOT = Path(".").resolve()
DATA_CSV = "data/load_series.csv"
OUT_BASE = PROJECT_ROOT / "results"
TUNE_OUT = OUT_BASE / "tune_prophet_optuna"
TUNE_OUT.mkdir(parents=True, exist_ok=True)

# WFV settings for a single trial (fast but meaningful)
WFV_COMMON_ARGS = [
    "--load_csv", DATA_CSV,
    "--initial_days", "14",
    "--period_days", "7",
    "--horizon_minutes", "60",
    "--max_folds", "10",     # limit folds for tuning speed
    "--downsample", "5T"    # coarse resolution for faster trials
]

# Where wfv_prophet will write metrics relative to the results_dir we give it.
WFV_METRICS_FILENAME = "prophet_wfv_metrics.csv"

def run_wfv_with_params(result_dir, params):
    """
    Calls wfv_prophet.py with a results_dir and a set of prophet params (dict).
    Blocks until completion. Raises CalledProcessError on non-zero exit.
    """
    results_dir_arg = str(result_dir)
    # base call; modify the path to wfv_prophet.py if needed
    cmd = ["python", "src/wfv_prophet.py", "--results_dir", results_dir_arg] + WFV_COMMON_ARGS

    # add our hyperparams as CLI flags
    cmd += ["--changepoint_prior_scale", str(params["changepoint_prior_scale"])]
    cmd += ["--seasonality_prior_scale", str(params["seasonality_prior_scale"])]
    cmd += ["--seasonality_mode", params["seasonality_mode"]]
    cmd += ["--yearly_seasonality", "1" if params["yearly_seasonality"] else "0"]
    cmd += ["--weekly_seasonality", "1" if params["weekly_seasonality"] else "0"]
    cmd += ["--daily_seasonality", "1" if params["daily_seasonality"] else "0"]
    cmd += ["--seasonality_fourier_order", str(params["seasonality_fourier_order"])]

    print("Running WFV command:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def read_wfv_metrics(result_dir):
    p = Path(result_dir) / WFV_METRICS_FILENAME
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # if file created, return aggregated metrics: mean mape and mean rmse
    return {
        "mape_mean": float(df["mape"].mean()),
        "rmse_mean": float(df["rmse"].mean())
    }

def objective(trial):
    # Search space
    cps = trial.suggest_loguniform("changepoint_prior_scale", 1e-3, 0.5)
    sps = trial.suggest_loguniform("seasonality_prior_scale", 1e-2, 10.0)
    mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
    yearly = trial.suggest_categorical("yearly_seasonality", [0, 1])
    weekly = trial.suggest_categorical("weekly_seasonality", [0, 1])
    daily = trial.suggest_categorical("daily_seasonality", [0, 1])
    fourier = trial.suggest_int("seasonality_fourier_order", 3, 30)

    params = {
        "changepoint_prior_scale": cps,
        "seasonality_prior_scale": sps,
        "seasonality_mode": mode,
        "yearly_seasonality": bool(yearly),
        "weekly_seasonality": bool(weekly),
        "daily_seasonality": bool(daily),
        "seasonality_fourier_order": fourier
    }

    trial_dir = TUNE_OUT / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_wfv_with_params(trial_dir, params)
    except subprocess.CalledProcessError as e:
        # if the trial fails, return a large objective value
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")

    metrics = read_wfv_metrics(trial_dir)
    if metrics is None:
        print("No metrics found for trial", trial.number)
        return float("inf")

    # Primary objective: minimize mean MAPE
    objective_value = metrics["mape_mean"]

    # Log trial info in trial user attrs for easier debugging later
    trial.set_user_attr("metrics", metrics)
    trial.set_user_attr("params", params)

    print(f"Trial {trial.number} -> MAPE mean: {metrics['mape_mean']:.4f}, RMSE mean: {metrics['rmse_mean']:.4f}")
    return objective_value

def main():
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    n_trials = 40  # start moderate; increase when you have resources
    study.optimize(objective, n_trials=n_trials)

    # save best params and full trials dataframe
    best = study.best_params
    print("BEST:", best, "value:", study.best_value)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(TUNE_OUT / "tune_prophet_optuna_trials.csv", index=False)

    # also save human-readable best params
    with open(TUNE_OUT / "best_prophet_params.json", "w") as fh:
        json.dump(best, fh, indent=2)

    print("All trial info saved to", TUNE_OUT)

if __name__ == "__main__":
    main()
