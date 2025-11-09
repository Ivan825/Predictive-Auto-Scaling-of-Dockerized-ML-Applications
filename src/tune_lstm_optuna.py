# src/tune_lstm_optuna.py
import optuna
import json
import os
from datetime import datetime
from wfv_lstm import run_wfv  # imports run_wfv function from wfv_lstm.py

RESULTS_ROOT = "results"

def objective(trial):
    # search space (example)
    params = {
        "window": trial.suggest_categorical("window", [30, 60, 120]),
        "units": trial.suggest_categorical("units", [32, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128, 256]),
        "epochs": 10,
        # other static args for run_wfv
        "load_csv": "data/load_series.csv",
        "initial_days": 14,
        "period_days": 7,
        "horizon_minutes": 60,
        "max_folds": 10,
        "downsample": "5T",
    }

    # create trial results folder
    trial_name = f"lstm_tune_{trial.number}"
    outdir = os.path.join(RESULTS_ROOT, trial_name)
    os.makedirs(outdir, exist_ok=True)

    # save trial params to JSON for reproducibility
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Call the training/validation runner directly and get metrics
    try:
        metrics_per_fold = run_wfv(
            load_csv=params["load_csv"],
            results_dir=outdir,
            initial_days=params["initial_days"],
            period_days=params["period_days"],
            horizon_minutes=params["horizon_minutes"],
            max_folds=params["max_folds"],
            downsample=params["downsample"],
            window=params["window"],
            epochs=params["epochs"],
            batch=params["batch"],
            lstm_units=params["units"],
            lstm_dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )
    except Exception as e:
        # tell Optuna the trial failed
        print("Trial failed with exception:", e)
        raise optuna.exceptions.TrialPruned()

    # metrics_per_fold should be a list of dicts: [{'fold':1,'mape':..,'rmse':..,'loss':..},...]
    # compute aggregate objective (mean MAPE)
    mape_vals = [m["mape"] for m in metrics_per_fold if m.get("mape") is not None]
    if not mape_vals:
        # bad trial (no valid metrics)
        return float("inf")
    mean_mape = sum(mape_vals) / len(mape_vals)
    return mean_mape


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial.params)
    print("Best value (mean MAPE):", study.best_value)
