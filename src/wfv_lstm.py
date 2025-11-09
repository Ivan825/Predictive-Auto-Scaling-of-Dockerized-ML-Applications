# src/wfv_lstm.py
"""
Walk-Forward Validation wrapper for LSTM.
This script is intentionally robust: it accepts extra hyperparameter CLI flags (used by the tuner)
and forwards them to the LSTM training/inference functions.

Expected behaviour:
 - Reads a CSV timeseries (--load_csv)
 - Runs WFV folds (initial_days, period_days, etc.)
 - Trains an LSTM per fold with provided hyperparameters
 - Saves forecasts and fold metrics into results_dir (same layout as other WFV scripts)
"""
import inspect
import argparse
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ---- IMPORT YOUR LSTM TRAINING/FORECASTING UTILITIES ----
# If you have an existing module (e.g., lstm_model.py) with functions like:
#   train_lstm_on_fold(train_series, model_path, scaler_path, ...)
#   lstm_forecast_from_model(model_path, scaler_path, train_end, ...)
# import them here. If not, fallback placeholders below will raise helpful errors.
try:
    # expected to exist in your project
    from lstm_model import train_lstm_on_fold, lstm_forecast_from_model
except Exception:
    # fallback placeholders: the script will error if these aren't implemented.
    def train_lstm_on_fold(*args, **kwargs):
        raise RuntimeError("train_lstm_on_fold not found - please implement or import correctly.")

    def lstm_forecast_from_model(*args, **kwargs):
        raise RuntimeError("lstm_forecast_from_model not found - please implement or import correctly.")
# ---------------------------------------------------------

def _mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def _rmse(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _safe_call_train(func, kw):
    """
    Call `func(**filtered_kw)` where filtered_kw contains only the names present in func's signature.
    Also supports passing the full kwargs if func accepts **kwargs.
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # If the target accepts **kwargs, pass everything.
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return func(**kw)

    # otherwise filter down to accepted parameters only
    accepted = {k: v for k, v in kw.items() if k in params}
    return func(**accepted)

def run_wfv(load_csv, results_dir, initial_window_days=14, initial_days=None,
            period_days=7, horizon_minutes=60,
            window=60, epochs=5, batch=128, max_folds=None, downsample=None,
            lstm_units=64, lstm_dropout=0.0, learning_rate=1e-3, optimizer="adam", num_layers=1, **kwargs):
    """
    Run walk-forward validation and write fold forecasts + metrics into results_dir.
    Accepts both `initial_window_days` and `initial_days` (for backwards compatibility).
    Also swallows extra kwargs so tuners passing extra flags won't crash.
    """
    # prefer explicit initial_days if provided (some tuner wrappers call it `initial_days`)
    if initial_days is not None:
        initial_window_days = initial_days

    _ensure_dir(results_dir)
    # save run params for reproducibility
    run_params = dict(
        load_csv=load_csv, initial_window_days=initial_window_days, period_days=period_days,
        horizon_minutes=horizon_minutes, window=window, epochs=epochs, batch=batch,
        max_folds=max_folds, downsample=downsample,
        lstm_units=lstm_units, lstm_dropout=lstm_dropout, learning_rate=learning_rate,
        optimizer=optimizer, num_layers=num_layers
    )
    try:
        with open(os.path.join(results_dir, "run_params.json"), "w") as fh:
            json.dump(run_params, fh, indent=2)
    except Exception:
        # non-fatal: continue even if params can't be written
        pass

    df = pd.read_csv(load_csv, parse_dates=['datetime']).set_index('datetime')
    if downsample:
        # bind to resample safely
        try:
            df = df.resample(downsample).sum()
        except Exception as e:
            print("[lstm_wfv] Downsample failed, continuing with original resolution:", e)

    series = df['load']
    # compute fold boundaries
    start = series.index.min()
    # initial end:
    train_end = start + pd.Timedelta(days=initial_window_days)
    fold = 0
    all_metrics = []

    # ensure metrics file exists (so tuner/readers at least find a file)
    metrics_path = os.path.join(results_dir, "lstm_wfv_metrics.csv")
    pd.DataFrame([], columns=["fold", "mape", "rmse", "loss"]).to_csv(metrics_path, index=False)

    # iterate folds until horizon goes beyond last timestamp
    while train_end + pd.Timedelta(minutes=horizon_minutes) <= series.index.max():
        fold += 1
        if max_folds and fold > int(max_folds):
            break

        print(f"[lstm_wfv] Fold {fold}: train_end {train_end}")
        # define train and test slices
        train_slice = series[:train_end]
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=period_days)

        # file paths for fold artifacts
        model_path = os.path.join(results_dir, f"lstm_model_fold_{fold}.h5")
        scaler_path = os.path.join(results_dir, f"lstm_scaler_fold_{fold}.pkl")

        # default fold metric values (set to NaN on failure)
        fold_metrics = {"fold": fold, "mape": np.nan, "rmse": np.nan, "loss": np.nan}

        # Prepare kwargs with aliases for training call
        train_call_kwargs = dict(
            train_series=train_slice,
            model_path=model_path,
            scaler_path=scaler_path,
            window=window,
            epochs=epochs,
            batch_size=batch,   # many implementations expect 'batch_size'
            batch=batch,        # alias if implementation expects 'batch'
            units=lstm_units,
            lstm_units=lstm_units,  # alias
            dropout=lstm_dropout,
            learning_rate=learning_rate,
            optimizer=optimizer,
            num_layers=num_layers,
            # include any extra kwargs from outer call so tuners' flags are available to forward if desired
            **kwargs
        )

        # TRAIN: call user's train function; catch errors and continue
        try:
            training_info = _safe_call_train(train_lstm_on_fold, train_call_kwargs)
        except Exception as e:
            print(f"[lstm_wfv] train_lstm_on_fold failed for fold {fold}: {e}")
            # write intermediate metrics and continue to next fold
            all_metrics.append(fold_metrics)
            try:
                pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            except Exception as e2:
                print(f"[lstm_wfv] Failed to write metrics csv after training failure: {e2}")
            train_end = train_end + pd.Timedelta(days=period_days)
            continue

        # FORECAST: call user's forecasting helper; catch errors and continue
        try:
            # call with the richer signature first (most helper implementations accept these)
            try:
                fc = lstm_forecast_from_model(model_path, scaler_path, train_end, horizon_minutes=horizon_minutes, df=df, window=window)
            except TypeError:
                # fallback if user func has different signature: try minimal call
                fc = lstm_forecast_from_model(model_path, scaler_path, train_end)
        except Exception as e:
            print(f"[lstm_wfv] lstm_forecast_from_model failed for fold {fold}: {e}")
            all_metrics.append(fold_metrics)
            try:
                pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            except Exception as e2:
                print(f"[lstm_wfv] Failed to write metrics csv after forecast failure: {e2}")
            train_end = train_end + pd.Timedelta(days=period_days)
            continue

        # Normalize fc into a DataFrame with y_true & y_pred (flexible about user's return type)
        try:
            if isinstance(fc, pd.Series):
                fc_df = fc.to_frame(name="forecast")
            elif isinstance(fc, pd.DataFrame):
                fc_df = fc.copy()
            else:
                # if it's array-like or list-like try to coerce
                try:
                    fc_df = pd.DataFrame(fc)
                except Exception:
                    raise RuntimeError("Forecast return type not recognized (not Series/DataFrame/array-like)")
        except Exception as e:
            print(f"[lstm_wfv] Could not coerce forecast to DataFrame for fold {fold}: {e}")
            all_metrics.append(fold_metrics)
            try:
                pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
            except Exception as e2:
                print(f"[lstm_wfv] Failed to write metrics csv after coercion failure: {e2}")
            train_end = train_end + pd.Timedelta(days=period_days)
            continue

        # Save raw forecast to CSV (make header friendly)
        fc_out_path = os.path.join(results_dir, f"lstm_fold_{fold}_forecast.csv")
        try:
            if isinstance(fc_df.index, pd.DatetimeIndex):
                fc_df.to_csv(fc_out_path, index=True)
            else:
                fc_df.to_csv(fc_out_path, index=True)
        except Exception as e:
            print(f"[lstm_wfv] Failed to save forecast for fold {fold}: {e}")

        # Attempt to align true values and predictions
        y_true = None
        y_pred = None
        try:
            # common column name heuristics
            if "y_true" in fc_df.columns and "y_pred" in fc_df.columns:
                y_true = fc_df["y_true"].values
                y_pred = fc_df["y_pred"].values
            elif "true" in fc_df.columns and "pred" in fc_df.columns:
                y_true = fc_df["true"].values
                y_pred = fc_df["pred"].values
            elif "load" in fc_df.columns and "forecast" in fc_df.columns:
                y_true = fc_df["load"].values
                y_pred = fc_df["forecast"].values
            elif "forecast" in fc_df.columns and isinstance(fc_df.index, pd.DatetimeIndex):
                # try to pick matching true values from series for the forecast timestamps
                idxs = fc_df.index
                y_pred = fc_df["forecast"].values
                # look up true values from original series for these timestamps (if present)
                y_true = series.reindex(idxs).values
            elif fc_df.shape[1] >= 2:
                # fallback: first column true, second column pred
                y_true = fc_df.iloc[:, 0].values
                y_pred = fc_df.iloc[:, 1].values
            elif fc_df.shape[1] == 1:
                # no true values returned, attempt to get true values from series after train_end
                if isinstance(fc_df.index, pd.DatetimeIndex):
                    idxs = fc_df.index
                    y_pred = fc_df.iloc[:, 0].values
                    y_true = series.reindex(idxs).values
                else:
                    # cannot align
                    y_true = None
                    y_pred = fc_df.iloc[:, 0].values
            else:
                y_true = None
                y_pred = None
        except Exception as e:
            print(f"[lstm_wfv] Error extracting y_true/y_pred for fold {fold}: {e}")
            y_true, y_pred = None, None

        # compute metrics if possible
        try:
            if y_pred is not None and y_true is not None and len(y_true) > 0:
                # drop nan pairs
                mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                if mask.sum() > 0:
                    y_true_clean = np.array(y_true)[mask]
                    y_pred_clean = np.array(y_pred)[mask]
                    fold_metrics["mape"] = _mape(y_true_clean, y_pred_clean)
                    fold_metrics["rmse"] = _rmse(y_true_clean, y_pred_clean)
                else:
                    print(f"[lstm_wfv] After removing NaNs there are no valid y_true/y_pred pairs for fold {fold}.")
            else:
                print(f"[lstm_wfv] Could not compute metrics for fold {fold}: y_true or y_pred missing.")
        except Exception as e:
            print(f"[lstm_wfv] Exception computing metrics for fold {fold}: {e}")

        # try to capture training loss if training_info returned it
        try:
            if 'training_info' in locals() and isinstance(training_info, dict):
                # many train functions return {'final_loss':..., ...} or history dict
                if "final_loss" in training_info and training_info["final_loss"] is not None:
                    fold_metrics["loss"] = float(training_info["final_loss"])
                elif "loss" in training_info and isinstance(training_info["loss"], (int, float)):
                    fold_metrics["loss"] = float(training_info["loss"])
                elif "history" in training_info and isinstance(training_info["history"], dict) and "loss" in training_info["history"]:
                    hist_loss = training_info["history"]["loss"]
                    if isinstance(hist_loss, (list, tuple)) and len(hist_loss) > 0:
                        fold_metrics["loss"] = float(hist_loss[-1])
            else:
                # if training_info isn't a dict or not present, ignore
                pass
        except Exception:
            # non-fatal
            pass

        # append and persist metrics after each fold (so partial result exists on crash)
        all_metrics.append(fold_metrics)
        try:
            pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
        except Exception as e:
            print(f"[lstm_wfv] Failed to write metrics csv: {e}")

        # advance train_end by period_days
        train_end = train_end + pd.Timedelta(days=period_days)

    # final message & ensure metrics path exists (if nothing succeeded write NaN row)
    metrics_df = pd.DataFrame(all_metrics)
    if metrics_df.empty:
        pd.DataFrame([{"mape": np.nan, "rmse": np.nan}]).to_csv(metrics_path, index=False)
    else:
        try:
            metrics_df.to_csv(metrics_path, index=False)
        except Exception as e:
            print(f"[lstm_wfv] Failed to write final metrics csv: {e}")

    print("[lstm_wfv] Saved metrics to", metrics_path)
    return all_metrics


def parse_args_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_csv", default="data/load_series.csv")
    parser.add_argument("--results_dir", default="results/lstm_wfv")
    parser.add_argument("--initial_days", type=int, default=14)
    parser.add_argument("--period_days", type=int, default=7)
    parser.add_argument("--horizon_minutes", type=int, default=60)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--max_folds", type=int, default=None)
    parser.add_argument("--downsample", default=None)

    # NEW hyperparameter CLI flags for tuning
    parser.add_argument("--lstm_units", type=int, default=64, help="Number of hidden units in LSTM")
    parser.add_argument("--lstm_dropout", type=float, default=0.0, help="Dropout for LSTM layers")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer name (adam, rmsprop, etc.)")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of stacked LSTM layers")

    args = parser.parse_args()
    run_wfv(load_csv=args.load_csv,
            results_dir=args.results_dir,
            initial_window_days=args.initial_days,
            period_days=args.period_days,
            horizon_minutes=args.horizon_minutes,
            window=args.window,
            epochs=args.epochs,
            batch=args.batch,
            max_folds=args.max_folds,
            downsample=args.downsample,
            lstm_units=args.lstm_units,
            lstm_dropout=args.lstm_dropout,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            num_layers=args.num_layers)

if __name__ == "__main__":
    parse_args_and_run()
