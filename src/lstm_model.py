# src/lstm_model.py
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

def make_sequences(series, window):
    x, y = [], []
    arr = np.array(series, dtype=float)
    for i in range(len(arr) - window):
        x.append(arr[i:i+window])
        y.append(arr[i+window])
    return np.array(x), np.array(y)

def build_lstm_model(input_shape, units=64, dropout=0.1, lr=1e-4, num_layers=1):
    """
    Simple stacked LSTM builder. `num_layers` controls stacking.
    """
    inp = tf.keras.Input(shape=input_shape)
    x = inp
    # if num_layers > 1 create intermediate LSTMs with return_sequences=True
    for i in range(num_layers - 1):
        x = layers.LSTM(units, return_sequences=True)(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
    # final LSTM layer
    if num_layers > 0:
        x = layers.LSTM(units, return_sequences=False)(x)
    else:
        x = layers.Flatten()(x)
    if dropout and dropout > 0:
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, name="out")(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=lr)
    # use standard loss and metrics (avoid custom objects)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

def _ensure_parent_dir(path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def _safe_model_save(model: tf.keras.Model, model_path: str):
    """
    Save model in native Keras format (.keras) by default. If model_path ends with .h5,
    also write an .h5 for backward compatibility. Avoid passing deprecated save_format unless necessary.
    """
    # canonical .keras path
    if model_path.endswith(".h5"):
        keras_path = model_path[:-3] + ".keras"
        h5_path = model_path
    elif model_path.endswith(".keras"):
        keras_path = model_path
        h5_path = model_path[:-6] + ".h5" if model_path.endswith(".keras") else model_path + ".h5"
    else:
        keras_path = model_path + ".keras"
        h5_path = model_path + ".h5"

    _ensure_parent_dir(keras_path)
    try:
        model.save(keras_path)   # native format
    except Exception as e:
        # try to at least save h5 if native save fails
        try:
            model.save(h5_path, save_format="h5")
        except Exception as e2:
            raise RuntimeError(f"Failed to save model as .keras and .h5: {e} / {e2}")

    # attempt to also provide h5 for older readers (non-fatal)
    try:
        if not os.path.exists(h5_path):
            model.save(h5_path, save_format="h5")
    except Exception:
        # ignore; .keras is canonical
        pass

def _load_model_robust(model_path: str):
    """
    Load model trying native .keras first, then provided model_path, then .h5.
    Use compile=False to avoid metric/optimizer deserialization issues. Try a small custom_objects mapping
    if basic load fails for legacy HDF5 files.
    """
    candidates = []
    if model_path.endswith(".keras"):
        candidates = [model_path, model_path.replace(".keras", ".h5")]
    elif model_path.endswith(".h5"):
        candidates = [model_path.replace(".h5", ".keras"), model_path, model_path]
    else:
        candidates = [model_path + ".keras", model_path, model_path + ".h5"]

    last_err = None
    for cand in candidates:
        try:
            if not os.path.exists(cand):
                continue
            # always try compile=False first (safer)
            m = tf.keras.models.load_model(cand, compile=False)
            return m
        except Exception as e:
            last_err = e
            # Try with custom_objects mapping to help legacy HDF5 models referencing metric names
            try:
                custom_objects = {
                    "mse": tf.keras.losses.MeanSquaredError(),
                    "mean_squared_error": tf.keras.losses.MeanSquaredError(),
                    "mean_absolute_percentage_error": tf.keras.metrics.MeanAbsolutePercentageError(),
                    "mae": tf.keras.metrics.MeanAbsoluteError()
                }
                m = tf.keras.models.load_model(cand, compile=False, custom_objects=custom_objects)
                return m
            except Exception as e2:
                last_err = e2
                continue
    raise RuntimeError(f"Could not load model from any candidate path {candidates}. Last error: {last_err}")

def train_lstm_on_fold(train_series: pd.Series, model_path: str, scaler_path: str,
                       window=60, epochs=5, batch_size=64, units=64, dropout=0.1,
                       learning_rate=1e-4, optimizer="adam", num_layers=1, **kwargs) -> Dict[str, Any]:
    """
    Trains and saves a model + scaler.
    Accepts extra kwargs (tuner may pass optimizer name etc). Writes scaler to scaler_path (joblib).
    Returns training info dict {'final_loss':..., 'history':...}
    """
    _ensure_parent_dir(model_path)
    _ensure_parent_dir(scaler_path)

    arr = np.array(train_series).reshape(-1, 1)
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr).flatten()
    joblib.dump(scaler, scaler_path)

    X, y = make_sequences(arr_scaled, window)
    if len(X) == 0:
        raise RuntimeError("Not enough data to form sequences")

    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = build_lstm_model(input_shape=(window, 1), units=units, dropout=dropout, lr=learning_rate, num_layers=num_layers)

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="loss"),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="loss")
    ]
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)

    # Save model robustly
    _safe_model_save(model, model_path)

    final_loss = float(history.history["loss"][-1]) if history.history.get("loss") else None
    return {"final_loss": final_loss, "history": history.history}

def lstm_forecast_from_model(model_path: str, scaler_path: str, train_end: pd.Timestamp,
                             horizon_minutes: int, df: pd.DataFrame, window=60, **kwargs):
    """
    Loads model and scaler, creates forecasts for the single horizon (train_end + horizon_minutes).
    Returns a DataFrame with columns ['y_true','y_pred'] for the forecast timestamps (one or more rows depending on freq).
    Accepts **kwargs to be tolerant of extra tuner flags.
    """
    if not os.path.exists(scaler_path):
        raise RuntimeError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    model = _load_model_robust(model_path)

    # Infer df frequency; fall back to minute-level assumption
    freq = pd.infer_freq(df.index)
    if freq is None:
        # fallback: use 1 minute steps assumption
        freq_td = pd.Timedelta(minutes=1)
    else:
        # convert freq string to Timedelta if possible
        try:
            freq_td = pd.tseries.frequencies.to_offset(freq).delta
        except Exception:
            freq_td = pd.Timedelta(minutes=1)

    # Determine how many steps equal horizon
    steps = max(1, int(pd.Timedelta(minutes=horizon_minutes) / freq_td))

    # build input window: last `window` values up to train_end (inclusive)
    # if train_end not exactly present, use <= train_end
    try:
        # prefer exact location
        idx_loc = df.index.get_loc(train_end)
    except KeyError:
        # pick last index <= train_end
        idx_loc = df.index.get_indexer([train_end], method="ffill")[0]
        if idx_loc == -1:
            idx_loc = len(df) - 1

    start_idx = max(0, idx_loc - window + 1)
    window_series = df['load'].iloc[start_idx:idx_loc+1].values
    if len(window_series) < window:
        # pad with earliest value in window_series or zeros
        pad_val = window_series[0] if len(window_series) > 0 else 0.0
        pad = np.full(window - len(window_series), pad_val)
        window_series = np.concatenate([pad, window_series])

    scaled = scaler.transform(np.array(window_series).reshape(-1, 1)).flatten()
    x = scaled.reshape((1, window, 1))

    preds_scaled = []
    for step in range(steps):
        p = model.predict(x, verbose=0).ravel()[0]
        preds_scaled.append(p)
        # slide the window: drop oldest, append p (in scaled space)
        x = np.roll(x, -1, axis=1)
        x[0, -1, 0] = p

    # inverse transform predictions to original units
    inv_preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    # gather true values for next steps (if available) — otherwise NaN
    y_true = []
    for i in range(1, steps + 1):
        pos = idx_loc + i
        if pos < len(df):
            y_true.append(float(df['load'].iloc[pos]))
        else:
            y_true.append(np.nan)

    # Build timestamps for each forecast step
    forecast_times = [df.index[idx_loc] + (i * freq_td) for i in range(1, steps + 1)]

    out_df = pd.DataFrame({"y_true": y_true, "y_pred": inv_preds}, index=pd.DatetimeIndex(forecast_times))
    return out_df
