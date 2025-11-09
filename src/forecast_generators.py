# src/forecast_generators.py
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def prophet_forecast_from_model(model_pkl_path, horizon_minutes=60):
    m = joblib.load(model_pkl_path)
    future = m.make_future_dataframe(periods=horizon_minutes, freq='T', include_history=False)
    fc = m.predict(future)
    return pd.Series(fc['yhat'].values, index=pd.to_datetime(fc['ds']))

def lstm_forecast_from_files(model_h5_path, scaler_pkl_path, load_csv, horizon_minutes=60, window=60):
    model = load_model(model_h5_path)
    scaler = joblib.load(scaler_pkl_path)
    df = pd.read_csv(load_csv, parse_dates=['datetime']).set_index('datetime')
    arr = df['load'].values.astype(float)
    s = scaler.transform(arr.reshape(-1,1)).flatten()
    seed = list(s[-window:])
    preds_s = []
    for _ in range(horizon_minutes):
        x = np.array(seed[-window:]).reshape((1,window,1))
        p_s = float(model.predict(x, verbose=0)[0,0])
        preds_s.append(p_s)
        seed.append(p_s)
    preds = scaler.inverse_transform(np.array(preds_s).reshape(-1,1)).flatten()
    last_ts = df.index.max()
    idx = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), periods=horizon_minutes, freq='T')
    return pd.Series(preds, index=idx)
