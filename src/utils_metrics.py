# src/utils_metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def smape(A, F):
    A = np.array(A).astype(float)
    F = np.array(F).astype(float)
    denom = (np.abs(A) + np.abs(F)) / 2.0
    diff = np.abs(A - F)
    denom[denom == 0] = 1.0
    return float(np.mean(diff / denom) * 100.0)

def mape(A, F):
    A = np.array(A).astype(float)
    F = np.array(F).astype(float)
    mask = A == 0
    A_adj = A.copy()
    A_adj[mask] = 1.0
    return float(np.mean(np.abs((A - F) / A_adj)) * 100.0)

def evaluate_forecast(true_series, pred_series):
    """
    Compute MAE, RMSE, MAPE, SMAPE between true_series and pred_series.
    Uses a portable way to compute RMSE (sqrt of MSE) so it works across sklearn versions.
    """
    t = pd.Series(true_series).astype(float).values
    p = pd.Series(pred_series).astype(float).values
    mae = mean_absolute_error(t, p)
    mse = mean_squared_error(t, p)        # mean squared error (standard API)
    rmse = float(np.sqrt(mse))           # robust RMSE computation
    return {
        'mae': float(mae),
        'rmse': rmse,
        'mape': mape(t, p),
        'smape': smape(t, p)
    }
