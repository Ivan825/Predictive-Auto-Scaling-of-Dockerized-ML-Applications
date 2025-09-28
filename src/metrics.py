import numpy as np, pandas as pd

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat)**2)))
def mape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    mask = y != 0
    return float(np.mean(np.abs((yhat[mask] - y[mask]) / y[mask])) * 100)

def smape(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    mask = denom != 0
    return float(np.mean(np.abs(yhat[mask]-y[mask]) / denom[mask]) * 100)

def scaling_scores(sim_df):
    over_rate = sim_df["over"].mean()*100
    under_events = sim_df["under"].sum()
    avg_lat = sim_df["lat_ms"].mean()
    p95_lat = sim_df["lat_ms"].quantile(0.95)
    avg_cpu = sim_df["cpu_pct"].mean()
    switches = (sim_df["containers"].diff()!=0).sum()
    cost_idx = sim_df["containers"].sum()  # proxy for container-minutes
    return {
        "over_%": over_rate,
        "under_events": int(under_events),
        "avg_latency_ms": float(avg_lat),
        "p95_latency_ms": float(p95_lat),
        "avg_cpu_%": float(avg_cpu),
        "switches": int(switches),
        "cost_index": int(cost_idx)
    }
