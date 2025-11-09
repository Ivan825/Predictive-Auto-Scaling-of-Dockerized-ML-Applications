# src/simulator_fast.py
import numpy as np
import pandas as pd
from math import ceil
from typing import Callable, Optional, Tuple

def _series_to_aligned_arrays(load_series: pd.Series, forecast_series: Optional[pd.Series], period_minutes: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align load_series and forecast_series (forecast is expected to be indexed in timestamps for next periods).
    Returns (timestamps_np, load_np, forecast_next_np) where forecast_next_np[i] is forecast for the next sample period ts_i + period.
    If forecast missing, forecast_next_np will be filled with load_np (reactive fallback).
    """
    idx = load_series.index
    load_np = load_series.values.astype(float)
    # Build target next timestamps
    next_idx = idx + pd.Timedelta(minutes=period_minutes)
    # Prepare forecast series reindexed to minute freq if needed, then ffill
    if forecast_series is None or len(forecast_series) == 0:
        fc_vals = None
    else:
        # Ensure forecast_series has a DatetimeIndex
        fc = forecast_series.sort_index()
        # forward-fill to minute resolution (safe)
        try:
            fc_min = fc.asfreq('T', method='ffill')
        except Exception:
            fc_min = fc.reindex(fc.index.union(next_idx)).sort_index().ffill()
        # take forecast values for next_idx
        fc_vals = fc_min.reindex(next_idx, method='ffill').values.astype(float)
    if forecast_series is None or len(forecast_series) == 0:
        forecast_next = load_np.copy()
    else:
        # if fc_vals has NaNs, fallback to load
        fc_vals = np.array([np.nan_to_num(v, nan=np.nan) for v in fc_vals])
        mask_nan = np.isnan(fc_vals)
        fc_vals[mask_nan] = load_np[mask_nan]
        forecast_next = fc_vals
    return np.array(idx.astype('datetime64[m]')), load_np, forecast_next

# -------------------------
# Vectorized (instant scaling) simulator
# -------------------------
def simulate_policy_fast_vectorized(load_series: pd.Series,
                                    forecast_series: Optional[pd.Series],
                                    capacity_per_container: float,
                                    buffer: int = 0,
                                    period_minutes: int = 1) -> pd.DataFrame:
    """
    Fast vectorized simulator.
    - No spin-up limit: containers at time t are set instantly to policy(predicted_required).
    - Reactive: buffer=0; Buffered: buffer>0.
    - Returns DataFrame with index timestamps and columns: load, required, containers, over_provision, under_provision
    """
    ts_np, load_np, forecast_next = _series_to_aligned_arrays(load_series, forecast_series, period_minutes=period_minutes)

    # compute required & predicted_required arrays (use ceil and ensure >=0)
    cap = float(max(1e-9, capacity_per_container))
    required = np.ceil(np.maximum(0.0, load_np) / cap).astype(int)
    pred_required = np.ceil(np.maximum(0.0, forecast_next) / cap).astype(int)

    # apply policy: containers = pred_required + buffer, at least 0
    containers = np.maximum(0, pred_required + int(buffer))

    # ensure containers >= 1 when load > 0 (consistent with your previous semantics)
    containers[(load_np > 0) & (containers == 0)] = 1

    over = np.maximum(0, containers - required)
    under = np.maximum(0, required - containers)

    df = pd.DataFrame({
        'timestamp': ts_np.astype('datetime64[m]'),
        'load': load_np,
        'required': required,
        'containers': containers,
        'over_provision': over,
        'under_provision': under
    })
    df = df.set_index('timestamp')
    return df

# -------------------------
# Numba JIT (preserves spin-up limit)
# -------------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

def simulate_policy_fast_numba(load_series: pd.Series,
                               forecast_series: Optional[pd.Series],
                               capacity_per_container: float,
                               buffer: int = 0,
                               period_minutes: int = 1,
                               spin_up_rate: int = 1,
                               start_containers: int = 0) -> pd.DataFrame:
    """
    Fast simulator using Numba if available. Preserves the "at most spin_up_rate containers can be added per step" rule.
    If numba isn't installed, falls back to a pure python accelerated loop (still faster than original because of numpy arrays).
    """
    ts_np, load_np, forecast_next = _series_to_aligned_arrays(load_series, forecast_series, period_minutes=period_minutes)
    cap = float(max(1e-9, capacity_per_container))
    required = np.ceil(np.maximum(0.0, load_np) / cap).astype(np.int64)
    pred_required = np.ceil(np.maximum(0.0, forecast_next) / cap).astype(np.int64)
    # ensure shapes match
    n = len(required)
    containers_out = np.zeros(n, dtype=np.int64)
    over_out = np.zeros(n, dtype=np.int64)
    under_out = np.zeros(n, dtype=np.int64)

    if NUMBA_AVAILABLE:
        return _simulate_numba_loop(ts_np, load_np, required, pred_required, containers_out, over_out, under_out, spin_up_rate, start_containers, buffer)
    else:
        # fallback pure-python loop over numpy arrays (still faster than pandas because of vectorized ops)
        cont = int(start_containers)
        for i in range(n):
            loadv = float(load_np[i])
            req = int(required[i])
            pred = int(pred_required[i])
            desired = max(0, pred + int(buffer))
            # spin-up: can increase by at most spin_up_rate per step
            if desired > cont:
                cont = min(desired, cont + spin_up_rate)
            else:
                cont = desired
            if loadv > 0 and cont == 0:
                cont = 1
            containers_out[i] = cont
            over_out[i] = max(0, cont - req)
            under_out[i] = max(0, req - cont)
        df = pd.DataFrame({
            'timestamp': ts_np.astype('datetime64[m]'),
            'load': load_np,
            'required': required,
            'containers': containers_out,
            'over_provision': over_out,
            'under_provision': under_out
        }).set_index('timestamp')
        return df

# Numba-compiled loop function
if NUMBA_AVAILABLE:
    @njit
    def _simulate_numba_loop(ts_np, load_np, required, pred_required, containers_out, over_out, under_out, spin_up_rate, start_containers, buffer):
        n = len(required)
        cont = start_containers
        for i in range(n):
            loadv = load_np[i]
            req = int(required[i])
            pred = int(pred_required[i])
            desired = pred + int(buffer)
            if desired > cont:
                # increase gradually
                if cont + spin_up_rate < desired:
                    cont = cont + spin_up_rate
                else:
                    cont = desired
            else:
                cont = desired
            if loadv > 0 and cont == 0:
                cont = 1
            containers_out[i] = cont
            over_out[i] = cont - req if cont - req > 0 else 0
            under_out[i] = req - cont if req - cont > 0 else 0
        # build dataframe after returning to python
        import pandas as pd
        df = pd.DataFrame({
            'timestamp': ts_np.astype('datetime64[m]'),
            'load': load_np,
            'required': required,
            'containers': containers_out,
            'over_provision': over_out,
            'under_provision': under_out
        }).set_index('timestamp')
        return df

