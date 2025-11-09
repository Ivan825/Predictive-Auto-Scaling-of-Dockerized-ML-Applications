# src/simulator_wrapper.py
import pandas as pd
from math import ceil
import os

def recommend_capacity(load_series, desired_avg_containers=5):
    """
    Recommend capacity per *time-step* of load_series (units = same as series).
    If load_series uses downsampling (e.g. 5T), average is per 5-minute bin.
    The returned capacity is in the same units (load units per sample period).
    """
    # load_series expected to be a pandas Series indexed by datetime
    avg_vol = float(load_series.mean())
    # Determine sample period in minutes (fallback to 1 if cannot determine)
    period_minutes = 1
    if hasattr(load_series.index, "to_series") and len(load_series.index) >= 2:
        try:
            diffs = load_series.index.to_series().diff().dropna()
            period_minutes = int(diffs.median() / pd.Timedelta(minutes=1))
            if period_minutes < 1:
                period_minutes = 1
        except Exception:
            period_minutes = 1
    cap = float(max(1.0, avg_vol / float(max(1, desired_avg_containers))))
    print(f"[simulator] Sample period (minutes): {period_minutes}, Avg load per sample: {avg_vol:.2f}")
    print(f"[simulator] Recommended capacity per container (per sample period): {cap:.2f}")
    return cap

def load_to_required_containers(load_value, capacity_per_container):
    """
    Compute required containers to serve load_value for a single sample period,
    returning at least 1 if load_value > 0 (so we don't simulate 0 containers while there's positive load).
    """
    required = int(ceil(max(0.0, float(load_value)) / float(max(1e-9, capacity_per_container))))
    if load_value > 0 and required == 0:
        required = 1
    return required

def simulate_policy(load_series, forecast_series, policy_fn, capacity_per_container, spin_up_delay=0, start_containers=0):
    """
    Simulate scaling policy over load_series (pandas Series indexed by datetime).
    forecast_series is a pandas Series indexed by datetimes (may be future timestamps).
    policy_fn signature: policy_fn(current_containers, predicted_required, now_ts, forecast_ts, load)
    Returns a DataFrame indexed by timestamp with columns:
      ['load','required','containers','over_provision','under_provision']
    """
    timestamps = list(load_series.index)
    records = []
    containers = int(start_containers)

    # Determine sample period in minutes (support downsampled series)
    if len(timestamps) >= 2:
        period_minutes = int((timestamps[1] - timestamps[0]).total_seconds() / 60)
        if period_minutes < 1:
            period_minutes = 1
    else:
        period_minutes = 1

    # Ensure forecast_series is a Series (may be empty)
    fc = forecast_series if isinstance(forecast_series, pd.Series) else pd.Series(forecast_series)

    for ts in timestamps:
        load = float(load_series.loc[ts])
        # required for this timestamp (how many containers are needed now)
        required_now = load_to_required_containers(load, capacity_per_container)

        # The forecast we want is for the next sample period (ts + period)
        next_ts = ts + pd.Timedelta(minutes=period_minutes)

        # Try to get forecast at next_ts; if not present, fallback to required_now (reactive)
        pred_required = None
        if (not fc.empty) and (next_ts in fc.index):
            try:
                pred_required = load_to_required_containers(float(fc.loc[next_ts]), capacity_per_container)
            except Exception:
                pred_required = None

        # If exact next_ts not found, try nearest with forward-fill within one sample period
        if pred_required is None and (not fc.empty):
            try:
                # try to find latest forecast <= next_ts
                idxs = fc.index[fc.index <= next_ts]
                if len(idxs) > 0:
                    chosen = idxs[-1]
                    pred_required = load_to_required_containers(float(fc.loc[chosen]), capacity_per_container)
            except Exception:
                pred_required = None

        if pred_required is None:
            pred_required = required_now

        desired = int(policy_fn(containers, pred_required, ts, next_ts, load))
        # Apply a simple spin-up rule (can increase by at most 1 container per sample)
        if desired > containers:
            containers = min(desired, containers + 1)
        else:
            containers = desired

        over = max(0, containers - required_now)
        under = max(0, required_now - containers)
        records.append({'timestamp': ts, 'load': load, 'required': required_now, 'containers': containers, 'over_provision': over, 'under_provision': under})

    df = pd.DataFrame(records).set_index('timestamp')
    return df

def reactive_policy(current_containers, predicted_required, now_ts, forecast_ts, load):
    # Guarantee at least 1 container if predicted_required > 0
    return max(0, int(predicted_required))

def buffered_policy(current_containers, predicted_required, now_ts, forecast_ts, load, buffer=1):
    # Keep at least 1 container when load exists
    return max(1, int(predicted_required) + int(buffer))
