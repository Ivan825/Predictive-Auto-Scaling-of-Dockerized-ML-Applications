import numpy as np, pandas as pd

def simulate_demand(start="2025-01-01", days=14, freq="1min",
                    base=18, daily_amp=10, weekly_amp=0.25,
                    spike_prob=0.003, spike_mag=(40,120), seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=int(days*24*60), freq=freq)
    t = np.arange(len(idx))

    # Daily seasonality (sinusoidal)
    hour = idx.hour + idx.minute/60
    daily = daily_amp * np.sin(2*np.pi*hour/24)

    # Weekly modulation (weekdays up, weekends down)
    dow = idx.dayofweek
    weekly = 1 + weekly_amp * np.where(dow < 5, 1, -1)

    # Trend + noise
    trend = np.linspace(0, 3, len(idx))
    noise = rng.normal(0, 1.8, len(idx))

    # Spikes/events
    spikes = np.zeros(len(idx))
    is_event = np.zeros(len(idx), dtype=int)
    spike_mask = rng.random(len(idx)) < spike_prob
    spikes[spike_mask] = rng.integers(spike_mag[0], spike_mag[1], spike_mask.sum())
    is_event[spike_mask] = 1

    req = (base + daily) * weekly + trend + noise + spikes
    req = np.clip(req, 0, None)

    df = pd.DataFrame({
        "timestamp": idx,
        "req_per_min": req,
        "hour": hour.astype(int),
        "dayofweek": dow,
        "is_event": is_event
    })
    return df

if __name__ == "__main__":
    simulate_demand(days=14).to_csv("data/scenario_mixed.csv", index=False)
    simulate_demand(days=14, daily_amp=7, spike_prob=0.0005).to_csv("data/scenario_smooth.csv", index=False)
    simulate_demand(days=14, daily_amp=12, spike_prob=0.01, spike_mag=(70,180)).to_csv("data/scenario_spiky.csv", index=False)
