# snippet to plot load (left) and containers (right axis) for Prophet vs LSTM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

RESULTS_DIR = "results"

def safe_read(path):
    return pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp') if os.path.exists(path) else None

sim_r_prophet = safe_read(os.path.join(RESULTS_DIR, "sim_reactive_prophet.csv"))
sim_b_prophet = safe_read(os.path.join(RESULTS_DIR, "sim_buffered_prophet.csv"))
sim_r_lstm = safe_read(os.path.join(RESULTS_DIR, "sim_reactive_lstm.csv"))
sim_b_lstm = safe_read(os.path.join(RESULTS_DIR, "sim_buffered_lstm.csv"))

# choose a small window to make plot legible (or set None to plot whole series)
window = slice(0, 200)

# compute an implied capacity per container from data so we can optionally plot provisioned capacity
def implied_capacity(df):
    # avoid division by zero: take mean(load)/mean(containers) if containers>0
    if df is None: 
        return None
    mean_load = df['load'].mean()
    mean_cont = df['containers'].replace(0, np.nan).mean()
    if np.isnan(mean_cont) or mean_cont == 0:
        return None
    return mean_load / mean_cont

cap_prophet = implied_capacity(sim_r_prophet)
cap_lstm = implied_capacity(sim_r_lstm)

plt.figure(figsize=(12,6))
ax = plt.gca()

# left axis: actual load
if sim_r_prophet is not None:
    ax.plot(sim_r_prophet.index[window], sim_r_prophet['load'][window], color='black', label='Actual Load', linewidth=1.5)

# right axis: containers
ax2 = ax.twinx()
# plot containers on right axis with distinct linestyles
if sim_r_prophet is not None:
    ax2.plot(sim_r_prophet.index[window], sim_r_prophet['containers'][window], '--', label='Prophet Reactive (containers)', color='tab:blue', alpha=0.9)
if sim_b_prophet is not None:
    ax2.plot(sim_b_prophet.index[window], sim_b_prophet['containers'][window], ':', label='Prophet Buffered (containers)', color='tab:orange', alpha=0.9)
if sim_r_lstm is not None:
    ax2.plot(sim_r_lstm.index[window], sim_r_lstm['containers'][window], '--', label='LSTM Reactive (containers)', color='tab:green', alpha=0.7)
if sim_b_lstm is not None:
    ax2.plot(sim_b_lstm.index[window], sim_b_lstm['containers'][window], ':', label='LSTM Buffered (containers)', color='tab:red', alpha=0.7)

# Optionally plot provisioned capacity (containers * implied_capacity) on left axis
if cap_prophet is not None and sim_r_prophet is not None:
    prov_prop_r = sim_r_prophet['containers'][window] * cap_prophet
    ax.plot(sim_r_prophet.index[window], prov_prop_r, '--', label='Prophet Reactive (provisioned capacity)', color='tab:blue', linewidth=0.8, alpha=0.5)
if cap_lstm is not None and sim_r_lstm is not None:
    prov_lstm_r = sim_r_lstm['containers'][window] * cap_lstm
    ax.plot(sim_r_lstm.index[window], prov_lstm_r, '--', label='LSTM Reactive (provisioned capacity)', color='tab:green', linewidth=0.8, alpha=0.5)

ax.set_xlabel('Time')
ax.set_ylabel('Load (requests / volume)')
ax2.set_ylabel('Containers (count)')
ax.set_title('Scaling Behavior Comparison (Prophet vs LSTM)')
# construct a combined legend
lines_ax, labels_ax = ax.get_legend_handles_labels()
lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
ax.legend(lines_ax + lines_ax2, labels_ax + labels_ax2, loc='upper right', fontsize='small')

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "scaling_behavior_prophet_vs_lstm_dualaxis.png")
plt.savefig(out_path)
plt.close()
print("Saved:", out_path)
