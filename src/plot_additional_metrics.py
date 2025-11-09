# src/plot_additional_metrics.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "savefig.bbox": "tight"
})

RESULTS_DIR = "results"

def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    return df

# ✅ Only these four CSVs
sim_r_prop = load_csv("sim_reactive_prophet.csv")
sim_b_prop = load_csv("sim_buffered_prophet.csv")
sim_r_lstm = load_csv("sim_reactive_lstm.csv")
sim_b_lstm = load_csv("sim_buffered_lstm.csv")

# ---- Utility ----
def implied_capacity(df):
    mean_load = df["load"].mean()
    mean_cont = df["containers"].replace(0, np.nan).mean()
    return None if np.isnan(mean_cont) or mean_cont == 0 else mean_load / mean_cont

cap_prop = implied_capacity(sim_r_prop)
cap_lstm = implied_capacity(sim_r_lstm)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# 1️⃣ Average Over/Under Provisioning
# ---------------------------
sims = [
    ("Prophet Reactive", sim_r_prop),
    ("Prophet Buffered", sim_b_prop),
    ("LSTM Reactive", sim_r_lstm),
    ("LSTM Buffered", sim_b_lstm),
]

labels, avg_over, avg_under = [], [], []
for name, df in sims:
    labels.append(name)
    avg_over.append(df["over_provision"].mean())
    avg_under.append(df["under_provision"].mean())

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, avg_over, width, label="Avg Over-Provision")
ax.bar(x + width / 2, avg_under, width, label="Avg Under-Provision")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=18)
ax.set_ylabel("Avg containers")
ax.set_title("Average Over/Under Provisioning by Model & Policy")
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "avg_over_under_provisioning.png"))
plt.close()

# ---------------------------
# 2️⃣ Cumulative Under-Provision
# ---------------------------
plt.figure(figsize=(12, 5))
for name, df in sims:
    plt.plot(df.index, df["under_provision"].cumsum(), label=name)
plt.xlabel("Time")
plt.ylabel("Cumulative under-provision (containers)")
plt.title("Cumulative Under-Provision Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "cumulative_under_provision.png"))
plt.close()

# ---------------------------
# 3️⃣ Distribution & CDF
# ---------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for name, df in sims:
    arr = df["under_provision"].values
    axs[0].hist(arr, bins=50, alpha=0.6, label=name)
    sorted_vals = np.sort(arr)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    axs[1].plot(sorted_vals, cdf, label=name)
axs[0].set_title("Histogram: Under-Provision Magnitudes")
axs[0].set_xlabel("Under-provision (containers)")
axs[1].set_title("CDF: Under-Provision")
axs[1].set_xlabel("Under-provision (containers)")
axs[0].legend(fontsize="small")
axs[1].legend(fontsize="small")
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "under_provision_distribution.png"))
plt.close()

# ---------------------------
# 4️⃣ Boxplot of Container Counts
# ---------------------------
fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(
    [sim_r_prop["containers"], sim_b_prop["containers"],
     sim_r_lstm["containers"], sim_b_lstm["containers"]],
    labels=["Prophet Reactive", "Prophet Buffered",
            "LSTM Reactive", "LSTM Buffered"],
    showfliers=False,
)
ax.set_ylabel("Containers (count)")
ax.set_title("Container Count Distribution by Model & Policy")
plt.xticks(rotation=18)
plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "containers_boxplot.png"))
plt.close()

# ---------------------------
# 5️⃣ Provisioned Capacity vs Load
# ---------------------------
plt.figure(figsize=(12, 5))
plt.plot(sim_r_prop.index, sim_r_prop["load"], color="black",
         label="Actual Load", linewidth=1.2)
plt.plot(sim_r_prop.index, sim_r_prop["containers"] * cap_prop,
         "--", label="Prophet provisioned capacity (reactive)")
plt.plot(sim_r_lstm.index, sim_r_lstm["containers"] * cap_lstm,
         ":", label="LSTM provisioned capacity (reactive)")
plt.xlabel("Time")
plt.ylabel("Load / Provisioned capacity (same units)")
plt.title("Actual Load vs Provisioned Capacity")
plt.legend(fontsize="small")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "provisioned_vs_load.png"))
plt.close()

print("✅ Additional plots saved to results/:")
for f in [
    "avg_over_under_provisioning.png",
    "cumulative_under_provision.png",
    "under_provision_distribution.png",
    "containers_boxplot.png",
    "provisioned_vs_load.png",
]:
    print(" -", f)
