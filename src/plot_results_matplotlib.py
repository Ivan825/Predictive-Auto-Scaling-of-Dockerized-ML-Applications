# src/plot_results_matplotlib.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight"
})

RESULTS_DIR = "results"

def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")
    return df

# ✅ Use only these 4 fixed simulation files
sim_r_prophet = load_csv("sim_reactive_prophet.csv")
sim_b_prophet = load_csv("sim_buffered_prophet.csv")
sim_r_lstm    = load_csv("sim_reactive_lstm.csv")
sim_b_lstm    = load_csv("sim_buffered_lstm.csv")

# ---- 1️⃣ Scaling Behavior Comparison ----
window = slice(0, 200)
plt.figure(figsize=(12, 5))
plt.plot(sim_r_prophet.index[window], sim_r_prophet["load"][window],
         color="black", label="Actual Load")
plt.plot(sim_r_prophet.index[window], sim_r_prophet["containers"][window],
         "--", label="Prophet Reactive")
plt.plot(sim_b_prophet.index[window], sim_b_prophet["containers"][window],
         ":", label="Prophet Buffered")
plt.plot(sim_r_lstm.index[window], sim_r_lstm["containers"][window],
         "--", color="green", label="LSTM Reactive", alpha=0.7)
plt.plot(sim_b_lstm.index[window], sim_b_lstm["containers"][window],
         ":", color="red", label="LSTM Buffered", alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Load / Containers")
plt.title("Scaling Behavior Comparison (Prophet vs LSTM)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "scaling_behavior_prophet_vs_lstm.png"))
plt.close()

# ---- 2️⃣ Policy Efficiency Comparison ----
def avg_over_under(df):
    return df["over_provision"].mean(), df["under_provision"].mean()

over_r_p, under_r_p = avg_over_under(sim_r_prophet)
over_b_p, under_b_p = avg_over_under(sim_b_prophet)
over_r_l, under_r_l = avg_over_under(sim_r_lstm)
over_b_l, under_b_l = avg_over_under(sim_b_lstm)

labels = ["Reactive Prophet", "Buffered Prophet",
          "Reactive LSTM", "Buffered LSTM"]
over = [over_r_p, over_b_p, over_r_l, over_b_l]
under = [under_r_p, under_b_p, under_r_l, under_b_l]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, over, width, label="Avg Over-Provision", color="tab:blue")
ax.bar(x + width / 2, under, width, label="Avg Under-Provision", color="tab:orange")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylabel("Avg Containers")
ax.set_title("Policy Efficiency Comparison (Prophet vs LSTM)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "policy_efficiency_prophet_vs_lstm.png"))
plt.close()

print("✅ Saved:")
print(" - results/scaling_behavior_prophet_vs_lstm.png")
print(" - results/policy_efficiency_prophet_vs_lstm.png")
