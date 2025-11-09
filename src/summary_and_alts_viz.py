# src/summary_and_alts_viz.py
"""
Summarize and produce alternative visualizations for long time-series simulation outputs.
Inputs (expected in results/): 
  - sim_reactive_prophet.csv
  - sim_buffered_prophet.csv
  - sim_reactive_lstm.csv
  - sim_buffered_lstm.csv

Outputs (saved to results/summary_*):
  - resampled CSVs (hourly, daily)
  - summary CSV (per-year metrics)
  - PNGs: hourly_heatmap, daily_percentiles, utilization_timeseries, event_counts, box_by_hour
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "summary_alts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "prophet_reactive": "sim_reactive_prophet.csv",
    "prophet_buffered": "sim_buffered_prophet.csv",
    "lstm_reactive": "sim_reactive_lstm.csv",
    "lstm_buffered": "sim_buffered_lstm.csv",
}

def load_sim_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, parse_dates=["timestamp"]).set_index("timestamp")
    # ensure numeric columns exist
    for c in ["load","containers","over_provision","under_provision","required"]:
        if c not in df.columns:
            df[c] = 0
    return df

def implied_capacity(df):
    mean_load = df["load"].mean()
    mean_cont = df["containers"].replace(0, np.nan).mean()
    if pd.isna(mean_cont) or mean_cont == 0:
        return None
    return mean_load / mean_cont

# load everything
sims = {k: load_sim_csv(v) for k,v in FILES.items() if (RESULTS_DIR / v).exists()}

# 1) Resample hourly and daily and save CSVs
resampled = {}
for k, df in sims.items():
    resampled[k+"_hour"] = df.resample("H").agg({
        "load":"sum", 
        "containers":"mean",
        "over_provision":"sum",
        "under_provision":"sum",
        "required":"mean"
    })
    resampled[k+"_day"] = df.resample("D").agg({
        "load":"sum",
        "containers":"mean",
        "over_provision":"sum",
        "under_provision":"sum",
        "required":"mean"
    })
    resampled[k+"_hour"].to_csv(OUT_DIR / f"{k}_hourly.csv")
    resampled[k+"_day"].to_csv(OUT_DIR / f"{k}_daily.csv")

# 2) Yearly summary CSV (mean metrics per year)
summary_rows = []
for k, df in sims.items():
    yr = df.resample("Y").agg({
        "load":"sum",
        "containers":"mean",
        "over_provision":"sum",
        "under_provision":"sum",
        "required":"mean"
    })
    yr.index = yr.index.year
    for y, row in yr.iterrows():
        summary_rows.append({
            "model_policy": k,
            "year": int(y),
            "total_load": float(row["load"]),
            "mean_containers": float(row["containers"]),
            "total_over": float(row["over_provision"]),
            "total_under": float(row["under_provision"]),
            "mean_required": float(row["required"])
        })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_DIR / "yearly_summary.csv", index=False)

# 3) Hour-of-day heatmap (average load or under-provision per hour across all days)
# we create a panel of heatmaps: load and under_provision for each sim
for key, df in sims.items():
    z = df.copy()
    z["date"] = z.index.date
    z["hour"] = z.index.hour
    pivot_load = z.pivot_table(index="date", columns="hour", values="load", aggfunc="sum", fill_value=0)
    pivot_under = z.pivot_table(index="date", columns="hour", values="under_provision", aggfunc="sum", fill_value=0)
    # save heatmaps
    plt.figure(figsize=(12,6))
    sns.heatmap(pivot_load.T, cmap="magma", cbar_kws={'label': 'load (sum per hour)'})
    plt.title(f"{key} - Hour-of-day heatmap (load). Rows=hour 0..23, columns=days")
    plt.xlabel("Day index (calendar days)")
    plt.ylabel("Hour of day")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{key}_heatmap_load.png")
    plt.close()

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot_under.T, cmap="viridis", cbar_kws={'label': 'under_provision (sum per hour)'})
    plt.title(f"{key} - Hour-of-day heatmap (under_provision)")
    plt.xlabel("Day index (calendar days)")
    plt.ylabel("Hour of day")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{key}_heatmap_under.png")
    plt.close()

# 4) Percentile ribbon over years (daily resampled)
for key, df_day in ((k,resampled[k+"_day"]) for k in sims.keys()):
    # compute rolling percentiles across days (window=365 days)
    pct10 = df_day["load"].rolling(365, min_periods=30).quantile(0.10)
    pct50 = df_day["load"].rolling(365, min_periods=30).quantile(0.50)
    pct90 = df_day["load"].rolling(365, min_periods=30).quantile(0.90)
    plt.figure(figsize=(12,4))
    plt.fill_between(df_day.index, pct10, pct90, color="lightgrey", label="10-90 pct")
    plt.plot(pct50.index, pct50.values, label="50 pct (median)", color="black")
    plt.title(f"{key} - Rolling daily percentiles (window=365 days)")
    plt.ylabel("Daily load (sum)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{key}_daily_percentiles.png")
    plt.close()

# 5) Utilization (%): load / (containers * implied_capacity) — resampled daily to be stable
util_plots = []
for key, df in sims.items():
    cap = implied_capacity(df)
    if cap is None:
        continue
    daily = df.resample("D").agg({"load":"sum","containers":"mean"})
    # provisioned capacity per day
    daily["prov_capacity"] = daily["containers"] * cap
    daily["util_pct"] = 100.0 * (daily["load"] / daily["prov_capacity"]).replace([np.inf, -np.inf], np.nan)
    daily["util_pct"] = daily["util_pct"].clip(lower=0, upper=1000)  # cap for plotting
    daily.to_csv(OUT_DIR / f"{key}_daily_utilization.csv")
    plt.figure(figsize=(12,4))
    plt.plot(daily.index, daily["util_pct"], label=f"{key} utilization (%)")
    plt.ylabel("Utilization (%)")
    plt.title(f"{key} - Daily Utilization % (load / prov_capacity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{key}_daily_utilization.png")
    plt.close()

# 6) Event counts: number of under-provision episodes exceeding thresholds (per year)
events = []
for key, df in sims.items():
    for thresh in [1,2,5,10]:  # containers
        # boolean series where under_provision >= thresh
        cond = df["under_provision"] >= thresh
        # count contiguous episodes (rising edges)
        episodes = ((cond.astype(int).diff() == 1).sum())
        events.append({
            "model_policy": key,
            "threshold": thresh,
            "episodes": int(episodes),
            "total_under_provision": int(df["under_provision"].sum())
        })
events_df = pd.DataFrame(events)
events_df.to_csv(OUT_DIR / "under_event_counts.csv", index=False)

# 7) Boxplot by hour-of-day for under_provision (to show distribution)
for key, df in sims.items():
    df2 = df.copy()
    df2["hour"] = df2.index.hour
    # group by hour
    groups = [df2[df2["hour"]==h]["under_provision"].values for h in range(24)]
    plt.figure(figsize=(12,4))
    plt.boxplot(groups, labels=list(range(24)), showfliers=False)
    plt.xlabel("Hour of day")
    plt.ylabel("under_provision (containers)")
    plt.title(f"{key} - Under-provision by hour of day (boxplots)")
    plt.savefig(OUT_DIR / f"{key}_box_by_hour_under.png")
    plt.close()

print("Saved summary artifacts to:", OUT_DIR.resolve())
print("Files: ")
for f in sorted(OUT_DIR.iterdir()):
    print(" -", f.name)
