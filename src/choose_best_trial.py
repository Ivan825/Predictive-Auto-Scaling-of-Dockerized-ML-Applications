# scripts/choose_best_trial.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUT_SUM = Path("results/tuning_summary.csv")
BEST_JSON = Path("results/best_lstm_trial.json")

rows = []
for d in sorted(Path("results").glob("*")):
    metrics_file = d / "lstm_wfv_metrics.csv"
    if not metrics_file.exists():
        continue
    try:
        df = pd.read_csv(metrics_file)
    except Exception as e:
        print("Failed to read", metrics_file, e)
        continue

    # safe metrics extraction
    def col_stats(col):
        if col not in df.columns or df[col].dropna().empty:
            return {"mean": np.nan, "median": np.nan, "std": np.nan, "count": 0}
        s = df[col].dropna().astype(float)
        return {"mean": float(s.mean()), "median": float(s.median()), "std": float(s.std(ddof=0)), "count": int(s.count())}

    mape_stats = col_stats("mape")
    rmse_stats = col_stats("rmse")

    # try to read params
    params = {}
    rp = d / "run_params.json"
    if rp.exists():
        try:
            params = json.loads(rp.read_text())
        except Exception:
            params = {}

    # count folds and successful folds (non-nan)
    total_folds = len(df) if not df.empty else 0
    valid_folds = int(df["mape"].count()) if "mape" in df.columns else 0

    # find last saved model/scaler
    model_files = sorted(d.glob("lstm_model_fold_*.h5"))
    scaler_files = sorted(d.glob("lstm_scaler_fold_*.pkl"))
    last_model = str(model_files[-1]) if model_files else None
    last_scaler = str(scaler_files[-1]) if scaler_files else None

    rows.append({
        "trial_dir": str(d),
        "total_folds": total_folds,
        "valid_folds": valid_folds,
        "mape_mean": mape_stats["mean"],
        "mape_median": mape_stats["median"],
        "mape_std": mape_stats["std"],
        "rmse_mean": rmse_stats["mean"],
        "rmse_median": rmse_stats["median"],
        "rmse_std": rmse_stats["std"],
        "last_model": last_model,
        "last_scaler": last_scaler,
        "run_params": params
    })

if not rows:
    print("No trials found in results/")
    raise SystemExit(1)

summary = pd.DataFrame(rows)
# drop trials with all-NaN MAPE (they are invalid)
valid_summary = summary[~summary["mape_mean"].isna()].copy()

# Primary ranking: mape_mean asc, then rmse_mean asc, then mape_std asc, prefer more valid_folds
valid_summary.sort_values(
    by=["mape_mean","rmse_mean","mape_std","valid_folds"],
    ascending=[True, True, True, False],
    inplace=True
)

summary.to_csv(OUT_SUM, index=False)
best = valid_summary.iloc[0]
print("Top trials (first 5):")
print(valid_summary[["trial_dir","mape_mean","rmse_mean","mape_std","valid_folds"]].head(5).to_string(index=False))
print("\nBest trial directory:", best["trial_dir"])
print("Best model:", best["last_model"])
print("Best scaler:", best["last_scaler"])
# write best metadata json
BEST_JSON.write_text(json.dumps({
    "trial_dir": best["trial_dir"],
    "mape_mean": float(best["mape_mean"]),
    "rmse_mean": None if pd.isna(best["rmse_mean"]) else float(best["rmse_mean"]),
    "last_model": best["last_model"],
    "last_scaler": best["last_scaler"],
    "run_params": best["run_params"]
}, indent=2))
print("\nSummary written to", OUT_SUM)
print("Best trial metadata written to", BEST_JSON)
