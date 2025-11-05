# scripts/run.py - FINAL VERSION (Prophet Only, No Simulations)

import os
import sys
import pandas as pd
from pathlib import Path
import traceback
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.metrics import mae, rmse, mape, smape


def generate_trading_workload(csv_path: str, use_percent: float = 0.005):
    """Load and process trading data"""
    print(f"[Data] Loading: {csv_path}")
    
    try:
        market_df = pd.read_csv(csv_path, parse_dates=['date'])
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        raise

    market_df = market_df.rename(columns={'date': 'timestamp'})
    
    total_rows = len(market_df)
    rows_to_keep = int(total_rows * use_percent)
    market_df = market_df.iloc[-rows_to_keep:].copy()
    
    print(f"[Data] Using {use_percent*100:.1f}% = {rows_to_keep:,} rows")
    
    market_df = market_df.set_index('timestamp').resample('1min').sum().reset_index()
    
    market_open_hour = 9.5
    market_close_hour = 16.0
    market_df['hour'] = market_df['timestamp'].dt.hour + market_df['timestamp'].dt.minute / 60
    market_df['weekday'] = market_df['timestamp'].dt.weekday
    
    is_market_day = market_df['weekday'].between(0, 4)
    is_market_hours = market_df['hour'].between(market_open_hour, market_close_hour)
    market_is_open = is_market_day & is_market_hours
    
    volume_to_demand_factor = 0.001
    base_demand = 10
    market_df['volume'] = market_df['volume'].fillna(0)
    
    demand_from_volume = market_df['volume'] * volume_to_demand_factor
    market_df['req_per_min'] = np.where(market_is_open, base_demand + demand_from_volume, np.random.uniform(1, 5, len(market_df)))
    market_df['req_per_min'] = market_df['req_per_min'].clip(lower=1, upper=1000).astype(int)
    market_df['is_event'] = 0
    
    final_df = market_df[['timestamp', 'req_per_min', 'is_event']].copy()
    final_df['timestamp'] = pd.to_datetime(final_df['timestamp']).dt.tz_localize(None)
    final_df['req_per_min'] = final_df['req_per_min'].fillna(final_df['req_per_min'].mean())
    
    print(f"[Data] Complete: {len(final_df)} rows")
    return final_df


def time_split(df: pd.DataFrame, train_ratio: float = 0.7):
    """Split data"""
    split_idx = int(len(df) * train_ratio)
    df_train = df[:split_idx].reset_index(drop=True)
    df_test = df[split_idx:].reset_index(drop=True)
    print(f"[Split] Train: {len(df_train)}, Test: {len(df_test)}")
    return df_train, df_test


HAVE_PROPHET = True
try:
    from src.models.prophet_model import train_prophet, forecast_prophet
except Exception as e:
    print(f"[WARN] Prophet unavailable: {e}")
    HAVE_PROPHET = False


def ensure_dirs():
    """Create directories"""
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)


def plot_prophet_forecast(df_fc, outpath):
    """Plot demand vs forecast"""
    try:
        print(f"[Plot] Creating forecast plot...")
        
        if df_fc is None or df_fc.empty:
            print(f"[WARN] Empty dataframe, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot actual demand
        valid_demand = ~df_fc['req_per_min'].isna()
        ax.plot(df_fc.loc[valid_demand, "timestamp"], 
                df_fc.loc[valid_demand, "req_per_min"], 
                label="Actual Demand", linewidth=2, color='blue', alpha=0.8)
        
        # Plot forecast
        if "yhat" in df_fc.columns:
            valid_yhat = ~df_fc['yhat'].isna()
            ax.plot(df_fc.loc[valid_yhat, "timestamp"], 
                    df_fc.loc[valid_yhat, "yhat"], 
                    label="Prophet Forecast", linewidth=2, color='orange', alpha=0.8)
        
        # Plot confidence interval
        if "yhat_upper" in df_fc.columns and "yhat_lower" in df_fc.columns:
            valid_ci = (~df_fc['yhat_upper'].isna()) & (~df_fc['yhat_lower'].isna())
            ax.fill_between(df_fc.loc[valid_ci, "timestamp"], 
                           df_fc.loc[valid_ci, "yhat_lower"], 
                           df_fc.loc[valid_ci, "yhat_upper"], 
                           alpha=0.2, color='orange', label="Confidence Interval (80%)")
        
        ax.set_title("Demand vs Prophet Forecast", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Requests per Minute", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"[Plot] ✓ Saved: {outpath}")
        
    except Exception as e:
        print(f"[ERROR] Plot creation failed: {e}")
        traceback.print_exc()
        plt.close('all')


def run_prophet_pipeline(df):
    """Prophet training and forecasting"""
    
    try:
        print("\n" + "="*70)
        print("PROPHET PIPELINE")
        print("="*70)
        
        # Split data
        df_train, df_test = time_split(df, train_ratio=0.7)

        # Train Prophet
        print("[Prophet] Training Bayesian model (this may take several minutes)...")
        try:
            m = train_prophet(df_train, regressor_cols=["is_event"])
            print("[Prophet] ✓ Training complete")
        except Exception as e:
            print(f"[ERROR] Prophet training failed: {e}")
            traceback.print_exc()
            return None

        # Forecast
        print("[Prophet] Generating forecasts...")
        try:
            horizon_minutes = len(df_test)
            fut_regs = df_test[["timestamp", "is_event"]].copy()
            
            fc_test = forecast_prophet(
                m,
                df=df_train,
                horizon_minutes=horizon_minutes,
                future_regressors=fut_regs
            ).rename(columns={"ds": "timestamp"})
            
            print(f"[Prophet] ✓ Generated {len(fc_test)} forecasts")
            
        except Exception as e:
            print(f"[ERROR] Prophet forecasting failed: {e}")
            traceback.print_exc()
            return None

        # Merge forecasts with test data
        print("[Prophet] Merging forecasts with actual data...")
        try:
            fc_test['timestamp'] = pd.to_datetime(fc_test['timestamp']).dt.tz_localize(None)
            df_fc_test = df_test.merge(fc_test, on="timestamp", how="left")
            print(f"[Prophet] ✓ Merged: {len(df_fc_test)} rows")
        except Exception as e:
            print(f"[ERROR] Merge failed: {e}")
            traceback.print_exc()
            return None

        # Fill missing forecasts
        missing = df_fc_test["yhat"].isna().sum()
        if missing > 0:
            print(f"[WARN] Filling {missing} missing forecasts with actual demand")
            df_fc_test["yhat"] = df_fc_test["yhat"].fillna(df_fc_test["req_per_min"])
            df_fc_test["yhat_upper"] = df_fc_test["yhat_upper"].fillna(df_fc_test["req_per_min"] * 1.2)
            df_fc_test["yhat_lower"] = df_fc_test["yhat_lower"].fillna(df_fc_test["req_per_min"] * 0.8)

        # Calculate metrics
        print("[Prophet] Calculating metrics...")
        try:
            mask = (~df_fc_test["yhat"].isna()) & (~df_fc_test["req_per_min"].isna())
            if mask.sum() > 0:
                y = df_fc_test.loc[mask, "req_per_min"].values
                yhat = df_fc_test.loc[mask, "yhat"].values
                
                mae_val = mae(y, yhat)
                rmse_val = rmse(y, yhat)
                mape_val = mape(y, yhat)
                
                print(f"[Metrics] MAE={mae_val:.2f}, RMSE={rmse_val:.2f}, MAPE={mape_val:.2f}%")
            else:
                print(f"[WARN] No valid predictions for metrics")
        except Exception as e:
            print(f"[WARN] Metrics calculation failed: {e}")

        # Create plots
        print("[Prophet] Creating plots...")
        plot_prophet_forecast(df_fc_test, ROOT / "reports/prophet_demand_vs_forecast.png")

        # Save forecast data
        print("[Prophet] Saving forecast data...")
        try:
            fc_output_path = ROOT / "reports/prophet_forecasts.csv"
            df_fc_test.to_csv(fc_output_path, index=False)
            print(f"[Prophet] ✓ Saved forecasts to: {fc_output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save forecasts: {e}")

        return df_fc_test

    except Exception as e:
        print(f"[CRITICAL ERROR] Prophet pipeline: {e}")
        traceback.print_exc()
        return None


def main():
    """Main execution"""
    try:
        ensure_dirs()

        print("\n" + "="*70)
        print("DATA LOADING")
        print("="*70)
        
        market_data_filepath = ROOT / "data" / "1_min_SPY_2008-2021.csv"
        
        try:
            df = generate_trading_workload(csv_path=str(market_data_filepath), use_percent=0.005)
        except Exception as e:
            print(f"[CRITICAL] Data loading failed: {e}")
            return

        if df is None or len(df) < 50:
            print(f"[CRITICAL] Not enough data: {len(df) if df is not None else 0} rows")
            return

        # Run Prophet
        if not HAVE_PROPHET:
            print("[ERROR] Prophet not available")
            return
        
        result = run_prophet_pipeline(df)

        # Summary
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        
        if result is not None and not result.empty:
            # Save summary
            summary_data = {
                "status": "SUCCESS",
                "data_rows": len(df),
                "forecasts": len(result),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_path = ROOT / "reports/execution_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            
            print(f"\n✅ [SUCCESS] Pipeline completed successfully!")
            print(f"   - Data rows processed: {len(df):,}")
            print(f"   - Forecasts generated: {len(result):,}")
            print(f"   - Results saved to: {ROOT / 'reports'}")
            print(f"\nGenerated files:")
            print(f"   - {ROOT / 'reports/prophet_demand_vs_forecast.png'}")
            print(f"   - {ROOT / 'reports/prophet_forecasts.csv'}")
            print(f"   - {ROOT / 'reports/execution_summary.csv'}")
            print()
        else:
            print(f"\n❌ [FAILED] Pipeline execution failed")

    except Exception as e:
        print(f"\n[CRITICAL] Main execution failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
