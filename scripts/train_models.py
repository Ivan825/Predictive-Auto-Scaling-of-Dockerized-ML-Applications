# E:/IIITB/Predictive-Auto-Scaling-of-Dockerized-ML-Applications/scripts/train_models.py
import pandas as pd
import joblib
import torch
from pathlib import Path
import sys

# Add the src directory to the Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.prophet_model import train_prophet
from src.models.lstm_model import train_lstm
from src.simulate_data import simulate_demand

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    """
    Loads the full dataset, splits it into train and test sets,
    trains models on the training set, and saves the models and the datasets.
    """
    print("Loading preprocessed Google Cluster Trace data...")
    df_raw = pd.read_csv(DATA_DIR / "google_trace_preprocessed.csv")
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

    # --- Train/Test Split ---
    print("Splitting data into train and test sets (80/20 split)...")
    train_size = int(len(df_raw) * 0.8)
    df_train = df_raw.iloc[:train_size].copy()
    df_test = df_raw.iloc[train_size:].copy()

    # Save the datasets for the experiment script
    df_train.to_csv(DATA_DIR / 'google_trace_train.csv', index=False)
    df_test.to_csv(DATA_DIR / 'google_trace_test.csv', index=False)
    print(f"Train set saved to {DATA_DIR / 'google_trace_train.csv'}")
    print(f"Test set saved to {DATA_DIR / 'google_trace_test.csv'}")

    # --- Prepare for Training ---
    # Rename columns to fit model requirements ('ds', 'y')
    df_train_prophet = df_train.rename(columns={'timestamp': 'ds', 'req_per_min': 'y'})
    df_train_lstm = df_train.rename(columns={'req_per_min': 'y'})


    # --- Train and Save Prophet Model ---
    print("\nTraining and saving Prophet model...")
    prophet_model = train_prophet(df_train_prophet)
    joblib.dump(prophet_model, MODELS_DIR / 'prophet_model.joblib')
    print(f"Prophet model saved to {MODELS_DIR / 'prophet_model.joblib'}")

    # --- Train and Save LSTM Model ---
    print("\nTraining and saving LSTM model...")
    # Note: train_lstm expects 'y' as the target column
    lstm_model, lstm_config = train_lstm(df_train_lstm)
    torch.save(lstm_model.state_dict(), MODELS_DIR / 'lstm_model.pt')
    joblib.dump(lstm_config, MODELS_DIR / 'lstm_config.joblib') # Save config too! 
    print(f"LSTM model saved to {MODELS_DIR / 'lstm_model.pt'}")
    print(f"LSTM config saved to {MODELS_DIR / 'lstm_config.joblib'}")

    print("\nModels trained and datasets saved successfully.")

if __name__ == "__main__":
    main()
