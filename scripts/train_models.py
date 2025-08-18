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
    Trains the Prophet and LSTM models on simulated data and saves them to disk.
    """
    print("Generating simulated demand data for training...")
    # Using one of the scenarios to generate training data
    df_train = pd.read_csv(DATA_DIR / "scenario_spiky.csv")
    # Rename columns to fit Prophet's requirements
    df_train.rename(columns={'timestamp': 'ds', 'req_per_min': 'y'}, inplace=True)
    df_train['ds'] = pd.to_datetime(df_train['ds'])

    # --- Train and Save Prophet Model ---
    print("Training and saving Prophet model...")
    prophet_model = train_prophet(df_train)
    joblib.dump(prophet_model, MODELS_DIR / 'prophet_model.joblib')
    print(f"Prophet model saved to {MODELS_DIR / 'prophet_model.joblib'}")

    # --- Train and Save LSTM Model ---
    print("Training and saving LSTM model...")
    lstm_model, lstm_config = train_lstm(df_train)
    torch.save(lstm_model.state_dict(), MODELS_DIR / 'lstm_model.pt')
    joblib.dump(lstm_config, MODELS_DIR / 'lstm_config.joblib') # Save config too!
    print(f"LSTM model saved to {MODELS_DIR / 'lstm_model.pt'}")
    print(f"LSTM config saved to {MODELS_DIR / 'lstm_config.joblib'}")

    print("\nModels trained and saved successfully.")

if __name__ == "__main__":
    main()
