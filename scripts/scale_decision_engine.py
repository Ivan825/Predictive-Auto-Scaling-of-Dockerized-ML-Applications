# E:/IIITB/Predictive-Auto-Scaling-of-Dockerized-ML-Applications/scale_decision_engine.py
import docker
import time
import joblib
import torch
import pandas as pd
import requests
import logging
from pathlib import Path

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Add src to path for imports ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# --- Import project modules ---
from src.models.prophet_model import forecast_prophet
from src.models.lstm_model import predict_lstm, LSTMRegressor # The class definition is needed
from src.policies import confidence_aware, simple_threshold
from src.safety_net import safety_override

# --- Constants ---
DOCKER_SERVICE_NAME = "ml_app"
PROMETHEUS_URL = "http://localhost:9090"
METRIC_QUERY = 'sum(rate(flask_http_requests_total{job="ml_app"}[1m]))' # Sum requests across all instances
MODEL_TYPE = "prophet"  # "prophet" or "lstm"
POLICY = "confidence_aware" # "confidence_aware" or "simple"
CAPACITY_PER_CONTAINER = 150  # Requests per minute per container
MIN_REPLICAS = 1
MAX_REPLICAS = 10
SLEEP_INTERVAL_SECONDS = 60
FORECAST_HORIZON_MINUTES = 15

# --- Load Models ---
logging.info("Loading models...")
MODELS_DIR = ROOT / "models"
prophet_model = joblib.load(MODELS_DIR / 'prophet_model.joblib')
lstm_model_state = torch.load(MODELS_DIR / 'lstm_model.pt')
lstm_config = joblib.load(MODELS_DIR / 'lstm_config.joblib')

# Recreate the LSTM model structure and load the state
lstm_model = LSTMRegressor(
    input_size=lstm_config['input_size'],
    hidden_size=lstm_config['hidden_size'],
    num_layers=lstm_config['num_layers']
)
lstm_model.load_state_dict(lstm_model_state)
lstm_model.eval() # Set to evaluation mode
logging.info("Models loaded successfully.")

# --- Docker Client ---
try:
    client = docker.from_env()
except docker.errors.DockerException:
    logging.error("Docker is not running or accessible. Please start Docker Desktop.")
    sys.exit(1)

def get_current_replicas(service_name):
    """Gets the current number of replicas for a service."""
    try:
        service = client.services.get(service_name)
        return service.attrs['Spec']['Mode']['Replicated']['Replicas']
    except docker.errors.NotFound:
        logging.error(f"Service {service_name} not found.")
        return 0
    except Exception as e:
        logging.error(f"Error getting current replicas: {e}")
        return 0

def scale_service(service_name, target_replicas):
    """Scales a Docker service to the target number of replicas."""
    target_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, int(target_replicas)))
    current_replicas = get_current_replicas(service_name)

    if current_replicas == target_replicas:
        logging.info(f"No scaling needed. Current replicas ({current_replicas}) match target ({target_replicas}).")
        return

    logging.info(f"Scaling {service_name} from {current_replicas} to {target_replicas} replicas...")
    try:
        service = client.services.get(service_name)
        service.scale(replicas=target_replicas)
        logging.info(f"Successfully scaled {service_name} to {target_replicas} replicas.")
    except docker.errors.NotFound:
        logging.error(f"Service {service_name} not found during scaling.")
    except Exception as e:
        logging.error(f"Error scaling service: {e}")

def get_metrics_from_prometheus():
    """Queries Prometheus to get historical demand data."""
    logging.info("Querying Prometheus for metrics...")
    try:
        # Query for the last 60 minutes of data to provide context for the forecast
        end_time = pd.Timestamp.now(tz='UTC')
        start_time = end_time - pd.Timedelta(minutes=60)
        
        response = requests.get(f'{PROMETHEUS_URL}/api/v1/query_range', params={
            'query': METRIC_QUERY,
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'step': '60s' # 1-minute resolution
        })
        response.raise_for_status()
        results = response.json()['data']['result']

        if not results:
            logging.warning("No data returned from Prometheus. Returning empty DataFrame.")
            return pd.DataFrame(columns=['ds', 'y'])

        # Convert to DataFrame
        data = results[0]['values']
        df = pd.DataFrame(data, columns=['unix_time', 'y'])
        df['ds'] = pd.to_datetime(df['unix_time'], unit='s')
        df['y'] = df['y'].astype(float)
        df = df[['ds', 'y']]
        
        logging.info(f"Successfully fetched {len(df)} data points from Prometheus.")
        return df

    except requests.exceptions.RequestException as e:
        logging.error(f"Could not connect to Prometheus at {PROMETHEUS_URL}. Error: {e}")
        return pd.DataFrame(columns=['ds', 'y'])
    except Exception as e:
        logging.error(f"An error occurred while fetching metrics: {e}")
        return pd.DataFrame(columns=['ds', 'y'])

def main_loop():
    """The main decision-making loop for the auto-scaler."""
    logging.info("Starting auto-scaler main loop...")
    while True:
        # 1. Get historical metrics
        metrics_df = get_metrics_from_prometheus()
        if metrics_df.empty:
            logging.warning("Metrics DataFrame is empty. Skipping this cycle.")
            time.sleep(SLEEP_INTERVAL_SECONDS)
            continue

        # 2. Get a forecast from the chosen model
        if MODEL_TYPE == "prophet":
            logging.info("Generating forecast with Prophet...")
            forecast_df = forecast_prophet(prophet_model, df=metrics_df, horizon_minutes=FORECAST_HORIZON_MINUTES)
            predicted_load = forecast_df['yhat'].iloc[-1]
            predicted_upper = forecast_df['yhat_upper'].iloc[-1]
            logging.info(f"Prophet forecast: predicted_load={predicted_load:.2f}, predicted_upper={predicted_upper:.2f}")

        elif MODEL_TYPE == "lstm":
            logging.info("Generating forecast with LSTM...")
            # LSTM requires the config for scaling
            predicted_load = predict_lstm(lstm_model, metrics_df, lstm_config)
            predicted_upper = predicted_load * 1.2 # LSTM doesn't provide confidence, add a 20% buffer
            logging.info(f"LSTM forecast: predicted_load={predicted_load:.2f}")
        else:
            logging.error(f"Unknown MODEL_TYPE: {MODEL_TYPE}")
            time.sleep(SLEEP_INTERVAL_SECONDS)
            continue

        # 3. Use a policy to get the target replica count
        if POLICY == "confidence_aware":
            target_replicas = confidence_aware(predicted_upper, CAPACITY_PER_CONTAINER)
            logging.info(f"Policy '{POLICY}' chose {target_replicas} replicas based on predicted_upper={predicted_upper:.2f}.")
        else: # Default to simple
            target_replicas = simple_threshold(predicted_load, CAPACITY_PER_CONTAINER)
            logging.info(f"Policy '{POLICY}' chose {target_replicas} replicas based on predicted_load={predicted_load:.2f}.")

        # 4. Apply safety net (optional, for future enhancement)
        # current_latency = get_latency_from_prometheus()
        # target_replicas = safety_override(target_replicas, current_latency, max_latency=500)

        # 5. Execute scaling action
        scale_service(DOCKER_SERVICE_NAME, target_replicas)

        logging.info(f"Cycle complete. Sleeping for {SLEEP_INTERVAL_SECONDS} seconds.")
        time.sleep(SLEEP_INTERVAL_SECONDS)

if __name__ == "__main__":
    main_loop()
