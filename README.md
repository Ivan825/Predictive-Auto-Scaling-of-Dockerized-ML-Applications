# Predictive Auto-Scaling Project: Runbook & Guide

This document provides a complete guide to understanding, setting up, and running the predictive auto-scaling project.

---

## 1. Project Structure

Here is a summary of what each file and directory in the project is responsible for.

-   `docker-compose.yml`: The main orchestration file. It defines and configures all the services (containers) of the application: `ml_app`, `prometheus`, `grafana`, and the `autoscaler`. It's the entry point for running the live system.
-   `Dockerfile`, `Dockerfile.autoscaler`, `Dockerfile.ml_app`: These files contain the instructions to build the Docker images for the different components of the system.
-   `prometheus.yml`: The configuration file for the Prometheus monitoring service. It tells Prometheus which targets to scrape for metrics (in this case, the `ml_app` service).
-   `requirements.txt`: A list of all the Python packages required to run the project's scripts.
-   `data/`: This directory holds the synthetic workload data (`.csv` files) used for training the forecasting models and for running offline simulations.
-   `models/`: This directory stores the trained and serialized machine learning models (e.g., `prophet_model.joblib`, `lstm_model.pt`) that are used by the live autoscaler and the simulation script.
-   `reports/`: This is the output directory for the offline simulation. When you run `run_experiment.py`, it will populate this folder with `.csv` summaries and `.png` plots that compare the performance of different scaling strategies.
-   `scripts/`: Contains high-level executable scripts.
    -   `train_models.py`: This script reads data from `data/`, trains the Prophet and LSTM forecasting models, and saves the trained artifacts into the `models/` directory. **This is the first script you need to run.**
    -   `scale_decision_engine.py`: This is the brain of the live auto-scaling system. It runs in a continuous loop within the `autoscaler` container, queries Prometheus for metrics, uses a trained model to predict future load, and executes scaling commands via the Docker API.
    -   `run_experiment.py`: An offline simulation and evaluation tool. It uses historical data to test how different scaling policies would have performed, allowing you to analyze and compare them without running the full live system. It generates files in the `reports/` directory.
-   `src/`: Contains the core Python source code.
    -   `ml_app.py`: A simple Flask web application that simulates an ML inference service. It exposes an endpoint for predictions and, crucially, exposes metrics for Prometheus to scrape.
    -   `policies.py`: A key module that implements the various scaling logic strategies (e.g., `simple_ceiling`, `buffered`, `confidence_aware`). The decision engine uses these functions to translate a load forecast into a required number of containers.
    -   `simulator.py`: The core simulation engine that is used by `run_experiment.py` to model how the system behaves under different conditions.
    -   `models/`: This sub-directory contains the Python code that defines the model architectures and training/prediction functions (e.g., for the LSTM model).
    -   `safety_net.py`: Implements logic for a reactive safety mechanism. This can override a predictive decision if real-time metrics (like latency or CPU) exceed critical thresholds.
    -   `simulate_data.py`: A utility script to generate the synthetic `.csv` workload patterns found in the `data/` directory.

---

## 2. How to Run the System

Follow these steps to get the project running.

### Prerequisites

-   **Docker and Docker Desktop:** Must be installed and running on your system.
-   **Python 3.8+:** Required to run the training and simulation scripts.

### Option 1: Run the Live Auto-Scaling System

This will launch the complete system with the autoscaler actively managing the number of ML application containers.

**Step 1: Install Python Dependencies**

Open your terminal and install all the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

**Step 2: Train the Predictive Models**

Before the autoscaler can run, it needs the trained models. Execute the training script:

```bash
python scripts/train_models.py
```

-   **Expected Output:** You will see console messages indicating that the Prophet and LSTM models are being trained. After a few moments, it will confirm that the models have been saved to the `models/` directory.

**Step 3: Launch the Entire System with Docker Compose**

Now, start all the services. This command will build the necessary Docker images and start the containers in the correct order.

```bash
docker-compose up --build
```

-   **What to Expect:**
    1.  Four services will be started: `ml_app`, `prometheus`, `grafana`, and `autoscaler`.
    2.  Your terminal will show a combined log output from all containers.
    3.  **To see the autoscaler in action**, watch for logs from the `autoscaler` service. You will see messages every 60 seconds like:
        -   `Querying Prometheus for metrics...`
        -   `Generating forecast with Prophet...`
        -   `Policy 'confidence_aware' chose X replicas...`
        -   `Scaling ml_app from Y to X replicas...`
    4.  You can access the supporting services in your web browser:
        -   **Prometheus:** `http://localhost:9090` (You can see the `ml_app` target and run queries).
        -   **Grafana:** `http://localhost:3000` (Login with `admin`/`admin`. You can configure it to use Prometheus as a data source).

To stop the system, press `Ctrl+C` in your terminal.

### Option 2: Run an Offline Simulation and Evaluation

If you only want to analyze and compare the performance of the different scaling policies without running the live system, use the experiment script.

**Step 1: Install Dependencies and Train Models**

Follow steps 1 and 2 from the "Live System" instructions above.

**Step 2: Run the Experiment Script**

Execute the simulation script:

```bash
python scripts/run_experiment.py
```

-   **What to Expect:**
    1.  The script will load a dataset from `data/`.
    2.  It will run simulations for both Prophet and LSTM models against multiple scaling policies (`static`, `simple`, `buffered`, `conf`, etc.).
    3.  The console will print out detailed performance metrics for each policy, including cost, under-provisioning events, and over-provisioning percentage.
    4.  Check the **`reports/`** directory. It will now contain:
        -   `metrics_summary.csv`: A CSV file with the final scores for all policies.
        -   Multiple `.png` image files visualizing the results, such as `prophet_containers_over_time.png` and `lstm_policy_bars.png`. These plots are excellent for understanding the trade-offs between different strategies.
