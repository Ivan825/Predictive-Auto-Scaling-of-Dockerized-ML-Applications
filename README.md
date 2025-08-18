# Predictive Auto-Scaling Project: Runbook & Guide

This document provides a complete guide to understanding, setting up, and running the unified predictive auto-scaling project.

---

## Part 1: How to Run Everything (Step-by-Step Guide)

This section explains what the system does, why it works, and how to run it.

### The 5 W's and 1 H: Understanding the System

*   **WHAT is this?**
    *   This is a predictive auto-scaling system for a Dockerized machine learning application. It automatically adjusts the number of running application containers based on *predicted* future demand, not just current demand.

*   **WHO is this for?**
    *   This system is for developers and operators running services (especially ML services) that experience variable demand and need to scale efficiently to maintain performance and control costs.

*   **WHEN does it operate?**
    *   The system runs continuously. The auto-scaler component checks for new metrics, runs a forecast, and makes a scaling decision every 60 seconds (by default).

*   **WHERE does it run?**
    *   The entire system is designed to run locally on your machine using Docker and Docker Compose. Each component (the ML app, Prometheus, the auto-scaler) runs in its own container.

*   **WHY is this approach useful?**
    *   **Proactive vs. Reactive:** Traditional auto-scaling is *reactive*—it scales up only after load has already increased, which can lead to slow response times for users. This system is *proactive*. It forecasts demand minutes into the future and scales up *before* the load arrives, ensuring low latency and a better user experience.
    *   **Efficiency:** By scaling down during predicted lulls, it prevents over-provisioning and saves computational resources and costs.
    *   **Intelligence:** It uses sophisticated forecasting models (like Prophet and LSTM) and scaling policies that can account for trends, seasonality, and uncertainty, making it much smarter than simple threshold-based scaling.

*   **HOW does it work?**
    1.  **Metrics Collection:** The `ml_app` (a Flask application) exposes performance metrics. A `Prometheus` container is configured to scrape and store these metrics, primarily the incoming request rate.
    2.  **Forecasting:** The `autoscaler` container runs the main `scale_decision_engine.py` script. In a loop, it queries the Prometheus database for the latest historical demand data.
    3.  **Prediction:** It feeds this historical data into a pre-trained forecasting model (e.g., `prophet_model.joblib`) to predict the demand for the next 15 minutes.
    4.  **Policy Decision:** The predicted demand (including the confidence interval, `yhat_upper`) is passed to a scaling policy function (e.g., `confidence_aware`). This policy calculates the optimal number of containers needed to handle the predicted load, adding a buffer for safety.
    5.  **Safety Check:** An optional safety net can verify the decision against operational constraints (e.g., maximum latency) to prevent risky scaling actions.
    6.  **Scaling Action:** The engine communicates with the Docker daemon on your host machine to command Docker Compose to scale the `ml_app` service up or down to the target number of replicas.

### How to Make It Happen: Execution Steps

Here is the sequence of commands to get the entire system running from scratch.

**Prerequisites:**
*   Docker Desktop is installed and running.
*   Python 3.8+ is installed.

**Step 1: Install Dependencies**

Install all the necessary Python packages using the new unified `requirements.txt` file.

```bash
pip install -r requirements.txt
```

**Step 2: Train and Save the Forecasting Models**

The live auto-scaler needs models to make predictions. Run the new training script. This will generate the `prophet_model.joblib` and `lstm_model.pt` files and save them in a new `/models` directory.

```bash
# Run the training script from the project root directory:
python scripts/train_models.py
```

**Step 3: Launch the Full System**

With the models trained and all files in place, you can now launch the entire stack using Docker Compose. This single command will build the necessary Docker images and start all the services defined in `docker-compose.yml`.

```bash
# Make sure Docker Desktop is running, then execute:
docker-compose up --build
```

**What will happen:**
*   Docker will build three images: one for your `ml_app`, one for the `autoscaler`, and one for `prometheus`.
*   It will start one container for Prometheus.
*   It will start one container for the `autoscaler`.
*   It will start with an initial number of `ml_app` containers (e.g., 1).
*   You will see logs from all containers in your terminal. Watch the `autoscaler` logs to see it making decisions and scaling the `ml_app` up and down.

You now have a fully operational, predictive auto-scaling system running on your machine!