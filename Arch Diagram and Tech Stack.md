# 4. Arch Diagram and Tech Stack

Author: Kanav Bhardwaj
Status: In Review
Category: Guide, Tech Spec

# Full Tech Stack for Predictive Auto-Scaling System

---

## 1. **Containerization & Orchestration**

| Tool | Purpose |
| --- | --- |
| **Docker** | Containerize ML inference applications and forecasting logic |
| **Docker Compose** | Define and manage multi-container configurations locally |
| **Docker CLI** | Direct scaling, monitoring, and container management |
| **(Optional) Kubernetes** | For advanced production-level orchestration |
| **Docker Swarm** | Lightweight orchestration if Kubernetes is overkill |

---

## 2. **Machine Learning Inference (Domain-Specific Models)**

| Use Case | Model Type | Library/Framework |
| --- | --- | --- |
| Defect detection | SVC, KNN, NN | `scikit-learn` |
| Fraud detection | XGBoost, LightGBM | `xgboost`, `lightgbm` |
| Algo trading | LSTM, Transformers | `keras`, `transformers` |
| General ML serving | REST API for predictions | `FastAPI`, `Flask` |

---

## 3. **Time-Series Forecasting Models (Predictive Scaling Logic)**

| Model | Use Case | Library |
| --- | --- | --- |
| **Prophet** | Trend + seasonality-based load forecasting | `prophet` |
| **LSTM** | Non-linear sequential forecasting | `keras`, `tensorflow` |
| **XGBoost** | Lag-based prediction using tabular data | `xgboost` |

---

## 4. **Monitoring & Metrics Collection**

| Tool | Purpose |
| --- | --- |
| **Prometheus** | Scrape metrics from containers (CPU, memory, req/min) |
| **Grafana** | Visual dashboards for real-time system performance |
| `docker stats` | CLI for local monitoring of container metrics |
| `psutil` | Collect system-level metrics if Prometheus is not used |

---

## 5. **Load Generation & Logging (for Forecast Training)**

| Tool/Script | Purpose |
| --- | --- |
| `simulate_data.py` | Create synthetic load patterns for training |
| `logging middleware` | Logs request timestamps, model latency, resource usage |
| `pandas`, `numpy` | Preprocess logs for modeling |

---

## 6. **Model Evaluation & Automation**

| Tool/Library | Purpose |
| --- | --- |
| `sklearn.metrics` | MAE, RMSE, MAPE, etc. for forecast evaluation |
| `cron` or `APScheduler` | Schedule retraining or prediction every N minutes |
| `joblib`, `pickle` | Save/load forecasting models |

---

## 7. **Scaling Trigger Logic**

| Component | Function |
| --- | --- |
| `scale_decision.py` | Maps predicted load → number of containers |
| `os.system` / `subprocess` | Executes Docker CLI commands (`docker run`, `docker stop`) |
| **(Optional)** Kubernetes HPA | Integrate with Horizontal Pod Autoscaler for production |

---

## 8. **Cloud Infrastructure (for Smart Manufacturing or Finance)**

| Provider | Services Used |
| --- | --- |
| **AWS EC2** | Host containers, Prometheus, Grafana, model APIs |
| **S3 (optional)** | Store model checkpoints and logs |
| **CloudWatch (optional)** | Extra monitoring & alerting |
| **(Optional)** GCP / Azure | Interchangeable setup |

---

## 9. **Development & Collaboration**

| Tool | Purpose |
| --- | --- |
| **VS Code / PyCharm** | Python development IDE |
| **Git + GitHub** | Version control, collaboration |
| **Jupyter Notebooks** | Rapid prototyping & visualization |

---

## 10. **Documentation & Reporting**

| Tool | Purpose |
| --- | --- |
| **Markdown / LaTeX / Notion** | For writing the research paper/report |
| **Draw.io / Lucidchart/Plant UML** | System architecture diagrams |
| **Matplotlib / Seaborn** | Forecast & metric visualizations |

---

## Optional Extensions (For Research Value)

| Extension | Benefit |
| --- | --- |
| **Reinforcement Learning scaler** | Novelty; learn optimal scaling from environment |
| **Model drift detection (with Grafana alerts)** | Trigger retraining only when accuracy drops |

# Arch diagram

![WhatsApp Image 2025-05-24 at 00.05.11_57f2c64a.jpg](4%20Arch%20Diagram%20and%20Tech%20Stack%201fa440e2284c80668122f85ab369d13b/WhatsApp_Image_2025-05-24_at_00.05.11_57f2c64a.jpg)

The architectural diagram presented using PlantUML captures the modular, layered design of our system for predictive auto-scaling of machine learning applications deployed in Docker containers. The system is built to support smart manufacturing use cases and follows IEEE best practices for architectural clarity, separation of concerns, and extensibility.

### 1. **Cloud Infrastructure Layer (AWS EC2)**

At the base is the **Docker Host**, a virtual machine hosted on AWS EC2. It runs all containerized components:

- **ML Inference Container**: Hosts the trained ML model responsible for real-time defect detection or other domain-specific inference tasks.
- **Scaling Agent Container**: Periodically evaluates forecasted demand and makes scale-up/scale-down decisions by invoking Docker APIs.

In addition to containers, **Prometheus** is deployed on the same or companion node to monitor container metrics, and **Grafana** is used for dashboard visualization of performance, usage, and scaling behavior.

---

### 2. **Monitoring & Logging Layer**

This layer collects real-time and historical data used for forecasting and system analysis:

- **Docker Stats**: Provides low-level container metrics (CPU, memory, etc.).
- **Logging Middleware**: Intercepts and logs request traffic and model latency data.
- **Request Logs** and **Resource Metrics**: Structured datasets that are used as inputs to the forecasting models and for evaluation purposes.

This comprehensive logging enables both forecasting accuracy and responsive fallback decisions.

---

### 3. **Forecasting Layer**

The forecasting component is a critical enabler of proactive scaling:

- **Forecast Scheduler**: Triggers prediction routines at fixed intervals (e.g., every 5–10 minutes).
- **Prophet Model**: A time-series model used to predict workload trends and seasonality.
- **LSTM Model**: A deep learning model suited for capturing complex temporal dependencies and bursty traffic patterns.

Both models are trained on logs and resource metrics, and output short-term predictions (e.g., 15–60 minutes ahead) of incoming request volume and resource usage.

---

### 4. **Scaling Logic and Execution Layer**

This layer decides **how many containers** should be active and initiates the scaling process:

- **Scale Decision Engine**: Receives predictions and system state, and invokes the appropriate scaling logic.
- **Scaling Policy Module**: Implements different auto-scaling policies (e.g., simple ceiling logic, buffered scaling, confidence-aware logic, etc.).
- **Docker CLI / API**: Executes the scale-up/scale-down actions using Docker commands, maintaining system elasticity.

This modular approach allows future enhancement through plug-and-play logic modules or policy-selecting reinforcement learning agents.

---

### 5. **Feedback Loop and Reactive Fallback**

In addition to forecasting-based scaling, the architecture includes a **reactive safety mechanism**:

- If forecasts are inaccurate (e.g., sudden spike missed), real-time metrics from Prometheus can trigger emergency scaling.
- This hybrid (predictive + reactive) design enhances system robustness and ensures reliable operation under unpredictable workloads.

---

### **Modularity and Extensibility**

The system is fully modular and containerized:

- Forecasting models, inference logic, and scaling agents are all independently deployable Docker containers.
- Monitoring and visualization are decoupled from core logic.
- The design is domain-agnostic and has been validated for transferability across smart manufacturing, finance, healthcare, and more.

---