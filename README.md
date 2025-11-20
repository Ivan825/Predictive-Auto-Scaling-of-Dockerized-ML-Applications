# Predictive Auto-Scaling of Dockerized ML Applications

This project implements a proof-of-concept framework for predictive auto-scaling in containerized environments. By combining **Facebook Prophet** (for seasonality-aware forecasting) and **Long Short-Term Memory (LSTM)** networks (for sequential pattern recognition), the system anticipates workload demand before it spikes.

The framework simulates a Docker-based machine learning environment to evaluate various scaling policies—ranging from simple reactive rules to advanced confidence-aware predictive strategies.

---

## Project Structure

```text
.
├── data/                   # Synthetic workload data (generated during runtime)
├── reports/                # Output plots and simulation metrics
├── scripts/
│   └── run_experiment.py   # Main entry point to run the full simulation
├── src/
│   ├── models/             # Prophet and LSTM model definitions
│   ├── policies.py         # Auto-scaling logic (Static, Buffered, Confidence-Aware)
│   ├── simulator.py        # Core event loop simulating minute-by-minute traffic
│   ├── metrics.py          # SLA violation and cost calculations
│   └── simulate_data.py    # Generator for synthetic workload traces
├── requirements.txt        # Python dependencies
└── README.md
```

## Simulation Workflow

The experiment runs through four distinct stages automatically:

1.  **Synthetic Data Generation**: Creates a 14-day workload trace featuring daily seasonality, random noise, and bursty "event" spikes (e.g., marketing promotions).

2.  **Model Training**:
    *   **Prophet**: Fits a decomposable time-series model handling trend and seasonality.
    *   **LSTM**: Trains a neural network on sliding windows of historical demand.

3.  **Policy Simulation**: Replays the test data minute-by-minute. The autoscaler decides the number of containers based on the forecast and the chosen policy (e.g., Buffered, Simple Ceiling).

4.  **Evaluation**: Calculates key metrics:
    *   **SLA Violations**: (Under-provisioning events)
    *   **Cost Index**: (Total container-uptime minutes)
    *   **Stability**: (Number of scaling switches)

---

## Installation & Setup

### 1. Prerequisites

*   Python 3.8+ (Python 3.11 recommended)
*   Git

### 2. Clone the Repository

```sh
git clone <YOUR_REPO_URL>
cd predictive-auto-scaling-of-dockerized-ml-applications
```

### 3. Environment Setup

**macOS / Linux**

```sh
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Windows**

```powershell
# Create a virtual environment
python -m venv .venv

# Activate the environment
# If using Command Prompt (cmd.exe):
.venv\Scripts\activate.bat
# If using PowerShell:
.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```sh
pip install -r requirements.txt
```
> **Note for Apple Silicon (M1/M2/M3) Users**: If you encounter issues installing PyTorch, use the platform-specific command: `pip install "torch>=2.2,<2.4" torchvision torchaudio`

---

## Usage

To run the complete experiment (Data Generation → Training → Simulation → Evaluation):

```sh
# Ensure your virtual environment is active
python -m scripts.run_experiment
```

### Expected Output

*   **Console Logs**: You will see training progress bars for Prophet and LSTM, followed by a summary table comparing the performance of all 5 scaling policies.
*   **Plots & Reports**: Check the `reports/` directory for:
    *   `metrics_summary.csv`: Raw performance numbers.
    *   `*_forecast.png`: Visualization of model predictions vs. actual demand.
    *   `*_policy_comparison.png`: Bar charts comparing cost vs. SLA violations.

---

## Auto-Scaling Policies Implemented

The simulator compares five distinct strategies:

*   **Static-2**: Fixed allocation (Baseline). Low complexity, high SLA violation risk.
*   **Simple Ceiling**: Scales strictly based on forecast / capacity. Efficient but reactive.
*   **Buffered**: Adds a safety margin (+15%) to the forecast. Balances cost and reliability.
*   **Confidence-Aware**: Scales based on the upper bound of the forecast's confidence interval (Prophet only). Zero violations, high cost.
*   **Policy-Aware**: Incremental scaling (add/remove 1 container at a time) to reduce system thrashing.