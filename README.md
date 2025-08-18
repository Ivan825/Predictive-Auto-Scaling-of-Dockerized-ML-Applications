# Auto-Scaling Simulation (Prophet + LSTM)

This project simulates demand, trains forecasting models (Prophet, LSTM), and evaluates different auto-scaling policies.

---

## 1. Prerequisites

- **macOS** (Intel or Apple Silicon)
- **Homebrew** installed
- Git + basic Python tooling

---

## 2. Install Python 3.11 and create virtual environment

```bash
# Install Python 3.11 via Homebrew
brew install python@3.11

# Navigate to project root
cd /path to root

# Remove any old virtualenv
rm -rf .venv

# Create new venv with Python 3.11
/opt/homebrew/bin/python3.11 -m venv .venv   # (on Intel Macs, path may be /usr/local/bin/python3.11)

# Activate venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

Install dependencies
Core libraries
pip install numpy pandas matplotlib

Prophet (time-series forecasting)
pip install prophet

PyTorch (for LSTM)

Apple Silicon (M1/M2/M3):

pip install "torch>=2.2,<2.4" torchvision torchaudio


Intel Mac (CPU only):

pip install "torch>=2.2,<2.4" --index-url https://download.pytorch.org/whl/cpu

Verify setup

Check Prophet:

python -c "import prophet; print('Prophet OK')"


Check PyTorch:

python -c "import torch; print('Torch version:', torch.__version__, '| MPS available:', torch.backends.mps.is_available())"

Run the experiment
# From project root
source .venv/bin/activate
python -m scripts.run_experiment
