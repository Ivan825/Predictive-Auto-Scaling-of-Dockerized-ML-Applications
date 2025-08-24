import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ---------- data utilities ----------

def make_windows(series, past=60, ahead=15):
    """
    Convert a 1-D time series into (X, y) where:
      - X: past `past` points
      - y: average of the next `ahead` points
    Returns:
      X: (N, past, 1) float32
      y: (N,) float32
    """
    X, y = [], []
    for i in range(past, len(series) - ahead + 1):
        X.append(series[i - past:i])
        y.append(series[i:i + ahead].mean())
    X = np.array(X, dtype=np.float32)[:, :, None]  # (N, past, 1)
    y = np.array(y, dtype=np.float32)              # (N,)
    return X, y

class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)    # (N, past, 1)
        self.y = torch.from_numpy(y)    # (N,)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

# ---------- model ----------

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, layer_dim=1, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        # x: (B, past, 1)
        out, _ = self.lstm(x)       # (B, past, hidden)
        last = out[:, -1, :]        # (B, hidden)
        return self.head(last).squeeze(-1)  # (B,)

# ---------- train / predict API (kept same names) ----------

def train_lstm(df, past=60, ahead=15, epochs=10, batch_size=128, lr=1e-3, hidden=64, device=None):
    """
    Trains a PyTorch LSTM to predict the *average* of the next `ahead` minutes.
    Returns: (model, cfg) where cfg stores past/ahead/device.
    """
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    series = df["y"].values.astype("float32")
    X, y = make_windows(series, past=past, ahead=ahead)
    if len(X) < 2:
        raise ValueError("Not enough data to make windows; increase data length or reduce past/ahead.")

    split = int(len(X) * 0.8)
    Xtr, ytr = X[:split], y[:split]
    Xval, yval = X[split:], y[split:]

    ds_tr = SeqDS(Xtr, ytr)
    ds_val = SeqDS(Xval, yval)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = LSTM(input_dim=1, hidden_dim=hidden, layer_dim=1, output_dim=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        # train
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)             # (B,)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(ds_tr)

        # quick validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in dl_val:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
            val_loss /= max(1, len(ds_val))
        # Uncomment for logs:
        # print(f"epoch {epoch+1}/{epochs}  train_mse={tr_loss:.4f}  val_mse={val_loss:.4f}")

    cfg = {
        "past": past, "ahead": ahead, "device": device, 
        "input_dim": 1, "hidden_dim": hidden, "layer_dim": 1, "output_dim": 1
    }
    return model, cfg

def predict_lstm(model, df, cfg):
    """
    Returns a single scalar: predicted *average* demand over the next `ahead` minutes,
    using the most recent `past` points in df.
    """
    device = cfg.get("device", "cpu")
    series = df["y"].values.astype("float32")
    past = cfg["past"]; ahead = cfg["ahead"]

    if len(series) < past:
        raise ValueError("Not enough points to run prediction with given past window.")

    x = series[-past:].reshape(1, past, 1)  # (1, past, 1)
    x_t = torch.from_numpy(x).to(device)

    model.eval()
    with torch.no_grad():
        yhat = model(x_t).cpu().numpy()[0].item()
    return float(yhat)
