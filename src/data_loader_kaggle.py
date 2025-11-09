# src/data_loader_kaggle.py
import pandas as pd
import os

def load_spy_csv(file_path: str):
    print(f"[data_loader] Loading SPY CSV: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if 'date' not in df.columns:
        raise ValueError("CSV must have a 'date' column")
    df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['datetime']).copy()
    if 'volume' not in df.columns:
        raise ValueError("CSV must have a 'volume' column")
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
    df = df.set_index('datetime').sort_index()
    vol = df['volume'].resample('1T').sum().fillna(0.0)
    print(f"[data_loader] Loaded {len(vol)} rows, {vol.index.min()} -> {vol.index.max()}")
    return vol

def save_load_series(series: pd.Series, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    series.rename('load').reset_index().to_csv(out_path, index=False)
    print(f"[data_loader] Saved load series to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_csv", default="data/load_series.csv")
    args = parser.parse_args()
    s = load_spy_csv(args.csv_path)
    save_load_series(s, args.out_csv)
