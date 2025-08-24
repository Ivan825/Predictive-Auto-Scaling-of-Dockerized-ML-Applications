
import pandas as pd
import plotly.express as px
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'google_trace_preprocessed.csv')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
OUTPUT_IMAGE_FILE = os.path.join(REPORTS_DIR, 'workload_distribution.png')

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Load the Data
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: The file {DATA_FILE} was not found.")
    exit()

# Ensure the 'req_per_min' column exists
if 'req_per_min' not in df.columns:
    print("Error: 'req_per_min' column not found in the data.")
    exit()

# 2. Statistical Summary
print("--- Workload Statistical Summary ---")
stats = df['req_per_min'].describe()
print(stats)
print("\n")

# 3. Quantile Analysis
print("--- Workload Quantile Analysis ---")
quantiles = df['req_per_min'].quantile([0.9, 0.95, 0.99])
print(quantiles)
print("\n")

# 4. Visualization
print(f"--- Generating Workload Distribution Histogram ---")
fig = px.histogram(df, x='req_per_min', nbins=100, title='Distribution of Requests per Minute')
fig.update_layout(
    xaxis_title="Requests per Minute",
    yaxis_title="Frequency",
    title_font_size=20,
    xaxis_title_font_size=14,
    yaxis_title_font_size=14
)

# Save the plot
try:
    fig.write_image(OUTPUT_IMAGE_FILE)
    print(f"Histogram saved to {OUTPUT_IMAGE_FILE}")
except Exception as e:
    print(f"Error saving plot: {e}")
    print("Please ensure you have the 'kaleido' package installed (`pip install kaleido`) for static image export.")

