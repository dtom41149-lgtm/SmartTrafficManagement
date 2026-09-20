import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_DIR / "results" / "cleaned_traffic_data.csv"
OUTPUT_DIR = PROJECT_DIR / "results"

print("===== CREATING DASHBOARD SUMMARY DATA =====")

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} records.")

# 1. Hourly traffic summary
hourly = (
    df.groupby("Hour Of Day")
    .agg(
        Average_Traffic_Density=("Traffic Density", "mean"),
        Average_Speed=("Speed", "mean"),
        Record_Count=("Traffic Density", "count")
    )
    .reset_index()
)

hourly.to_csv(
    OUTPUT_DIR / "dashboard_hourly_analysis.csv",
    index=False
)

# 2. Peak vs non-peak summary
peak = (
    df.groupby("Is Peak Hour")
    .agg(
        Average_Traffic_Density=("Traffic Density", "mean"),
        Average_Speed=("Speed", "mean"),
        Record_Count=("Traffic Density", "count")
    )
    .reset_index()
)

peak["Period"] = peak["Is Peak Hour"].map({
    0: "Non-Peak",
    1: "Peak"
})

peak.to_csv(
    OUTPUT_DIR / "dashboard_peak_analysis.csv",
    index=False
)

# 3. Traffic density distribution
bins = [0, 0.15, 0.30, float("inf")]
labels = ["Low", "Moderate", "High"]

df["Traffic Level"] = pd.cut(
    df["Traffic Density"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

distribution = (
    df["Traffic Level"]
    .value_counts()
    .reindex(labels, fill_value=0)
    .reset_index()
)

distribution.columns = ["Traffic Level", "Record Count"]

distribution.to_csv(
    OUTPUT_DIR / "dashboard_traffic_distribution.csv",
    index=False
)

print("\nCreated:")
print("1. results/dashboard_hourly_analysis.csv")
print("2. results/dashboard_peak_analysis.csv")
print("3. results/dashboard_traffic_distribution.csv")

print("\nDashboard summary data created successfully.")