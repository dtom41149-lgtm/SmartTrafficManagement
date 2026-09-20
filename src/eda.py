import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "results" / "cleaned_traffic_data.csv"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    print("Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    return df


def basic_analysis(df):
    print("\n===== DATASET INFORMATION =====")
    print(df.info())

    print("\n===== NUMERICAL SUMMARY =====")
    print(df.describe())

    print("\n===== CATEGORICAL VALUES =====")

    for column in [
        "City",
        "Vehicle Type",
        "Weather",
        "Economic Condition",
        "Day Of Week"
    ]:
        print(f"\n{column}:")
        print(df[column].value_counts())


def create_visualizations(df):

    # 1. Traffic Density Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["Traffic Density"], bins=50)
    plt.xlabel("Traffic Density")
    plt.ylabel("Number of Records")
    plt.title("Traffic Density Distribution")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "traffic_density_distribution.png")
    plt.close()

    # 2. Average Traffic Density by Hour
    hourly_density = df.groupby("Hour Of Day")["Traffic Density"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_density.index, hourly_density.values, marker="o")
    plt.xlabel("Hour Of Day")
    plt.ylabel("Average Traffic Density")
    plt.title("Average Traffic Density by Hour")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "traffic_density_by_hour.png")
    plt.close()

    # 3. Average Speed by Hour
    hourly_speed = df.groupby("Hour Of Day")["Speed"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(hourly_speed.index, hourly_speed.values, marker="o")
    plt.xlabel("Hour Of Day")
    plt.ylabel("Average Speed")
    plt.title("Average Vehicle Speed by Hour")
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "speed_by_hour.png")
    plt.close()

    # 4. Peak vs Non-Peak Traffic Density
    peak_density = df.groupby("Is Peak Hour")["Traffic Density"].mean()

    plt.figure(figsize=(8, 6))
    plt.bar(
        ["Non-Peak", "Peak"],
        [
            peak_density.get(0, 0),
            peak_density.get(1, 0)
        ]
    )
    plt.xlabel("Traffic Period")
    plt.ylabel("Average Traffic Density")
    plt.title("Peak vs Non-Peak Traffic Density")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "peak_vs_nonpeak.png")
    plt.close()

    print("\nEDA graphs saved successfully.")


def main():
    df = load_data()
    basic_analysis(df)
    create_visualizations(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()