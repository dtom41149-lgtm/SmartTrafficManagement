import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "results" / "cleaned_traffic_data.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "model_ready_data.csv"


def main():

    print("===== IMPROVED FEATURE ENGINEERING =====")

    print("\nLoading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Rows loaded: {len(df)}")

    target = "Traffic Density"

    # -----------------------------
    # Time-based feature engineering
    # -----------------------------

    # Convert hour into cyclic features.
    # This helps the model understand that 23:00 and 00:00
    # are close to each other.
    import numpy as np

    df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour Of Day"] / 24)
    df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour Of Day"] / 24)

    # Create broad time-of-day categories
    def get_time_period(hour):
        if 6 <= hour < 10:
            return "Morning Peak"
        elif 10 <= hour < 16:
            return "Daytime"
        elif 16 <= hour < 20:
            return "Evening Peak"
        elif 20 <= hour < 24:
            return "Night"
        else:
            return "Late Night"

    df["Time Period"] = df["Hour Of Day"].apply(get_time_period)

    # Speed-related feature
    df["Speed_Peak_Interaction"] = (
        df["Speed"] * df["Is Peak Hour"]
    )

    feature_columns = [
        "City",
        "Vehicle Type",
        "Weather",
        "Economic Condition",
        "Day Of Week",
        "Time Period",
        "Hour Of Day",
        "Hour_Sin",
        "Hour_Cos",
        "Speed",
        "Is Peak Hour",
        "Random Event Occurred",
        "Speed_Peak_Interaction"
    ]

    model_df = df[feature_columns + [target]].copy()

    model_df = model_df.dropna().reset_index(drop=True)

    print("\nNew features created:")
    print("- Hour_Sin")
    print("- Hour_Cos")
    print("- Time Period")
    print("- Speed_Peak_Interaction")

    print(f"\nTarget variable: {target}")
    print(f"Final rows: {len(model_df)}")

    model_df.to_csv(OUTPUT_PATH, index=False)

    print("\nModel-ready dataset saved to:")
    print(OUTPUT_PATH)

    print("\nImproved feature engineering completed successfully.")


if __name__ == "__main__":
    main()