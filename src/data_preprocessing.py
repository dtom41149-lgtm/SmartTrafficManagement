import pandas as pd
from pathlib import Path


# Dataset location
DATA_PATH = Path(r"C:\Users\akars\Desktop\datasetD\futuristic_city_traffic.csv")

# Project result folder
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Output file
CLEANED_DATA_PATH = RESULTS_DIR / "cleaned_traffic_data.csv"


def load_data():
    """Load the original traffic dataset."""
    print("Loading traffic dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


def preprocess_data(df):
    """Perform basic data quality checks and preprocessing."""

    print("\nChecking missing values...")
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values: {missing_values}")

    print("\nChecking duplicate rows...")
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    # Remove exact duplicate rows if present
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Rows after removing duplicates: {len(df)}")
    else:
        print("No duplicate rows found.")

    return df


def save_data(df):
    """Save the cleaned dataset."""
    df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"\nCleaned dataset saved to:")
    print(CLEANED_DATA_PATH)


def main():
    df = load_data()
    df = preprocess_data(df)
    save_data(df)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()