import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_DIR / "results" / "model_ready_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "traffic_density_model.pkl"
RESULTS_PATH = PROJECT_DIR / "results" / "model_metrics.txt"


def main():

    print("===== MODEL EVALUATION =====")

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Traffic Density"])
    y = df["Traffic Density"]

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # Use the same test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"\nTesting records: {len(X_test)}")

    print("\nGenerating predictions...")
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\n===== MODEL PERFORMANCE =====")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")

    # Save metrics
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("Smart Traffic Management - Model Evaluation\n")
        file.write("=" * 50 + "\n")
        file.write(f"MAE  : {mae:.6f}\n")
        file.write(f"RMSE : {rmse:.6f}\n")
        file.write(f"R2   : {r2:.6f}\n")

    print("\nMetrics saved to:")
    print(RESULTS_PATH)

    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()