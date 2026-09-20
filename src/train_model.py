import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "results" / "model_ready_data.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "traffic_density_model.pkl"


def main():

    print("===== IMPROVED TRAFFIC MODEL TRAINING =====")

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Rows loaded: {len(df)}")

    X = df.drop(columns=["Traffic Density"])
    y = df["Traffic Density"]

    categorical_features = [
        "City",
        "Vehicle Type",
        "Weather",
        "Economic Condition",
        "Day Of Week",
        "Time Period"
    ]

    numerical_features = [
        "Hour Of Day",
        "Hour_Sin",
        "Hour_Cos",
        "Speed",
        "Is Peak Hour",
        "Random Event Occurred",
        "Speed_Peak_Interaction"
    ]

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    print("\nTraining improved HistGradientBoosting model...")

    pipeline.fit(X_train, y_train)

    print("\nModel training completed.")

    joblib.dump(pipeline, MODEL_PATH)

    print("\nImproved model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()