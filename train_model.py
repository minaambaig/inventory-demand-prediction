from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_inventory_data.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "inventory_demand_model.joblib"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])

    return data


def prepare_data(data):
    features = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Inventory Level",
        "Price",
        "Discount",
        "Weather Condition",
        "Holiday/Promotion",
        "Competitor Pricing",
        "Seasonality",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear",
        "Quarter",
        "IsWeekend",
        "PriceDifference",
        "DiscountedPrice",
        "Lag_1",
        "Lag_7",
        "Lag_14",
        "Lag_30",
        "RollingMean_7",
        "RollingMean_30"
    ]

    target = "Units Sold"

    # Time-based split
    split_date = pd.Timestamp("2023-10-01")

    train_data = data[data["Date"] < split_date].copy()
    test_data = data[data["Date"] >= split_date].copy()

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    categorical_features = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality"
    ]

    categorical_indices = [
        features.index(column)
        for column in categorical_features
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        categorical_indices
    )


def train_model():
    data = load_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        categorical_indices
    ) = prepare_data(data)

    model = CatBoostRegressor(
        iterations=700,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_indices
    )

    predictions = model.predict(X_test)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    r2 = r2_score(y_test, predictions)

    print("\nMODEL PERFORMANCE")
    print("-----------------")
    print(f"MAE:  {mae:.2f} units")
    print(f"RMSE: {rmse:.2f} units")
    print(f"R²:   {r2:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)

    model_package = {
        "model": model,
        "features": features
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()