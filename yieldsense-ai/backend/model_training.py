"""
Yield Prediction Module — Model Training

Trains a regression model on soil/weather/farm features and saves
all artifacts (model, encoders, feature names, metrics) to the /models directory.

Usage:
    python model_training.py --data ../data/sample_crop_yield_data.csv
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from data_preprocessing import preprocess  # noqa: E402

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)


def build_model():
    """Builds XGBoost Regressor or sklearn GradientBoostingRegressor fallback."""
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            objective="reg:squarederror",
        )
        return model, "XGBoost"

    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            loss="squared_error",
        )
        return (
            model,
            "GradientBoosting (sklearn fallback)"
        )


def train(data_path: str):
    """Loads dataset, preprocesses features, trains regressor, and exports model artifacts."""
    print("\n" + "=" * 60)
    print("YIELD PREDICTION MODEL TRAINING")
    print("=" * 60)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"\nDataset: {data_path}")

    X, y, encoders, feature_cols = preprocess(data_path)

    print("\n" + "=" * 60)
    print("DATA CHECK")
    print("=" * 60)
    print(f"Number of rows     : {len(X)}")
    print(f"Number of features: {len(feature_cols)}")

    y_array = np.asarray(y, dtype=float)

    print("\nTarget statistics:")
    print(f"Mean  : {np.mean(y_array):.4f}")
    print(f"Min   : {np.min(y_array):.4f}")
    print(f"Max   : {np.max(y_array):.4f}")

    if np.isnan(y_array).any() or np.isinf(y_array).any():
        raise ValueError("Target contains NaN or Inf values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, model_name = build_model()
    print(f"\nModel: {model_name}")

    print("Training model...")
    model.fit(X_train, y_train)

    raw_preds = model.predict(X_test)
    preds = np.maximum(0.0, np.asarray(raw_preds, dtype=float))

    print("\n" + "=" * 60)
    print("PREDICTION EVALUATION")
    print("=" * 60)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)

    metrics = {
        "model": model_name,
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "target_min": round(float(np.min(y_array)), 4),
        "target_max": round(float(np.max(y_array)), 4),
        "target_mean": round(float(np.mean(y_array)), 4),
        "prediction_min": round(float(np.min(preds)), 4),
        "prediction_max": round(float(np.max(preds)), 4),
        "prediction_mean": round(float(np.mean(preds)), 4),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "yield_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts to:", MODEL_DIR)
    print(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop yield model.")
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "sample_crop_yield_data.csv"
        ),
        help="Path to training CSV."
    )
    args = parser.parse_args()
    train(args.data)
