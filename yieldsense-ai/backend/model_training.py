"""
Yield Prediction Module — model training.

Trains a regression model to predict yield_tonnes_per_ha from weather,
soil, and farm-management features, then saves the model + encoders +
feature list + metrics to /models so the API can load them.

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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def build_model():
    """Prefer XGBoost (per the tech stack); fall back to sklearn's
    GradientBoosting if xgboost isn't installed, so this always runs."""
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        ), "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42,
        ), "GradientBoosting (sklearn fallback — install xgboost for the real thing)"


def train(data_path: str):
    X, y, encoders, feature_cols = preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model, model_name = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "model": model_name,
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2": round(r2_score(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "yield_model.joblib"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "sample_crop_yield_data.csv"),
    )
    args = parser.parse_args()
    train(args.data)
