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

# Make sure local modules can be imported
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import preprocess  # noqa: E402


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)


# ============================================================
# MODEL
# ============================================================

def build_model():
    """
    Prefer XGBoost.

    If XGBoost is not installed, fall back to sklearn
    GradientBoostingRegressor.
    """

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
            "GradientBoosting "
            "(sklearn fallback — install xgboost for the real thing)"
        )


# ============================================================
# TRAINING
# ============================================================

def train(data_path: str):
    """
    Load data, preprocess it, train the regression model,
    evaluate it, and save all required model artifacts.
    """

    print("\n" + "=" * 60)
    print("YIELD PREDICTION MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. CHECK DATA FILE
    # --------------------------------------------------------

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"\nDataset not found:\n{data_path}\n"
            f"\nPlease check the --data path."
        )

    print(f"\nDataset: {data_path}")

    # --------------------------------------------------------
    # 2. PREPROCESS DATA
    # --------------------------------------------------------

    X, y, encoders, feature_cols = preprocess(data_path)

    print("\n" + "=" * 60)
    print("DATA CHECK")
    print("=" * 60)

    print(f"Number of rows     : {len(X)}")
    print(f"Number of features: {len(feature_cols)}")

    print("\nFeatures used by model:")
    for i, feature in enumerate(feature_cols, start=1):
        print(f"{i:2}. {feature}")

    # --------------------------------------------------------
    # 3. TARGET CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TARGET CHECK")
    print("=" * 60)

    print("Target column: yield_tonnes_per_ha")

    # Convert target to numpy array safely
    y_array = np.asarray(y, dtype=float)

    print("\nTarget statistics:")

    print(f"Count : {len(y_array)}")
    print(f"Mean  : {np.mean(y_array):.6f}")
    print(f"Std   : {np.std(y_array):.6f}")
    print(f"Min   : {np.min(y_array):.6f}")
    print(f"Max   : {np.max(y_array):.6f}")
    print(f"Median: {np.median(y_array):.6f}")

    print("\nFirst 10 target values:")
    print(y_array[:10].tolist())

    # --------------------------------------------------------
    # 4. CHECK FOR INVALID TARGET VALUES
    # --------------------------------------------------------

    if np.isnan(y_array).any():
        raise ValueError(
            "Target column contains NaN values. "
            "Please clean yield_tonnes_per_ha."
        )

    if np.isinf(y_array).any():
        raise ValueError(
            "Target column contains infinite values. "
            "Please clean yield_tonnes_per_ha."
        )

    # --------------------------------------------------------
    # 5. CHECK TARGET VARIATION
    # --------------------------------------------------------

    unique_targets = np.unique(y_array)

    print(f"\nUnique target values: {len(unique_targets)}")

    if len(unique_targets) <= 1:
        raise ValueError(
            "\nERROR: yield_tonnes_per_ha contains only one unique value.\n"
            "A regression model cannot learn meaningful predictions "
            "from a constant target."
        )

    # Warn if most values are zero
    zero_count = int(np.sum(y_array == 0))
    zero_percentage = (zero_count / len(y_array)) * 100

    print(f"Zero target values : {zero_count}")
    print(f"Zero percentage    : {zero_percentage:.2f}%")

    if zero_percentage > 90:
        print(
            "\nWARNING: More than 90% of target values are zero."
        )
        print(
            "This may cause the model to predict values close to zero."
        )

    # --------------------------------------------------------
    # 6. CHECK FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE CHECK")
    print("=" * 60)

    X_array = np.asarray(X, dtype=float)

    print(f"X shape: {X_array.shape}")
    print(f"y shape: {y_array.shape}")

    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "Number of feature rows does not match target rows."
        )

    if np.isnan(X_array).any():
        raise ValueError(
            "Feature matrix contains NaN values after preprocessing."
        )

    if np.isinf(X_array).any():
        raise ValueError(
            "Feature matrix contains infinite values after preprocessing."
        )

    print("Feature matrix contains no NaN values.")
    print("Feature matrix contains no infinite values.")

    # --------------------------------------------------------
    # 7. TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # --------------------------------------------------------
    # 8. BUILD MODEL
    # --------------------------------------------------------

    model, model_name = build_model()

    print(f"\nModel: {model_name}")

    # --------------------------------------------------------
    # 9. TRAIN MODEL
    # --------------------------------------------------------

    print("\nTraining model...")

    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # 10. TEST PREDICTIONS
    # --------------------------------------------------------

    preds = model.predict(X_test)

    preds = np.asarray(preds, dtype=float)

    print("\n" + "=" * 60)
    print("PREDICTION CHECK")
    print("=" * 60)

    print("\nFirst 10 actual values:")
    print(np.asarray(y_test)[:10].tolist())

    print("\nFirst 10 predicted values:")
    print(preds[:10].tolist())

    print(f"\nMinimum prediction: {float(np.min(preds)):.6f}")
    print(f"Maximum prediction: {float(np.max(preds)):.6f}")
    print(f"Mean prediction   : {float(np.mean(preds)):.6f}")

    # --------------------------------------------------------
    # 11. CHECK PREDICTION PROBLEM
    # --------------------------------------------------------

    negative_predictions = int(np.sum(preds < 0))

    print(f"Negative predictions: {negative_predictions}")

    if np.allclose(preds, 0):
        print(
            "\nWARNING: Model predictions are approximately ZERO."
        )
        print(
            "Check the target values and preprocessing pipeline."
        )

    if negative_predictions > 0:
        print(
            "\nWARNING: Model produced negative yield predictions."
        )
        print(
            "The API can clamp these to 0, but the training data "
            "or model configuration should be checked."
        )

    # --------------------------------------------------------
    # 12. METRICS
    # --------------------------------------------------------

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
        "negative_predictions": negative_predictions,
    }

    # --------------------------------------------------------
    # 13. CREATE MODEL DIRECTORY
    # --------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("SAVING MODEL FILES")
    print("=" * 60)

    # --------------------------------------------------------
    # 14. SAVE MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "yield_model.joblib"
    )

    joblib.dump(model, model_path)

    print(f"Saved model    : {model_path}")

    # --------------------------------------------------------
    # 15. SAVE ENCODERS
    # --------------------------------------------------------

    encoder_path = os.path.join(
        MODEL_DIR,
        "encoders.joblib"
    )

    joblib.dump(encoders, encoder_path)

    print(f"Saved encoders : {encoder_path}")

    # --------------------------------------------------------
    # 16. SAVE FEATURE COLUMNS
    # --------------------------------------------------------

    feature_path = os.path.join(
        MODEL_DIR,
        "feature_cols.json"
    )

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Saved features : {feature_path}")

    # --------------------------------------------------------
    # 17. SAVE METRICS
    # --------------------------------------------------------

    metrics_path = os.path.join(
        MODEL_DIR,
        "metrics.json"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics  : {metrics_path}")

    # --------------------------------------------------------
    # 18. FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(json.dumps(metrics, indent=2))

    print("\nModel files created:")
    print("  1. yield_model.joblib")
    print("  2. encoders.joblib")
    print("  3. feature_cols.json")
    print("  4. metrics.json")

    print("\n" + "=" * 60)

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train crop yield prediction model."
    )

    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "sample_crop_yield_data.csv"
        ),
        help="Path to the crop yield CSV dataset."
    )

    args = parser.parse_args()

    try:
        train(args.data)

    except Exception as e:
        print("\n" + "=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        print("=" * 60)

        raise
