"""
Data Preprocessing Module (Modules 2 & 3/4 from the brief:
Data Collection + Weather/Soil Analysis feature prep)

Takes the raw crop/weather/soil CSV and returns model-ready features.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["region", "crop"]
NUMERIC_COLS = [
    "year", "rainfall_mm", "avg_temperature_c", "humidity_pct",
    "soil_ph", "soil_nitrogen_kg_ha", "soil_phosphorus_kg_ha",
    "soil_potassium_kg_ha", "irrigation_pct_area", "fertilizer_kg_ha",
    "farm_area_ha",
]
TARGET_COL = "yield_tonnes_per_ha"


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Handle missing values: numeric -> median, categorical -> mode
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Drop obvious outliers using IQR on the target, if present
    if TARGET_COL in df.columns:
        q1, q3 = df[TARGET_COL].quantile([0.01, 0.99])
        df = df[(df[TARGET_COL] >= q1) & (df[TARGET_COL] <= q3)]

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Simple derived features that tend to help tree models
    df["npk_total"] = (
        df["soil_nitrogen_kg_ha"] + df["soil_phosphorus_kg_ha"] + df["soil_potassium_kg_ha"]
    )
    df["rainfall_temp_ratio"] = df["rainfall_mm"] / (df["avg_temperature_c"] + 1e-6)
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict[str, LabelEncoder] | None = None):
    """Label-encodes categorical columns. Pass `encoders` at inference time
    to reuse the encoders fit during training (unseen categories fall back
    to the most common training category)."""
    df = df.copy()
    fitted = {}
    for col in CATEGORICAL_COLS:
        if encoders is not None and col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            fitted[col] = le
    return df, (encoders or fitted)


def preprocess(path: str):
    """Full pipeline: load -> clean -> engineer -> encode. Returns (X, y, encoders)."""
    df = load_raw_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df, encoders = encode_categoricals(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, encoders, feature_cols
