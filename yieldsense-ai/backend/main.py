"""
YieldSense AI — API Gateway / Backend.

Implements:
  - Yield Prediction Module   (POST /api/predict)
  - Weather Analysis Module   (POST /api/weather-analysis)
  - Soil Analysis Module      (POST /api/soil-analysis)
  - Recommendation Module     (embedded in /api/predict response)

Run:
    uvicorn main:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import json
import os

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data_preprocessing import CATEGORICAL_COLS, engineer_features


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models"
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="YieldSense AI API",
    version="0.1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODEL ARTIFACTS
# ============================================================

_model = None
_encoders = None
_feature_cols = None
_metrics = {}


# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

@app.on_event("startup")
def load_artifacts():
    """
    Load trained model, encoders, feature columns and metrics
    when the API starts.
    """

    global _model
    global _encoders
    global _feature_cols
    global _metrics

    try:
        model_path = os.path.join(
            MODEL_DIR,
            "yield_model.joblib"
        )

        encoder_path = os.path.join(
            MODEL_DIR,
            "encoders.joblib"
        )

        feature_path = os.path.join(
            MODEL_DIR,
            "feature_cols.json"
        )

        metrics_path = os.path.join(
            MODEL_DIR,
            "metrics.json"
        )

        _model = joblib.load(model_path)

        _encoders = joblib.load(
            encoder_path
        )

        with open(
            feature_path,
            "r",
            encoding="utf-8"
        ) as f:
            _feature_cols = json.load(f)

        with open(
            metrics_path,
            "r",
            encoding="utf-8"
        ) as f:
            _metrics = json.load(f)

        print("=" * 60)
        print("YIELDSENSE AI MODEL LOADED")
        print("=" * 60)

        print("Model loaded successfully.")
        print("Model type:", type(_model).__name__)
        print("Number of features:", len(_feature_cols))

        print("\nFeatures expected by model:")
        for feature in _feature_cols:
            print(" -", feature)

        print("\nModel metrics:")
        print(json.dumps(_metrics, indent=2))

        print("=" * 60)

    except FileNotFoundError as exc:

        _model = None
        _encoders = None
        _feature_cols = None

        print("=" * 60)
        print("MODEL FILES NOT FOUND")
        print("=" * 60)

        print(f"Missing file: {exc}")
        print(
            "Run model_training.py to generate the model artifacts."
        )

        print("=" * 60)


    except Exception as exc:

        _model = None
        _encoders = None
        _feature_cols = None

        print("=" * 60)
        print("ERROR LOADING MODEL")
        print("=" * 60)

        print(str(exc))

        print("=" * 60)


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class YieldPredictionRequest(BaseModel):

    region: str = Field(
        ...,
        examples=["North"]
    )

    crop: str = Field(
        ...,
        examples=["Wheat"]
    )

    year: int = Field(
        ...,
        examples=[2026]
    )

    rainfall_mm: float
    avg_temperature_c: float
    humidity_pct: float

    soil_ph: float
    soil_nitrogen_kg_ha: float
    soil_phosphorus_kg_ha: float
    soil_potassium_kg_ha: float

    irrigation_pct_area: float
    fertilizer_kg_ha: float

    farm_area_ha: float


class WeatherAnalysisRequest(BaseModel):

    rainfall_mm: float
    avg_temperature_c: float
    humidity_pct: float
    crop: str


class SoilAnalysisRequest(BaseModel):

    soil_ph: float
    soil_nitrogen_kg_ha: float
    soil_phosphorus_kg_ha: float
    soil_potassium_kg_ha: float


# ============================================================
# CONSTANTS
# ============================================================

IDEAL_PH_RANGE = (6.0, 7.0)

IDEAL_RAINFALL_RANGE = (
    700,
    1300
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

def _recommendations(
    payload: YieldPredictionRequest,
    predicted_yield: float
) -> list[str]:

    tips = []

    # Soil pH
    if payload.soil_ph < IDEAL_PH_RANGE[0]:

        tips.append(
            "Soil is acidic — consider liming "
            "to raise pH toward 6.0-7.0."
        )

    elif payload.soil_ph > IDEAL_PH_RANGE[1]:

        tips.append(
            "Soil is alkaline — consider sulfur "
            "amendments or acid-forming fertilizer."
        )

    # Rainfall
    if payload.rainfall_mm < IDEAL_RAINFALL_RANGE[0]:

        tips.append(
            "Rainfall is below the ideal band — "
            "plan supplemental irrigation."
        )

    elif payload.rainfall_mm > IDEAL_RAINFALL_RANGE[1]:

        tips.append(
            "Rainfall is high — ensure adequate "
            "field drainage to avoid waterlogging."
        )

    # Irrigation
    if payload.irrigation_pct_area < 40:

        tips.append(
            "Irrigation coverage is low — expanding it "
            "could meaningfully lift yield."
        )

    # Fertilizer
    if payload.fertilizer_kg_ha < 100:

        tips.append(
            "Fertilizer application is on the low side "
            "for this crop's typical needs."
        )

    # Nitrogen
    if payload.soil_nitrogen_kg_ha < 80:

        tips.append(
            "Nitrogen levels are low — a nitrogen-rich "
            "fertilizer split-application is advised."
        )

    # Default
    if not tips:

        tips.append(
            "Current conditions look well-balanced "
            "for this crop; maintain current practices."
        )

    return tips


# ============================================================
# RISK LEVEL
# ============================================================

def _risk_level(
    payload: YieldPredictionRequest
) -> str:

    score = 0

    # Rainfall
    if (
        payload.rainfall_mm < 500
        or payload.rainfall_mm > 1800
    ):
        score += 1

    # Temperature
    if (
        payload.avg_temperature_c > 35
        or payload.avg_temperature_c < 10
    ):
        score += 1

    # Soil pH
    if (
        payload.soil_ph < 5.0
        or payload.soil_ph > 8.5
    ):
        score += 1

    # Irrigation
    if payload.irrigation_pct_area < 20:

        score += 1

    return {
        0: "Low",
        1: "Low",
        2: "Moderate",
        3: "High",
        4: "High"
    }[score]


# ============================================================
# API STATUS
# ============================================================

@app.get("/api/status")
def root():

    return {
        "status": "ok",
        "service": "YieldSense AI API",
        "model_loaded": _model is not None,
        "model_metrics": _metrics
    }


# ============================================================
# YIELD PREDICTION
# ============================================================

@app.post("/api/predict")
def predict_yield(
    payload: YieldPredictionRequest
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if _model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model not trained yet. "
                "Run `python model_training.py` first."
            )
        )

    if _encoders is None:

        raise HTTPException(
            status_code=503,
            detail="Model encoders are not available."
        )

    if _feature_cols is None:

        raise HTTPException(
            status_code=503,
            detail="Model feature configuration is not available."
        )

    # --------------------------------------------------------
    # Convert request to DataFrame
    # --------------------------------------------------------

    row = pd.DataFrame(
        [payload.model_dump()]
    )

    print("\n" + "=" * 60)
    print("NEW YIELD PREDICTION REQUEST")
    print("=" * 60)

    print("\nInput data:")

    print(
        row.to_dict(
            orient="records"
        )[0]
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    try:

        row = engineer_features(row)

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Feature engineering failed: {str(exc)}"
        )

    # --------------------------------------------------------
    # Encode categorical columns
    # --------------------------------------------------------

    try:

        for col in CATEGORICAL_COLS:

            if col not in row.columns:
                continue

            if col not in _encoders:

                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Encoder for categorical column "
                        f"'{col}' not found."
                    )
                )

            le = _encoders[col]

            known = set(
                le.classes_
            )

            # Unknown categories are replaced
            # with the first known category.
            row[col] = row[col].apply(
                lambda value:
                value
                if value in known
                else le.classes_[0]
            )

            row[col] = le.transform(
                row[col]
            )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Categorical encoding failed: {str(exc)}"
        )

    # --------------------------------------------------------
    # Verify required features
    # --------------------------------------------------------

    missing_features = [
        col
        for col in _feature_cols
        if col not in row.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Required model features are missing.",
                "missing_features": missing_features
            }
        )

    # --------------------------------------------------------
    # Keep exactly the trained feature order
    # --------------------------------------------------------

    row = row[_feature_cols]

    # --------------------------------------------------------
    # Debug: Features sent to model
    # --------------------------------------------------------

    print("\nFEATURES SENT TO MODEL:")

    print(
        row.to_dict(
            orient="records"
        )[0]
    )

    print(
        "\nFeature order:"
    )

    print(
        list(row.columns)
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    try:

        raw_prediction = float(
            _model.predict(row)[0]
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(exc)}"
        )

    # --------------------------------------------------------
    # IMPORTANT DEBUG OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL PREDICTION DEBUG")
    print("=" * 60)

    print(
        "RAW MODEL PREDICTION:",
        raw_prediction
    )

    print(
        "MODEL PREDICTION TYPE:",
        type(raw_prediction).__name__
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Prevent negative yield
    # --------------------------------------------------------

    predicted = max(
        raw_prediction,
        0.0
    )

    # --------------------------------------------------------
    # Calculate total production
    # --------------------------------------------------------

    total_production = round(
        predicted * payload.farm_area_ha,
        2
    )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = _recommendations(
        payload,
        predicted
    )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    risk = _risk_level(
        payload
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = {
        "predicted_yield_tonnes_per_ha": round(
            predicted,
            2
        ),

        # Raw value is included specifically to
        # diagnose the previous zero-prediction issue.
        "raw_model_prediction": round(
            raw_prediction,
            4
        ),

        "estimated_total_production_tonnes": (
            total_production
        ),

        "risk_level": risk,

        "recommendations": recommendations,

        "model_metrics": _metrics
    }

    print("\nFINAL API RESPONSE:")

    print(
        json.dumps(
            response,
            indent=2
        )
    )

    print("=" * 60 + "\n")

    return response


# ============================================================
# WEATHER ANALYSIS
# ============================================================

@app.post("/api/weather-analysis")
def weather_analysis(
    payload: WeatherAnalysisRequest
):

    rainfall_status = (
        "Below ideal"
        if payload.rainfall_mm < IDEAL_RAINFALL_RANGE[0]
        else "Above ideal"
        if payload.rainfall_mm > IDEAL_RAINFALL_RANGE[1]
        else "Ideal"
    )

    temperature_status = (
        "Too cold"
        if payload.avg_temperature_c < 15
        else "Too hot"
        if payload.avg_temperature_c > 32
        else "Favorable"
    )

    return {
        "crop": payload.crop,

        "rainfall_status": rainfall_status,

        "temperature_status": temperature_status,

        "humidity_pct": payload.humidity_pct,

        "impact_summary": (
            f"{rainfall_status} rainfall and "
            f"{temperature_status.lower()} temperatures "
            f"for {payload.crop}."
        )
    }


# ============================================================
# SOIL ANALYSIS
# ============================================================

@app.post("/api/soil-analysis")
def soil_analysis(
    payload: SoilAnalysisRequest
):

    ph_status = (
        "Acidic"
        if payload.soil_ph < IDEAL_PH_RANGE[0]
        else "Alkaline"
        if payload.soil_ph > IDEAL_PH_RANGE[1]
        else "Optimal"
    )

    npk_total = (
        payload.soil_nitrogen_kg_ha
        + payload.soil_phosphorus_kg_ha
        + payload.soil_potassium_kg_ha
    )

    fertility = (
        "Low"
        if npk_total < 150
        else "Moderate"
        if npk_total < 300
        else "High"
    )

    return {
        "ph_status": ph_status,

        "npk_total_kg_ha": round(
            npk_total,
            2
        ),

        "fertility_rating": fertility
    }


# ============================================================
# SERVE FRONTEND
# ============================================================

FRONTEND_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "frontend"
)


# Mount frontend LAST so it does not shadow /api/* routes.
if os.path.isdir(FRONTEND_DIR):

    app.mount(
        "/",
        StaticFiles(
            directory=FRONTEND_DIR,
            html=True
        ),
        name="frontend"
    )

else:

    print(
        f"WARNING: Frontend directory not found: "
        f"{FRONTEND_DIR}"
    )
