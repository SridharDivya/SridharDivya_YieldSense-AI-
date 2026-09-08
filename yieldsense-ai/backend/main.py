"""
YieldSense AI — API Gateway / backend.

Implements, at MVP scope:
  - Yield Prediction Module   (POST /api/predict)
  - Weather Analysis Module   (POST /api/weather-analysis)
  - Soil Analysis Module      (POST /api/soil-analysis)
  - Recommendation Module     (embedded in /api/predict response)

Run:
    uvicorn main:app --reload --port 8000
Docs (auto-generated):
    http://localhost:8000/docs
"""
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_preprocessing import CATEGORICAL_COLS, engineer_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="YieldSense AI API", version="0.1.0")

# Allow the frontend (any origin, for local dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load trained artifacts at startup -------------------------------------
_model = None
_encoders = None
_feature_cols = None
_metrics = {}


@app.on_event("startup")
def load_artifacts():
    global _model, _encoders, _feature_cols, _metrics
    try:
        _model = joblib.load(os.path.join(MODEL_DIR, "yield_model.joblib"))
        _encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
        with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
            _feature_cols = json.load(f)
        with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
            _metrics = json.load(f)
    except FileNotFoundError:
        # Model not trained yet — /api/predict will return a clear error
        pass


# ---- Schemas -----------------------------------------------------------------
class YieldPredictionRequest(BaseModel):
    region: str = Field(..., examples=["North"])
    crop: str = Field(..., examples=["Wheat"])
    year: int = Field(..., examples=[2026])
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


# ---- Helpers -------------------------------------------------------------
IDEAL_PH_RANGE = (6.0, 7.0)
IDEAL_RAINFALL_RANGE = (700, 1300)  # mm, generic reference band


def _recommendations(payload: YieldPredictionRequest, predicted_yield: float) -> list[str]:
    tips = []
    if payload.soil_ph < IDEAL_PH_RANGE[0]:
        tips.append("Soil is acidic — consider liming to raise pH toward 6.0-7.0.")
    elif payload.soil_ph > IDEAL_PH_RANGE[1]:
        tips.append("Soil is alkaline — consider sulfur amendments or acid-forming fertilizer.")

    if payload.rainfall_mm < IDEAL_RAINFALL_RANGE[0]:
        tips.append("Rainfall is below the ideal band — plan supplemental irrigation.")
    elif payload.rainfall_mm > IDEAL_RAINFALL_RANGE[1]:
        tips.append("Rainfall is high — ensure adequate field drainage to avoid waterlogging.")

    if payload.irrigation_pct_area < 40:
        tips.append("Irrigation coverage is low — expanding it could meaningfully lift yield.")

    if payload.fertilizer_kg_ha < 100:
        tips.append("Fertilizer application is on the low side for this crop's typical needs.")

    if payload.soil_nitrogen_kg_ha < 80:
        tips.append("Nitrogen levels are low — a nitrogen-rich fertilizer split-application is advised.")

    if not tips:
        tips.append("Current conditions look well-balanced for this crop; maintain current practices.")

    return tips


def _risk_level(payload: YieldPredictionRequest) -> str:
    score = 0
    if payload.rainfall_mm < 500 or payload.rainfall_mm > 1800:
        score += 1
    if payload.avg_temperature_c > 35 or payload.avg_temperature_c < 10:
        score += 1
    if payload.soil_ph < 5.0 or payload.soil_ph > 8.5:
        score += 1
    if payload.irrigation_pct_area < 20:
        score += 1
    return {0: "Low", 1: "Low", 2: "Moderate", 3: "High", 4: "High"}[score]


# ---- Routes ------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "YieldSense AI API", "model_metrics": _metrics}


@app.post("/api/predict")
def predict_yield(payload: YieldPredictionRequest):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python model_training.py` first.",
        )

    row = pd.DataFrame([payload.model_dump()])
    row = engineer_features(row)

    for col in CATEGORICAL_COLS:
        le = _encoders[col]
        known = set(le.classes_)
        row[col] = row[col].apply(lambda v: v if v in known else le.classes_[0])
        row[col] = le.transform(row[col])

    row = row[_feature_cols]
    predicted = float(_model.predict(row)[0])
    total_production = round(predicted * payload.farm_area_ha, 2)

    return {
        "predicted_yield_tonnes_per_ha": round(predicted, 2),
        "estimated_total_production_tonnes": total_production,
        "risk_level": _risk_level(payload),
        "recommendations": _recommendations(payload, predicted),
        "model_metrics": _metrics,
    }


@app.post("/api/weather-analysis")
def weather_analysis(payload: WeatherAnalysisRequest):
    rainfall_status = (
        "Below ideal" if payload.rainfall_mm < IDEAL_RAINFALL_RANGE[0]
        else "Above ideal" if payload.rainfall_mm > IDEAL_RAINFALL_RANGE[1]
        else "Ideal"
    )
    temp_status = (
        "Too cold" if payload.avg_temperature_c < 15
        else "Too hot" if payload.avg_temperature_c > 32
        else "Favorable"
    )
    return {
        "crop": payload.crop,
        "rainfall_status": rainfall_status,
        "temperature_status": temp_status,
        "humidity_pct": payload.humidity_pct,
        "impact_summary": f"{rainfall_status} rainfall and {temp_status.lower()} temperatures for {payload.crop}.",
    }


@app.post("/api/soil-analysis")
def soil_analysis(payload: SoilAnalysisRequest):
    ph_status = (
        "Acidic" if payload.soil_ph < IDEAL_PH_RANGE[0]
        else "Alkaline" if payload.soil_ph > IDEAL_PH_RANGE[1]
        else "Optimal"
    )
    npk_total = payload.soil_nitrogen_kg_ha + payload.soil_phosphorus_kg_ha + payload.soil_potassium_kg_ha
    fertility = "Low" if npk_total < 150 else "Moderate" if npk_total < 300 else "High"
    return {
        "ph_status": ph_status,
        "npk_total_kg_ha": npk_total,
        "fertility_rating": fertility,
    }
