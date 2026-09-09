"""
YieldSense AI — API Gateway & FastAPI Backend Servicing Predictions
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="YieldSense AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_encoders = None
_feature_cols = None
_metrics = {}


@app.on_event("startup")
def load_artifacts():
    global _model, _encoders, _feature_cols, _metrics

    try:
        model_path = os.path.join(MODEL_DIR, "yield_model.joblib")
        encoder_path = os.path.join(MODEL_DIR, "encoders.joblib")
        feature_path = os.path.join(MODEL_DIR, "feature_cols.json")
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")

        _model = joblib.load(model_path)
        _encoders = joblib.load(encoder_path)

        with open(feature_path, "r", encoding="utf-8") as f:
            _feature_cols = json.load(f)

        with open(metrics_path, "r", encoding="utf-8") as f:
            _metrics = json.load(f)

        print("=" * 60)
        print("YIELDSENSE AI MODEL LOADED SUCCESSFULLY")
        print("=" * 60)

    except Exception as exc:
        _model, _encoders, _feature_cols = None, None, None
        print(f"WARNING: Failed to load model artifacts ({exc}). Please train model first.")


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


IDEAL_PH_RANGE = (6.0, 7.0)
IDEAL_RAINFALL_RANGE = (700, 1300)


def _recommendations(payload: YieldPredictionRequest, predicted_yield: float) -> list[str]:
    tips = []
    if payload.soil_ph < IDEAL_PH_RANGE[0]:
        tips.append("Soil is acidic — consider liming to raise pH toward 6.0-7.0.")
    elif payload.soil_ph > IDEAL_PH_RANGE[1]:
        tips.append("Soil is alkaline — consider sulfur amendments or acid-forming fertilizer.")

    if payload.rainfall_mm < IDEAL_RAINFALL_RANGE[0]:
        tips.append("Rainfall is below ideal band — plan supplemental irrigation.")
    elif payload.rainfall_mm > IDEAL_RAINFALL_RANGE[1]:
        tips.append("Rainfall is high — ensure adequate field drainage.")

    if payload.irrigation_pct_area < 40:
        tips.append("Irrigation coverage is low — expanding it could increase yield.")

    if payload.fertilizer_kg_ha < 100:
        tips.append("Fertilizer application rate is relatively low.")

    if payload.soil_nitrogen_kg_ha < 80:
        tips.append("Nitrogen levels are low — split-application of nitrogen is recommended.")

    if not tips:
        tips.append("Current growth conditions are well-balanced.")

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


@app.get("/api/status")
def root():
    return {
        "status": "ok",
        "service": "YieldSense AI API",
        "model_loaded": _model is not None,
        "model_metrics": _metrics,
    }


@app.post("/api/predict")
def predict_yield(payload: YieldPredictionRequest):
    if _model is None or _encoders is None or _feature_cols is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Train the model first."
        )

    row = pd.DataFrame([payload.model_dump()])

    try:
        row = engineer_features(row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {exc}")

    try:
        for col in CATEGORICAL_COLS:
            if col in row.columns and col in _encoders:
                le = _encoders[col]
                known = set(le.classes_)
                row[col] = row[col].apply(lambda v: v if v in known else le.classes_[0])
                row[col] = le.transform(row[col])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Categorical encoding failed: {exc}")

    row = row[_feature_cols]

    try:
        raw_prediction = float(_model.predict(row)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model execution failed: {exc}")

    predicted = max(0.0, raw_prediction)
    total_production = round(predicted * payload.farm_area_ha, 2)
    recommendations = _recommendations(payload, predicted)
    risk = _risk_level(payload)

    return {
        "predicted_yield_tonnes_per_ha": round(predicted, 2),
        "raw_model_prediction": round(raw_prediction, 4),
        "estimated_total_production_tonnes": total_production,
        "risk_level": risk,
        "recommendations": recommendations,
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
        "npk_total_kg_ha": round(npk_total, 2),
        "fertility_rating": fertility,
    }


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
