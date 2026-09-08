# YieldSense AI — Starter Implementation

A working MVP scaffold for the project described in your brief: crop yield
prediction using weather + soil + farm-management data, served by a FastAPI
backend and a simple dashboard frontend.

This has been **tested and runs end-to-end** in this environment (synthetic
data → trained model → metrics). You'll run it again on your own machine
after installing dependencies.

```
yieldsense-ai/
├── data/
│   ├── generate_sample_data.py     # makes a synthetic dataset to develop against
│   └── sample_crop_yield_data.csv  # 4,000-row generated dataset (already created)
├── backend/
│   ├── data_preprocessing.py       # cleaning, feature engineering, encoding
│   ├── model_training.py           # trains the yield prediction model
│   ├── main.py                     # FastAPI app: predict / weather / soil endpoints
│   └── requirements.txt
├── frontend/
│   └── index.html                  # single-file dashboard, calls the API
└── models/                         # trained model + encoders land here
```

## 1. Run it locally

```bash
cd yieldsense-ai/backend
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the model (uses the sample dataset by default)
python model_training.py

# Start the API
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (or `python -m http.server`
from the `frontend/` folder) and fill in the form. It calls `http://localhost:8000`.

Check `http://localhost:8000/docs` for interactive Swagger API docs — useful
for testing `/api/predict`, `/api/weather-analysis`, `/api/soil-analysis`
without the frontend.

## 2. What's already implemented (maps to brief section 4)

| Module in brief | What's here |
|---|---|
| Data Collection | `generate_sample_data.py` (synthetic) + `data_preprocessing.py` loads any CSV with the same schema |
| Weather Analysis | `/api/weather-analysis` endpoint |
| Soil Analysis | `/api/soil-analysis` endpoint |
| Yield Prediction | `/api/predict` — trained XGBoost/GradientBoosting regressor |
| Recommendation | Rule-based tips embedded in the `/api/predict` response |
| Analytics Dashboard | `frontend/index.html` (minimal — extend per step 5 below) |
| User Management | Not built — see step 6 |

## 3. Swap in real data (recommended next step)

The synthetic dataset gets the pipeline running, but for a real project you
want actual data. Three sources, matching your brief:

1. **FAOSTAT Crop Production** — faostat.fao.org, "Production → Crops and
   livestock products". Free CSV/bulk download, country/regional yield,
   area harvested, production by crop and year.
2. **USDA NASS Quick Stats** — quickstats.nass.usda.gov. US county/state
   level yield, acreage, and production, with a public API.
3. **Kaggle "Crop Yield Prediction Dataset"** — search Kaggle for this
   title; several versions exist combining crop, rainfall, temperature,
   soil, and yield in one table, closest in shape to what this code expects.

To use real data: download a CSV, rename/remap its columns to match
`data_preprocessing.py`'s `NUMERIC_COLS` / `CATEGORICAL_COLS` / `TARGET_COL`
(or edit those lists to match your columns), then run:

```bash
python model_training.py --data ../data/your_real_data.csv
```

For weather, you'd also want a live source like Open-Meteo (free, no key)
or NOAA/OpenWeatherMap for the "Weather APIs" integration shown in the
architecture diagram — that's a separate fetch step feeding into the same
feature columns at prediction time.

## 4. Model performance (on the sample data)

Running `model_training.py` right now prints something like:

```json
{
  "model": "GradientBoosting (sklearn fallback — install xgboost for the real thing)",
  "mae": 0.76,
  "rmse": 1.63,
  "r2": 0.988
}
```

Install `xgboost` (already in `requirements.txt`) and it'll use that instead
— it's what the brief's tech stack calls for. The R² is very high here
because the synthetic data has a clean underlying formula; expect it to
drop with real, noisier agricultural data, which is normal.

## 5. Extending the dashboard (Module 6: Analytics Dashboard)

The current `index.html` only shows a single prediction. To build out the
full "Analytics Dashboard Module" from the brief (trend charts, farm
comparisons, seasonal reports):
- Add a `/api/history` endpoint in `main.py` that reads stored past
  predictions/records from a database and returns aggregates.
- On the frontend, add a charting library (Chart.js or Recharts, as listed
  in your tech stack) to plot yield trends over time and crop comparisons.
- If you want the React/Next.js frontend named in the brief instead of
  plain HTML, scaffold it with `npx create-next-app@latest` and port the
  form/fetch logic from `index.html` into a page component — the API
  contract (`POST /api/predict`) stays the same either way.

## 6. Adding User Management (Module 1)

Not built yet — this needs a database and auth. Suggested path with your
stated stack (PostgreSQL + JWT):
1. Add SQLAlchemy models: `User`, `Farm`, `PredictionRecord`.
2. Add `fastapi-users` or a hand-rolled JWT flow (`/api/auth/register`,
   `/api/auth/login`) — issue a JWT, protect `/api/predict` with a
   dependency that checks it.
3. Store each prediction against the logged-in user's farm so the
   dashboard (step 5) has real history to chart.

## 7. Deployment (Milestone 4 in the brief)

```bash
# from yieldsense-ai/backend
docker build -t yieldsense-api .
docker run -p 8000:8000 yieldsense-api
```

(Add a `Dockerfile` with `FROM python:3.11-slim`, `COPY . .`,
`RUN pip install -r requirements.txt`, `CMD ["uvicorn", "main:app",
"--host", "0.0.0.0", "--port", "8000"]` — happy to generate this file
too if you want it.) Push the image to AWS ECR / Azure Container Registry
and run it on ECS, App Service, or a plain VM behind nginx, per your
brief's "Cloud & DevOps" section.

## 8. Suggested week-by-week order (matches brief section 5)

- **Week 1-2**: Get this scaffold running locally, swap in one real dataset
  (start with Kaggle's, it's the most ready-to-use), wire up basic auth.
- **Week 3-4**: Retrain with real data, tune the model (try XGBoost
  hyperparameter search), add the weather/soil endpoints' logic more
  rigorously (real agronomic thresholds, not the placeholder ones here).
- **Week 5-6**: Build out the dashboard with charts and history, flesh out
  recommendation logic.
- **Week 7-8**: Test, containerize, deploy, write it up.
