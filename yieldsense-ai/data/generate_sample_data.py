"""
Generates a synthetic crop yield dataset that mimics the structure of the
real open-source datasets referenced in the project brief (FAOSTAT, USDA,
Kaggle Crop Yield Prediction Dataset).

Use this to get the pipeline running end-to-end immediately. Swap it out
for real downloaded data later (see README.md, section "Real datasets").
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 4000
crops = ["Wheat", "Rice", "Maize", "Soybean", "Cotton", "Sugarcane"]
regions = ["North", "South", "East", "West", "Central"]

df = pd.DataFrame({
    "region": np.random.choice(regions, N),
    "crop": np.random.choice(crops, N),
    "year": np.random.randint(2010, 2025, N),
    "rainfall_mm": np.round(np.random.normal(900, 250, N).clip(100, 2500), 1),
    "avg_temperature_c": np.round(np.random.normal(24, 5, N).clip(5, 42), 1),
    "humidity_pct": np.round(np.random.normal(60, 15, N).clip(10, 100), 1),
    "soil_ph": np.round(np.random.normal(6.5, 0.8, N).clip(4.0, 9.0), 2),
    "soil_nitrogen_kg_ha": np.round(np.random.normal(120, 35, N).clip(10, 300), 1),
    "soil_phosphorus_kg_ha": np.round(np.random.normal(45, 15, N).clip(5, 150), 1),
    "soil_potassium_kg_ha": np.round(np.random.normal(150, 40, N).clip(10, 400), 1),
    "irrigation_pct_area": np.round(np.random.uniform(0, 100, N), 1),
    "fertilizer_kg_ha": np.round(np.random.normal(180, 60, N).clip(0, 500), 1),
    "farm_area_ha": np.round(np.random.exponential(8, N).clip(0.5, 200), 2),
})

# Crop-specific base yield (tonnes/ha) so the target has real signal to learn
crop_base_yield = {
    "Wheat": 3.2, "Rice": 4.0, "Maize": 5.5,
    "Soybean": 2.6, "Cotton": 1.8, "Sugarcane": 65.0,
}
df["base_yield"] = df["crop"].map(crop_base_yield)

# Synthetic yield formula: base yield adjusted by weather/soil/inputs + noise
rainfall_effect = 1 - (np.abs(df["rainfall_mm"] - 1000) / 3000)
temp_effect = 1 - (np.abs(df["avg_temperature_c"] - 24) / 60)
ph_effect = 1 - (np.abs(df["soil_ph"] - 6.5) / 8)
fert_effect = 0.7 + 0.3 * (df["fertilizer_kg_ha"] / 250).clip(0, 1.3)
irrigation_effect = 0.85 + 0.15 * (df["irrigation_pct_area"] / 100)
noise = np.random.normal(1, 0.08, N)

df["yield_tonnes_per_ha"] = np.round(
    df["base_yield"] * rainfall_effect * temp_effect * ph_effect
    * fert_effect * irrigation_effect * noise,
    2,
).clip(0.1, None)

df = df.drop(columns=["base_yield"])
df.to_csv("/home/claude/yieldsense-ai/data/sample_crop_yield_data.csv", index=False)
print(f"Wrote {len(df)} rows -> data/sample_crop_yield_data.csv")
print(df.head())
