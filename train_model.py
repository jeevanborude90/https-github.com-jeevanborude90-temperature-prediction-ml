"""
Train the temperature prediction model.

Run:  python train_model.py

What it does
1. Generates a weather dataset (see NOTE below).
2. Trains three regression models and compares them on a held-out test set.
3. Saves the best model plus metadata (feature ranges, metrics) into ./model/

NOTE ON DATA: To keep the project self-contained (no downloads, no API keys),
the dataset is simulated from simple, realistic weather relationships:
seasonal cycle, day/night cycle, and the well-known fact that higher humidity,
cloud cover and rainfall tend to lower the air temperature. You can replace
`generate_dataset()` with `pd.read_csv("your_real_data.csv")` as long as the
column names match FEATURES and TARGET below.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

FEATURES = [
    "month",
    "hour",
    "humidity",
    "wind_speed",
    "pressure",
    "cloud_cover",
    "rainfall",
]
TARGET = "temperature"

# Human-friendly info used by the web page (labels, units, slider ranges)
FEATURE_INFO = {
    "month":       {"label": "Month",          "unit": "",     "min": 1,   "max": 12,   "step": 1,   "default": 5},
    "hour":        {"label": "Hour of day",    "unit": "h",    "min": 0,   "max": 23,   "step": 1,   "default": 14},
    "humidity":    {"label": "Humidity",       "unit": "%",    "min": 10,  "max": 100,  "step": 1,   "default": 55},
    "wind_speed":  {"label": "Wind speed",     "unit": "km/h", "min": 0,   "max": 40,   "step": 1,   "default": 12},
    "pressure":    {"label": "Air pressure",   "unit": "hPa",  "min": 990, "max": 1030, "step": 1,   "default": 1010},
    "cloud_cover": {"label": "Cloud cover",    "unit": "%",    "min": 0,   "max": 100,  "step": 1,   "default": 30},
    "rainfall":    {"label": "Rainfall",       "unit": "mm",   "min": 0,   "max": 50,   "step": 1,   "default": 0},
}


def generate_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)

    month = rng.integers(1, 13, n)
    hour = rng.integers(0, 24, n)

    # Seasonal humidity (monsoon months are more humid) + lower in the afternoon
    humidity = (
        55
        + 20 * np.sin(2 * np.pi * (month - 7) / 12)
        - 8 * np.sin(2 * np.pi * (hour - 9) / 24)
        + rng.normal(0, 12, n)
    )
    humidity = np.clip(humidity, 10, 100)

    cloud_cover = np.clip(
        30 + 25 * np.sin(2 * np.pi * (month - 7) / 12) + rng.normal(0, 25, n), 0, 100
    )
    rainfall = np.clip(
        rng.exponential(2.0, n) * (1 + np.sin(2 * np.pi * (month - 7) / 12)) * (cloud_cover / 50),
        0,
        50,
    )
    wind_speed = np.clip(rng.gamma(3.0, 4.0, n), 0, 40)
    pressure = np.clip(
        1010 - 4 * np.sin(2 * np.pi * (month - 7) / 12) + rng.normal(0, 5, n), 990, 1030
    )

    # Physical-style relationship for temperature (degrees Celsius)
    seasonal = 26 + 6 * np.sin(2 * np.pi * (month - 3) / 12)       # peaks around May/June
    diurnal = 5 * np.sin(2 * np.pi * (hour - 9) / 24)               # peaks around 15:00
    temperature = (
        seasonal
        + diurnal
        - 0.08 * (humidity - 60)
        - 0.03 * (cloud_cover - 40)
        - 0.12 * rainfall
        - 0.05 * (wind_speed - 12)
        - 0.10 * (pressure - 1010)
        + rng.normal(0, 1.2, n)
    )

    return pd.DataFrame(
        {
            "month": month,
            "hour": hour,
            "humidity": humidity.round(1),
            "wind_speed": wind_speed.round(1),
            "pressure": pressure.round(1),
            "cloud_cover": cloud_cover.round(1),
            "rainfall": rainfall.round(1),
            "temperature": temperature.round(2),
        }
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = generate_dataset()
    df.to_csv(os.path.join(MODEL_DIR, "weather_data.csv"), index=False)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42),
    }

    results = {}
    best_name, best_model, best_mae = None, None, float("inf")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        results[name] = {
            "r2": round(float(r2_score(y_test, pred)), 4),
            "mae": round(float(mae), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
        }
        print(f"{name:20s} R2={results[name]['r2']:.4f}  MAE={mae:.3f}  RMSE={results[name]['rmse']:.3f}")
        if mae < best_mae:
            best_name, best_model, best_mae = name, model, mae

    print(f"\nBest model: {best_name}")

    # Feature importance (tree models) or absolute scaled coefficients (linear)
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    else:
        importance = np.abs(best_model.coef_) * X_train.std().values
        importance = importance / importance.sum()

    meta = {
        "best_model": best_name,
        "metrics": results,
        "features": FEATURES,
        "feature_info": FEATURE_INFO,
        "feature_importance": {f: round(float(i), 4) for f, i in zip(FEATURES, importance)},
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "data_note": "Simulated weather dataset based on realistic seasonal, daily and humidity/cloud/rain effects.",
    }

    joblib.dump(best_model, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved model/model.joblib and model/meta.json")


if __name__ == "__main__":
    main()
