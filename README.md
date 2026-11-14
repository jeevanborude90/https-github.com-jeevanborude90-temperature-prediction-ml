# Temperature Prediction using Machine Learning

**Created by Jeevan Borude**

A machine learning web app that predicts air temperature from weather features.
Viewers can run a **live hands-on experiment**: change humidity (or any other feature) and instantly see a new prediction and a curve showing how that feature affects temperature.

## Features

- Predicts temperature from: month, hour, humidity, wind speed, air pressure, cloud cover, rainfall
- Compares 3 algorithms (Linear Regression, Random Forest, Gradient Boosting) and uses the best one
- Interactive sliders with live prediction and "warmer / cooler than baseline" indicator
- "Live experiment" chart: vary one feature and see the model's response
- Model metrics (R², MAE, RMSE) and feature importance shown on the page
- Flask backend (REST API) + HTML/CSS/JS frontend, served together

## Project structure

```
app.py              Flask backend (API + serves the web page)
train_model.py      Creates the dataset, trains and saves the model
static/             Frontend (index.html, style.css, app.js)
model/              Saved model, metrics and dataset
requirements.txt    Python dependencies
render.yaml         One-click deployment settings for Render
```

## Run locally

```bash
pip install -r requirements.txt
python train_model.py      # optional, a trained model is already included
python app.py
```

Open http://localhost:5000

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/meta` | Features, slider ranges, model metrics |
| POST | `/api/predict` | Body `{"features": {...}}` returns the predicted temperature |
| POST | `/api/sweep` | Vary one feature across its range and return the curve |
| GET | `/api/health` | Health check |

Example:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"month":5,"hour":14,"humidity":55,"wind_speed":12,"pressure":1010,"cloud_cover":30,"rainfall":0}}'
```

## Deploy (GitHub + Render)

1. Upload this folder to a new **public** GitHub repository.
2. On https://render.com choose **New + > Blueprint**, connect the repository and click **Apply**.
   Render reads `render.yaml`, installs dependencies, trains the model and starts the app.
3. Your public link will look like `https://jeevan-temperature-predictor.onrender.com`.

Note: on Render's free plan the app sleeps after inactivity, so the first visit can take 30 to 60 seconds to wake up.

## About the data

The dataset is simulated from realistic weather relationships (seasonal cycle, day/night cycle, and the effect of humidity, cloud cover, rain, wind and pressure on temperature), so the project works with no downloads or API keys.
To use real data, replace `generate_dataset()` in `train_model.py` with `pd.read_csv("your_file.csv")` using the same column names.
