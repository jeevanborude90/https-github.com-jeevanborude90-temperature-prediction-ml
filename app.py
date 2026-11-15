"""
Flask backend for the Temperature Prediction project.

Serves BOTH the API and the web page, so one deployment (e.g. on Render) is enough.

Endpoints
  GET  /                -> the web page
  GET  /api/meta        -> feature list, slider ranges, model metrics
  POST /api/predict     -> {"features": {...}} -> predicted temperature
  POST /api/sweep       -> vary ONE feature across its range, return the curve
  GET  /api/health      -> simple health check
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Works with BOTH layouts: files inside static/ and model/ folders, or all files
# loose in one folder (what happens when uploading through GitHub's web page).
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(os.path.join(STATIC_DIR, "index.html")):
    STATIC_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "model")
if not os.path.exists(os.path.join(MODEL_DIR, "model.joblib")) and os.path.exists(
    os.path.join(BASE_DIR, "model.joblib")
):
    MODEL_DIR = BASE_DIR
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# Only these files are ever served to the browser (never app.py or the model).
PUBLIC_FILES = {"index.html", "style.css", "app.js"}

app = Flask(__name__, static_folder=None)


def load_artifacts():
    """Load the trained model. If it is missing or was saved with an
    incompatible library version, retrain automatically."""
    try:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        return model, meta
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load saved model ({exc}). Retraining...")
        import train_model

        train_model.main()
        new_dir = os.path.join(BASE_DIR, "model")
        model = joblib.load(os.path.join(new_dir, "model.joblib"))
        with open(os.path.join(new_dir, "meta.json")) as f:
            meta = json.load(f)
        return model, meta


model, meta = load_artifacts()
FEATURES = meta["features"]
INFO = meta["feature_info"]


def clean_features(raw):
    """Validate user input: every feature must be a number inside its allowed range."""
    if not isinstance(raw, dict):
        raise ValueError("'features' must be an object.")
    values = {}
    for name in FEATURES:
        if name not in raw:
            raise ValueError(f"Missing feature: {name}")
        try:
            value = float(raw[name])
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{name}' must be a number.")
        if not np.isfinite(value):
            raise ValueError(f"Feature '{name}' must be a finite number.")
        lo, hi = INFO[name]["min"], INFO[name]["max"]
        values[name] = min(max(value, lo), hi)  # clamp to the training range
    return values


def predict_rows(rows):
    frame = pd.DataFrame(rows, columns=FEATURES)
    return model.predict(frame)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    if filename not in PUBLIC_FILES:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/meta")
def get_meta():
    return jsonify(meta)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        values = clean_features(payload.get("features"))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400
    temp = float(predict_rows([values])[0])
    return jsonify({"temperature": round(temp, 2), "features": values})


@app.route("/api/sweep", methods=["POST"])
def sweep():
    payload = request.get_json(silent=True) or {}
    feature = payload.get("feature")
    if feature not in FEATURES:
        return jsonify({"error": f"Unknown feature: {feature}"}), 400
    try:
        base = clean_features(payload.get("features"))
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    lo, hi = INFO[feature]["min"], INFO[feature]["max"]
    xs = np.linspace(lo, hi, 25)
    rows = []
    for x in xs:
        row = dict(base)
        row[feature] = float(x)
        rows.append(row)
    ys = predict_rows(rows)
    return jsonify(
        {
            "feature": feature,
            "x": [round(float(x), 2) for x in xs],
            "y": [round(float(y), 2) for y in ys],
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
