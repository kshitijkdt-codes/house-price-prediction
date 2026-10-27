"""Flask API for the House Price Prediction project.

Run (from backend/, after `python train_model.py`):
    python app.py

Endpoints
    GET  /health   -> {"status": "ok"}
    GET  /options  -> values for the website dropdowns (taken from the trained model)
    POST /predict  -> JSON in, predicted price out
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
FEATURE_LABELS = {
    "location_enc": "Location", "type_enc": "Property type", "status_enc": "Building status",
    "area_insqft": "Area", "bhk": "Bedrooms (BHK)",
}

app = Flask(__name__)
CORS(app)  # lets the React site (a different port) call this API

STATE = {}  # filled lazily by get_state(); tests can fill it directly


def load_artifacts(model_dir: Path = MODEL_DIR) -> dict:
    return {
        "model": joblib.load(model_dir / "xgb_model.pkl"),
        "location": joblib.load(model_dir / "location_encoder.pkl"),
        "type": joblib.load(model_dir / "type_encoder.pkl"),
        "status": joblib.load(model_dir / "status_encoder.pkl"),
        "metrics": json.loads((model_dir / "metrics.json").read_text()),
    }


def get_state():
    if not STATE:
        try:
            STATE.update(load_artifacts())
        except FileNotFoundError:
            return None
    return STATE


def not_trained():
    return jsonify({"error": "Model not trained yet. Run `python train_model.py` in the backend folder."}), 503


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/options")
def options():
    state = get_state()
    if state is None:
        return not_trained()
    return jsonify({
        "locations": sorted(state["location"].classes_.tolist()),
        "propertyTypes": sorted(state["type"].classes_.tolist()),
        "buildingStatuses": sorted(state["status"].classes_.tolist()),
    })


def number_field(data, name, low, high, integer=False):
    raw = data.get(name)
    if isinstance(raw, bool) or raw is None:
        raise ValueError(f"'{name}' is required and must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number")
    if integer and value != int(value):
        raise ValueError(f"'{name}' must be a whole number")
    if not (low <= value <= high):
        raise ValueError(f"'{name}' must be between {low} and {high}")
    return int(value) if integer else value


def category_field(data, name, encoder):
    value = data.get(name)
    if not isinstance(value, str) or value not in encoder.classes_:
        raise ValueError(f"Unknown or missing '{name}': {value!r}")
    return int(encoder.transform([value])[0])


@app.post("/predict")
def predict():
    state = get_state()
    if state is None:
        return not_trained()

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON body"}), 400

    try:
        area = number_field(data, "area", 100, 20000)
        bhk = number_field(data, "bedrooms", 1, 10, integer=True)
        loc = category_field(data, "location", state["location"])
        ptype = category_field(data, "propertyType", state["type"])
        status = category_field(data, "buildingStatus", state["status"])
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    X = np.array([[loc, ptype, status, area, bhk]], dtype=float)
    metrics = state["metrics"]
    raw_pred = float(state["model"].predict(X)[0])
    if metrics.get("log_target"):  # the model was trained on log(1 + price)
        raw_pred = float(np.expm1(raw_pred))
    price_lakhs = max(raw_pred, 0.0)
    spread = metrics.get("rel_error_p80", 0.1)
    importance = sorted(metrics.get("feature_importance", {}).items(), key=lambda kv: kv[1], reverse=True)

    return jsonify({
        "predictedPriceLakhs": round(price_lakhs, 2),
        "predictedPrice": round(price_lakhs * 100000),  # in rupees
        "pricePerSqft": round(price_lakhs * 100000 / area),
        "rangeLakhs": {"low": round(price_lakhs * (1 - spread), 2), "high": round(price_lakhs * (1 + spread), 2)},
        "rangeNote": f"80% of test predictions were within +/-{spread * 100:.0f}% of the real price",
        "modelR2": metrics.get("r2"),
        "topFactors": [FEATURE_LABELS.get(name, name) for name, _ in importance[:3]],
        "isSampleData": metrics.get("is_synthetic_sample", False),
    }), 200


if __name__ == "__main__":
    # debug is OFF by default: debug mode lets anyone on your network run code.
    app.run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
