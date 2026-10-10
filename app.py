"""
app.py - Flask deep-learning API (digit classifier).

Endpoints
  GET  /            service description and list of endpoints
  GET  /health      liveness check (used by the Docker HEALTHCHECK)
  GET  /model-info  model metadata (architecture, parameters, test accuracy)
  GET  /samples     one held-out example per digit (pixels + true label) for quick testing
  POST /predict     JSON {"pixels": [64 numbers]} or {"instances": [[64 numbers], ...]}

Run locally:   python app.py                (Flask development server)
Production:    gunicorn --bind 0.0.0.0:5000 app:app
"""
import json
import os
import time

import numpy as np
from flask import Flask, jsonify, request

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "model", "model.npz"))
MAX_BATCH = 100
N_FEATURES = 64

app = Flask(__name__)

_m = np.load(MODEL_PATH, allow_pickle=False)
W1, b1, W2, b2 = (_m[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2"))
SAMPLES_X, SAMPLES_Y = _m["samples_x"], _m["samples_y"]
META = json.loads(str(_m["meta"]))
STARTED = time.time()


def predict_proba(x):
    """x: array (n, 64) with pixel values 0-16 -> softmax probabilities (n, 10)."""
    x = np.asarray(x, dtype=np.float32) / 16.0
    h = np.maximum(0.0, x @ W1 + b1)
    z = h @ W2 + b2
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def bad_request(message, status=400):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return jsonify({
        "service": "Flask deep-learning API (digit classifier)",
        "model": META["name"],
        "endpoints": {"GET /health": "liveness check", "GET /model-info": "model metadata",
                      "GET /samples": "example inputs with true labels",
                      "POST /predict": 'body: {"pixels": [64 numbers 0-16]} or {"instances": [[...], ...]}'},
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "uptime_seconds": round(time.time() - STARTED, 1)})


@app.get("/model-info")
def model_info():
    return jsonify(META)


@app.get("/samples")
def samples():
    return jsonify([{"label": int(y), "pixels": [float(v) for v in x]} for x, y in zip(SAMPLES_X, SAMPLES_Y)])


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return bad_request('Request body must be a JSON object with "pixels" or "instances".')
    if "pixels" in data:
        rows = [data["pixels"]]
    elif "instances" in data:
        rows = data["instances"]
        if not isinstance(rows, list) or not rows:
            return bad_request('"instances" must be a non-empty list of 64-value lists.')
        if len(rows) > MAX_BATCH:
            return bad_request(f"At most {MAX_BATCH} instances per request.")
    else:
        return bad_request('Provide "pixels" (one image) or "instances" (list of images).')
    try:
        x = np.array(rows, dtype=np.float64)
    except (ValueError, TypeError):
        return bad_request("All pixel values must be numbers.")
    if x.ndim != 2 or x.shape[1] != N_FEATURES:
        return bad_request(f"Each image needs exactly {N_FEATURES} pixel values (flattened 8x8).")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 16:
        return bad_request("Pixel values must be finite numbers between 0 and 16.")

    start = time.perf_counter()
    probs = predict_proba(x)
    elapsed = (time.perf_counter() - start) * 1000
    results = [{"digit": int(p.argmax()), "confidence": round(float(p.max()), 4),
                "probabilities": [round(float(v), 4) for v in p]} for p in probs]
    body = {"predictions": results, "count": len(results), "inference_ms": round(elapsed, 3)}
    if "pixels" in data:
        body.update(results[0])
    return jsonify(body)


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(405)
def not_allowed(_):
    return jsonify({"error": "Method not allowed for this route."}), 405


@app.errorhandler(500)
def server_error(_):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
