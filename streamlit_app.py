"""
streamlit_app.py - Streamlit user interface for the deep-learning digit classifier.

The UI contains no model. It calls the Flask API (separate container) over HTTP:
    GET  /health, /model-info, /samples      POST /predict
The API address comes from the environment variable API_URL
(docker-compose sets API_URL=http://api:5000; default for local runs: http://localhost:5000).

Run locally:  API_URL=http://localhost:5000 streamlit run streamlit_app.py
"""
import os

import numpy as np
import requests
import streamlit as st

TIMEOUT = 5


def api_url():
    return os.environ.get("API_URL", "http://localhost:5000").rstrip("/")


def api_get(path):
    r = requests.get(api_url() + path, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def api_post(path, payload):
    r = requests.post(api_url() + path, json=payload, timeout=TIMEOUT)
    body = r.json()
    if r.status_code != 200:
        raise ValueError(body.get("error", f"HTTP {r.status_code}"))
    return body


def digit_image(pixels, scale=20):
    arr = np.array(pixels, dtype=float).reshape(8, 8) / 16.0 * 255.0
    return np.kron(arr, np.ones((scale, scale))).astype(np.uint8)


def show_result(result, true_label=None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted digit", str(result["digit"]))
    c2.metric("Confidence", f"{result['confidence'] * 100:.1f} %")
    if true_label is not None:
        c3.metric("True label", str(true_label))
        if result["digit"] == true_label:
            st.success("The prediction is correct.")
        else:
            st.warning("The prediction differs from the true label.")
    st.bar_chart({"probability": result["probabilities"]})
    st.caption(f"Inference time in the API: {result['inference_ms']} ms")


st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="wide")
st.title("Deep Learning Digit Classifier")
st.caption("Streamlit UI  ->  Flask API  ->  NumPy neural network  (UI and API run in separate Docker containers)")

# ---------------------------------------------------------------- sidebar: API status
with st.sidebar:
    st.header("API connection")
    st.write(f"Address: `{api_url()}`")
    try:
        health = api_get("/health")
        info = api_get("/model-info")
        st.success(f"API reachable (status: {health['status']})")
        st.metric("Model parameters", f"{info['parameters']:,}")
        st.metric("Model test accuracy", f"{info['test_accuracy'] * 100:.2f} %")
        st.caption(info["architecture"])
    except Exception as exc:  # the UI must not crash when the API is down
        st.error(f"API not reachable: {exc}")
        st.stop()

tab_sample, tab_pixels, tab_about = st.tabs(["Sample digits", "Enter pixel values", "About"])

# ---------------------------------------------------------------- tab 1: sample digits
with tab_sample:
    if "samples" not in st.session_state:
        st.session_state["samples"] = api_get("/samples")
    samples = {s["label"]: s["pixels"] for s in st.session_state["samples"]}
    digit = st.selectbox("Choose a held-out sample digit", list(range(10)), key="digit")
    left, right = st.columns([1, 3])
    left.image(digit_image(samples[digit]), caption=f"8x8 image of a {digit}", clamp=True)
    with right:
        if st.button("Predict", key="predict_sample", type="primary"):
            try:
                show_result(api_post("/predict", {"pixels": samples[digit]}), true_label=digit)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------- tab 2: pixel values
with tab_pixels:
    st.write("Enter 64 pixel values (0-16, flattened 8x8 image), separated by commas.")
    default = ", ".join(str(int(v)) for v in samples[8])
    text = st.text_area("Pixel values", value=default, height=120, key="pixels_text")
    if st.button("Predict pixel values", key="predict_text"):
        try:
            values = [float(v) for v in text.replace("\n", ",").split(",") if v.strip()]
            show_result(api_post("/predict", {"pixels": values}))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

# ---------------------------------------------------------------- tab 3: about
with tab_about:
    st.markdown(
        "**Architecture**\n\n"
        "- `ui` container: this Streamlit application (port 8501)\n"
        "- `api` container: Flask + gunicorn serving the NumPy model (port 5000)\n"
        "- Both containers are started by Docker Compose and talk to each other over a private "
        "network using the service name `api`.\n\n"
        "**Model**: a neural network (64 -> 64 ReLU -> 10 softmax) trained on the UCI handwritten digits."
    )
