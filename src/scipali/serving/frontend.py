r"""Streamlit frontend for the PaliGemma2 ScienceQA API.

A small UI over the /predict endpoint: upload an image, type a question and
comma-separated choices (plus optional hint/lecture), and see the predicted
answer letter. Points at the API via the ``API_URL`` env var so the same UI
works against a local server or the deployed Cloud Run service.

This module imports no project code, so it is launched standalone via ``uvx``
in its own environment (Streamlit's Starlette server pins a newer ``starlette``
than the FastAPI API does, so they can't share one venv):

    API_URL=http://localhost:8000 \\
      uvx --with requests --with pillow \\
      streamlit run src/scipali/serving/frontend.py
"""

import base64
import io
import os

import requests  # type: ignore[import-untyped]
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="ScienceQA · PaliGemma2", page_icon="🔬")
st.title("🔬 ScienceQA — PaliGemma2")
st.caption(f"Backend: {API_URL}")

# Health badge so it's obvious whether the backend is up and the model loaded.
try:
    health = requests.get(f"{API_URL}/", timeout=5).json()
    if health.get("model_loaded") == "True":
        st.success("Backend up · model loaded")
    else:
        st.warning("Backend up · model not loaded yet (loads on first prediction)")
except requests.RequestException as exc:
    st.error(f"Backend unreachable: {exc}")

with st.form("predict"):
    image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
    question = st.text_input(
        "Question", "Which property do these objects have in common?"
    )
    choices_raw = st.text_input("Choices (comma-separated)", "soft, salty, sticky")
    hint = st.text_area("Hint (optional)", "")
    lecture = st.text_area("Lecture (optional)", "")
    submitted = st.form_submit_button("Predict")

if submitted:
    if image_file is None:
        st.error("PaliGemma is image-conditioned — please upload an image.")
        st.stop()
    choices = [c.strip() for c in choices_raw.split(",") if c.strip()]
    if len(choices) < 2:
        st.error("Provide at least two choices.")
        st.stop()

    image = Image.open(image_file).convert("RGB")
    st.image(image, width=240)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = {
        "question": question,
        "choices": choices,
        "hint": hint,
        "lecture": lecture,
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
    }
    with st.spinner("Querying the model (first call may be slow while it loads)…"):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=600)
            resp.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    letter = resp.json()["prediction"]
    idx = ord(letter) - ord("A")
    answer = choices[idx] if 0 <= idx < len(choices) else "(out of range)"
    st.metric("Predicted answer", f"{letter} — {answer}")
