"""
Streamlit demo — one page.

Upload a mouth photo → original with detection box, Grad-CAM overlay,
predicted disease + confidence, plain-language recommendation.

Run:  streamlit run app.py
"""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

import config
from src.pipeline import OralLesionPipeline

st.set_page_config(page_title="Oral Lesion Screening", layout="wide")


@st.cache_resource(show_spinner="Loading models…")
def get_pipeline() -> OralLesionPipeline:
    return OralLesionPipeline()


st.title("🦷 Oral Lesion Screening")
st.caption(
    "Screening aid, not a diagnosis. Two-stage: lesion detector → disease classifier."
)

uploaded = st.file_uploader("Upload a mouth photo", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

file_bytes = np.frombuffer(uploaded.read(), np.uint8)
bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if bgr is None:
    st.error("Could not read that image.")
    st.stop()

pipe = get_pipeline()
with st.spinner("Analysing…"):
    res = pipe.analyze(bgr)

if not res.lesion_found and not res.used_fallback:
    st.image(res.boxed_image, caption="Uploaded image", use_container_width=True)
    st.success(f"✅ {res.recommendation}")
    st.stop()

if res.used_fallback:
    st.error(f"🟠 {res.recommendation}")
    st.caption(
        "Detector found no lesion; this is a low-confidence central-region "
        "guess, not a localized detection."
    )
else:
    st.warning(f"⚠️ {res.recommendation}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Detection")
    st.image(res.boxed_image, use_container_width=True)
with c2:
    st.subheader("Grad-CAM (why)")
    st.image(res.gradcam_overlay, use_container_width=True)

st.subheader(f"Predicted: {res.disease}  ·  {res.confidence:.1%} confidence")
st.progress(min(max(res.confidence, 0.0), 1.0))

st.subheader("All class probabilities")
for name, p in sorted(
    zip(config.CLASS_NAMES, res.probs), key=lambda x: x[1], reverse=True
):
    st.write(f"{name}: {p:.1%}")
    st.progress(min(max(float(p), 0.0), 1.0))

st.caption(f"Detector confidence threshold: {res.detector_conf}")
