import os
import json
import joblib
import numpy as np
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Prediction")
st.warning(
    "Educational demonstration only. This prediction is not a medical diagnosis "
    "and should not be used for clinical decisions."
)

MODEL_PATH = "models/breast_cancer_model.joblib"
META_PATH = "models/metadata.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error("Model files are missing. Run `python src/train.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

data = load_breast_cancer(as_frame=True)
defaults = data.data.median()

st.write("Enter the 30 diagnostic feature values. Median values are provided as defaults.")

values = {}
cols = st.columns(2)

for i, feature in enumerate(metadata["feature_names"]):
    with cols[i % 2]:
        values[feature] = st.number_input(
            feature,
            value=float(defaults[feature]),
            format="%.5f"
        )

if st.button("Predict", type="primary"):
    X = np.array([[values[f] for f in metadata["feature_names"]]], dtype=float)
    prediction = int(model.predict(X)[0])
    probability = model.predict_proba(X)[0]

    label = metadata["target_mapping"][str(prediction)]
    confidence = float(probability[prediction]) * 100

    if label == "benign":
        st.success(f"Model prediction: {label.upper()}")
    else:
        st.error(f"Model prediction: {label.upper()}")

    st.write(f"Model probability for predicted class: {confidence:.2f}%")
    st.caption(
        "This is a machine-learning output on a public research dataset, "
        "not a clinical assessment."
    )
