"""
Heart Disease Prediction System - Streamlit web app
----------------------------------------------------
Loads svc_trained_model.pkl (trained in Heart_Disease_Prediction.ipynb) and
recreates the same LabelEncoders (fit on the same fixed category lists) used
during training, so predictions match the notebook exactly.
Educational demo only - NOT a medical diagnostic tool.
"""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """Same fixed category lists as the notebook (Step 5.1) -> identical codes."""
    return {
        "Gender": LabelEncoder().fit(["Male", "Female"]),
        "ChestPain": LabelEncoder().fit(["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]),
        "ExerciseAngina": LabelEncoder().fit(["Yes", "No"]),
        "Thalassemia": LabelEncoder().fit(["Normal", "Fixed Defect", "Reversible Defect"]),
    }

model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction** notebook
    (UCI Cleveland dataset). Enter the patient's details below and the trained
    Support Vector Classifier (SVC) will predict whether heart disease is present.

    *Model accuracy on held-out test data: **80%** (61 test patients).*
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
with col2:
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"], index=3)
    thalassemia = st.selectbox("Thalassemia (Thallium Stress Test)", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Gender": [gender],
        "ChestPain": [chest_pain],
        "ExerciseAngina": [exercise_angina],
        "Thalassemia": [thalassemia],
    })
    encoded = pd.DataFrame({col: encoders[col].transform(user_input[col]) for col in user_input.columns})

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ❌ HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from the UCI Cleveland Heart Disease dataset using only 4 input attributes. "
    "This is NOT a medical tool and must not be used for diagnosis."
)
