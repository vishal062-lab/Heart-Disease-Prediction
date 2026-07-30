"""
Heart Disease Prediction - Web App (Streamlit)
================================================
Loads the trained model (heart_disease_model.pkl) and scaler (scaler.pkl)
saved by heart_disease_prediction.py, and lets the user enter patient
details to get an instant prediction.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found! Pehle 'heart_disease_prediction.py' run karo "
        "taaki 'heart_disease_model.pkl' aur 'scaler.pkl' generate ho jaayein, "
        "usi folder mein jahan yeh app.py hai."
    )
    st.stop()

# ---------------------------------------------------------------------------
# PAGE CONFIG & TITLE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction")
st.write("Apna health data bharo aur turant prediction dekho ki heart disease ka risk hai ya nahi.")

st.divider()

# ---------------------------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        cp = st.selectbox(
            "Chest Pain Type",
            options=[
                "Typical Angina", "Atypical Angina",
                "Non-anginal Pain", "Asymptomatic"
            ],
        )
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
        restecg = st.selectbox(
            "Resting ECG Result",
            options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
        )

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"])
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment", options=["Upsloping", "Flat", "Downsloping"])
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

# ---------------------------------------------------------------------------
# ENCODE INPUTS (must match how the training data was encoded)
# ---------------------------------------------------------------------------
sex_map = {"Male": 1, "Female": 0}
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
fbs_map = {"No": 0, "Yes": 1}
restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
exang_map = {"No": 0, "Yes": 1}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

if submitted:
    features = np.array([[
        age,
        sex_map[sex],
        cp_map[cp],
        trestbps,
        chol,
        fbs_map[fbs],
        restecg_map[restecg],
        thalach,
        exang_map[exang],
        oldpeak,
        slope_map[slope],
        ca,
        thal_map[thal],
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, "predict_proba") else None

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ **Result: Heart Disease Risk Detected**")
    else:
        st.success(f"✅ **Result: No Heart Disease Detected**")

    if proba is not None:
        st.metric("Model Confidence (probability of disease)", f"{proba*100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))

    st.caption(
        "⚠️ Disclaimer: Yeh sirf ek ML learning project hai, actual medical diagnosis "
        "ke liye doctor se consult karo. Yeh app real clinical use ke liye nahi hai."
    )