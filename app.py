
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load saved model and features
# -----------------------------
model = joblib.load("loan_Approval_predict.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to predict loan approval.")

# -----------------------------
# Create input fields dynamically
# -----------------------------
user_input = {}

for feature in features:
    user_input[feature] = st.number_input(
        label=f"{feature}",
        value=0.0
    )

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.write("### Prediction Probability")
    st.write(f"Approved: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Not Approved: **{prediction_proba[0][0]:.2f}*
