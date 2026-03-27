import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and features
model = joblib.load('loan_Approval_predict.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title="Loan Approval Predictor")

st.title("🏦 Loan Approval Prediction App")

st.write("Enter applicant details to predict loan approval status.")

# Create input fields dynamically
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.write(f"Approval Probability: {probability[0][1]:.2f}")
