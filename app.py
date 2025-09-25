# Gender -> 1 Male  0 Female
# Churn -> 1 Yes    0 No
# Transformation pipeline is within model.pkl
# Model is exported as model.pkl
# Order of X -> "Age", "Tenure", "MonthlyCharges", "Gender", "TechSupport", "ContractType", "InternetService"

import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model/best_model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Please enter the values and hit the prediction button to get the predition.")

st.divider()

age = st.number_input("Enter age", min_value=16, max_value=100, value=30)

tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)

monthlycharges = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

techsupport = st.selectbox("Tech Support", ["Yes", "No"])

ContractType = st.selectbox("Enter the Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])

InternetService = st.selectbox("Enter the Internet Service", ["Fiber Optic", "DSL"])

st.divider()

predict_button = st.button("Predict")

if predict_button:
    columns = ['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'TechSupport', 'ContractType', 'InternetService']
    X = [age, gender, tenure, monthlycharges, techsupport, ContractType, InternetService]

    X_df = pd.DataFrame([X], columns=columns)

    prediction = model.predict(X_df)[0]

    predicted = 'Yes' if prediction == 1 else 'No'

    st.write(f"Predicted: {predicted}")

else:
    st.write("Please enter the values and use predict button.")