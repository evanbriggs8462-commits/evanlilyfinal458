
import streamlit as st
import pickle
import pandas as pd

# Load model + column structure
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("Loan Approval Prediction App")

# Example user inputs
fico = st.number_input("FICO Score", 300, 850, 650)
income = st.number_input("Monthly Gross Income", 0, 30000, 5000)
loan_amt = st.number_input("Requested Loan Amount", 500, 50000, 10000)

if st.button("Predict Approval"):
    # Build a dataframe of one sample
    df = pd.DataFrame([{
        "FICO_score": fico,
        "Monthly_Gross_Income": income,
        "Requested_Loan_Amount": loan_amt
    }])

    # Align columns to training schema
    df = df.reindex(columns=model_columns, fill_value=0)

    # Predict probability
    prob = model.predict_proba(df)[0][1]
    st.write(f"Approval Probability: **{prob:.2%}**")
