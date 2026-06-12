import streamlit as st
import pandas as pd
import joblib

st.title("📁 Batch Prediction")

model = joblib.load("models/final_churn_model.pkl")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    probs = model.predict_proba(df)[:, 1]
    preds = model.predict(df)

    df["churn_probability"] = probs
    df["prediction"] = preds

    st.dataframe(df)

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Predictions",
        csv,
        "batch_predictions.csv",
        "text/csv"
    )