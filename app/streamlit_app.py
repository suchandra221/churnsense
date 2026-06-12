import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="ChurnSense", page_icon="📊", layout="wide")

st.title("📊 ChurnSense Dashboard")

@st.cache_resource
def load_model():
    return joblib.load("models/final_churn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

churn_rate = (df["Churn"].value_counts(normalize=True)["Yes"] * 100).round(2)

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers", len(df))
c2.metric("Churn Rate", f"{churn_rate}%")
c3.metric("Total Features", df.shape[1])

st.subheader("Churn Distribution")
fig = px.histogram(df, x="Churn", color="Churn")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Contract Type vs Churn")
fig2 = px.histogram(df, x="Contract", color="Churn", barmode="group")
st.plotly_chart(fig2, use_container_width=True)