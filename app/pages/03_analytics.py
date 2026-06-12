import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Analytics")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

st.subheader("Monthly Charges Distribution")
fig = px.histogram(df, x="MonthlyCharges", color="Churn")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Tenure Distribution")
fig2 = px.histogram(df, x="tenure", color="Churn")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Internet Service vs Churn")
fig3 = px.histogram(df, x="InternetService", color="Churn", barmode="group")
st.plotly_chart(fig3, use_container_width=True)