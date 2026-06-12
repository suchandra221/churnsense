import sys
sys.path.append(".")

import pandas as pd
from src.model import load_model, predict_single, predict_batch

def sample_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }

def test_model_loads():
    model = load_model()
    assert model is not None

def test_single_prediction():
    model = load_model()
    result = predict_single(model, sample_customer())

    assert "prediction" in result
    assert "churn_probability" in result
    assert 0 <= result["churn_probability"] <= 1

def test_batch_prediction():
    model = load_model()
    df = pd.DataFrame([sample_customer(), sample_customer()])
    result = predict_batch(model, df)

    assert "prediction" in result.columns
    assert "churn_probability" in result.columns
    assert len(result) == 2