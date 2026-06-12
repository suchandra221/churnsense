import joblib
import pandas as pd

def load_model(path="models/final_churn_model.pkl"):
    return joblib.load(path)

def predict_single(model, customer_data):
    df = pd.DataFrame([customer_data])
    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "churn_probability": float(probability)
    }

def predict_batch(model, df):
    probabilities = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    result = df.copy()
    result["prediction"] = predictions
    result["churn_probability"] = probabilities

    return result