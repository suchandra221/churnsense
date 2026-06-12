import pandas as pd

def prepare_input(data):
    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df