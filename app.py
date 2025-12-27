import gradio as gr
import pandas as pd
import joblib
import os

# Load model artifacts
MODEL_PATH = "isolation_forest.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

REQUIRED_COLUMNS = [
    "Timestamp",
    "TransactionID",
    "AccountID",
    "Amount",
    "Merchant",
    "TransactionType",
    "Location"
]

def detect_anomalies(file):
    if file is None:
        return pd.DataFrame({"Error": ["No file uploaded"]})

    df = pd.read_csv(file)

    # Validate columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        return pd.DataFrame({
            "Error": [f"Missing columns: {', '.join(missing_cols)}"]
        })

    # Feature engineering (same as training)
    account_stats = df.groupby("AccountID").agg(
        total_transactions=("TransactionID", "count"),
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        max_amount=("Amount", "max")
    ).reset_index()

    X = scaler.transform(
        account_stats[
            ["total_transactions", "total_amount", "avg_amount", "max_amount"]
        ]
    )

    preds = model.predict(X)
    account_stats["Anomaly"] = [
        "Anomalous" if p == -1 else "Normal" for p in preds
    ]

    return account_stats


app = gr.Interface(
    fn=detect_anomalies,
    inputs=gr.File(label="Upload Transaction CSV"),
    outputs=gr.Dataframe(label="Account-Level Anomaly Results"),
    title="💳 Financial Anomaly Detection System",
    description="Upload transaction-level financial data to detect anomalous accounts using Isolation Forest."
)

if __name__ == "__main__":
    app.launch()
