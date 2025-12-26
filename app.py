import gradio as gr
import pandas as pd
import joblib

# Load model artifacts
model = joblib.load("isolation_forest.pkl")
scaler = joblib.load("scaler.pkl")

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
    df = pd.read_csv(file.name)

    # Validate columns
    if not all(col in df.columns for col in REQUIRED_COLUMNS):
        return pd.DataFrame({
            "Error": ["Uploaded CSV does not contain required columns"]
        })

    # Feature engineering (same as training)
    account_stats = df.groupby("AccountID").agg(
        total_transactions=("TransactionID", "count"),
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        max_amount=("Amount", "max")
    ).reset_index()

    X = scaler.transform(
        account_stats[[
            "total_transactions",
            "total_amount",
            "avg_amount",
            "max_amount"
        ]]
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
    app.launch(server_name="0.0.0.0", server_port=7860)
