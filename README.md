💳 Financial Anomaly Detection System

An end-to-end Machine Learning–based Financial Anomaly Detection system that identifies suspicious accounts from transaction-level data using Isolation Forest.
The project includes data preprocessing, feature engineering, model training, and interactive deployment using Gradio / Streamlit, and is ready for Hugging Face deployment.

🚀 Project Overview

Financial fraud and abnormal transaction behavior are critical risks in digital payment systems.
This project detects anomalous accounts by analyzing transaction patterns such as:

Frequency of transactions
Total transaction amount
Average transaction value
Maximum transaction value

Using unsupervised learning (Isolation Forest), the system flags accounts that significantly deviate from normal behavior.


🧠 Machine Learning Approach

Model: Isolation Forest (Unsupervised Anomaly Detection)
Why Isolation Forest?

Works well with unlabeled data
Efficient for high-dimensional datasets
Designed specifically for anomaly detection

Features Used (Account-Level Aggregation)
Feature	Description
total_transactions	Number of transactions per account
total_amount	Total transaction value
avg_amount	Average transaction amount
max_amount	Maximum transaction amount

📂 Dataset
Input CSV Format (Transaction-Level)

The system expects a CSV file with the following columns:

Timestamp
TransactionID
AccountID
Amount
Merchant
TransactionType
Location


You can upload the original transaction dataset (financial_anomaly_data.csv) directly into the application.

🛠️ Tech Stack

Python
Pandas, NumPy
Scikit-learn
Joblib
Gradio / Streamlit
Hugging Face Spaces


🖥️ Application Interfaces
1️⃣ Gradio Web Interface

Upload transaction CSV

Automatically aggregates data to account level

Flags each account as Normal / Anomalous

Displays results in an interactive table

2️⃣ Streamlit Dashboard

CSV upload

Data preview

Account-level risk analysis

Metrics showing anomalous account counts

📁 Project Structure
financial-anomaly-detection/
│
├── app.py                     # Gradio / Streamlit application
├── financial_anomaly_data.csv # Sample dataset
├── isolation_forest.pkl       # Trained ML model
├── scaler.pkl                 # Feature scaler
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
└── .gradio/                   # Gradio runtime files


▶️ How to Run Locally
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Gradio App
python app.py

The app will launch at:

http://127.0.0.1:7860


☁️ Deployment (Hugging Face)

Upload the following files to your Hugging Face Space:

app.py

requirements.txt
isolation_forest.pkl
scaler.pkl
Select Gradio as the Space SDK
The app will auto-launch after build

📊 Output

The system returns an account-level anomaly report:

AccountID	total_transactions	total_amount	avg_amount	max_amount	Anomaly
A1023	45	850000	18888	250000	Anomalous


🎯 Key Learnings

End-to-end ML pipeline development
Unsupervised anomaly detection
Feature engineering from transactional data
Model serialization & reuse
Web-based ML deployment
Debugging real-world deployment issues

🔮 Future Improvements

Add anomaly score visualization
Explainability (why an account was flagged)
Time-based anomaly detection
User-level dashboards
API endpoint for real-time detection


👨‍💻 Author

Anirudha Kolay
Computer Science (AI & ML) | Data & ML Enthusiast
Built with ❤️ for real-world ML deployment experience
