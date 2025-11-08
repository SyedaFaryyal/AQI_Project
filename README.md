Pearls AQI Predictor (Hopsworks Integrated)
Developed by: Syeda Faryal Fatima
Organization: 10Pearls
Project Type: Machine Learning + MLOps + Streamlit + Hopsworks
📖 Overview

Pearls AQI Predictor is a fully automated Air Quality Index (AQI) forecasting system built for Karachi.
It fetches live pollution and weather data, processes it into meaningful features, trains forecasting models, and deploys them into a Streamlit dashboard.
The entire system is powered by Hopsworks Feature Store and Model Registry for end-to-end MLOps integration.

🚀 Key Features

✅ Automated feature pipeline fetching and aggregating AQI + weather data
✅ Automated model training with Random Forest for next 3-day AQI forecast
✅ Models automatically pushed to Hopsworks Model Registry
✅ Streamlit Web App to visualize predictions, feature importance, and local explanations
✅ CI/CD pipeline via GitHub Actions to retrain and refresh data automatically
✅ Interactive EDA Dashboard with pollutant trends and model performance
✅ Hazardous AQI alerts integrated based on real-time predictions

⚙️ System Architecture

Feature Pipeline (hourly)

Fetches live AQI data via OpenAQ API

Retrieves weather data via Open-Meteo

Aggregates and builds ML-ready features

Pushes data into Hopsworks Feature Store

Training Pipeline (daily)

Loads latest features from Feature Store

Trains models for 1, 2, and 3-day forecasts

Evaluates models using RMSE, MAE, R²

Uploads models to Hopsworks Model Registry

Streamlit Dashboard

Visualizes live AQI predictions

Provides SHAP and LIME explainability views

Includes EDA section with correlations, trends, and model metrics

Automation (CI/CD)

Hourly Feature Pipeline via GitHub Actions

Daily Model Training Pipeline automation

🧠 Models Used
Model	Purpose	Framework	Evaluation Metrics
RandomForestRegressor	AQI Forecast (Day 1, 2, 3)	Scikit-learn	RMSE, MAE, R²

Future improvements may include TensorFlow and PyTorch-based deep learning models.

📊 Performance Metrics (Hopsworks Synced)
Forecast Day	RMSE	MAE	R²
Day 1	~6.0	~4.4	0.93
Day 2	~4.4	~3.3	0.96
Day 3	~7.3	~4.5	0.92

All metrics are automatically fetched from the Hopsworks Model Registry and displayed in the EDA Dashboard.

💡 Explainability

SHAP: Displays global feature importance (which pollutants and weather features most influence AQI).

LIME: Provides local interpretability for specific prediction samples.

Both visualizations are fully interactive in Streamlit.

⚠️ AQI Hazard Levels
AQI Range	Category	Color	Health Advisory
0–50	Good	🟢 Green	Air quality is satisfactory
51–100	Moderate	🟡 Yellow	Acceptable but minor risk
101–150	Unhealthy for Sensitive Groups	🟠 Orange	Avoid outdoor exertion
151–200	Unhealthy	🔴 Red	General public may experience health effects
201–300	Very Unhealthy	🟣 Purple	Emergency conditions
301–500	Hazardous	⚫ Maroon	Health warning of emergency conditions

When predictions exceed 150 AQI, the app automatically triggers a Hazard Alert in the Streamlit dashboard.

🧩 Tech Stack

Language: Python 3.10

ML Framework: Scikit-learn

Visualization: Streamlit, Matplotlib, Seaborn

Explainability: SHAP, LIME

MLOps: Hopsworks Feature Store & Model Registry

Automation: GitHub Actions

Environment Management: Python venv

APIs: OpenAQ, Open-Meteo

🧱 Folder Structure
AQI_Project/
│
├── feature_pipeline/
│   ├── fetch_raw.py
│   ├── compute_features.py
│   └── run_feature_pipeline.py
│
├── training_pipeline/
│   └── train_models.py
│
├── web_app/
│   └── streamlit_app.py
│
├── model_registry/
│   └── load_model.py
│
├── scripts/
│   ├── run_hourly_features.ps1
│   └── run_daily_training.ps1
│
├── data/
│   ├── features.csv
│   └── models/
│
├── .github/workflows/
│   └── ci_cd.yml
│
├── .env
├── requirements.txt
└── README.md

🧪 Local Setup Guide

Clone the repository

git clone https://github.com/YOUR_USERNAME/Pearls_AQI_Predictor.git
cd Pearls_AQI_Predictor


Create a new virtual environment

python -m venv .venv
.venv\Scripts\activate   # Windows


Install dependencies

pip install -r requirements.txt


Configure your .env file

CITY=Karachi
LOCAL_FEATURE_STORE=0
HOPSWORKS_HOST=https://c.app.hopsworks.ai
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT=default


Run feature pipeline

python -m feature_pipeline.run_feature_pipeline


Train and upload models

python training_pipeline/train_models.py


Run Streamlit app

streamlit run web_app/streamlit_app.py

⚙️ CI/CD Pipeline

GitHub Actions automates:

Hourly feature updates

Daily model retraining and registry upload

You can view and monitor runs in the Actions tab of your GitHub repo.

📈 Hopsworks Integration

Feature Store: All feature data is stored and versioned automatically.

Model Registry: Each model upload includes performance metrics.

Models can be explored here:
https://c.app.hopsworks.ai

🏁 Results & Achievements

Fully automated AQI prediction pipeline

Real-time integration with APIs

Hopsworks-based data and model versioning

CI/CD automation with GitHub Actions

Interactive dashboards with model explainability

Professional-grade MLOps workflow

👩‍💻 Author

Syeda Faryal Fatima
Data Science Trainee at 10Pearls
📍 Karachi, Pakistan
🌐 Project: Pearls AQI Predictor (Hopsworks Integrated)

🏆 Final Remarks

This project demonstrates a complete end-to-end MLOps pipeline, integrating real-world AQI data, automated training, model explainability, and deployment — achieving AI-driven environmental forecasting with transparency and automation.
