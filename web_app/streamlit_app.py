import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from model_registry.load_model import load_model_for_day


st.set_page_config(page_title="Karachi AQI Forecast", layout="wide")

# Load feature data
@st.cache_data
def load_data():
    path = "data/features.csv"
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"])
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq="h")
    return df

df = load_data()

# Sidebar
st.sidebar.title("🌍 AQI Forecast Dashboard")
st.sidebar.write("City: **Karachi**")
day_choice = st.sidebar.selectbox("Select Forecast Day", [1, 2, 3])
st.sidebar.write("Day 1 = Tomorrow, Day 2 = +2 days, Day 3 = +3 days")

# Load model
model = load_model_for_day(day_choice)

# Prepare features
ignore_cols = ["time", "timestamp", "target_day1", "target_day2", "target_day3"]
X = df[[c for c in df.columns if c not in ignore_cols and df[c].dtype != "object"]].fillna(0)

# Predictions
preds = model.predict(X)
latest_pred = preds[-1]

# Display
st.title("🌫️ Karachi Air Quality Forecast")
st.metric(label=f"Predicted AQI for Day {day_choice}", value=f"{latest_pred:.1f}")

st.line_chart(pd.DataFrame({
    "timestamp": df["timestamp"],
    "Predicted AQI": preds
}).set_index("timestamp"))

# SHAP explanation
st.subheader("🔍 Feature Importance (SHAP)")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st.write("Top contributing features:")
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.clf()
except Exception as e:
    st.warning(f"Could not generate SHAP plot: {e}")

# LIME explanation
st.subheader("💡 Local Explanation (LIME)")
try:
    sample_index = np.random.randint(0, len(X))
    lime_explainer = LimeTabularExplainer(
        X.values, mode="regression", feature_names=X.columns.tolist()
    )
    exp = lime_explainer.explain_instance(X.values[sample_index], model.predict)
    st.write(f"Explaining prediction for sample #{sample_index}")
    st.components.v1.html(exp.as_html(), height=600)
except Exception as e:
    st.warning(f"Could not generate LIME explanation: {e}")

st.caption("Developed with ❤️ for AQI Forecasting (Karachi)")
