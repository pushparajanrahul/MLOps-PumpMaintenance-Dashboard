import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.preprocess import engineer_features

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_path = os.path.join(current_dir, 'models', 'pump_failure_model.pkl')
scaler_path = os.path.join(current_dir, 'models', 'scaler.pkl')
feature_names_path = os.path.join(current_dir, 'models', 'feature_names.pkl')

# Load the model, scaler, and feature names
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_names = joblib.load(feature_names_path)

# Load the model and scaler
# model = joblib.load('../models/pump_failure_model.pkl')
# scaler = joblib.load('../models/scaler.pkl')

st.set_page_config(page_title="Pump Failure Prediction", layout="wide")

st.title("Industrial Pump Failure Prediction")

# Sidebar for user input
st.sidebar.header("Pump Parameters")


# Function to get user input
def get_user_input():
    vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 5.0, 1.5)
    temperature = st.sidebar.slider("Temperature (°C)", 50, 100, 75)
    pressure = st.sidebar.slider("Pressure (PSI)", 40, 60, 50)
    flow_rate = st.sidebar.slider("Flow Rate (gal/min)", 80, 120, 100)
    power = st.sidebar.slider("Power (kW)", 8, 12, 10)
    rpm = st.sidebar.slider("RPM", 1700, 1900, 1800)
    hours_since_maintenance = st.sidebar.number_input("Hours Since Last Maintenance", 0, 10000, 1000)

    data = {
        'timestamp': datetime.now(),
        'vibration': vibration,
        'temperature': temperature,
        'pressure': pressure,
        'flow_rate': flow_rate,
        'power': power,
        'rpm': rpm,
        'hours_since_maintenance': hours_since_maintenance
    }
    return pd.DataFrame(data, index=[0])


# Get user input
user_input = get_user_input()


# Make prediction
def predict(df):
    # Ensure df has at least two rows for rolling calculations
    if len(df) == 1:
        df = pd.concat([df] * 2, ignore_index=True)

    df_engineered = engineer_features(df)

    # Use the loaded feature names
    X = df_engineered[feature_names]

    # Ensure the order of features matches the model's expectation
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]
    return probability


def predict_single(df):
    df_engineered = engineer_features(df)
    X = df_engineered[feature_names]
    X_scaled = scaler.transform(X)
    probability = model.predict_proba(X_scaled)[0][1]
    return probability


def predict_batch(df):
    df_engineered = engineer_features(df)
    X = df_engineered[feature_names]
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    return probabilities

failure_probability = predict(user_input)

# Display prediction
st.header("Failure Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Failure Probability", f"{failure_probability:.2%}")
col2.metric("Status", "High Risk" if failure_probability > 0.7 else "Normal")
col3.metric("Recommended Action", "Immediate Maintenance" if failure_probability > 0.7 else "Regular Operation")

# Visualize current pump status
st.header("Current Pump Status")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=failure_probability,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Failure Probability"},
    gauge={
        'axis': {'range': [None, 1]},
        'steps': [
            {'range': [0, 0.5], 'color': "lightgreen"},
            {'range': [0.5, 0.7], 'color': "yellow"},
            {'range': [0.7, 1], 'color': "red"}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 0.7}}))

st.plotly_chart(fig)

# Historical Data Simulation
st.header("Simulated Historical Data")

st.header("Simulated Historical Data")


def simulate_historical_data(days=7):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')

    data = []
    for date in date_range:
        row = user_input.copy()
        row['timestamp'] = date
        for col in ['vibration', 'temperature', 'pressure', 'flow_rate', 'power', 'rpm']:
            row[col] += np.random.normal(0, row[col].values[0] * 0.1)  # Add some random noise
        data.append(row)

    return pd.concat(data, ignore_index=True)


historical_data = simulate_historical_data()
historical_probabilities = predict_batch(historical_data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=historical_data['timestamp'], y=historical_probabilities, mode='lines', name='Failure Probability'))
fig.update_layout(title="Historical Failure Probability", xaxis_title="Date", yaxis_title="Probability")
st.plotly_chart(fig)

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Input Data")
    st.write(user_input)