import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import numpy as np
from datetime import timedelta, date

# --- 1. Live Exchange Rate Fetcher ---
@st.cache_data(ttl=3600) # Caches the rate for 1 hour to keep the app fast
def get_live_exchange_rate():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()['rates']['INR']
    except Exception:
        # Failsafe just in case the Wi-Fi drops during your presentation
        return 90.67 

# --- 2. Page Config & Sidebar ---
st.set_page_config(page_title="Gold Price Prediction", layout="wide")

EXCHANGE_RATE = get_live_exchange_rate()
OUNCE_TO_GRAMS = 31.1035
GRAMS_IN_SOVEREIGN = 8.0

st.sidebar.header("Dashboard Settings")
st.sidebar.caption(f"Live USD to INR: ₹{EXCHANGE_RATE:.2f}")

currency = st.sidebar.radio("Select Currency", ("USD", "INR"))
weight_unit = st.sidebar.radio("Select Weight Unit", ("Ounce (oz)", "Gram (g)", "Sovereign (8g)"))

# --- 3. Conversion Logic ---
currency_multiplier = EXCHANGE_RATE if currency == "INR" else 1.0
symbol = "₹" if currency == "INR" else "$"

if weight_unit == "Gram (g)":
    weight_divisor = OUNCE_TO_GRAMS
    unit_label = "g"
elif weight_unit == "Sovereign (8g)":
    weight_divisor = OUNCE_TO_GRAMS / GRAMS_IN_SOVEREIGN
    unit_label = "sov"
else: 
    weight_divisor = 1.0
    unit_label = "oz"

final_multiplier = currency_multiplier / weight_divisor

# --- 4. Load Data (REPLACE THIS WITH YOUR LSTM MODEL DATA) ---
# Generating dummy data so the app runs out-of-the-box for testing
@st.cache_data
def load_data():
    dates = [date.today() - timedelta(days=x) for x in range(30, -1, -1)]
    base_price = 2030 # Base USD per Ounce
    actuals = [base_price + np.random.randint(-30, 30) for _ in range(31)]
    predictions = [a + np.random.randint(-15, 15) for a in actuals]
    return pd.DataFrame({'Date': dates, 'Actual_USD_Ounce': actuals, 'Predicted_USD_Ounce': predictions})

df = load_data()

# --- 5. Apply Conversions to Dataframe ---
df['Actual_Converted'] = df['Actual_USD_Ounce'] * final_multiplier
df['Predicted_Converted'] = df['Predicted_USD_Ounce'] * final_multiplier

# --- 6. Extract Daily Metrics ---
yesterday_actual = df['Actual_Converted'].iloc[-2]
yesterday_predicted = df['Predicted_Converted'].iloc[-2]
today_actual = df['Actual_Converted'].iloc[-1]
today_predicted = df['Predicted_Converted'].iloc[-1]

# Placeholder: Replace this variable with your actual LSTM model prediction for tomorrow
tomorrow_predicted_base_oz = df['Predicted_USD_Ounce'].iloc[-1] + np.random.randint(-10, 10) 
tomorrow_predicted = tomorrow_predicted_base_oz * final_multiplier

# Calculate Yesterday's Accuracy
accuracy = (1 - abs(yesterday_actual - yesterday_predicted) / yesterday_actual) * 100

# --- 7. Main Dashboard UI ---
st.title("Gold Price Prediction Using Machine Learning and LSTM")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Yesterday's Metrics")
    st.metric("Actual Price", f"{symbol}{yesterday_actual:.2f} / {unit_label}")
    st.metric("Predicted Price", f"{symbol}{yesterday_predicted:.2f} / {unit_label}")
    st.metric("Model Accuracy", f"{accuracy:.2f}%")

with col2:
    st.success("Today's Metrics")
    st.metric("Actual Price", f"{symbol}{today_actual:.2f} / {unit_label}")
    st.metric("Predicted Price", f"{symbol}{today_predicted:.2f} / {unit_label}")

with col3:
    st.warning("Tomorrow's Forecast")
    st.metric("Predicted Price", f"{symbol}{tomorrow_predicted:.2f} / {unit_label}")

st.markdown("---")

# --- 8. Accuracy Graph ---
st.subheader("Historical Accuracy: Actual vs Predicted")

fig = px.line(
    df, 
    x='Date', 
    y=['Actual_Converted', 'Predicted_Converted'], 
    labels={'value': f'Price ({symbol}/{unit_label})', 'variable': 'Legend'},
    color_discrete_sequence=['#1f77b4', '#ff7f0e'] # Blue for Actual, Orange for Predicted
)

# Clean up the legend names for the presentation
fig.for_each_trace(lambda t: t.update(name=t.name.replace("_Converted", "").replace("Actual", "Actual Price").replace("Predicted", "Predicted Price")))

st.plotly_chart(fig, use_container_width=True)
