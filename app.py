import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# 1. SETUP PAGE
st.set_page_config(page_title="Gold Price MLOps", layout="wide")
st.title("🏆 Gold Price Prediction: MLOps Pipeline")

# 2. LOAD SAVED ASSETS
@st.cache_resource
def load_assets():
    model = load_model('gold_price_prediction_model_2.keras')
    scaler = joblib.load('gold_price_prediction_scaler_2.gz')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("✅ Model and Scaler Loaded Successfully")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# 3. SIDEBAR CONTROLS
st.sidebar.header("MLOps Configuration")
days_lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=90, value=60)

if st.sidebar.button("Run Prediction Pipeline"):
    
    # 4. DATA INGESTION (Live)
    with st.spinner("Fetching live market data..."):
        df = yf.download('GC=F', period='6mo', interval='1d')
        
        if isinstance(df.columns, pd.MultiIndex):
            data = df.xs('Close', axis=1, level=0)
        else:
            data = df['Close']
            
        raw_prices = data.values.reshape(-1, 1)
        
    # 5. PREPROCESSING
    last_60_days = raw_prices[-days_lookback:]
    last_60_days_scaled = scaler.transform(last_60_days)
    X_input = last_60_days_scaled.reshape(1, days_lookback, 1)
    
    # 6. INFERENCE
    predicted_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
    
    # Get Dates and Last Price
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    last_actual_price = raw_prices[-1][0]
    diff = predicted_price - last_actual_price
    
    # 7. DISPLAY METRICS
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Last Close")
        st.metric(label=f"Price on {last_date.strftime('%Y-%m-%d')}", value=f"${last_actual_price:.2f}")
        
    with col2:
        st.subheader("Prediction")
        st.metric(label=f"Forecast for {next_date.strftime('%Y-%m-%d')}", value=f"${predicted_price:.2f}", delta=f"${diff:.2f}")
        
    with col3:
        st.subheader("Trend")
        if diff > 0:
            st.success("📈 Bullish (Up)")
        else:
            st.error("📉 Bearish (Down)")

    # 8. DATA TABLE (Last 10 Days) -- NEW FEATURE
    st.markdown("---")
    st.subheader("Recent Market Data (Last 10 Days)")
    
    # Get last 10 days and reverse them so the newest is on top
    last_10_days = data.tail(10).sort_index(ascending=False)
    
    # Create a nice dataframe for display
    display_df = pd.DataFrame({
        "Date": last_10_days.index.strftime('%Y-%m-%d'),
        "Close Price ($)": [f"${x:.2f}" for x in last_10_days.values]
    })
    
    # Show as a clean table that fills the width
    st.table(display_df)

    # 9. GRAPH
    st.subheader("Market Trend (Last 6 Months)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, raw_prices, label="Actual Price", color='blue')
    ax.scatter([next_date], [predicted_price], color='red', s=100, label="Prediction", zorder=5)
    ax.plot([last_date, next_date], [last_actual_price, predicted_price], color='red', linestyle='--')
    ax.legend()
    st.pyplot(fig)
        
    st.sidebar.info("App Updated: Shows Last 10 Days Data")

else:
    st.info("👈 Click 'Run Prediction Pipeline' in the sidebar to start.")