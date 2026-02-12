import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import time

# 1. SETUP PAGE
st.set_page_config(page_title="Sunil's Gold Price", layout="wide")
st.title("🏆 Gold Price Prediction")

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
    
    # Start Timer
    start_time = time.time()
    
    # 4. DATA INGESTION (Live)
    with st.spinner("Fetching live market data..."):
        df = yf.download('GC=F', period='6mo', interval='1d')
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        model_data = df['Close'].values.reshape(-1, 1)
        
    # 5. PREPROCESSING
    last_60_days = model_data[-days_lookback:]
    last_60_days_scaled = scaler.transform(last_60_days)
    X_input = last_60_days_scaled.reshape(1, days_lookback, 1)
    
    # 6. INFERENCE
    predicted_scaled = model.predict(X_input)
    
    # Calibration Offset
    calibration_value = 0 
    predicted_price = scaler.inverse_transform(predicted_scaled)[0][0] + calibration_value
    
    # Stop Timer
    end_time = time.time()
    latency = end_time - start_time
    
    # Get Dates and Last Price
    last_date = df.index[-1]
    next_date = last_date + pd.Timedelta(days=1)
    last_actual_price = model_data[-1][0]
    diff = predicted_price - last_actual_price
    
    # CONVERSIONS (New Feature)
    # 1 Troy Ounce = 31.1035 Grams
    price_per_gram = last_actual_price / 31.1035
    predicted_per_gram = predicted_price / 31.1035
    
    # Current System Time
    current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # 7. DISPLAY METRICS
    st.markdown(f"### 🕒 Last Updated: {current_time}")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Last Close")
        # Added explicit 'per Ounce' label
        st.metric(label=f"Price/Ounce ({last_date.strftime('%Y-%m-%d')})", value=f"${last_actual_price:.2f}/oz")
        
    with col2:
        st.subheader("Prediction")
        st.metric(label=f"Forecast/Ounce ({next_date.strftime('%Y-%m-%d')})", value=f"${predicted_price:.2f}/oz", delta=f"${diff:.2f}")

    with col3:
        st.subheader("Per Gram")
        # Added useful conversion for context
        st.metric(label="Current Price per Gram", value=f"${price_per_gram:.2f}/g")
        
    with col4:
        st.subheader("Performance")
        st.info(f"⚡ Latency: {latency:.4f}s")
        if diff > 0:
            st.success("📈 Trend: Up")
        else:
            st.error("📉 Trend: Down")

    # 8. DATA TABLE
    st.markdown("---")
    st.subheader("Recent Market Data (Last 10 Days)")
    
    try:
        last_10_days = df.tail(10).sort_index(ascending=False)
        display_df = pd.DataFrame({
            "Date": last_10_days.index.strftime('%Y-%m-%d'),
            "Open ($/oz)": [f"${x:.2f}" for x in last_10_days['Open'].values.flatten()],
            "High ($/oz)": [f"${x:.2f}" for x in last_10_days['High'].values.flatten()],
            "Low ($/oz)": [f"${x:.2f}" for x in last_10_days['Low'].values.flatten()],
            "Close ($/oz)": [f"${x:.2f}" for x in last_10_days['Close'].values.flatten()],
            "Volume": [f"{int(x):,}" for x in last_10_days['Volume'].values.flatten()]
        })
        display_df.set_index("Date", inplace=True)
        st.table(display_df)
    except Exception as e:
        st.error(f"Could not display data table: {e}")

    # 9. GRAPH
    st.subheader("Market Trend (Last 6 Months)")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['Close'], label="Actual Price ($/oz)", color='blue')
    ax.scatter([next_date], [predicted_price], color='red', s=100, label="Prediction", zorder=5)
    ax.plot([last_date, next_date], [last_actual_price, predicted_price], color='red', linestyle='--')
    ax.legend()
    st.pyplot(fig)
    
    st.sidebar.success(f"✅ Pipeline Executed in {latency:.2f}s")

else:
    st.info("👈 Click 'Run Prediction Pipeline' in the sidebar to start.")
