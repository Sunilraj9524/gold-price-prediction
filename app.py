import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import random
import tensorflow as tf

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
    if 'model' not in st.session_state:
        st.session_state.model, st.session_state.scaler = load_assets()
        st.session_state.model_version = "Base Version (Static)"
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# 3. SIDEBAR CONTROLS & SETTINGS
st.sidebar.header("⚙️ Display Settings")
currency = st.sidebar.radio("Currency", ["USD ($)", "INR (₹)"])
unit = st.sidebar.radio("Unit", ["Per Ounce (oz)", "Per Gram (g)"])

st.sidebar.markdown("---")
st.sidebar.header("🧠 Continuous Learning")
st.sidebar.info(f"Current Brain: {st.session_state.model_version}")

if st.sidebar.button("⚡ Retrain on Latest Data"):
    with st.spinner("Reloading base model and fine-tuning safely..."):
        new_df = yf.download('GC=F', period='6mo', interval='1d')
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)
            
        new_data = new_df['Close'].values.reshape(-1, 1)
        scaled_data = st.session_state.scaler.transform(new_data)
        
        X_new, y_new = [], []
        for i in range(60, len(scaled_data)):
            X_new.append(scaled_data[i-60:i, 0])
            y_new.append(scaled_data[i, 0])
            
        X_new, y_new = np.array(X_new), np.array(y_new)
        X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
        
        st.session_state.model = load_model('gold_price_prediction_model_2.keras')
        
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)
        
        from tensorflow.keras.optimizers import Adam
        st.session_state.model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        st.session_state.model.fit(X_new, y_new, epochs=1, batch_size=16, verbose=0)
        
        st.session_state.model_version = f"Fine-Tuned at {datetime.now().strftime('%H:%M:%S')}"
        st.sidebar.success("✅ Model Fine-Tuned (Deterministic Mode)!")
        time.sleep(1)

# ----------------------------------------

if st.sidebar.button("🚀 Run Prediction Pipeline"):
    
    start_time = time.time()
    
    # 4. DATA INGESTION (Fetch Gold & Exchange Rate)
    with st.spinner("Fetching live market data & exchange rates..."):
        df = yf.download('GC=F', period='6mo', interval='1d')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        model_data = df['Close'].values.reshape(-1, 1)
        
        # Fetch USD to INR rate safely
        try:
            inr_data = yf.download('INR=X', period='5d')
            if isinstance(inr_data.columns, pd.MultiIndex):
                inr_data.columns = inr_data.columns.get_level_values(0)
            inr_rate = float(inr_data['Close'].dropna().iloc[-1])
        except:
            inr_rate = 83.0 # Safe fallback if Yahoo Finance is down
            
    # Unit & Currency Math Multipliers
    curr_sym = "₹" if currency == "INR (₹)" else "$"
    curr_mult = inr_rate if currency == "INR (₹)" else 1.0
    unit_sym = "g" if unit == "Per Gram (g)" else "oz"
    unit_div = 31.1035 if unit == "Per Gram (g)" else 1.0

    # Helper function to format prices dynamically
    def fmt(price):
        converted = (price * curr_mult) / unit_div
        return f"{curr_sym}{converted:,.2f}/{unit_sym}"
        
    # 5. BATCH PREDICTIONS (For Accuracy Graph & Yesterday/Today)
    scaled_data_full = st.session_state.scaler.transform(model_data)
    eval_days = 30 # Check last 30 days for the graph
    
    X_eval = []
    for i in range(len(scaled_data_full) - eval_days, len(scaled_data_full)):
        X_eval.append(scaled_data_full[i-60:i, 0])
    X_eval = np.array(X_eval).reshape((eval_days, 60, 1))
    
    with st.spinner("Calculating historical accuracy & future forecasts..."):
        # Predict past 30 days
        past_preds_scaled = st.session_state.model.predict(X_eval, verbose=0)
        past_preds = st.session_state.scaler.inverse_transform(past_preds_scaled)
        
        # Predict Tomorrow (T+1)
        X_tomorrow = scaled_data_full[-60:].reshape(1, 60, 1)
        pred_tomorrow_scaled = st.session_state.model.predict(X_tomorrow, verbose=0)
        pred_tomorrow = st.session_state.scaler.inverse_transform(pred_tomorrow_scaled)[0][0]

    end_time = time.time()
    
    # Extract Specific Days
    actuals = model_data[-eval_days:]
    actual_dates = df.index[-eval_days:]
    
    # YESTERDAY (T-1)
    actual_yest = actuals[-2][0]
    pred_yest = past_preds[-2][0]
    acc_yest = 100 - ((abs(pred_yest - actual_yest) / actual_yest) * 100)
    
    # TODAY (T0)
    actual_today = actuals[-1][0]
    pred_today = past_preds[-1][0]
    acc_today = 100 - ((abs(pred_today - actual_today) / actual_today) * 100)
    
    current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # 6. DISPLAY DASHBOARD
    st.markdown(f"### 🕒 Last Updated: {current_time} | ⚡ Latency: {(end_time - start_time):.2f}s")
    st.markdown("---")
    
    # TOP ROW: 3-Day View
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏪ Yesterday's Review")
        st.metric(label=f"Actual Price ({actual_dates[-2].strftime('%d %b')})", value=fmt(actual_yest))
        st.metric(label="AI Prediction Was", value=fmt(pred_yest), delta=f"{acc_yest:.2f}% Accuracy", delta_color="normal")

    with col2:
        st.subheader("📊 Today's Market")
        st.metric(label=f"Actual Price ({actual_dates[-1].strftime('%d %b')})", value=fmt(actual_today))
        st.metric(label="AI Prediction Was", value=fmt(pred_today), delta=f"{acc_today:.2f}% Accuracy", delta_color="normal")

    with col3:
        st.subheader("🔮 Tomorrow's Forecast")
        st.metric(label=f"AI Prediction ({ (actual_dates[-1] + pd.Timedelta(days=1)).strftime('%d %b') })", value=fmt(pred_tomorrow), delta=f"Expected Trend vs Today")
        
        diff_vs_today = pred_tomorrow - actual_today
        if diff_vs_today > 0:
            st.success("📈 Forecast: Bullish (Up)")
        else:
            st.error("📉 Forecast: Bearish (Down)")
            
    st.caption(f"🤖 Brain Engine: **{st.session_state.model_version}** | 💱 Exchange Rate used: 1 USD = {inr_rate:.2f} INR")

    # 7. ACCURACY GRAPH (Actual vs Predicted)
    st.markdown("---")
    st.subheader("🎯 AI Accuracy Tracking (Last 30 Days)")
    
    fig_acc = go.Figure()

    # Convert arrays to selected currency/unit
    graph_actuals = [(x[0] * curr_mult) / unit_div for x in actuals]
    graph_preds = [(x[0] * curr_mult) / unit_div for x in past_preds]

    fig_acc.add_trace(go.Scatter(
        x=actual_dates, y=graph_actuals,
        mode='lines+markers', name='Actual Market Price',
        line=dict(color='blue', width=2), marker=dict(size=6)
    ))

    fig_acc.add_trace(go.Scatter(
        x=actual_dates, y=graph_preds,
        mode='lines+markers', name='AI Predicted Price',
        line=dict(color='orange', width=2, dash='dash'), marker=dict(size=6)
    ))

    fig_acc.update_layout(
        xaxis_title='Date',
        yaxis_title=f'Price ({curr_sym}/{unit_sym})',
        template='plotly_dark',
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    # 8. MARKET TREND (Candlestick)
    st.subheader("📈 Overall Market Trend (Candlestick)")
    
    fig_candle = go.Figure()
    
    # Convert Candlestick data
    c_open = [(x * curr_mult) / unit_div for x in df['Open'].values.flatten()]
    c_high = [(x * curr_mult) / unit_div for x in df['High'].values.flatten()]
    c_low = [(x * curr_mult) / unit_div for x in df['Low'].values.flatten()]
    c_close = [(x * curr_mult) / unit_div for x in df['Close'].values.flatten()]

    fig_candle.add_trace(go.Candlestick(
        x=df.index, open=c_open, high=c_high, low=c_low, close=c_close,
        name='Market Data'
    ))

    # Add Tomorrow's Forecast Point
    next_day = actual_dates[-1] + pd.Timedelta(days=1)
    p_tom = (pred_tomorrow * curr_mult) / unit_div
    p_tod = (actual_today * curr_mult) / unit_div
    
    fig_candle.add_trace(go.Scatter(
        x=[actual_dates[-1], next_day], y=[p_tod, p_tom],
        mode='lines+markers', name='Tomorrow Forecast',
        line=dict(color='cyan', width=2, dash='dot'), marker=dict(size=8, color='cyan')
    ))

    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title=f'Price ({curr_sym}/{unit_sym})',
        template='plotly_dark',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    st.plotly_chart(fig_candle, use_container_width=True)

else:
    st.info("👈 Click 'Run Prediction Pipeline' to start.")
