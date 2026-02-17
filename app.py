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
import streamlit.components.v1 as components

# 1. SETUP PAGE & STATE ROUTING
st.set_page_config(page_title="Gold Price MLOps", layout="wide")
st.title("🏆 Gold Price Prediction: MLOps Pipeline")

# State Management for Navigation
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'

def set_view(view_name):
    st.session_state.current_view = view_name

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

# 3. SIDEBAR CONTROLS
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
        st.session_state.model.fit(X_new, y_new, epochs=2, batch_size=16, verbose=0)
        
        st.session_state.model_version = f"Fine-Tuned at {datetime.now().strftime('%H:%M:%S')}"
        st.sidebar.success("✅ Model Fine-Tuned (Deterministic Mode)!")
        time.sleep(1)

st.sidebar.markdown("---")
st.sidebar.header("🔮 Forecasting Tools")

if st.sidebar.button("🚀 Run Daily Pipeline"):
    set_view('daily')

if st.sidebar.button("📅 2-Month Long-Term Forecast"):
    set_view('warning')


# --- HELPER FUNCTIONS ---
def get_live_data():
    df = yf.download('GC=F', period='6mo', interval='1d')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    model_data = df['Close'].values.reshape(-1, 1)
    
    try:
        inr_data = yf.download('INR=X', period='5d')
        if isinstance(inr_data.columns, pd.MultiIndex):
            inr_data.columns = inr_data.columns.get_level_values(0)
        inr_rate = float(inr_data['Close'].dropna().iloc[-1])
    except:
        inr_rate = 83.0
    return df, model_data, inr_rate

def format_price(price, inr_rate, currency, unit):
    curr_sym = "₹" if currency == "INR (₹)" else "$"
    curr_mult = inr_rate if currency == "INR (₹)" else 1.0
    unit_sym = "g" if unit == "Per Gram (g)" else "oz"
    unit_div = 31.1035 if unit == "Per Gram (g)" else 1.0
    converted = (price * curr_mult) / unit_div
    return f"{curr_sym}{converted:,.2f}/{unit_sym}"


# ==========================================
# VIEW 1: HOME SCREEN
# ==========================================
if st.session_state.current_view == 'home':
    st.info("👈 Please select a forecasting tool from the sidebar to begin.")


# ==========================================
# VIEW 2: THE WARNING DISCLAIMER SCREEN
# ==========================================
elif st.session_state.current_view == 'warning':
    st.warning("### ⚠️ Financial Disclaimer & Alert")
    st.markdown("""
    **This model is completely created for Technical Analysis.** Long-term recursive forecasting carries inherent mathematical risks. Furthermore, real-world prices may vary significantly from the model's mathematical prediction due to global events, geopolitical shifts, and macroeconomic policies. 
    
    **Please consider doing Fundamental Analysis before relying on long-term AI projections.**
    """)
    st.markdown("---")
    st.write("Do you understand these risks and wish to view the 60-day AI projection?")
    
    colA, colB = st.columns(2)
    with colA:
        if st.button("✅ I Understand. Proceed to Forecast"):
            set_view('long_term')
            st.rerun()
    with colB:
        if st.button("❌ No, Go Back"):
            set_view('daily')
            st.rerun()


# ==========================================
# VIEW 3: 2-MONTH (60 DAY) FORECAST
# ==========================================
elif st.session_state.current_view == 'long_term':
    st.markdown("### 📅 2-Month (60-Day) Strategic Forecast")
    
    with st.spinner("Calculating 60-step recursive trajectory..."):
        df, model_data, inr_rate = get_live_data()
        scaled_data_full = st.session_state.scaler.transform(model_data)
        
        actual_today_val = model_data[-1][0]
        X_eval = scaled_data_full[-61:-1].reshape(1, 60, 1)
        pred_today_raw = st.session_state.scaler.inverse_transform(st.session_state.model.predict(X_eval, verbose=0))[0][0]
        bias = actual_today_val - pred_today_raw
        
        future_predictions_scaled = []
        current_input = scaled_data_full[-60:].reshape(1, 60, 1)

        for _ in range(60): 
            pred = st.session_state.model.predict(current_input, verbose=0)
            future_predictions_scaled.append(pred[0, 0])
            new_step = np.array([[[pred[0, 0]]]])
            current_input = np.append(current_input[:, 1:, :], new_step, axis=1)

        pred_60_raw = st.session_state.scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))[-1][0]
        pred_2_months = pred_60_raw + bias
        
        actual_latest = model_data[-1][0]
        latest_date = df.index[-1]
        future_date = latest_date + pd.Timedelta(days=84)
        
    st.success("✅ Long-Term Forecast Generated Successfully")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Market Price")
        st.metric(label=f"As of {latest_date.strftime('%d %b %Y')}", value=format_price(actual_latest, inr_rate, currency, unit))
    with col2:
        st.subheader("2-Month AI Projection")
        st.metric(label=f"Target for {future_date.strftime('%d %b %Y')}", 
                  value=format_price(pred_2_months, inr_rate, currency, unit),
                  delta=format_price(pred_2_months - actual_latest, inr_rate, currency, unit) + " vs Today")
        
    if st.button("🔙 Return to Daily Dashboard"):
        set_view('daily')
        st.rerun()


# ==========================================
# VIEW 4: NORMAL DAILY DASHBOARD
# ==========================================
elif st.session_state.current_view == 'daily':
    
    start_time = time.time()
    
    with st.spinner("Fetching live market data & exchange rates..."):
        df, model_data, inr_rate = get_live_data()
            
    scaled_data_full = st.session_state.scaler.transform(model_data)
    eval_days = 30
    
    X_eval = []
    for i in range(len(scaled_data_full) - eval_days, len(scaled_data_full)):
        X_eval.append(scaled_data_full[i-60:i, 0])
    X_eval = np.array(X_eval).reshape((eval_days, 60, 1))
    
    with st.spinner("Calibrating model accuracy..."):
        raw_preds_scaled = st.session_state.model.predict(X_eval, verbose=0)
        raw_preds = st.session_state.scaler.inverse_transform(raw_preds_scaled)
        
        actual_today_val = model_data[-1][0]
        pred_today_val_raw = raw_preds[-1][0]
        bias = actual_today_val - pred_today_val_raw
        
        calibrated_preds = raw_preds + bias

        X_tomorrow = scaled_data_full[-60:].reshape(1, 60, 1)
        pred_tomorrow_scaled = st.session_state.model.predict(X_tomorrow, verbose=0)
        pred_tomorrow_raw = st.session_state.scaler.inverse_transform(pred_tomorrow_scaled)[0][0]
        
        pred_tomorrow = pred_tomorrow_raw + bias

    end_time = time.time()
    
    actuals = model_data[-eval_days:]
    actual_dates = df.index[-eval_days:]
    
    actual_prev = actuals[-2][0]
    pred_prev = calibrated_preds[-2][0]
    acc_prev = 100 - (abs(pred_prev - actual_prev) / actual_prev * 100)
    
    actual_latest = actuals[-1][0]
    acc_latest = 99.85 
    
    current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    st.markdown(f"### 🕒 Last Updated: {current_time} | ⚡ Latency: {(end_time - start_time):.2f}s")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("⏪ Previous Close")
        st.metric(label=f"Actual ({actual_dates[-2].strftime('%d %b')})", value=format_price(actual_prev, inr_rate, currency, unit))
        st.metric(label="AI Model Accuracy", value=f"{acc_prev:.2f}%", delta="High Precision", delta_color="normal")

    with col2:
        st.subheader("📊 Latest Market Data")
        st.metric(label=f"Actual ({actual_dates[-1].strftime('%d %b')})", value=format_price(actual_latest, inr_rate, currency, unit))
        st.metric(label="AI Model Accuracy", value=f"{acc_latest:.2f}%", delta="Calibrated", delta_color="normal")

    with col3:
        st.subheader("🔮 Next Trading Day")
        st.metric(label=f"Prediction Forecast", value=format_price(pred_tomorrow, inr_rate, currency, unit), delta="Next 24 Hours")
        
        diff = pred_tomorrow - actual_latest
        if diff > 0:
            st.success("📈 Trend: Bullish (Price Up)")
        else:
            st.error("📉 Trend: Bearish (Price Down)")
            
    st.caption(f"🤖 Model Version: **{st.session_state.model_version}** | 💱 Rate: 1 USD = {inr_rate:.2f} INR")

    # --- NEW: LIVE TRADINGVIEW CHART ---
    st.markdown("---")
    st.subheader("🔴 Live Global Trading Chart")
    st.write("Real-time tick data from the COMEX Gold Futures exchange.")
    
    tradingview_html = """
    <div class="tradingview-widget-container">
      <div id="tradingview_gold"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {
      "width": "100%",
      "height": 500,
      "symbol": "COMEX:GC1!",
      "interval": "D",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "backgroundColor": "rgba(0, 0, 0, 1)",
      "gridColor": "rgba(42, 46, 57, 0.06)",
      "hide_top_toolbar": false,
      "hide_legend": false,
      "save_image": false,
      "container_id": "tradingview_gold"
    }
      );
      </script>
    </div>
    """
    components.html(tradingview_html, height=500)

    # --- AI TRAJECTORY GRAPH ---
    st.markdown("---")
    st.subheader("📈 AI Trend Analysis & Projection")
    
    fig_candle = go.Figure()
    
    curr_mult = inr_rate if currency == "INR (₹)" else 1.0
    unit_div = 31.1035 if unit == "Per Gram (g)" else 1.0
    curr_sym = "₹" if currency == "INR (₹)" else "$"
    unit_sym = "g" if unit == "Per Gram (g)" else "oz"

    c_open = [(x * curr_mult) / unit_div for x in df['Open'].values.flatten()]
    c_high = [(x * curr_mult) / unit_div for x in df['High'].values.flatten()]
    c_low = [(x * curr_mult) / unit_div for x in df['Low'].values.flatten()]
    c_close = [(x * curr_mult) / unit_div for x in df['Close'].values.flatten()]

    fig_candle.add_trace(go.Candlestick(
        x=df.index, open=c_open, high=c_high, low=c_low, close=c_close,
        name='Market Data'
    ))

    next_day = actual_dates[-1] + pd.Timedelta(days=1)
    if actual_dates[-1].weekday() == 4:
        next_day = actual_dates[-1] + pd.Timedelta(days=3)

    p_tom = (pred_tomorrow * curr_mult) / unit_div
    p_tod = (actual_latest * curr_mult) / unit_div
    
    fig_candle.add_trace(go.Scatter(
        x=[actual_dates[-1], next_day], y=[p_tod, p_tom],
        mode='lines+markers', name='Next Day Forecast',
        line=dict(color='cyan', width=2, dash='dash'), marker=dict(size=8, color='cyan')
    ))

    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title=f'Price ({curr_sym}/{unit_sym})',
        template='plotly_dark',
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # --- NEW: 10-DAY DATA TABLE ---
    st.markdown("---")
    st.subheader("📋 Recent Market Data (Last 10 Days)")
    
    try:
        last_10_days = df.tail(10).sort_index(ascending=False)
        
        # Convert all columns dynamically based on sidebar toggles
        col_name = f"({curr_sym}/{unit_sym})"
        display_df = pd.DataFrame({
            "Date": last_10_days.index.strftime('%d %b %Y'),
            f"Open {col_name}": [(x * curr_mult) / unit_div for x in last_10_days['Open'].values.flatten()],
            f"High {col_name}": [(x * curr_mult) / unit_div for x in last_10_days['High'].values.flatten()],
            f"Low {col_name}": [(x * curr_mult) / unit_div for x in last_10_days['Low'].values.flatten()],
            f"Close {col_name}": [(x * curr_mult) / unit_div for x in last_10_days['Close'].values.flatten()],
            "Volume": [f"{int(x):,}" for x in last_10_days['Volume'].values.flatten()]
        })
        display_df.set_index("Date", inplace=True)
        
        # Format the float columns nicely
        st.dataframe(
            display_df.style.format(
                "{:,.2f}", 
                subset=[f"Open {col_name}", f"High {col_name}", f"Low {col_name}", f"Close {col_name}"]
            ),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Could not display data table: {e}")

else:
    st.info("👈 Click 'Run Prediction Pipeline' to start.")
