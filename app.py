import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

# 1. SETUP PAGE
st.set_page_config(page_title="Sunil raj's Gold Price MLOps", layout="wide")
st.title("🏆 Gold Price technical analysis for prediction")

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
st.sidebar.header("MLOps Configuration")
days_lookback = st.sidebar.slider("Lookback Period (Days)", min_value=30, max_value=90, value=60)

# --- CONTINUOUS LEARNING ---
st.sidebar.markdown("---")
st.sidebar.header("Continuous Learning")
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
        
        from tensorflow.keras.optimizers import Adam
        st.session_state.model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        st.session_state.model.fit(X_new, y_new, epochs=1, batch_size=16, verbose=0)
        
        st.session_state.model_version = f"Fine-Tuned at {datetime.now().strftime('%H:%M:%S')}"
        st.sidebar.success("✅ Model Fine-Tuned and Stabilized!")
        time.sleep(1)

# ----------------------------------------

if st.sidebar.button("Run Prediction Pipeline"):
    
    start_time = time.time()
    
    # 4. DATA INGESTION
    with st.spinner("Fetching live market data..."):
        df = yf.download('GC=F', period='6mo', interval='1d')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        model_data = df['Close'].values.reshape(-1, 1)
        
    # 5. PREPROCESSING
    last_60_days = model_data[-days_lookback:]
    last_60_days_scaled = st.session_state.scaler.transform(last_60_days)
    X_input = last_60_days_scaled.reshape(1, days_lookback, 1)
    
    # 6. RECURSIVE INFERENCE (30 DAYS)
    future_predictions_scaled = []
    current_input = X_input.copy()

    with st.spinner("Calculating 30-day future trajectory..."):
        for _ in range(30):
            pred = st.session_state.model.predict(current_input, verbose=0)
            future_predictions_scaled.append(pred[0, 0])
            new_step = np.array([[[pred[0, 0]]]])
            current_input = np.append(current_input[:, 1:, :], new_step, axis=1)

    future_predictions = st.session_state.scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    pred_tomorrow = future_predictions[0][0]
    pred_week = future_predictions[6][0]
    pred_month = future_predictions[29][0]
    
    end_time = time.time()
    latency = end_time - start_time
    
    last_date = df.index[-1]
    last_actual_price = model_data[-1][0]
    
    diff_tomorrow = pred_tomorrow - last_actual_price
    error_percentage = (abs(diff_tomorrow) / last_actual_price) * 100
    accuracy_percentage = 100 - error_percentage
    
    current_time = datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # 7. DISPLAY METRICS
    st.markdown(f"### 🕒 Last Updated: {current_time} | ⚡ Latency: {latency:.2f}s")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Last Market Close")
        st.metric(label=f"Price ({last_date.strftime('%Y-%m-%d')})", value=f"${last_actual_price:.2f}/oz")
    with col2:
        st.subheader("Model Accuracy")
        st.metric(label="Expected Reliability", value=f"{accuracy_percentage:.2f}%", delta=f"-{error_percentage:.2f}% Error", delta_color="inverse")
    with col3:
        st.subheader("Current Trend")
        if diff_tomorrow > 0:
            st.success("📈 Market is Bullish")
        else:
            st.error("📉 Market is Bearish")

    st.markdown("### 🔮 Future Trajectory Forecast")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        st.info("Tomorrow (Day 1)")
        st.metric(label="T+1 Forecast", value=f"${pred_tomorrow:.2f}/oz", delta=f"${diff_tomorrow:.2f} vs Today")
    with f_col2:
        st.warning("Next Week (Day 7)")
        st.metric(label="T+7 Forecast", value=f"${pred_week:.2f}/oz", delta=f"${(pred_week - last_actual_price):.2f} vs Today")
    with f_col3:
        st.error("Next Month (Day 30)")
        st.metric(label="T+30 Forecast", value=f"${pred_month:.2f}/oz", delta=f"${(pred_month - last_actual_price):.2f} vs Today")
        
    st.caption(f"🤖 Prediction generated using: **{st.session_state.model_version}**")

    # 8. INTERACTIVE CANDLESTICK GRAPH (PLOTLY)
    st.markdown("---")
    st.subheader("Market Trend & 30-Day Projection (Candlestick)")
    
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
    
    # Create the Plotly figure
    fig = go.Figure()

    # Add Historical Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].values.flatten(),
        high=df['High'].values.flatten(),
        low=df['Low'].values.flatten(),
        close=df['Close'].values.flatten(),
        name='Historical Data'
    ))

    # Add Future 30-Day Prediction Line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions.flatten(),
        mode='lines+markers',
        name='30-Day AI Forecast',
        line=dict(color='cyan', width=2, dash='dash'),
        marker=dict(size=4, color='cyan')
    ))

    # Format the layout
    fig.update_layout(
        xaxis_rangeslider_visible=False, # Hides the messy slider at the bottom
        yaxis_title='Gold Price ($/oz)',
        xaxis_title='Date',
        template='plotly_dark', # Looks amazing in Streamlit's dark mode
        margin=dict(l=0, r=0, t=30, b=0),
        height=500
    )
    
    # Display interactive chart
    st.plotly_chart(fig, use_container_width=True)

    # 9. DATA TABLE
    st.subheader("Recent Market Data (Last 10 Days)")
    try:
        last_10_days = df.tail(10).sort_index(ascending=False)
        display_df = pd.DataFrame({
            "Date": last_10_days.index.strftime('%Y-%m-%d'),
            "Open ($)": [f"${x:.2f}" for x in last_10_days['Open'].values.flatten()],
            "Close ($)": [f"${x:.2f}" for x in last_10_days['Close'].values.flatten()],
            "Volume": [f"{int(x):,}" for x in last_10_days['Volume'].values.flatten()]
        })
        display_df.set_index("Date", inplace=True)
        st.table(display_df)
    except Exception as e:
        st.error(f"Could not display data table: {e}")

else:
    st.info("👈 Click 'Run Prediction Pipeline' to start.")
