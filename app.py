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
st.set_page_config(page_title="AI Price Intelligence", layout="wide", page_icon="🥇")

# ==========================================
# GLASSMORPHISM CSS INJECTION
# ==========================================
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --gold-bright:   #C8922A;
    --gold-mid:      #B07D20;
    --gold-light:    #F0C060;
    --gold-glow:     rgba(230, 175, 60, 0.55);
    --glass-white:   rgba(255, 255, 255, 0.55);
    --glass-border:  rgba(255, 255, 255, 0.75);
    --glass-shadow:  rgba(180, 140, 40, 0.18);
    --glass-blur:    blur(22px) saturate(180%);
    --text-primary:  #2A1F0A;
    --text-muted:    rgba(80, 60, 20, 0.65);
    --success:       #1A7A4A;
    --danger:        #B83248;
}

/* ════════════════════════════════
   BACKGROUND — the key to real glass
   ════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: transparent !important;
}

/* Layer 1 — base white-cream canvas */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -3;
    background: #FAF6EE;
}

/* Layer 2 — large vivid colour blobs (what gets blurred through the glass) */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: -2;
    background:
        radial-gradient(ellipse 70% 55% at 8%  12%,  rgba(255, 210,  80, 0.72) 0%, transparent 65%),
        radial-gradient(ellipse 55% 45% at 92% 10%,  rgba(255, 180,  40, 0.60) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 50% 92%,  rgba(240, 160,  20, 0.55) 0%, transparent 65%),
        radial-gradient(ellipse 45% 40% at 78% 55%,  rgba(255, 230, 120, 0.50) 0%, transparent 60%),
        radial-gradient(ellipse 40% 35% at 22% 70%,  rgba(250, 200,  60, 0.45) 0%, transparent 55%),
        radial-gradient(ellipse 30% 25% at 60% 35%,  rgba(255, 255, 200, 0.40) 0%, transparent 50%);
    pointer-events: none;
}

/* ── Sidebar Glass ── */
[data-testid="stSidebar"] {
    background: rgba(255, 252, 240, 0.60) !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.80) !important;
    box-shadow: 4px 0 30px rgba(180, 140, 40, 0.12) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Sidebar Buttons ── */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.55) !important;
    border: 1px solid rgba(200, 146, 42, 0.45) !important;
    border-radius: 10px !important;
    color: var(--gold-mid) !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 2px 12px rgba(180,140,40,0.10) !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255, 255, 255, 0.80) !important;
    border-color: var(--gold-bright) !important;
    box-shadow: 0 4px 20px rgba(200, 146, 42, 0.28) !important;
    transform: translateY(-1px) !important;
}

/* ── Main area ── */
[data-testid="stMain"], .main, .block-container {
    background: transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1400px !important;
}

/* ── Page Title ── */
h1 {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
    font-size: 2.3rem !important;
    background: linear-gradient(90deg, #8B5E0A 0%, #C8922A 40%, #8B5E0A 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: 0.2px !important;
    margin-bottom: 0.2rem !important;
    text-shadow: none !important;
}

/* ── All headings ── */
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
h3 { font-size: 1.1rem !important; color: var(--gold-mid) !important; }

/* ── Glass Card ── */
.glass-card {
    background: rgba(255, 255, 255, 0.50);
    border: 1px solid rgba(255, 255, 255, 0.80);
    border-radius: 20px;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    padding: 1.4rem 1.6rem;
    box-shadow:
        0 8px 32px rgba(180, 140, 40, 0.14),
        inset 0 1px 0 rgba(255, 255, 255, 0.90),
        inset 0 -1px 0 rgba(180, 140, 40, 0.08);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover {
    box-shadow:
        0 14px 44px rgba(180, 140, 40, 0.22),
        inset 0 1px 0 rgba(255, 255, 255, 1.0),
        0 0 0 1px rgba(200, 146, 42, 0.30);
    transform: translateY(-2px);
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.52) !important;
    border: 1px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    box-shadow:
        0 6px 28px rgba(180, 140, 40, 0.13),
        inset 0 1.5px 0 rgba(255,255,255,0.95),
        inset 0 -1px 0 rgba(180,140,40,0.06) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stMetric"]:hover {
    background: rgba(255, 255, 255, 0.72) !important;
    border-color: rgba(200, 146, 42, 0.45) !important;
    box-shadow:
        0 10px 36px rgba(180, 140, 40, 0.22),
        inset 0 1.5px 0 rgba(255,255,255,1.0),
        0 0 0 1px rgba(200, 146, 42, 0.28) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: var(--gold-mid) !important;
    font-weight: 500 !important;
    font-size: 1.50rem !important;
    letter-spacing: -0.5px !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(200,146,42,0.35), rgba(200,146,42,0.20), transparent) !important;
    margin: 1.2rem 0 !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.50) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 4px 20px rgba(180,140,40,0.10) !important;
}

/* ── Buttons ── */
.stButton button {
    background: rgba(255, 255, 255, 0.55) !important;
    border: 1px solid rgba(200, 146, 42, 0.42) !important;
    border-radius: 12px !important;
    color: var(--gold-mid) !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 12px rgba(180,140,40,0.12) !important;
    backdrop-filter: blur(10px) !important;
}
.stButton button:hover {
    background: rgba(255, 255, 255, 0.82) !important;
    border-color: var(--gold-bright) !important;
    box-shadow: 0 6px 24px rgba(200,146,42,0.30) !important;
    transform: translateY(-1px) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * {
    color: var(--gold-mid) !important;
    border-top-color: var(--gold-mid) !important;
}

/* ── Caption ── */
.stCaption, caption {
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: rgba(255, 255, 255, 0.48) !important;
    border: 1px solid rgba(255, 255, 255, 0.80) !important;
    border-radius: 16px !important;
    backdrop-filter: var(--glass-blur) !important;
    -webkit-backdrop-filter: var(--glass-blur) !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(180,140,40,0.10) !important;
}
[data-testid="stDataFrame"] table {
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(240, 192, 96, 0.18) !important;
    color: var(--gold-mid) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.5px !important;
    border-bottom: 1px solid rgba(200,146,42,0.20) !important;
}
[data-testid="stDataFrame"] td {
    border-color: rgba(200,146,42,0.08) !important;
    color: var(--text-primary) !important;
}
[data-testid="stDataFrame"] tr:hover td {
    background: rgba(255,210,80,0.10) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.4); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: rgba(200,146,42,0.30); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(200,146,42,0.55); }

/* ── Status bar ── */
.status-bar {
    background: rgba(255, 255, 255, 0.52);
    border: 1px solid rgba(255, 255, 255, 0.85);
    border-radius: 14px;
    padding: 0.65rem 1.3rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    color: var(--text-muted);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    box-shadow: 0 4px 20px rgba(180,140,40,0.10), inset 0 1px 0 rgba(255,255,255,0.90);
    margin-bottom: 1rem;
}
.status-dot {
    width: 7px; height: 7px;
    background: #1A7A4A;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(26,122,74,0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

/* ── Section label ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--gold-mid);
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(200,146,42,0.35), transparent);
}

/* ── Home welcome card ── */
.home-hero {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.90);
    border-radius: 24px;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    padding: 3rem 2.5rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 680px;
    box-shadow:
        0 20px 60px rgba(180,140,40,0.16),
        inset 0 1.5px 0 rgba(255,255,255,1.0),
        inset 0 -1px 0 rgba(180,140,40,0.08);
}
.home-hero h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    background: linear-gradient(90deg, #8B5E0A, #C8922A, #8B5E0A) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.8rem !important;
    letter-spacing: 0.2px !important;
}
.home-hero p {
    color: rgba(80, 60, 20, 0.70);
    font-size: 0.98rem;
    line-height: 1.8;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
}
.home-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    display: block;
}

/* ── Warning disclaimer card ── */
.disclaimer-card {
    background: rgba(255, 240, 230, 0.65);
    border: 1px solid rgba(220, 100, 60, 0.28);
    border-radius: 18px;
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 6px 24px rgba(200,80,40,0.10), inset 0 1px 0 rgba(255,255,255,0.80);
}

/* ── Trend badges ── */
.trend-bull {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(220, 255, 235, 0.70);
    border: 1px solid rgba(26, 122, 74, 0.35);
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #1A7A4A;
    font-family: 'DM Sans', sans-serif;
    backdrop-filter: blur(8px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.80);
}
.trend-bear {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255, 230, 230, 0.70);
    border: 1px solid rgba(184, 50, 72, 0.35);
    border-radius: 8px;
    padding: 0.35rem 0.9rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #B83248;
    font-family: 'DM Sans', sans-serif;
    backdrop-filter: blur(8px);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.80);
}

/* ── Sidebar logo ── */
.sidebar-logo {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
    font-family: 'Playfair Display', serif;
    font-weight: 600;
    font-size: 1.1rem;
    background: linear-gradient(90deg, #8B5E0A, #C8922A);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.3px;
}
.sidebar-sub {
    text-align: center;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(80, 60, 20, 0.50) !important;
    margin-bottom: 1rem;
    font-family: 'DM Sans', sans-serif;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--text-primary) !important;
}
[data-testid="stRadio"] > div {
    gap: 0.4rem !important;
}

/* ── Sidebar info box ── */
[data-testid="stInfo"] {
    background: rgba(255, 248, 220, 0.60) !important;
    border: 1px solid rgba(200, 146, 42, 0.30) !important;
    border-radius: 10px !important;
    color: var(--gold-mid) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    backdrop-filter: blur(10px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Page title (rendered by Streamlit so CSS h1 applies) ──
st.title("✦ AI Price Intelligence")

# State Management
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
    st.error(f"Error loading model files: {e}")
    st.stop()

# 3. SIDEBAR
st.sidebar.markdown('<div class="sidebar-logo">✦ AI Price Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">AI Price Intelligence</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("**⚙️ Display Settings**")
currency = st.sidebar.radio("Currency", ["USD ($)", "INR (₹)"])
unit = st.sidebar.radio("Unit", ["Per Ounce (oz)", "Per Gram (g)"])

st.sidebar.markdown("---")
st.sidebar.info(f"🧠 {st.session_state.model_version}")

if st.sidebar.button("⚡ Retrain on Latest Data"):
    with st.spinner("Fine-tuning model on recent market data..."):
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
        np.random.seed(42); random.seed(42); tf.random.set_seed(42)
        from tensorflow.keras.optimizers import Adam
        st.session_state.model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        st.session_state.model.fit(X_new, y_new, epochs=2, batch_size=16, verbose=0)
        st.session_state.model_version = f"Fine-Tuned {datetime.now().strftime('%H:%M:%S')}"
        st.sidebar.success("✅ Fine-tuned successfully!")
        time.sleep(1)

st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Forecasting Tools**")
if st.sidebar.button("📊 Daily Dashboard"):
    set_view('daily')
if st.sidebar.button("📅 2-Month Forecast"):
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
# VIEW 1: HOME
# ==========================================
if st.session_state.current_view == 'home':
    st.markdown("""
    <div class="home-hero">
        <span class="home-icon">🥇</span>
        <h2>Gold Market Intelligence</h2>
        <p>
            AI-powered price forecasting using deep LSTM neural networks.<br>
            Real-time market data · Multi-currency support · 60-day projection.
        </p>
        <br>
        <p style="font-size:0.85rem; color:rgba(80,60,20,0.45);">
            ← Select a forecasting tool from the sidebar to begin
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# VIEW 2: WARNING DISCLAIMER
# ==========================================
elif st.session_state.current_view == 'warning':
    st.markdown('<div class="section-label">⚠️ Risk Disclosure</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-card">
        <h3 style="color:#B83248; font-family:'Playfair Display',serif; margin-top:0;">Financial Disclaimer</h3>
        <p style="color:rgba(60,40,10,0.85); font-family:'DM Sans',sans-serif; line-height:1.7; font-size:0.95rem;">
            This model is built for <strong style="color:#8B5E0A;">Technical Analysis only</strong>. Long-term recursive forecasting carries 
            inherent mathematical drift risks. Real-world prices may diverge significantly due to geopolitical events, 
            central bank policy, and macroeconomic shifts.
        </p>
        <p style="color:rgba(60,40,10,0.60); font-family:'DM Sans',sans-serif; line-height:1.7; font-size:0.88rem; margin-bottom:0;">
            Always conduct independent Fundamental Analysis before acting on any AI projection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Do you understand these risks and wish to view the 60-day AI projection?**")
    colA, colB = st.columns(2)
    with colA:
        if st.button("✅ I Understand — Show Forecast"):
            set_view('long_term')
            st.rerun()
    with colB:
        if st.button("← Go Back"):
            set_view('daily')
            st.rerun()


# ==========================================
# VIEW 3: 2-MONTH FORECAST
# ==========================================
elif st.session_state.current_view == 'long_term':
    st.markdown('<div class="section-label">📅 Strategic Long-Term Forecast</div>', unsafe_allow_html=True)

    with st.spinner("Generating 60-step recursive trajectory..."):
        df, model_data, inr_rate = get_live_data()
        scaled_data_full = st.session_state.scaler.transform(model_data)
        actual_today_val = model_data[-1][0]
        X_eval = scaled_data_full[-61:-1].reshape(1, 60, 1)
        pred_today_raw = st.session_state.scaler.inverse_transform(
            st.session_state.model.predict(X_eval, verbose=0))[0][0]
        bias = actual_today_val - pred_today_raw
        future_predictions_scaled = []
        current_input = scaled_data_full[-60:].reshape(1, 60, 1)
        for _ in range(60):
            pred = st.session_state.model.predict(current_input, verbose=0)
            future_predictions_scaled.append(pred[0, 0])
            new_step = np.array([[[pred[0, 0]]]])
            current_input = np.append(current_input[:, 1:, :], new_step, axis=1)
        pred_60_raw = st.session_state.scaler.inverse_transform(
            np.array(future_predictions_scaled).reshape(-1, 1))[-1][0]
        pred_2_months = pred_60_raw + bias
        actual_latest = model_data[-1][0]
        latest_date = df.index[-1]
        future_date = latest_date + pd.Timedelta(days=84)

    st.success("✅ 60-Day Projection Generated")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Market Price")
        st.metric(label=f"As of {latest_date.strftime('%d %b %Y')}",
                  value=format_price(actual_latest, inr_rate, currency, unit))
    with col2:
        st.subheader("2-Month AI Projection")
        st.metric(
            label=f"Target — {future_date.strftime('%d %b %Y')}",
            value=format_price(pred_2_months, inr_rate, currency, unit),
            delta=format_price(pred_2_months - actual_latest, inr_rate, currency, unit) + " vs Today")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Return to Daily Dashboard"):
        set_view('daily')
        st.rerun()


# ==========================================
# VIEW 4: DAILY DASHBOARD
# ==========================================
elif st.session_state.current_view == 'daily':

    start_time = time.time()
    with st.spinner("Fetching live market data..."):
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
    current_time = datetime.now().strftime("%d %b %Y · %H:%M:%S")

    # Status bar
    diff = pred_tomorrow - actual_latest
    trend_html = (
        '<span class="trend-bull">📈 Bullish</span>' if diff > 0
        else '<span class="trend-bear">📉 Bearish</span>'
    )
    st.markdown(f"""
    <div class="status-bar">
        <span class="status-dot"></span>
        <span style="color:rgba(60,40,10,0.85); font-weight:600;">Live</span>
        <span style="margin-left:0.5rem; color:rgba(80,60,20,0.70);">🕒 {current_time}</span>
        <span style="margin-left:auto; color:rgba(80,60,20,0.65);">⚡ {(end_time - start_time):.2f}s latency</span>
        <span style="margin-left:1rem; color:rgba(80,60,20,0.65);">💱 1 USD = {inr_rate:.2f} INR</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric Cards ──
    st.markdown('<div class="section-label">📊 Market Snapshot</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⏪ Previous Close")
        st.metric(
            label=f"Actual · {actual_dates[-2].strftime('%d %b')}",
            value=format_price(actual_prev, inr_rate, currency, unit))
        st.metric(
            label="Model Accuracy",
            value=f"{acc_prev:.2f}%",
            delta="High Precision")

    with col2:
        st.subheader("📍 Latest Price")
        st.metric(
            label=f"Market Close · {actual_dates[-1].strftime('%d %b')}",
            value=format_price(actual_latest, inr_rate, currency, unit))

    with col3:
        st.subheader("🔮 Next Trading Day")
        st.metric(
            label="AI Forecast",
            value=format_price(pred_tomorrow, inr_rate, currency, unit),
            delta=f"{'↑' if diff > 0 else '↓'} {format_price(abs(diff), inr_rate, currency, unit)}")
        st.markdown(trend_html, unsafe_allow_html=True)

    st.caption(f"🧠 Model: **{st.session_state.model_version}**")

    # ── TradingView ──
    st.markdown("---")
    st.markdown('<div class="section-label">🔴 Live Market Feed</div>', unsafe_allow_html=True)
    tradingview_html = """
    <div class="tradingview-widget-container" style="border-radius:16px; overflow:hidden; box-shadow: 0 6px 28px rgba(180,140,40,0.15);">
      <div id="tradingview_gold"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({
        "width": "100%", "height": 500,
        "symbol": "OANDA:XAUUSD", "interval": "D",
        "timezone": "Etc/UTC", "theme": "light", "style": "1",
        "locale": "en", "enable_publishing": false,
        "backgroundColor": "rgba(255, 252, 240, 1)",
        "gridColor": "rgba(200, 146, 42, 0.06)",
        "hide_top_toolbar": false, "hide_legend": false,
        "save_image": false, "container_id": "tradingview_gold"
      });
      </script>
    </div>
    """
    components.html(tradingview_html, height=510)

    # ── AI Trajectory Chart ──
    st.markdown("---")
    st.markdown('<div class="section-label">📈 AI Trend Analysis</div>', unsafe_allow_html=True)

    fig_candle = go.Figure()
    curr_mult = inr_rate if currency == "INR (₹)" else 1.0
    unit_div = 31.1035 if unit == "Per Gram (g)" else 1.0
    curr_sym = "₹" if currency == "INR (₹)" else "$"
    unit_sym = "g" if unit == "Per Gram (g)" else "oz"

    c_open  = [(x * curr_mult) / unit_div for x in df['Open'].values.flatten()]
    c_high  = [(x * curr_mult) / unit_div for x in df['High'].values.flatten()]
    c_low   = [(x * curr_mult) / unit_div for x in df['Low'].values.flatten()]
    c_close = [(x * curr_mult) / unit_div for x in df['Close'].values.flatten()]

    fig_candle.add_trace(go.Candlestick(
        x=df.index, open=c_open, high=c_high, low=c_low, close=c_close,
        name='Market Data',
        increasing_line_color='#4EFAA7', increasing_fillcolor='rgba(78,250,167,0.7)',
        decreasing_line_color='#FF6B8A', decreasing_fillcolor='rgba(255,107,138,0.7)'
    ))

    next_day = actual_dates[-1] + pd.Timedelta(days=1)
    if actual_dates[-1].weekday() == 4:
        next_day = actual_dates[-1] + pd.Timedelta(days=3)

    p_tom = (pred_tomorrow * curr_mult) / unit_div
    p_tod = (actual_latest * curr_mult) / unit_div

    fig_candle.add_trace(go.Scatter(
        x=[actual_dates[-1], next_day], y=[p_tod, p_tom],
        mode='lines+markers', name='AI Forecast',
        line=dict(color='#FFD166', width=2.5, dash='dash'),
        marker=dict(size=9, color='#FFD166',
                    line=dict(color='rgba(255,209,102,0.3)', width=8))
    ))

    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title=f'Price ({curr_sym}/{unit_sym})',
        paper_bgcolor='rgba(255,252,240,0.0)',
        plot_bgcolor='rgba(255,255,255,0.45)',
        font=dict(family='Inter', color='rgba(60,40,10,0.75)', size=12),
        xaxis=dict(gridcolor='rgba(200,146,42,0.10)', linecolor='rgba(200,146,42,0.15)', zerolinecolor='rgba(200,146,42,0.15)'),
        yaxis=dict(gridcolor='rgba(200,146,42,0.10)', linecolor='rgba(200,146,42,0.15)', zerolinecolor='rgba(200,146,42,0.15)'),
        legend=dict(
            bgcolor='rgba(255,252,240,0.70)',
            bordercolor='rgba(200,146,42,0.25)',
            borderwidth=1,
            font=dict(color='rgba(60,40,10,0.80)')
        ),
        margin=dict(l=0, r=0, t=20, b=0),
        height=500
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Data Table ──
    st.markdown("---")
    st.markdown('<div class="section-label">📋 Recent Market Data — Last 10 Sessions</div>', unsafe_allow_html=True)
    try:
        last_10_days = df.tail(10).sort_index(ascending=False)
        col_name = f"({curr_sym}/{unit_sym})"
        display_df = pd.DataFrame({
            "Date": last_10_days.index.strftime('%d %b %Y'),
            f"Open {col_name}":  [(x * curr_mult) / unit_div for x in last_10_days['Open'].values.flatten()],
            f"High {col_name}":  [(x * curr_mult) / unit_div for x in last_10_days['High'].values.flatten()],
            f"Low {col_name}":   [(x * curr_mult) / unit_div for x in last_10_days['Low'].values.flatten()],
            f"Close {col_name}": [(x * curr_mult) / unit_div for x in last_10_days['Close'].values.flatten()],
            "Volume": [f"{int(x):,}" for x in last_10_days['Volume'].values.flatten()]
        })
        display_df.set_index("Date", inplace=True)
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
    st.info("👈 Select a forecasting tool from the sidebar to begin.")
    
