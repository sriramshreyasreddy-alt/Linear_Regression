# =============================================================================
# app.py
# Streamlit Dashboard — Apple (AAPL) Next-Day Stock Price Predictor
# =============================================================================
#
# Dashboard Sections:
#   1. 🏠 Overview          — Project intro, dataset info, key stats
#   2. 📈 Historical Data   — Candlestick price chart, volume chart
#   3. 🔬 Feature Analysis  — Engineered features table + moving average chart
#   4. 🤖 Model & Metrics  — Train model button, MAE/RMSE/R² per split
#   5. 🎯 Prediction        — Next-day closing price prediction card
#   6. 📊 Evaluation Plots  — Actual vs Predicted, Residual plot
# =============================================================================

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Add src/ to path so we can import our modules ─────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader          import load_data
from preprocessing        import preprocess
from feature_engineering  import engineer_features, get_feature_list
from train_model          import train, split_data, FEATURES, TARGET
from evaluation           import compute_metrics
from predict              import load_model, predict_next_close, predict_batch

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/NASDAQ100_Historical_Data.csv"
MODEL_PATH  = "models/linear_regression.pkl"
SCALER_PATH = "models/scaler.pkl"

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title  = "AAPL Stock Predictor",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1c1f2e;
        border: 1px solid #2d3147;
        border-radius: 10px;
        padding: 16px 20px;
    }

    /* Section headers */
    .section-header {
        font-size: 22px;
        font-weight: 700;
        color: #e0e0e0;
        border-left: 4px solid #00d4aa;
        padding-left: 12px;
        margin: 24px 0 16px 0;
    }

    /* Prediction card */
    .pred-card {
        background: linear-gradient(135deg, #1a2a3a, #0d1b2a);
        border: 1px solid #00d4aa;
        border-radius: 14px;
        padding: 28px 32px;
        text-align: center;
    }
    .pred-price {
        font-size: 52px;
        font-weight: 800;
        color: #00d4aa;
        margin: 8px 0;
    }
    .pred-label {
        font-size: 14px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .pred-change-pos { color: #26de81; font-size: 22px; font-weight: 700; }
    .pred-change-neg { color: #ff4757; font-size: 22px; font-weight: 700; }

    /* Info box */
    .info-box {
        background-color: #1c1f2e;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #2d3147;
        margin-bottom: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Remove default padding */
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg",
             width=50)
    st.title("AAPL Predictor")
    st.caption("Next-Day Closing Price Forecast")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Overview",
            "📈 Historical Data",
            "🔬 Feature Analysis",
            "🤖 Model & Metrics",
            "🎯 Prediction",
            "📊 Evaluation Plots",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Date range filter (used in Historical Data page)
    st.subheader("📅 Date Filter")
    year_start = st.slider("Start Year", min_value=2000, max_value=2025, value=2015)
    year_end   = st.slider("End Year",   min_value=2000, max_value=2026, value=2026)

    st.divider()
    st.caption("📌 Dataset: NASDAQ-100 Historical")
    st.caption("🤖 Model: Linear Regression")
    st.caption("📦 Built with Streamlit + Scikit-learn")


# =============================================================================
# Data Loading (cached)
# =============================================================================

@st.cache_data(show_spinner=False)
def get_full_pipeline():
    """Run the full data pipeline and return all artifacts."""
    raw   = load_data(DATA_PATH)
    clean = preprocess(raw)
    feat  = engineer_features(clean)
    return raw, clean, feat


@st.cache_resource(show_spinner=False)
def get_model():
    """Load trained model and scaler from disk."""
    return load_model()


def model_is_trained():
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading AAPL data..."):
    try:
        raw_df, clean_df, feat_df = get_full_pipeline()
        data_loaded = True
    except FileNotFoundError as e:
        st.error(f"❌ {e}")
        st.stop()


# =============================================================================
# Helper: Apply date filter
# =============================================================================

def filter_by_year(df, start, end):
    return df[(df["Date"].dt.year >= start) & (df["Date"].dt.year <= end)]


# =============================================================================
# PAGE 1 — Overview
# =============================================================================

if page == "🏠 Overview":

    st.title("📈 Apple Stock Price Predictor")
    st.markdown("**Next-Day Closing Price Prediction using Linear Regression**")
    st.divider()

    # ── Key stats row ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    latest  = clean_df.iloc[-1]
    oldest  = clean_df.iloc[0]
    all_time_high  = clean_df["High"].max()
    all_time_low   = clean_df["Low"].min()

    col1.metric("Total Trading Days",  f"{len(clean_df):,}")
    col2.metric("Latest Close",        f"${latest['Close']:.2f}")
    col3.metric("All-Time High",       f"${all_time_high:.2f}")
    col4.metric("All-Time Low",        f"${all_time_low:.2f}")

    st.divider()

    # ── Project description ────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)
        st.markdown("""
        This application demonstrates a complete **end-to-end Machine Learning pipeline**
        for predicting Apple's (AAPL) next trading day's closing stock price.

        **Pipeline steps:**
        1. 📥 Load & filter AAPL data from NASDAQ-100 dataset
        2. 🧹 Clean and preprocess raw data
        3. ⚙️ Engineer 15 predictive features
        4. 📊 Chronological train/validation/test split
        5. 🤖 Train a Linear Regression model
        6. 📉 Evaluate with MAE, RMSE, R²
        7. 🎯 Predict next-day closing price
        """)

    with col_b:
        st.markdown('<div class="section-header">Dataset Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-box">
            <b>Source:</b> Kaggle — NASDAQ-100 Historical<br><br>
            <b>Ticker:</b> AAPL (Apple Inc.)<br><br>
            <b>Date Range:</b> {clean_df['Date'].min().date()} → {clean_df['Date'].max().date()}<br><br>
            <b>Total Rows:</b> {len(clean_df):,}<br><br>
            <b>Features:</b> {len(get_feature_list())} engineered features<br><br>
            <b>Target:</b> Next Day Closing Price
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Feature list ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Engineered Features</div>', unsafe_allow_html=True)

    feat_info = {
        "Daily_Return"      : "( Close − Open ) / Open",
        "Daily_Range"       : "High − Low",
        "MA5"               : "5-day Moving Average of Close",
        "MA10"              : "10-day Moving Average of Close",
        "MA20"              : "20-day Moving Average of Close",
        "Rolling_Volatility": "20-day Rolling Std Dev of Close",
        "Momentum"          : "Close − Close 5 days ago",
        "Prev_Close"        : "Previous day's closing price",
        "Prev_Volume"       : "Previous day's volume",
        "Prev_Return"       : "Previous day's daily return",
    }

    feat_df_display = pd.DataFrame(
        [(k, v) for k, v in feat_info.items()],
        columns=["Feature", "Formula / Description"]
    )
    st.dataframe(feat_df_display, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 2 — Historical Data
# =============================================================================

elif page == "📈 Historical Data":

    st.title("📈 AAPL Historical Price Data")
    st.divider()

    filtered = filter_by_year(clean_df, year_start, year_end)

    if filtered.empty:
        st.warning("No data available for the selected date range.")
        st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period Open",   f"${filtered.iloc[0]['Open']:.2f}")
    c2.metric("Period Close",  f"${filtered.iloc[-1]['Close']:.2f}")
    c3.metric("Period High",   f"${filtered['High'].max():.2f}")
    c4.metric("Period Low",    f"${filtered['Low'].min():.2f}")

    st.divider()

    # ── Candlestick Chart ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Candlestick Price Chart</div>', unsafe_allow_html=True)

    fig_candle = go.Figure(data=[go.Candlestick(
        x     = filtered["Date"],
        open  = filtered["Open"],
        high  = filtered["High"],
        low   = filtered["Low"],
        close = filtered["Close"],
        increasing_line_color = "#26de81",
        decreasing_line_color = "#ff4757",
        name  = "AAPL",
    )])

    fig_candle.update_layout(
        template       = "plotly_dark",
        xaxis_title    = "Date",
        yaxis_title    = "Price (USD)",
        height         = 450,
        xaxis_rangeslider_visible = False,
        margin         = dict(l=0, r=0, t=10, b=0),
        legend         = dict(orientation="h"),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ── Volume Chart ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Trading Volume</div>', unsafe_allow_html=True)

    fig_vol = go.Figure(go.Bar(
        x     = filtered["Date"],
        y     = filtered["Volume"],
        marker_color = "#5271ff",
        opacity      = 0.7,
        name         = "Volume",
    ))
    fig_vol.update_layout(
        template    = "plotly_dark",
        xaxis_title = "Date",
        yaxis_title = "Volume",
        height      = 280,
        margin      = dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Raw Data Table ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Raw Data Table</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        .sort_values("Date", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=350,
    )


# =============================================================================
# PAGE 3 — Feature Analysis
# =============================================================================

elif page == "🔬 Feature Analysis":

    st.title("🔬 Feature Analysis")
    st.divider()

    filtered_feat = filter_by_year(feat_df, year_start, year_end)

    # ── Moving Averages Chart ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Moving Averages (MA5 / MA10 / MA20)</div>', unsafe_allow_html=True)

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=filtered_feat["Date"], y=filtered_feat["Close"],
                                name="Close",  line=dict(color="#aaaaaa", width=1), opacity=0.6))
    fig_ma.add_trace(go.Scatter(x=filtered_feat["Date"], y=filtered_feat["MA5"],
                                name="MA5",    line=dict(color="#ffd32a", width=1.5)))
    fig_ma.add_trace(go.Scatter(x=filtered_feat["Date"], y=filtered_feat["MA10"],
                                name="MA10",   line=dict(color="#ff9f43", width=1.5)))
    fig_ma.add_trace(go.Scatter(x=filtered_feat["Date"], y=filtered_feat["MA20"],
                                name="MA20",   line=dict(color="#ee5a24", width=1.5)))

    fig_ma.update_layout(
        template    = "plotly_dark",
        xaxis_title = "Date",
        yaxis_title = "Price (USD)",
        height      = 380,
        margin      = dict(l=0, r=0, t=10, b=0),
        legend      = dict(orientation="h"),
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # ── Daily Return & Volatility ──────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Daily Return</div>', unsafe_allow_html=True)
        fig_ret = px.histogram(
            filtered_feat, x="Daily_Return", nbins=80,
            color_discrete_sequence=["#00d4aa"],
            template="plotly_dark",
        )
        fig_ret.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Daily Return", yaxis_title="Frequency",
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Rolling Volatility (20-day)</div>', unsafe_allow_html=True)
        fig_v = go.Figure(go.Scatter(
            x=filtered_feat["Date"], y=filtered_feat["Rolling_Volatility"],
            fill="tozeroy", line=dict(color="#ff6b81"), name="Volatility",
        ))
        fig_v.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date", yaxis_title="Std Dev (USD)",
        )
        st.plotly_chart(fig_v, use_container_width=True)

    # ── Momentum ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Momentum (Close vs 5 days ago)</div>', unsafe_allow_html=True)
    colors = ["#26de81" if v >= 0 else "#ff4757" for v in filtered_feat["Momentum"]]
    fig_mom = go.Figure(go.Bar(
        x=filtered_feat["Date"], y=filtered_feat["Momentum"],
        marker_color=colors, name="Momentum",
    ))
    fig_mom.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="Momentum (USD)",
    )
    st.plotly_chart(fig_mom, use_container_width=True)

    # ── Correlation Heatmap ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = get_feature_list() + ["Next_Close"]
    corr      = filtered_feat[corr_cols].corr().round(2)

    fig_heat = px.imshow(
        corr,
        color_continuous_scale = "RdBu_r",
        zmin=-1, zmax=1,
        template="plotly_dark",
        text_auto=True,
    )
    fig_heat.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Feature Table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Engineered Feature Table (Latest 100 rows)</div>', unsafe_allow_html=True)
    display_cols = ["Date", "Close", "Daily_Return", "Daily_Range",
                    "MA5", "MA10", "MA20", "Rolling_Volatility",
                    "Momentum", "Prev_Close", "Next_Close"]
    st.dataframe(
        filtered_feat[display_cols].tail(100).sort_values("Date", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=350,
    )


# =============================================================================
# PAGE 4 — Model & Metrics
# =============================================================================

elif page == "🤖 Model & Metrics":

    st.title("🤖 Model Training & Metrics")
    st.divider()

    # ── Model info ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
        st.markdown("""
        | Parameter        | Value                          |
        |------------------|-------------------------------|
        | Algorithm        | Linear Regression              |
        | Scaler           | StandardScaler                 |
        | Features         | 15 engineered features         |
        | Target           | Next Day Closing Price         |
        | Train Period     | 2010-01-01 → 2022-12-31        |
        | Validation Period| 2023-01-01 → 2023-12-31        |
        | Test Period      | 2024-01-01 → present           |
        | Split Strategy   | Chronological (no shuffling)   |
        | Serialization    | Joblib (.pkl)                  |
        """)

    with col2:
        st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)
        if model_is_trained():
            st.success("✅ Model is trained and ready")
            model_size = os.path.getsize(MODEL_PATH)
            st.caption(f"Model size: {model_size / 1024:.1f} KB")
        else:
            st.warning("⚠️ Model not trained yet")
            st.caption("Click the button below to train.")

    st.divider()

    # ── Train button ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Train Model</div>', unsafe_allow_html=True)

    if st.button("🚀 Train Linear Regression Model", type="primary", use_container_width=True):
        with st.spinner("Training in progress... please wait"):
            try:
                model, scaler, splits = train(feat_df)
                X_train_s, X_val_s, X_test_s, y_train, y_val, y_test = splits
                st.success("✅ Model trained and saved successfully!")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    st.divider()

    # ── Metrics ────────────────────────────────────────────────────────────────
    if model_is_trained():
        st.markdown('<div class="section-header">Evaluation Metrics</div>', unsafe_allow_html=True)

        with st.spinner("Computing metrics..."):
            try:
                model, scaler = get_model()

                # Rebuild splits
                X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(feat_df)

                X_train_s = scaler.transform(X_train_df)
                X_val_s   = scaler.transform(X_val_df)
                X_test_s  = scaler.transform(X_test_df)

                splits_data = [
                    ("Train (2010–2022)",      X_train_s, y_train),
                    ("Validation (2023)",       X_val_s,   y_val),
                    ("Test (2024–present)",     X_test_s,  y_test),
                ]

                for label, X, y in splits_data:
                    y_pred  = model.predict(X)
                    metrics = compute_metrics(y, y_pred, label=label)

                    st.markdown(f"**{label}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE  (USD)",  f"${metrics['mae']:.4f}")
                    m2.metric("RMSE (USD)",  f"${metrics['rmse']:.4f}")
                    m3.metric("R² Score",    f"{metrics['r2']:.4f}")
                    st.divider()

                # ── Feature importance (coefficients) ──────────────────────
                st.markdown('<div class="section-header">Feature Coefficients</div>', unsafe_allow_html=True)

                coeff_df = pd.DataFrame({
                    "Feature"     : FEATURES,
                    "Coefficient" : model.coef_,
                }).sort_values("Coefficient", ascending=True)

                fig_coeff = px.bar(
                    coeff_df,
                    x            = "Coefficient",
                    y            = "Feature",
                    orientation  = "h",
                    color        = "Coefficient",
                    color_continuous_scale = "RdBu",
                    template     = "plotly_dark",
                )
                fig_coeff.update_layout(
                    height = 450,
                    margin = dict(l=0, r=0, t=10, b=0),
                    coloraxis_showscale = False,
                )
                st.plotly_chart(fig_coeff, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing metrics: {e}")
    else:
        st.info("Train the model first to see metrics.")


# =============================================================================
# PAGE 5 — Prediction
# =============================================================================

elif page == "🎯 Prediction":

    st.title("🎯 Next-Day Price Prediction")
    st.divider()

    if not model_is_trained():
        st.warning("⚠️ Model not trained yet. Go to **🤖 Model & Metrics** and train the model first.")
        st.stop()

    try:
        model, scaler = get_model()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    # ── Latest row ─────────────────────────────────────────────────────────────
    latest_row  = feat_df.iloc[-1]
    latest_date = latest_row["Date"]
    last_close  = float(latest_row["Close"])

    prediction  = predict_next_close(latest_row, model, scaler)
    change      = round(prediction - last_close, 4)
    change_pct  = round((change / last_close) * 100, 4)
    direction   = "▲" if change >= 0 else "▼"
    clr_class   = "pred-change-pos" if change >= 0 else "pred-change-neg"

    # ── Prediction card ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Last Known Trading Day</div>
            <div style="font-size:18px; color:#ccc; margin:6px 0;">{latest_date.date()}</div>
            <div class="pred-label" style="margin-top:20px;">Last Closing Price</div>
            <div style="font-size:36px; font-weight:700; color:#e0e0e0; margin:6px 0;">${last_close:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Predicted Next-Day Close</div>
            <div class="pred-price">${prediction:.2f}</div>
            <div class="{clr_class}">{direction} ${abs(change):.2f} &nbsp; ({abs(change_pct):.2f}%)</div>
            <div style="margin-top:16px; color:#666; font-size:12px;">
                Based on Linear Regression model trained on 2010–2022 data
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Input features used for this prediction ────────────────────────────────
    st.markdown('<div class="section-header">Input Features Used</div>', unsafe_allow_html=True)

    input_data = {feat: round(float(latest_row[feat]), 4) for feat in FEATURES}
    input_df   = pd.DataFrame(input_data.items(), columns=["Feature", "Value"])
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Recent predictions vs actuals ──────────────────────────────────────────
    st.markdown('<div class="section-header">Recent Predictions vs Actual (Last 60 Days)</div>', unsafe_allow_html=True)

    recent     = feat_df.tail(60).copy()
    X_recent   = scaler.transform(recent[FEATURES])
    preds_rec  = model.predict(X_recent)

    fig_rec = go.Figure()
    fig_rec.add_trace(go.Scatter(
        x=recent["Date"], y=recent["Next_Close"],
        name="Actual Next Close", line=dict(color="#aaaaaa", width=1.5),
    ))
    fig_rec.add_trace(go.Scatter(
        x=recent["Date"], y=preds_rec,
        name="Predicted", line=dict(color="#00d4aa", width=2, dash="dot"),
    ))
    fig_rec.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="Price (USD)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_rec, use_container_width=True)


# =============================================================================
# PAGE 6 — Evaluation Plots
# =============================================================================

elif page == "📊 Evaluation Plots":

    st.title("📊 Model Evaluation Plots")
    st.divider()

    if not model_is_trained():
        st.warning("⚠️ Train the model first from the **🤖 Model & Metrics** page.")
        st.stop()

    try:
        model, scaler = get_model()
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    # ── Rebuild test split ─────────────────────────────────────────────────────
    _, _, X_test_df, _, _, y_test = split_data(feat_df)
    X_test_s = scaler.transform(X_test_df)
    y_pred   = model.predict(X_test_s)
    residuals= np.array(y_test) - y_pred

    # ── Split selector ─────────────────────────────────────────────────────────
    split_choice = st.selectbox(
        "Select split to visualize",
        ["Test (2024–present)", "Validation (2023)", "Train (2010–2022)"]
    )

    if split_choice == "Validation (2023)":
        _, X_sel_df, _, _, y_sel, _ = split_data(feat_df)
        X_sel_s = scaler.transform(X_sel_df)
    elif split_choice == "Train (2010–2022)":
        X_sel_df, _, _, y_sel, _, _ = split_data(feat_df)
        X_sel_s = scaler.transform(X_sel_df)
    else:
        X_sel_df = X_test_df
        y_sel    = y_test
        X_sel_s  = X_test_s

    y_sel_pred = model.predict(X_sel_s)
    residuals  = np.array(y_sel) - y_sel_pred
    metrics    = compute_metrics(y_sel, y_sel_pred, label=split_choice)

    # ── Metrics row ────────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"${metrics['mae']:.4f}")
    m2.metric("RMSE", f"${metrics['rmse']:.4f}")
    m3.metric("R²",   f"{metrics['r2']:.4f}")

    st.divider()

    # ── Actual vs Predicted ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Actual vs Predicted Closing Price</div>', unsafe_allow_html=True)

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        y=list(y_sel), name="Actual",
        line=dict(color="#5271ff", width=1.5),
    ))
    fig_avp.add_trace(go.Scatter(
        y=y_sel_pred, name="Predicted",
        line=dict(color="#ff9f43", width=1.5, dash="dash"),
    ))
    fig_avp.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Trading Day Index",
        yaxis_title="Closing Price (USD)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_avp, use_container_width=True)

    # ── Scatter: Predicted vs Actual ───────────────────────────────────────────
    st.markdown('<div class="section-header">Predicted vs Actual Scatter</div>', unsafe_allow_html=True)

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=list(y_sel), y=y_sel_pred,
        mode="markers",
        marker=dict(color="#00d4aa", opacity=0.5, size=5),
        name="Predictions",
    ))
    # Perfect prediction line
    mn = min(list(y_sel)); mx = max(list(y_sel))
    fig_scatter.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#ff4757", dash="dash", width=1.5),
        name="Perfect Prediction",
    ))
    fig_scatter.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Actual Price (USD)",
        yaxis_title="Predicted Price (USD)",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Residual Plots ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Residuals vs Predicted</div>', unsafe_allow_html=True)
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_sel_pred, y=residuals,
            mode="markers",
            marker=dict(color="#5271ff", opacity=0.5, size=4),
            name="Residuals",
        ))
        fig_res.add_hline(y=0, line_color="#ff4757", line_dash="dash", line_width=1.5)
        fig_res.update_layout(
            template="plotly_dark", height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Predicted (USD)",
            yaxis_title="Residual (USD)",
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Residual Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            x=residuals, nbins=60,
            color_discrete_sequence=["#9b59b6"],
            template="plotly_dark",
        )
        fig_hist.add_vline(x=0, line_color="#ff4757", line_dash="dash", line_width=1.5)
        fig_hist.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Residual (USD)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True)