"""Local Streamlit dashboard for Gujarat TOP price forecasts and shock alerts."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    COMMODITIES,
    FEATURE_REGIMES,
    MASTER_CSV,
    MODELS_DIR,
    PREDICTIONS_CSV,
)

st.set_page_config(
    page_title="Gujarat Agri Market Intelligence",
    page_icon="🌾",
    layout="wide",
)

VADER_ANALYZER = SentimentIntensityAnalyzer()


@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame | None:
    if not MASTER_CSV.exists():
        return None
    df = pd.read_csv(MASTER_CSV, parse_dates=["Date"])
    return df.sort_values(["Commodity", "Date"])


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_CSV.exists():
        return None
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["Date"])
    return df.sort_values(["Commodity", "Date"])


@st.cache_resource
def load_models_for_commodity(crop: str) -> dict:
    """Load all trained models for a commodity."""
    models = {}
    crop_lower = crop.lower()
    
    # Regression models
    reg_models = {
        'xgboost': f'{crop_lower}_reg_xgboost_multimodal.joblib',
        'ridge': f'{crop_lower}_reg_ridge_multimodal.joblib',
    }
    
    for name, filename in reg_models.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[f'reg_{name}'] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name} regressor: {e}")
    
    # Classification models
    clf_models = {
        'random_forest': f'{crop_lower}_clf_random_forest_multimodal.joblib',
        'logreg': f'{crop_lower}_clf_logreg_multimodal.joblib',
    }
    
    for name, filename in clf_models.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[f'clf_{name}'] = joblib.load(path)
            except Exception as e:
                st.warning(f"Could not load {name} classifier: {e}")
    
    # BiLSTM models (PyTorch)
    bilstm_reg_path = MODELS_DIR / f'{crop_lower}_reg_bilstm_multimodal.pt'
    bilstm_scaler_path = MODELS_DIR / f'{crop_lower}_reg_bilstm_multimodal_scaler.joblib'
    
    if bilstm_reg_path.exists() and bilstm_scaler_path.exists():
        try:
            # Load the model state dict
            models['bilstm_state'] = torch.load(bilstm_reg_path, map_location='cpu')
            models['bilstm_scaler'] = joblib.load(bilstm_scaler_path)
        except Exception as e:
            st.warning(f"Could not load BiLSTM model: {e}")
    
    return models


def _price_figure(history: pd.DataFrame, preds: pd.DataFrame | None, crop: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Modal_Price"],
            name="Actual modal price",
            mode="lines",
            line=dict(color="#1b5e20", width=2),
        )
    )
    if preds is not None:
        overlay = preds.loc[preds["Commodity"] == crop].copy()
        series = [
            ("pred_price_ridge_multimodal", "Ridge (multimodal)", "#1565c0", "dash"),
            ("pred_price_xgboost_multimodal", "XGBoost (multimodal)", "#ef6c00", "dash"),
            ("pred_price_bilstm_multimodal", "Bi-LSTM (multimodal)", "#6a1b9a", "dot"),
        ]
        for col, label, color, dash in series:
            if col not in overlay.columns:
                continue
            subset = overlay.dropna(subset=[col])
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"],
                    y=subset[col],
                    name=label,
                    mode="lines",
                    line=dict(color=color, width=2, dash=dash),
                )
            )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="INR / quintal",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def _shock_figure(history: pd.DataFrame, preds: pd.DataFrame | None, crop: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["Date"],
            y=history["Modal_Price"],
            name="Modal price",
            mode="lines",
            line=dict(color="#37474f", width=1.8),
        )
    )
    actual_shocks = history.loc[history.get("Target_Shock", 0).fillna(0) == 1]
    if not actual_shocks.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_shocks["Date"],
                y=actual_shocks["Modal_Price"],
                name="Realized shock (z > 1.75)",
                mode="markers",
                marker=dict(color="#c62828", size=10, symbol="x"),
            )
        )
    if preds is not None:
        overlay = preds.loc[preds["Commodity"] == crop].copy()
        prob_col = None
        for candidate in ("pred_shock_rf_multimodal", "pred_shock_bilstm_multimodal", "pred_shock_logreg_multimodal"):
            if candidate in overlay.columns:
                prob_col = candidate
                break
        if prob_col is not None:
            flagged = overlay.loc[overlay[prob_col].fillna(0) >= 0.5].dropna(subset=["Modal_Price"], how="all")
            if "Modal_Price" not in flagged.columns:
                flagged = flagged.merge(history[["Date", "Modal_Price"]], on="Date", how="left")
            flagged = flagged.dropna(subset=["Modal_Price"])
            if not flagged.empty:
                fig.add_trace(
                    go.Scatter(
                        x=flagged["Date"],
                        y=flagged["Modal_Price"],
                        name="Classifier shock alert",
                        mode="markers",
                        marker=dict(color="#ff1744", size=13, symbol="triangle-up"),
                    )
                )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Date",
        yaxis_title="INR / quintal",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def construct_feature_vector(
    crop_history: pd.DataFrame,
    modal_price: float,
    min_price: float,
    max_price: float,
    arrival_volume: float,
    active_mandis: int,
    max_temp: float,
    rainfall: float,
    news_headline: str,
    target_date: datetime,
) -> pd.DataFrame:
    """
    Construct a feature vector matching the training schema.
    Uses historical data from crop_history to compute lags and rolling stats.
    """
    # Compute VADER sentiment from news headline
    news_sentiment = VADER_ANALYZER.polarity_scores(news_headline)['compound']
    
    # Get recent history for lag features
    recent_prices = crop_history['Modal_Price'].tail(30).tolist()
    
    # Compute lag features
    price_lag_1 = recent_prices[-1] if len(recent_prices) >= 1 else modal_price
    price_lag_3 = recent_prices[-3] if len(recent_prices) >= 3 else modal_price
    price_lag_7 = recent_prices[-7] if len(recent_prices) >= 7 else modal_price
    price_lag_14 = recent_prices[-14] if len(recent_prices) >= 14 else modal_price
    
    # Compute rolling statistics
    if len(recent_prices) >= 7:
        price_roll_mean_7 = np.mean(recent_prices[-7:])
        price_roll_std_7 = np.std(recent_prices[-7:])
    else:
        price_roll_mean_7 = modal_price
        price_roll_std_7 = 0.0
    
    if len(recent_prices) >= 30:
        price_roll_mean_30 = np.mean(recent_prices[-30:])
        price_roll_std_30 = np.std(recent_prices[-30:])
    else:
        price_roll_mean_30 = modal_price
        price_roll_std_30 = price_roll_std_7
    
    # Intraday spread
    intraday_spread = max_price - min_price
    
    # Cyclical time encoding
    month = target_date.month
    day_of_week = target_date.weekday()
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    day_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    day_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    
    # Get recent rainfall and sentiment for rolling features
    recent_rainfall = crop_history['Rainfall_mm'].tail(7).tolist() if 'Rainfall_mm' in crop_history.columns else []
    recent_sentiment = crop_history['News_Sentiment_Score'].tail(7).tolist() if 'News_Sentiment_Score' in crop_history.columns else []
    recent_disruption = crop_history['Event_Disruption_Flag'].tail(7).tolist() if 'Event_Disruption_Flag' in crop_history.columns else []
    
    rainfall_roll_7d = sum(recent_rainfall) if recent_rainfall else rainfall
    sentiment_roll_mean_7d = np.mean(recent_sentiment) if recent_sentiment else news_sentiment
    disruption_count_7d = sum(recent_disruption) if recent_disruption else 0
    
    # Construct feature dictionary matching MULTIMODAL_FEATURES
    features = {
        'Price_Lag_1': price_lag_1,
        'Price_Lag_3': price_lag_3,
        'Price_Lag_7': price_lag_7,
        'Price_Lag_14': price_lag_14,
        'Price_Roll_Mean_7': price_roll_mean_7,
        'Price_Roll_Std_7': price_roll_std_7,
        'Price_Roll_Mean_30': price_roll_mean_30,
        'Price_Roll_Std_30': price_roll_std_30,
        'Intraday_Spread': intraday_spread,
        'Month_Sin': month_sin,
        'Month_Cos': month_cos,
        'Max_Temp_C': max_temp,
        'Rainfall_mm': rainfall,
        'Rainfall_Roll_7D': rainfall_roll_7d,
        'News_Sentiment_Score': news_sentiment,
        'Sentiment_Roll_Mean_7D': sentiment_roll_mean_7d,
        'Disruption_Count_7D': disruption_count_7d,
    }
    
    return pd.DataFrame([features])


def tab_historical_evaluation():
    """Tab 1: Historical Evaluation & Backtesting (2023-2025)"""
    st.header("📊 Historical Evaluation & Backtesting (2023–2025)")
    
    master = load_master()
    preds = load_predictions()

    if master is None:
        st.error("⚠️ Processed master CSV not found. Run the training pipeline first:")
        st.code("python -m src.train --rebuild-data", language="bash")
        return

    crop = st.selectbox("Select Crop", COMMODITIES, index=0, key="hist_crop")
    crop_hist = master.loc[master["Commodity"] == crop].copy()
    
    if crop_hist.empty:
        st.warning(f"No historical data for {crop}")
        return
    
    latest = crop_hist.iloc[-2] if len(crop_hist) > 1 else crop_hist.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest modal price", f"₹{latest['Modal_Price']:.0f}")
    c2.metric("7-day mean", f"₹{latest.get('Price_Roll_Mean_7', float('nan')):.0f}")
    c3.metric("News sentiment", f"{latest.get('News_Sentiment_Score', 0.0):+.2f}")
    c4.metric("7-day rainfall", f"{latest.get('Rainfall_Roll_7D', 0.0):.1f} mm")

    st.subheader("Actual vs Forecasted Prices")
    if preds is None:
        st.warning("No saved predictions yet. Charts will show history only until you run `python -m src.train`.")
    st.plotly_chart(_price_figure(crop_hist, preds, crop), use_container_width=True)

    st.subheader("Price-Shock Alerts")
    st.plotly_chart(_shock_figure(crop_hist, preds, crop), use_container_width=True)

    # Date explorer
    dates = crop_hist["Date"].dt.date
    selected = st.slider(
        "Explore News & Weather by Date",
        min_value=dates.min(),
        max_value=dates.max(),
        value=dates.max(),
        key="hist_date_slider"
    )
    
    day = crop_hist.loc[crop_hist["Date"].dt.date == selected]
    if not day.empty:
        row = day.iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Max temp", f"{row.get('Max_Temp_C', float('nan')):.1f} °C")
        d2.metric("Rainfall", f"{row.get('Rainfall_mm', float('nan')):.1f} mm")
        d3.metric("Sentiment", f"{row.get('News_Sentiment_Score', 0.0):+.2f}")
        disrupted = int(row.get("Event_Disruption_Flag", 0) or 0)
        d4.metric("Disruption event", "Yes" if disrupted else "No")

        if disrupted:
            st.error("🔴 Disruption flag is on for this date (unseasonal rain, heatwave, or sharply negative news tone).")
        elif row.get("News_Sentiment_Score", 0) < -0.2:
            st.warning("⚠️ News/weather sentiment is negative, but below the disruption threshold.")
        else:
            st.success("✅ No disruption event encoded for this date.")


def tab_live_predictor():
    """Tab 2: Live / 2026 Market Shock & Price Predictor"""
    st.header("🔮 Live / 2026 Market Shock & Price Predictor")
    st.caption("Interactive What-If Scenario Tester for Viva Demo")
    
    master = load_master()
    if master is None:
        st.error("⚠️ Master dataset not found. Train models first with `python -m src.train --rebuild-data`")
        return
    
    with st.form("prediction_form"):
        st.subheader("📝 Input Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌾 Crop & Date Selection**")
            crop = st.selectbox("Select Crop", COMMODITIES, index=0, key="live_crop")
            
            # Allow dates up to 2026
            target_date = st.date_input(
                "Prediction Date",
                value=datetime(2026, 3, 15),
                min_value=datetime(2023, 6, 1),
                max_value=datetime(2026, 12, 31)
            )
            
            st.markdown("**💰 Market Parameters**")
            modal_price = st.number_input("Recent Modal Price (₹/Quintal)", value=2100.0, min_value=100.0, max_value=10000.0, step=50.0)
            min_price = st.number_input("Min Price (₹/Quintal)", value=1950.0, min_value=100.0, max_value=10000.0, step=50.0)
            max_price = st.number_input("Max Price (₹/Quintal)", value=2250.0, min_value=100.0, max_value=10000.0, step=50.0)
            
        with col2:
            st.markdown("**📦 Supply Parameters**")
            arrival_volume = st.number_input("Mandi Arrival Volume (Tonnes)", value=450.0, min_value=0.0, max_value=5000.0, step=10.0)
            active_mandis = st.number_input("Active Mandis Reporting", value=15, min_value=1, max_value=100, step=1)
            
            st.markdown("**🌤️ Weather Conditions**")
            max_temp = st.number_input("Today's Max Temperature (°C)", value=36.0, min_value=10.0, max_value=50.0, step=0.5)
            rainfall = st.number_input("Rainfall in past 24 hrs (mm)", value=0.0, min_value=0.0, max_value=200.0, step=1.0)
        
        st.markdown("**📰 News Headline Input**")
        news_headline = st.text_area(
            "Enter today's agriculture news headline",
            value="Heavy unseasonal downpour damages standing onion crops in Gujarat mandis",
            height=100
        )
        
        submit = st.form_submit_button("🚀 Run Multimodal AI Prediction", use_container_width=True)
    
    if submit:
        with st.spinner("Loading models and computing predictions..."):
            # Load models for the selected crop
            models = load_models_for_commodity(crop)
            
            if not models:
                st.error(f"⚠️ No trained models found for {crop}. Please train models first.")
                return
            
            # Get historical data for this crop
            crop_hist = master.loc[master["Commodity"] == crop].copy()
            
            if crop_hist.empty:
                st.error(f"⚠️ No historical data found for {crop}")
                return
            
            # Construct feature vector
            try:
                features = construct_feature_vector(
                    crop_hist,
                    modal_price,
                    min_price,
                    max_price,
                    arrival_volume,
                    active_mandis,
                    max_temp,
                    rainfall,
                    news_headline,
                    target_date
                )
                
                # Get feature names in correct order
                feature_cols = FEATURE_REGIMES['multimodal']
                X = features[feature_cols].values
                
                # Make predictions
                predictions = {}
                
                # XGBoost prediction
                if 'reg_xgboost' in models:
                    try:
                        pred_xgb = models['reg_xgboost'].predict(X)[0]
                        predictions['XGBoost'] = pred_xgb
                    except Exception as e:
                        st.warning(f"XGBoost prediction failed: {e}")
                
                # Ridge prediction
                if 'reg_ridge' in models:
                    try:
                        pred_ridge = models['reg_ridge'].predict(X)[0]
                        predictions['Ridge'] = pred_ridge
                    except Exception as e:
                        st.warning(f"Ridge prediction failed: {e}")
                
                # Classification (shock detection)
                shock_probability = 0.0
                if 'clf_random_forest' in models:
                    try:
                        shock_prob_rf = models['clf_random_forest'].predict_proba(X)[0][1]
                        shock_probability = max(shock_probability, shock_prob_rf)
                    except Exception as e:
                        st.warning(f"Random Forest classification failed: {e}")
                
                if not predictions:
                    st.error("⚠️ No predictions could be generated. Check model availability.")
                    return
                
                # Use the best available prediction (prefer XGBoost)
                forecast_price = predictions.get('XGBoost', predictions.get('Ridge', modal_price))
                price_change = forecast_price - modal_price
                price_change_pct = (price_change / modal_price) * 100
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                # Main metrics
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    "Forecasted Modal Price (Tomorrow)",
                    f"₹{forecast_price:.0f}",
                    delta=f"{price_change:+.0f} ({price_change_pct:+.1f}%)"
                )
                m2.metric("Current Price", f"₹{modal_price:.0f}")
                m3.metric("Shock Probability", f"{shock_probability*100:.1f}%")
                
                # Shock status banner
                is_shock = shock_probability >= 0.5
                if is_shock:
                    st.error("🚨 **PRICE SHOCK DETECTED - RISK LEVEL HIGH**")
                    st.markdown("Market volatility expected. High probability of extreme price movement.")
                else:
                    st.success("✅ **NORMAL MARKET REGIME**")
                    st.markdown("Market conditions appear stable with normal price dynamics.")
                
                # Feature breakdown
                st.subheader("📊 Feature Breakdown")
                f1, f2, f3, f4 = st.columns(4)
                
                vader_score = features['News_Sentiment_Score'].values[0]
                f1.metric("VADER Sentiment", f"{vader_score:+.3f}")
                f2.metric("Rainfall Impact", f"{rainfall:.1f} mm")
                
                price_to_arrival = modal_price / arrival_volume if arrival_volume > 0 else 0
                f3.metric("Price/Arrival Ratio", f"{price_to_arrival:.2f}")
                f4.metric("7-Day Price Volatility", f"₹{features['Price_Roll_Std_7'].values[0]:.0f}")
                
                # Model comparison
                if len(predictions) > 1:
                    st.subheader("🤖 Model Comparison")
                    model_df = pd.DataFrame([
                        {"Model": k, "Predicted Price (₹)": f"{v:.0f}", "Change from Current": f"{v-modal_price:+.0f}"}
                        for k, v in predictions.items()
                    ])
                    st.dataframe(model_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                import traceback
                st.code(traceback.format_exc())


def main() -> None:
    st.title("🌾 Multimodal Agricultural Market Intelligence")
    st.caption("Gujarat Mandi Prices · Onion, Potato, Tomato · Advanced ML Forecasting System")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Historical Evaluation & Backtesting", "🔮 Live 2026 Predictor"])
    
    with tab1:
        tab_historical_evaluation()
    
    with tab2:
        tab_live_predictor()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system combines:
        - 🌾 Gujarat Mandi Price Data
        - 🌤️ Ahmedabad Weather Data
        - 📰 Google News RSS + VADER Sentiment
        - 🤖 ML Models (Ridge, XGBoost, Bi-LSTM)
        
        **Training Pipeline:**
        ```bash
        python -m src.train --rebuild-data
        ```
        
        **Data Sources:**
        - Mandi Prices: 2023-2025
        - Weather: Open-Meteo API
        - News: Google News RSS Feed
        """)
        
        st.markdown("---")
        st.markdown("**Model Status:**")
        master = load_master()
        preds = load_predictions()
        
        if master is not None:
            st.success(f"✅ Master dataset loaded ({len(master)} rows)")
        else:
            st.error("❌ Master dataset missing")
        
        if preds is not None:
            st.success(f"✅ Predictions available ({len(preds)} rows)")
        else:
            st.warning("⚠️ No predictions yet")


if __name__ == "__main__":
    main()
