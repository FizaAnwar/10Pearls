"""
app_streamlit.py
A simple Streamlit dashboard that loads latest model and shows predictions for next 72 hours.
Run: streamlit run app_streamlit.py
"""

import os, joblib, time, pandas as pd, numpy as np, streamlit as st
from datetime import datetime, timedelta

PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

st.set_page_config(layout="wide", page_title="Pearls AQI Predictor")

st.title("Pearls AQI Predictor — 3-day forecast")

city = st.sidebar.text_input("City", "Islamabad")
if st.sidebar.button("Fetch latest features (local CSV)"):
    st.info("This app reads local data/data/features.csv produced by fetch_features.py")

if not os.path.exists(os.path.join(DATA_DIR, 'features.csv')):
    st.warning("No features.csv found. Run fetch_features.py first (or use sample dataset).")
else:
    df = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['timestamp']).sort_values('timestamp')
    st.write("Latest data snapshot:")
    st.dataframe(df.tail(10))

    # Load scaler and model
    if os.path.exists(os.path.join(MODEL_DIR, 'scaler.joblib')) and os.path.exists(os.path.join(MODEL_DIR, 'model_rf.joblib')):
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        model = joblib.load(os.path.join(MODEL_DIR, 'model_rf.joblib'))
        # Prepare last row as input and predict horizon by iterating (simple persistence of features)
        last = df.iloc[-1:].copy()
        feature_cols = ['temp','humidity','wind_speed','pm25','pm10','no2','o3','year','month','day','hour','weekday','pm25_change','aqi_change_rate','pm25_3h_mean','pm25_24h_mean']
        feature_cols = [c for c in feature_cols if c in df.columns]
        X0 = last[feature_cols].values.astype(float)
        Xs = scaler.transform(X0)
        pred = model.predict(Xs)[0]
        st.metric("Predicted AQI (after horizon)", float(pred))
        st.line_chart(pd.DataFrame({'predicted_aqi':[pred]}))
    else:
        st.warning("Model artifacts not found. Run train_model.py first.")
