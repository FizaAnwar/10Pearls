"""
train_model.py
Loads features from data/features.csv, prepares training examples for forecasting next 72 hours (3 days),
trains sklearn RandomForest and a simple TensorFlow MLP, evaluates, and saves best model to models/.
USAGE:
  python train_model.py --horizon_hours 72
"""

import os, joblib, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from datetime import timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def create_supervised(df, horizon_hours=72, feature_cols=None, target_col='aqi_pm25'):
    # For simplicity create samples where X = features at time t, y = aqi at t+horizon (single-step forecast)
    rows = []
    for i in range(len(df) - horizon_hours):
        X = df.iloc[i][feature_cols].values.astype(float)
        y = df.iloc[i + horizon_hours][target_col]
        rows.append((X, y))
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    return X, y

def build_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_preds(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon_hours', type=int, default=72)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['timestamp'])
    # Drop rows with NaNs for simplicity
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['temp','humidity','wind_speed','pm25','pm10','no2','o3','year','month','day','hour','weekday','pm25_change','aqi_change_rate','pm25_3h_mean','pm25_24h_mean']
    feature_cols = [c for c in feature_cols if c in df.columns]
    X, y = create_supervised(df, horizon_hours=args.horizon_hours, feature_cols=feature_cols, target_col='aqi_pm25')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features simply
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)
    metrics_rf = evaluate_preds(y_test, y_pred_rf)
    print("RF metrics:", metrics_rf)

    # Neural Net
    mlp = build_mlp(X_train_s.shape[1])
    history = mlp.fit(X_train_s, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=0)
    y_pred_mlp = mlp.predict(X_test_s).squeeze()
    metrics_mlp = evaluate_preds(y_test, y_pred_mlp)
    print("MLP metrics:", metrics_mlp)

    # Choose best by RMSE
    best = ('rf', rf, metrics_rf['rmse'])
    if metrics_mlp['rmse'] < best[2]:
        best = ('mlp', mlp, metrics_mlp['rmse'])

    print("Best:", best[0])
    # Save models and scaler
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    if best[0] == 'rf':
        joblib.dump(rf, os.path.join(MODEL_DIR, 'model_rf.joblib'))
    else:
        mlp.save(os.path.join(MODEL_DIR, 'model_mlp'))

    # Save evaluation report
    report = {'rf': metrics_rf, 'mlp': metrics_mlp, 'best': best[0]}
    with open(os.path.join(MODEL_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print('Saved models and report to', MODEL_DIR)

if __name__ == '__main__':
    main()
