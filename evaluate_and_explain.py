"""
evaluate_and_explain.py
Loads test set and trained RandomForest model, computes metrics, and generates SHAP feature importances.
Produces a small HTML file with SHAP summary.
"""

import os, joblib, json
import pandas as pd, numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')

def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def main():
    df = pd.read_csv(os.path.join(DATA_DIR, 'features.csv'), parse_dates=['timestamp']).dropna().reset_index(drop=True)
    feature_cols = ['temp','humidity','wind_speed','pm25','pm10','no2','o3','year','month','day','hour','weekday','pm25_change','aqi_change_rate','pm25_3h_mean','pm25_24h_mean']
    feature_cols = [c for c in feature_cols if c in df.columns]
    horizon = 72
    X = []
    y = []
    for i in range(len(df)-horizon):
        X.append(df.iloc[i][feature_cols].values.astype(float))
        y.append(df.iloc[i+horizon]['aqi_pm25'])
    X = np.stack(X); y = np.array(y)
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    Xs = scaler.transform(X)
    rf_path = os.path.join(MODEL_DIR, 'model_rf.joblib')
    if not os.path.exists(rf_path):
        print("RandomForest model not found at", rf_path); return
    rf = joblib.load(rf_path)
    y_pred = rf.predict(Xs)
    metrics = evaluate(y, y_pred)
    print("Evaluation:", metrics)
    # SHAP
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(Xs)
    # save a small HTML
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, Xs, feature_names=feature_cols, show=False)
    plt.tight_layout()
    out_png = os.path.join(MODEL_DIR, 'shap_summary.png')
    plt.savefig(out_png)
    print("Saved SHAP summary to", out_png)
    with open(os.path.join(MODEL_DIR, 'explain_report.json'), 'w') as f:
        json.dump({"metrics": metrics, "features": feature_cols}, f, indent=2)

if __name__ == '__main__':
    main()
