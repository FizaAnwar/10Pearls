# Pearls AQI Predictor — Project scaffold

This repository contains a full scaffold for the AQI forecasting project described in the slides. It includes:

- `fetch_features.py` — fetches hourly weather and pollutant data and computes derived features. Produces `data/features.csv`.
- `train_model.py` — trains a RandomForest and a TensorFlow MLP to forecast AQI `horizon_hours` ahead. Saves best model into `models/`.
- `evaluate_and_explain.py` — evaluates model(s) and produces SHAP explanation image.
- `app_streamlit.py` — a minimal Streamlit app to display predictions.
- GitHub Actions example workflow in `.github/workflows/train.yml` (saved here as `.github_workflow_train.yml`).
- `requirements.txt` to install dependencies.

## Quickstart (local testing)

1. Create a Python venv and install packages:
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Generate synthetic / API-backed features:
   ```bash
   python fetch_features.py --city Islamabad --start 2024-01-01 --end 2024-10-01 --api_keys '{"openweather":"YOUR_KEY","aqicn":"YOUR_KEY"}'
   ```

3. Train models:
   ```bash
   python train_model.py --horizon_hours 72
   ```

4. Evaluate & explain:
   ```bash
   python evaluate_and_explain.py
   ```

5. Run Streamlit app:
   ```bash
   streamlit run app_streamlit.py
   ```

## Notes & Extensions
- Integrate with Feast, Hopsworks, or Vertex AI feature store by replacing CSV persistence with their ingestion SDKs.
- For production-grade AQI computing, replace `compute_aqi_from_pm25` with official EPA or regional AQI breakpoints that combine pollutants.
- CI/CD: GitHub Actions example demonstrates hourly feature runs and nightly training. Consider Airflow for orchestration, and containerize components for reproducibility.
- Alerts: add a scheduled job that checks predicted AQI and calls SNS/SMTP/Push API when hazardous levels are predicted.

