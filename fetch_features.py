"""
fetch_features.py
Fetch raw pollutant & weather data for a city and compute time-based and derived features.
Saves output CSV to data/features.csv (append mode for backfilling).
USAGE:
  python fetch_features.py --city Islamabad --start 2024-01-01 --end 2024-10-20 --api_keys '{"openweather":"YOUR_KEY"}'
Notes:
 - Replace API calls with your API keys.
 - The script includes a fallback to synthetic data when APIs are not configured (useful for testing).
"""

import argparse, os, json, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_from_openweather(city, timestamp, api_key):
    # OpenWeather One Call historical example (requires paid tier for long history). Placeholder implementation.
    # Return dict with temperature, humidity, wind_speed, etc.
    # In production, call the API and parse response.
    return {
        "temp": 20 + 5*np.sin(timestamp.hour / 24 * 2*np.pi),
        "humidity": 60 + 10*np.cos(timestamp.hour / 24 * 2*np.pi),
        "wind_speed": 2 + 0.5*np.random.randn(),
    }

def fetch_from_aqicn(city, timestamp, api_key):
    # Placeholder for AQICN pollutant fetch (PM2.5, PM10, NO2, O3...)
    return {
        "pm25": max(0, 30 + 20*np.sin(timestamp.day / 30 * 2*np.pi) + 5*np.random.randn()),
        "pm10": max(0, 40 + 15*np.cos(timestamp.day / 30 * 2*np.pi) + 5*np.random.randn()),
        "no2": max(0, 10 + 8*np.random.randn()),
        "o3": max(0, 20 + 7*np.random.randn())
    }

def compute_aqi_from_pm25(pm25):
    # Simplified linear mapping for demo. For production, use official EPA breakpoints.
    if pm25 <= 12:
        return 50 * (pm25 / 12)
    elif pm25 <= 35.4:
        return 50 + (50 * (pm25 - 12) / (35.4 - 12))
    elif pm25 <= 55.4:
        return 100 + (50 * (pm25 - 35.4) / (55.4 - 35.4))
    else:
        return 150 + (100 * (pm25 - 55.4) / 100.0)

def generate_rows(city, start_date, end_date, api_keys):
    rows = []
    cur = start_date
    while cur <= end_date:
        weather = fetch_from_openweather(city, cur, api_keys.get("openweather"))
        pollutants = fetch_from_aqicn(city, cur, api_keys.get("aqicn"))
        row = {
            "city": city,
            "timestamp": cur,
            "year": cur.year,
            "month": cur.month,
            "day": cur.day,
            "hour": cur.hour,
            "weekday": cur.weekday(),
            **weather,
            **pollutants
        }
        row["aqi_pm25"] = compute_aqi_from_pm25(row["pm25"])
        rows.append(row)
        cur += timedelta(hours=1)
    return rows

def add_derived_features(df):
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['pm25_change'] = df['pm25'].diff().fillna(0)
    df['aqi_change_rate'] = df['aqi_pm25'].pct_change().fillna(0)
    # rolling features
    df['pm25_3h_mean'] = df['pm25'].rolling(3, min_periods=1).mean()
    df['pm25_24h_mean'] = df['pm25'].rolling(24, min_periods=1).mean()
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', default='Islamabad')
    parser.add_argument('--start', default=None)
    parser.add_argument('--end', default=None)
    parser.add_argument('--api_keys', default='{}', help='JSON string with api keys (openweather, aqicn)')
    args = parser.parse_args()
    api_keys = json.loads(args.api_keys)

    if args.start and args.end:
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
    else:
        # default: last 7 days hourly
        end = datetime.utcnow()
        start = end - timedelta(days=7)

    rows = generate_rows(args.city, start, end, api_keys)
    df = pd.DataFrame(rows)
    df = add_derived_features(df)

    out_file = os.path.join(DATA_DIR, 'features.csv')
    if os.path.exists(out_file):
        df_existing = pd.read_csv(out_file, parse_dates=['timestamp'])
        df_combined = pd.concat([df_existing, df]).drop_duplicates(subset=['city', 'timestamp']).sort_values('timestamp')
        df_combined.to_csv(out_file, index=False)
        print(f"Appended {len(df)} rows. Total rows now: {len(df_combined)}")
    else:
        df.to_csv(out_file, index=False)
        print(f"Wrote {len(df)} rows to {out_file}")

if __name__ == '__main__':
    main()
