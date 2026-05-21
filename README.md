# Smart Grid DL — Power Load Intelligence Platform

A production-grade deep learning platform for smart grid analysis. Upload a city's
hourly power-load CSV and get instant 24-hour load forecasting and anomaly detection.

## What it does

| Capability | Model | Metric |
|---|---|---|
| 24-hour load forecast | Bidirectional LSTM | MAE 72.9 MW · sMAPE 6.0% |
| Anomaly detection | LSTM Autoencoder | Anomaly rate ~0.42% on test |

## Architecture

```
                         CSV Upload
                             │
              ┌──────────────▼──────────────┐
              │      FastAPI  (port 8000)    │
              │  POST /forecast              │
              │  POST /anomalies             │
              │  POST /full-analysis         │
              └──────┬─────────────┬─────────┘
                     │             │
          ┌──────────▼──┐   ┌──────▼────────┐
          │ LSTM         │   │ LSTM           │
          │ Forecaster   │   │ Autoencoder    │
          │ (168h → 24h) │   │ (24h windows)  │
          └──────────────┘   └────────────────┘
                     │             │
              ┌──────▼─────────────▼─────────┐
              │    Streamlit  (port 8501)     │
              │  ┌─────────────────────────┐  │
              │  │  KPI cards              │  │
              │  │  24 h forecast chart    │  │
              │  │  Load heatmap           │  │
              │  │  Anomaly timeline       │  │
              │  └─────────────────────────┘  │
              └───────────────────────────────┘
```

## Quick start (local)

**Step 1 — install dependencies**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Step 2 — start the API**
```bash
uvicorn api.main:app --reload --port 8000
```

**Step 3 — start the dashboard** (separate terminal)
```bash
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501), upload
`data/processed/panama_features.parquet` exported as CSV, and click **▶ Run Analysis**.

## Docker

```bash
# Build
docker build -t smart-grid-dl .

# Run (both services in one container)
docker run -p 8000:8000 -p 8501:8501 smart-grid-dl
```

FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model load status |
| `POST` | `/forecast` | 24 h LSTM forecast |
| `POST` | `/anomalies` | Autoencoder anomaly scan |
| `POST` | `/full-analysis` | Both models combined |

All `POST` endpoints accept `multipart/form-data` with a `file` field containing the CSV.

## Model performance

| Model | MAE | RMSE | sMAPE |
|-------|-----|------|-------|
| LSTM Forecaster | 72.9 MW | 97.6 MW | 6.0 % |

| Model | Test anomaly rate | P99 threshold |
|-------|-------------------|---------------|
| LSTM Autoencoder | 0.42 % | 0.0428 (MAE) |

## Expected CSV format

The uploaded CSV must include these columns (see `src/inference/predictor.py` for
the full list):

```
timestamp, load, temperature, humidity,
is_weekend, is_holiday,
hour_sin, hour_cos, dayofweek_sin, dayofweek_cos, month_sin, month_cos,
lag_1, lag_24, lag_168,
rolling_mean_24, rolling_std_24, rolling_mean_168
```

Minimum **168 rows** (one week of hourly data) for the LSTM window.

## Project structure

```
smart-grid-dl/
├── src/
│   ├── inference/predictor.py   ← inference entry point (run_forecast / run_anomaly_detection)
│   ├── models/
│   │   ├── config.py            ← all hyperparameters and file paths
│   │   ├── lstm/                ← model weights, scalers, metrics
│   │   └── autoencoder/         ← model weights, scaler, threshold
│   └── training/                ← training scripts
├── api/main.py                  ← FastAPI application
├── app/dashboard.py             ← Streamlit dashboard
├── data/processed/              ← panama_features.parquet
├── notebooks/                   ← EDA and feature engineering
├── Dockerfile
└── start.sh
```

## Screenshot

_Dashboard screenshot will be added after first deployment._
