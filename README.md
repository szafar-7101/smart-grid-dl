# GridSense.AI — Smart Grid Power Load Intelligence

A production-ready deep learning platform for smart grid analysis built on the **Panama Electrical Load Dataset** (Jan 2015 – Jun 2020, 47,880 hourly observations). Upload a grid feature CSV and get instant 24-hour load forecasting, 7-day extended forecasting, and LSTM Autoencoder anomaly detection — all through an interactive Streamlit dashboard backed by a FastAPI inference server.

---

## What it does

| Capability | Model | Result |
|---|---|---|
| 24-hour load forecast | Bidirectional LSTM | MAE **72.9 MW** · RMSE **97.6 MW** · sMAPE **6.0%** |
| 7-day (168 h) extended forecast | LSTM (iterative) | Seasonal-naïve feature approximation |
| Anomaly detection | LSTM Autoencoder | P99 threshold · **0.42%** anomaly rate on test set |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CSV Upload                      │
└──────────────────────┬──────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │     FastAPI  :8000         │
         │                           │
         │  POST /forecast           │
         │  POST /anomalies          │
         │  POST /full-analysis      │
         │   GET /health             │
         └──────┬──────────┬─────────┘
                │          │
   ┌────────────▼──┐  ┌────▼──────────────┐
   │  LSTM          │  │  LSTM Autoencoder  │
   │  Forecaster    │  │  Anomaly Detector  │
   │  168 h → 24 h  │  │  24 h windows      │
   └────────────────┘  └────────────────────┘
                │          │
         ┌──────▼──────────▼──────┐
         │   Streamlit  :8501      │
         │                         │
         │  ▪ 6 KPI metric cards   │
         │  ▪ 24 h forecast chart  │
         │  ▪ 168 h forecast chart │
         │  ▪ Load heatmap         │
         │  ▪ Anomaly timeline     │
         └─────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 (3.11 recommended for Docker) |
| pip | ≥ 23 |
| Git | any recent version |

> **Apple Silicon (M1/M2/M3):** TensorFlow ships a native ARM wheel (`tensorflow-metal` optional). Everything works out of the box with `pip install`.

> **GPU:** The models are small enough to run on CPU in seconds. No GPU required.

---

## Quick start — local

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/smart-grid-dl.git
cd smart-grid-dl

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                   # makes `src.*` importable from any directory
```

### 2. Add model artefacts and processed data

The trained model weights and processed dataset are **not stored in Git** (they are large binary files). You have two options:

**Option A — Receive pre-trained artefacts** *(fastest)*

Place the following files exactly where shown:

```
src/models/lstm/
    lstm_model.keras
    feature_scaler.pkl
    target_scaler.pkl
    best_params.json
    metrics.json

src/models/autoencoder/
    ae_model.keras
    ae_feature_scaler.pkl
    threshold.json
    ae_metrics.json

data/processed/
    panama_features.parquet
```

**Option B — Train from scratch**

Put the raw Panama dataset (`dataset.csv` or `dataset.parquet`) in `data/`, then run the notebooks in order:

```bash
# 1. Clean and engineer features
jupyter notebook notebooks/data_cleaning.ipynb
jupyter notebook notebooks/feature_engineering.ipynb

# 2. Train the LSTM forecaster (~10 min on CPU, runs Optuna HPO)
python -m src.training.train_lstm

# 3. Train the autoencoder anomaly detector (~5 min on CPU)
python -m src.training.train_autoencoder
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

You should see:

```
INFO  Smart Grid DL API v0.1.0 starting …
INFO  Models pre-loaded successfully.
INFO  Uvicorn running on http://127.0.0.1:8000
```

### 4. Start the dashboard

Open a **second terminal** (same virtualenv):

```bash
streamlit run app/dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Using the dashboard

1. **Upload** — click *Upload feature CSV* in the sidebar and select `data/processed/panama_features.parquet` exported as CSV (or any CSV with the required columns).
2. **Run** — click **▶ Run Analysis**.
3. **Explore** — six KPI cards appear instantly, followed by four charts stacked vertically:
   - 24-hour load forecast with ±73 MW confidence band
   - 168-hour (7-day) extended forecast
   - Historical average load heatmap (hour × day of week)
   - Anomaly timeline coloured by severity

### Exporting the parquet to CSV

```python
import pandas as pd
df = pd.read_parquet("data/processed/panama_features.parquet")
df.to_csv("panama_features.csv", index=True)
```

---

## API reference

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns model load status |
| `POST` | `/forecast` | 24 h LSTM load forecast |
| `POST` | `/anomalies` | Autoencoder anomaly scan |
| `POST` | `/full-analysis` | All three models in one call |

All `POST` endpoints accept `multipart/form-data` with a single `file` field containing the CSV.

**Example with curl:**

```bash
curl -X POST http://localhost:8000/full-analysis \
  -F "file=@panama_features.csv;type=text/csv" | python3 -m json.tool
```

**Example with Python:**

```python
import httpx

with open("panama_features.csv", "rb") as f:
    resp = httpx.post(
        "http://localhost:8000/full-analysis",
        files={"file": ("data.csv", f, "text/csv")},
        timeout=180,
    )

data = resp.json()
print(data["forecast"]["predictions"][:3])     # [988.9, 961.3, 939.94]
print(data["anomalies"]["anomaly_rate"])        # 0.15
print(data["extended_forecast"]["horizon_hours"])  # 168
```

---

## Required CSV columns

The uploaded CSV must contain at minimum the following columns. Order does not matter.

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime string | Optional — enables time-axis labels |
| `load` | float | Actual load in MW (required for anomaly detection) |
| `temperature` | float | Ambient temperature |
| `humidity` | float | Relative humidity |
| `is_weekend` | 0 / 1 | Weekend flag |
| `is_holiday` | 0 / 1 | Public holiday flag |
| `hour_sin`, `hour_cos` | float | Cyclic encoding of hour-of-day |
| `dayofweek_sin`, `dayofweek_cos` | float | Cyclic encoding of day-of-week |
| `month_sin`, `month_cos` | float | Cyclic encoding of month |
| `lag_1` | float | Load 1 hour ago |
| `lag_24` | float | Load 24 hours ago |
| `lag_168` | float | Load 168 hours ago |
| `rolling_mean_24` | float | 24 h rolling mean of load |
| `rolling_std_24` | float | 24 h rolling standard deviation of load |
| `rolling_mean_168` | float | 168 h rolling mean of load |

**Minimum rows:** 192 (168-hour LSTM window + 24-hour seasonal reference buffer).

---

## Docker

### Build and run

```bash
docker build -t gridsense-ai .
docker run -p 8000:8000 -p 8501:8501 gridsense-ai
```

Both services start automatically via `start.sh`.

| Service | URL |
|---|---|
| Streamlit dashboard | http://localhost:8501 |
| FastAPI + Swagger | http://localhost:8000/docs |

### Mount local artefacts

If you want to inject pre-trained models into the container without rebuilding:

```bash
docker run -p 8000:8000 -p 8501:8501 \
  -v $(pwd)/src/models:/app/src/models \
  -v $(pwd)/data:/app/data \
  gridsense-ai
```

---

## Model details

### LSTM Forecaster

| Hyperparameter | Value |
|---|---|
| Architecture | BiLSTM(256) → LSTM(64) → Dense(24) |
| Loss function | Huber (δ = 0.166) |
| Optimiser | Adam (lr = 2.33 × 10⁻⁴) |
| Dropout | 0.5 / 0.3 |
| Batch size | 64 |
| Input window | 168 h (1 week) |
| Forecast horizon | 24 h |
| Training epochs | 58 (early stopping) |
| Train / val / test split | 70 / 15 / 15 % (chronological) |

**Test-set results:**

| Metric | Value |
|---|---|
| MAE | 72.9 MW |
| RMSE | 97.6 MW |
| sMAPE | 6.0 % |

### LSTM Autoencoder

| Hyperparameter | Value |
|---|---|
| Encoder | LSTM(128) → LSTM(64) → LSTM(32 bottleneck) |
| Decoder | RepeatVector → LSTM(64) → LSTM(128) → TimeDistributed Dense(17) |
| Input features | 17 (16 load features + load target) |
| Window length | 24 h |
| Stride | 24 h |
| Anomaly threshold | P99 of training reconstruction errors = 0.0428 |

**Test-set results:**

| Metric | Value |
|---|---|
| Anomaly rate | 0.42 % |
| P99 threshold | 0.0428 (MAE) |

---

## Project structure

```
smart-grid-dl/
│
├── src/
│   ├── inference/
│   │   └── predictor.py          ← public inference API (run_forecast / run_anomaly_detection / run_full_pipeline)
│   ├── models/
│   │   ├── config.py             ← single source of truth for hyperparameters and paths
│   │   ├── lstm/
│   │   │   └── lstm.py           ← model architecture
│   │   └── autoencoder/
│   │       └── autoencoder.py    ← model architecture + threshold utilities
│   └── training/
│       ├── train_lstm.py         ← full training pipeline with Optuna HPO + MLflow
│       └── train_autoencoder.py  ← autoencoder training pipeline
│
├── api/
│   └── main.py                   ← FastAPI application
│
├── app/
│   └── dashboard.py              ← Streamlit dashboard (GridSense.AI)
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   └── eda.ipynb
│
├── data/
│   └── processed/                ← panama_features.parquet (gitignored)
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── start.sh
```

---

## Re-training

To tune and re-train the LSTM with Optuna HPO:

```bash
python -m src.training.train_lstm          # runs 20 Optuna trials then trains final model
```

To skip HPO and use default hyperparameters (faster, good for debugging):

```python
from src.training.train_lstm import train
train(run_hpo=False)
```

MLflow logs are written to `mlruns/` and can be viewed with:

```bash
mlflow ui --port 5000
```

---

## Development

```bash
# Lint
ruff check src/ api/ app/

# Format
black src/ api/ app/

# Tests
pytest
```

---

## Dataset

**Panama Electrical Load Dataset** — hourly electricity demand for Panama from January 2015 to June 2020, available on Kaggle. The raw file goes in `data/`, and the feature engineering notebook generates `data/processed/panama_features.parquet`.
