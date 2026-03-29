# Smart Grid DL — Deep Learning Power Load Intelligence Platform

A production-grade deep learning platform for smart grid analysis.
Upload a city's power load CSV and get instant time series analysis,
anomaly detection, and multi-horizon load forecasting.

## Features
- Multi-horizon load forecasting (24h, 7d, 30d) using LSTM + Temporal Transformer
- Anomaly detection using LSTM Autoencoder with explainable flagging
- Time series decomposition (trend, seasonality, residual)
- Interactive Plotly dashboard with heatmaps, forecast ribbons, and SHAP charts
- Full MLOps pipeline: MLflow tracking, Optuna tuning, DVC data versioning
- Production API built with FastAPI, deployed on Render

## Tech Stack
Python · TensorFlow · FastAPI · Streamlit · Plotly · MLflow · Optuna · DVC · Docker

## Live Demo
_Link will be added after deployment_

## Setup
```bash
git clone https://github.com/YOUR_USERNAME/smart-grid-dl.git
cd smart-grid-dl
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure
```
smart-grid-dl/
├── src/          # All ML logic: ingestion, features, models, training, inference
├── api/          # FastAPI backend
├── app/          # Streamlit frontend
├── data/         # Data files (tracked by DVC)
├── tests/        # Unit and integration tests
└── notebooks/    # EDA and exploration
```