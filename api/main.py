"""
api/main.py
===========
FastAPI application for Smart Grid DL.

Run:
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /health         — liveness check and model-load status
    POST /forecast       — 24-hour LSTM load forecast from CSV upload
    POST /anomalies      — Autoencoder anomaly detection from CSV upload
    POST /full-analysis  — Both models in one request
    GET  /docs           — Interactive Swagger UI (built-in)
"""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

import src.inference.predictor as predictor

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Lifespan: pre-load models at startup for low first-request latency
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the model cache at startup; log a warning if files are absent."""
    logger.info("Smart Grid DL API v{} starting …", __version__)
    try:
        predictor._load_models()
        logger.info("Models pre-loaded successfully.")
    except predictor.ModelNotFoundError as exc:
        logger.warning("Model pre-load skipped — files not found:\n{}", exc)
    except Exception as exc:
        logger.error("Unexpected error during model pre-load: {}", exc)
    yield
    logger.info("API shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Grid DL",
    description=(
        "Power load forecasting (LSTM) and anomaly detection (LSTM Autoencoder) "
        "for smart grid time-series data."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(predictor.ModelNotFoundError)
async def _handle_model_not_found(_request, exc: predictor.ModelNotFoundError):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(predictor.InputValidationError)
async def _handle_input_validation(_request, exc: predictor.InputValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def _handle_generic(_request, exc: Exception):
    logger.error("Unhandled exception: {}", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
    )


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    timestamp: str


class ForecastResponse(BaseModel):
    predictions: list[float]
    timestamps: list[str]
    horizon_hours: int
    rows_used: int


class AnomalyResponse(BaseModel):
    anomaly_rate: float
    n_anomalies: int
    flagged_timestamps: list[Optional[str]]
    severities: list[str]
    reconstruction_errors: list[float]
    threshold: float
    rows_analyzed: int


class ExtendedForecastResponse(BaseModel):
    predictions: list[float]
    timestamps: list[str]
    horizon_hours: int


class FullAnalysisResponse(BaseModel):
    forecast: ForecastResponse
    extended_forecast: ExtendedForecastResponse
    anomalies: AnomalyResponse


# ---------------------------------------------------------------------------
# CSV parsing helper
# ---------------------------------------------------------------------------


async def _parse_csv(file: UploadFile) -> pd.DataFrame:
    """
    Decode and parse an uploaded CSV file into a DataFrame.

    If a 'timestamp' column is present it is coerced to datetime and set as index.

    Raises
    ------
    HTTPException (400) if the file cannot be decoded or parsed.
    """
    try:
        raw = await file.read()
        df  = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {exc}")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp").sort_index()

    return df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness probe.

    Always returns HTTP 200. Check `models_loaded` to know whether
    the model artefacts are present and the inference cache is warm.
    """
    return HealthResponse(
        status="ok",
        models_loaded=predictor.models_ready(),
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(file: UploadFile = File(...)) -> ForecastResponse:
    """
    Run the 24-hour LSTM load forecaster.

    Upload a CSV containing at least the 16 LSTM feature columns and 168 rows.
    Returns 24 hourly MW predictions inverse-transformed to real scale.
    """
    df     = await _parse_csv(file)
    result = predictor.run_forecast(df)
    return ForecastResponse(**result, rows_used=len(df))


@app.post("/anomalies", response_model=AnomalyResponse)
async def anomalies(file: UploadFile = File(...)) -> AnomalyResponse:
    """
    Run the LSTM Autoencoder anomaly detector.

    Upload a CSV with all 16 feature columns plus 'load'.
    Returns per-window reconstruction errors, anomaly flags, and severity ratings.
    """
    df     = await _parse_csv(file)
    result = predictor.run_anomaly_detection(df)
    return AnomalyResponse(**result, rows_analyzed=len(df))


@app.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(file: UploadFile = File(...)) -> FullAnalysisResponse:
    """
    Run the 24 h forecaster, 168 h extended forecast, and anomaly detector.

    Accepts a CSV with all 16 feature columns + 'load', minimum 192 rows.
    Returns all three model outputs in a single response.
    """
    df     = await _parse_csv(file)
    result = predictor.run_full_pipeline(df)
    return FullAnalysisResponse(
        forecast=ForecastResponse(**result["forecast"], rows_used=len(df)),
        extended_forecast=ExtendedForecastResponse(**result["extended_forecast"]),
        anomalies=AnomalyResponse(**result["anomalies"], rows_analyzed=len(df)),
    )
