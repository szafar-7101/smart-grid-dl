"""
src/inference/predictor.py
==========================
Self-contained inference module for Smart Grid DL.

Loads all model artefacts from disk on the first call (lazy + cached).
No I/O happens at import time — the module is always importable even when
model files have not yet been generated.

Public API
----------
    run_forecast(df)           -> dict   (24 h)
    run_extended_forecast(df)  -> dict   (168 h iterative)
    run_anomaly_detection(df)  -> dict
    run_full_pipeline(df)      -> dict   (all three combined)
    models_ready()             -> bool
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger("smart_grid.predictor")  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Paths — computed from this file, no hardcoded strings
# ---------------------------------------------------------------------------

_SRC_DIR    = Path(__file__).resolve().parent.parent   # .../src/
_MODELS_DIR = _SRC_DIR / "models"                      # .../src/models/

_LSTM_DIR              = _MODELS_DIR / "lstm"
_LSTM_MODEL_PATH       = _LSTM_DIR / "lstm_model.keras"
_LSTM_FEAT_SCALER_PATH = _LSTM_DIR / "feature_scaler.pkl"
_LSTM_TGT_SCALER_PATH  = _LSTM_DIR / "target_scaler.pkl"

_AE_DIR          = _MODELS_DIR / "autoencoder"
_AE_MODEL_PATH   = _AE_DIR / "ae_model.keras"
_AE_SCALER_PATH  = _AE_DIR / "ae_feature_scaler.pkl"
_AE_THRESH_PATH  = _AE_DIR / "threshold.json"

# ---------------------------------------------------------------------------
# Feature / sequence constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "temperature", "humidity",
    "is_weekend", "is_holiday",
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "month_sin", "month_cos",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_168",
]

AE_FEATURE_COLUMNS: list[str] = FEATURE_COLUMNS + ["load"]

SEQUENCE_LENGTH  = 168   # LSTM input window: 1 week of hourly data
AE_SEQ_LENGTH    = 24    # Autoencoder window — matches saved model input shape (None, 24, 17)
AE_STRIDE        = 24    # Hours between autoencoder windows
FORECAST_HORIZON = 24    # Hours predicted by the LSTM

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PredictorError(Exception):
    """Base class for all predictor errors."""


class ModelNotFoundError(PredictorError):
    """Required model or scaler file is missing from disk."""


class InputValidationError(PredictorError):
    """Input DataFrame failed column or length validation."""


# ---------------------------------------------------------------------------
# Module-level cache — populated on first call to _load_models()
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _load_models() -> None:
    """
    Load all model artefacts into the module cache.

    Called automatically by each public function on its first invocation.
    Subsequent calls return immediately — the cache is checked first.

    Raises
    ------
    ModelNotFoundError
        If any required file is absent from disk.
    """
    if _cache:
        return

    _assert_files_exist()

    import tensorflow as tf  # deferred — avoids TF startup cost at import time

    logger.info("Loading LSTM model from {}", _LSTM_MODEL_PATH)
    _cache["lstm_model"] = tf.keras.models.load_model(str(_LSTM_MODEL_PATH))

    logger.info("Loading Autoencoder from {}", _AE_MODEL_PATH)
    _cache["ae_model"] = tf.keras.models.load_model(str(_AE_MODEL_PATH))

    with _LSTM_FEAT_SCALER_PATH.open("rb") as fh:
        _cache["lstm_feat_scaler"] = pickle.load(fh)

    with _LSTM_TGT_SCALER_PATH.open("rb") as fh:
        _cache["lstm_tgt_scaler"] = pickle.load(fh)

    with _AE_SCALER_PATH.open("rb") as fh:
        _cache["ae_scaler"] = pickle.load(fh)

    thresh_data = json.loads(_AE_THRESH_PATH.read_text())
    _cache["ae_threshold"] = float(thresh_data["threshold"])

    logger.info("All models loaded and cached.")


def _assert_files_exist() -> None:
    """Raise ModelNotFoundError listing every missing artefact path."""
    required = {
        "LSTM model":          _LSTM_MODEL_PATH,
        "LSTM feature scaler": _LSTM_FEAT_SCALER_PATH,
        "LSTM target scaler":  _LSTM_TGT_SCALER_PATH,
        "AE model":            _AE_MODEL_PATH,
        "AE feature scaler":   _AE_SCALER_PATH,
        "AE threshold":        _AE_THRESH_PATH,
    }
    missing = [f"  • {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        raise ModelNotFoundError(
            "Missing model artefacts — run the training scripts first:\n"
            + "\n".join(missing)
        )


def models_ready() -> bool:
    """Return True only when every required model artefact exists on disk."""
    return all(p.exists() for p in [
        _LSTM_MODEL_PATH, _LSTM_FEAT_SCALER_PATH, _LSTM_TGT_SCALER_PATH,
        _AE_MODEL_PATH, _AE_SCALER_PATH, _AE_THRESH_PATH,
    ])


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate(df: pd.DataFrame, required_cols: list[str], min_rows: int) -> None:
    """Raise InputValidationError if df is missing columns or is too short."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise InputValidationError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Present columns: {list(df.columns)}"
        )
    if len(df) < min_rows:
        raise InputValidationError(
            f"DataFrame has {len(df)} rows; need at least {min_rows}."
        )


# ---------------------------------------------------------------------------
# Sequence builders
# ---------------------------------------------------------------------------


def _lstm_sequence(X_scaled: np.ndarray) -> np.ndarray:
    """Return the last SEQUENCE_LENGTH rows as a (1, 168, 16) inference batch."""
    return X_scaled[-SEQUENCE_LENGTH:][np.newaxis].astype(np.float32)


def _ae_windows(
    X_scaled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding windows of length AE_SEQ_LENGTH over X_scaled, spaced AE_STRIDE apart.

    Returns
    -------
    windows     : (n_windows, AE_SEQ_LENGTH, n_features) float32
    end_indices : (n_windows,) int — last-row index of each window in the source df
    """
    windows: list[np.ndarray] = []
    ends: list[int] = []
    T = len(X_scaled)
    for start in range(0, T - AE_SEQ_LENGTH + 1, AE_STRIDE):
        end = start + AE_SEQ_LENGTH
        windows.append(X_scaled[start:end])
        ends.append(end - 1)
    if not windows:
        raise InputValidationError(
            f"Not enough rows to form AE windows "
            f"(need ≥ {AE_SEQ_LENGTH}, got {T})."
        )
    return np.array(windows, dtype=np.float32), np.array(ends, dtype=np.int32)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------


def _severity(error: float, threshold: float) -> str:
    """
    Classify an anomalous window by how far its error exceeds the P99 threshold.

    Severity tiers (fraction above threshold):
        < 25 %  → "low"
        25–75 % → "medium"
        > 75 %  → "high"
    """
    excess = (error - threshold) / threshold
    if excess < 0.25:
        return "low"
    if excess < 0.75:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def _forecast_timestamps(df: pd.DataFrame) -> list[str]:
    """Return ISO timestamps for the 24-hour forecast horizon."""
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        base = df.index[-1]
        return [(base + pd.Timedelta(hours=h + 1)).isoformat() for h in range(FORECAST_HORIZON)]
    return [f"T+{h + 1}h" for h in range(FORECAST_HORIZON)]


def _window_timestamp(df: pd.DataFrame, row_idx: int) -> str | None:
    """Return ISO timestamp of row_idx, or None when there is no DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[int(row_idx)].isoformat()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_forecast(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the LSTM forecaster on the most recent 168-hour window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all 16 FEATURE_COLUMNS.
        Must have at least SEQUENCE_LENGTH (168) rows.
        A pd.DatetimeIndex enables ISO timestamp output; otherwise labels are "T+Nh".

    Returns
    -------
    dict
        predictions   : list[float]  — 24 hourly load values in real MW (≥ 0)
        timestamps    : list[str]    — ISO strings or "T+Nh" step labels
        horizon_hours : int          — always 24
    """
    _load_models()
    _validate(df, FEATURE_COLUMNS, SEQUENCE_LENGTH)

    X        = df[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = _cache["lstm_feat_scaler"].transform(X)
    seq      = _lstm_sequence(X_scaled)                                  # (1, 168, 16)

    raw  = _cache["lstm_model"].predict(seq, verbose=0)                  # (1, 24)
    mw   = _cache["lstm_tgt_scaler"].inverse_transform(
        raw.reshape(-1, 1)
    ).flatten()

    predictions = [max(0.0, round(float(v), 2)) for v in mw]

    logger.info(
        "Forecast — peak {:.1f} MW  trough {:.1f} MW",
        max(predictions), min(predictions),
    )
    return {
        "predictions":   predictions,
        "timestamps":    _forecast_timestamps(df),
        "horizon_hours": FORECAST_HORIZON,
    }


def run_anomaly_detection(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the LSTM Autoencoder anomaly detector over the full DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all 17 AE_FEATURE_COLUMNS (16 FEATURE_COLUMNS + 'load').
        Must have at least AE_SEQ_LENGTH (24) rows.
        A DatetimeIndex enables timestamp output for flagged windows.

    Returns
    -------
    dict
        anomaly_rate          : float          — % of windows flagged (0–100)
        n_anomalies           : int
        flagged_timestamps    : list[str|None] — one entry per anomalous window
        severities            : list[str]      — "low" / "medium" / "high" per anomaly
        reconstruction_errors : list[float]    — MAE per window (ALL windows, not just flagged)
        threshold             : float          — P99 threshold used for flagging
    """
    _load_models()
    _validate(df, AE_FEATURE_COLUMNS, AE_SEQ_LENGTH)

    X        = df[AE_FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = _cache["ae_scaler"].transform(X)

    windows, end_idxs = _ae_windows(X_scaled)                           # (n, 24, 17)
    recon    = _cache["ae_model"].predict(windows, verbose=0)            # (n, 24, 17)
    errors   = np.mean(np.abs(windows - recon), axis=(1, 2))            # (n,)  MAE per window

    threshold = _cache["ae_threshold"]
    flags     = errors > threshold

    flagged_ts = [
        _window_timestamp(df, end_idxs[i])
        for i in range(len(errors))
        if flags[i]
    ]
    severities = [
        _severity(float(errors[i]), threshold)
        for i in range(len(errors))
        if flags[i]
    ]

    n_anom = int(flags.sum())
    rate   = round(n_anom / max(len(errors), 1) * 100, 2)

    logger.info(
        "Anomaly detection — {}/{} windows flagged ({:.2f}%)",
        n_anom, len(errors), rate,
    )
    return {
        "anomaly_rate":          rate,
        "n_anomalies":           n_anom,
        "flagged_timestamps":    flagged_ts,
        "severities":            severities,
        "reconstruction_errors": [round(float(e), 6) for e in errors],
        "threshold":             round(float(threshold), 6),
    }


def run_extended_forecast(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the LSTM iteratively for 7 × 24 = 168 hours ahead.

    Each 24-hour chunk is predicted from the previous 168-hour window.
    For synthetic future rows the approach is:
    - Time cyclic features (hour_sin/cos, etc.): computed analytically from future timestamps
    - Weather features (temperature, humidity): seasonal-naïve — same 24h from 7 days ago
    - is_weekend / is_holiday: derived from future date where possible
    - Lag and rolling features: updated from the expanding prediction series

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all 16 FEATURE_COLUMNS.
        Needs at least SEQUENCE_LENGTH + 24 = 192 rows.

    Returns
    -------
    dict
        predictions   : list[float]  — 168 hourly MW predictions
        timestamps    : list[str]    — ISO strings or "T+Nh" labels
        horizon_hours : int          — always 168
    """
    _load_models()
    _validate(df, FEATURE_COLUMNS, SEQUENCE_LENGTH + 24)

    has_dt = isinstance(df.index, pd.DatetimeIndex)

    # Expanding load series — used to compute future lags and rolling stats
    if "load" in df.columns:
        load_series: list[float] = list(df["load"].values[-SEQUENCE_LENGTH:])
    else:
        load_series = [0.0] * SEQUENCE_LENGTH

    # Seasonal reference: the 24 rows that were 168 h ago — used for weather features
    seasonal_ref = df[FEATURE_COLUMNS].iloc[-SEQUENCE_LENGTH: -SEQUENCE_LENGTH + 24].values.copy()

    # Sliding window of feature rows fed to the LSTM (168 rows × 16 features)
    working_X = df[FEATURE_COLUMNS].values[-SEQUENCE_LENGTH:].copy().astype(np.float32)

    # Column index lookup (build once)
    col = {name: FEATURE_COLUMNS.index(name) for name in FEATURE_COLUMNS}

    all_preds: list[float] = []
    all_ts: list[str] = []
    last_ts = df.index[-1] if has_dt else None

    for _step in range(7):
        # --- forward pass ---
        X_sc  = _cache["lstm_feat_scaler"].transform(working_X)
        seq   = X_sc[np.newaxis].astype(np.float32)              # (1, 168, 16)
        raw   = _cache["lstm_model"].predict(seq, verbose=0)      # (1, 24)
        mw    = _cache["lstm_tgt_scaler"].inverse_transform(raw.reshape(-1, 1)).flatten()
        preds = [max(0.0, round(float(v), 2)) for v in mw]
        all_preds.extend(preds)
        load_series.extend(preds)

        # --- timestamps for this 24-h chunk ---
        if has_dt:
            future_idx = pd.date_range(
                start=last_ts + pd.Timedelta(hours=1), periods=24, freq="h"
            )
            all_ts.extend([t.isoformat() for t in future_idx])
            last_ts = future_idx[-1]
        else:
            n = len(all_preds)
            all_ts.extend([f"T+{n - 24 + i + 1}h" for i in range(24)])

        # --- build 24 synthetic feature rows for next window ---
        new_rows = seasonal_ref.copy().astype(np.float32)   # weather base from 7 days ago

        # Override time-cyclic and binary features analytically when we have timestamps
        if has_dt:
            hrs  = future_idx.hour.values.astype(float)
            dow  = future_idx.dayofweek.values.astype(float)
            mths = future_idx.month.values.astype(float)
            new_rows[:, col["hour_sin"]]      = np.sin(2 * np.pi * hrs / 24)
            new_rows[:, col["hour_cos"]]      = np.cos(2 * np.pi * hrs / 24)
            new_rows[:, col["dayofweek_sin"]] = np.sin(2 * np.pi * dow / 7)
            new_rows[:, col["dayofweek_cos"]] = np.cos(2 * np.pi * dow / 7)
            new_rows[:, col["month_sin"]]     = np.sin(2 * np.pi * mths / 12)
            new_rows[:, col["month_cos"]]     = np.cos(2 * np.pi * mths / 12)
            new_rows[:, col["is_weekend"]]    = (dow >= 5).astype(np.float32)
            new_rows[:, col["is_holiday"]]    = 0.0

        # Override lag and rolling features from the expanding load series
        n_total = len(load_series)
        for i in range(24):
            idx = n_total - 24 + i   # position of this hour in load_series

            new_rows[i, col["lag_1"]]   = load_series[idx - 1]   if idx >= 1   else 0.0
            new_rows[i, col["lag_24"]]  = load_series[idx - 24]  if idx >= 24  else 0.0
            new_rows[i, col["lag_168"]] = load_series[idx - 168] if idx >= 168 else 0.0

            w24  = load_series[max(0, idx - 23): idx + 1]
            w168 = load_series[max(0, idx - 167): idx + 1]
            new_rows[i, col["rolling_mean_24"]]  = float(np.mean(w24))
            new_rows[i, col["rolling_std_24"]]   = float(np.std(w24)) if len(w24) > 1 else 0.0
            new_rows[i, col["rolling_mean_168"]] = float(np.mean(w168))

        # Slide the 168-row window: drop oldest 24, append the new 24
        working_X = np.vstack([working_X[24:], new_rows])   # (168, 16)

    return {
        "predictions":   all_preds,
        "timestamps":    all_ts,
        "horizon_hours": 168,
    }


def run_full_pipeline(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run all three models: 24h forecast, 168h extended forecast, and anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must satisfy all three functions: all 16 FEATURE_COLUMNS plus 'load',
        minimum 192 rows (SEQUENCE_LENGTH + 24).

    Returns
    -------
    dict
        forecast           : dict — output of run_forecast(df)
        extended_forecast  : dict — output of run_extended_forecast(df)
        anomalies          : dict — output of run_anomaly_detection(df)
    """
    logger.info("Full pipeline — {} rows", len(df))
    return {
        "forecast":          run_forecast(df),
        "extended_forecast": run_extended_forecast(df),
        "anomalies":         run_anomaly_detection(df),
    }
