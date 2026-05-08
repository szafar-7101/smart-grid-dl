"""
src/inference/predictor.py
==========================
Inference engine for the Smart Grid forecasting + anomaly detection system.

Single responsibility: accept a raw DataFrame → return structured predictions.
All model logic lives here. The API and dashboard are thin wrappers over this.

Pipeline (in order):
    1. Load both trained models and scalers from disk  (lazy, cached on first call)
    2. Validate the input DataFrame has the required columns
    3. Scale features using the same scalers used during training
    4. Build 168-hour sliding-window sequences
    5. Run LSTM  → 24-hour load forecasts (scaled)
    6. Run Autoencoder → per-timestep reconstruction errors
    7. Apply P99 threshold from threshold.json → anomaly flags
    8. Inverse-transform LSTM output → real megawatts
    9. Return a clean Python dictionary

Usage
-----
    from src.inference.predictor import Predictor

    predictor = Predictor()                      # loads models once
    result    = predictor.predict(df)            # df: raw feature DataFrame
"""

from __future__ import annotations

import json
import logging
import pickle
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — single source of truth for paths, column names, sequence params
# ---------------------------------------------------------------------------
from src.config import (
    # LSTM artefacts
    LSTM_MODEL_FILE,
    LSTM_FEATURE_SCALER_FILE,
    LSTM_SCALER_FILE,          # target (inverse-transform) scaler
    # Autoencoder artefacts
    AE_MODEL_FILE,
    AE_FEATURE_SCALER,
    AE_THRESHOLD_FILE,
    # Feature / sequence definitions
    FEATURE_COLUMNS,           # 16 features used by the LSTM
    AE_FEATURE_COLUMNS,        # 17 features used by the AE (includes 'load')
    TARGET_COLUMN,
    SEQUENCE_LENGTH,           # 168
    FORECAST_HORIZON,          # 24
    AE_SEQUENCE_LENGTH,        # 168
)

logger = logging.getLogger("smart_grid.predictor")


# ---------------------------------------------------------------------------
# Custom exceptions — lets callers distinguish error types cleanly
# ---------------------------------------------------------------------------

class PredictorError(Exception):
    """Base class for all predictor errors."""


class ModelLoadError(PredictorError):
    """Raised when a model or scaler file cannot be loaded from disk."""


class InputValidationError(PredictorError):
    """Raised when the input DataFrame fails column or length validation."""


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Stateless inference engine.  Models and scalers are loaded once on first
    access (via cached_property) and reused across all subsequent calls.

    Parameters
    ----------
    lstm_model_path : Path, optional
        Override the default LSTM model path from config.
    ae_model_path : Path, optional
        Override the default autoencoder model path from config.

    Notes
    -----
    Thread safety: TensorFlow/Keras model inference is not thread-safe by
    default.  Wrap calls in a lock if serving from a multi-threaded API.
    """

    def __init__(
        self,
        lstm_model_path: Path | None = None,
        ae_model_path: Path | None = None,
    ) -> None:
        self._lstm_model_path = Path(lstm_model_path or LSTM_MODEL_FILE)
        self._ae_model_path   = Path(ae_model_path   or AE_MODEL_FILE)

    # ------------------------------------------------------------------
    # Lazy-loaded artefacts (loaded once, cached forever)
    # ------------------------------------------------------------------

    @cached_property
    def _lstm_model(self):
        return _load_keras_model(self._lstm_model_path, label="LSTM")

    @cached_property
    def _ae_model(self):
        return _load_keras_model(self._ae_model_path, label="Autoencoder")

    @cached_property
    def _lstm_feature_scaler(self):
        return _load_pickle(LSTM_FEATURE_SCALER_FILE, label="LSTM feature scaler")

    @cached_property
    def _lstm_target_scaler(self):
        return _load_pickle(LSTM_SCALER_FILE, label="LSTM target scaler")

    @cached_property
    def _ae_feature_scaler(self):
        return _load_pickle(AE_FEATURE_SCALER, label="AE feature scaler")

    @cached_property
    def _ae_threshold(self) -> float:
        return _load_ae_threshold(AE_THRESHOLD_FILE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Run full inference pipeline on a raw feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all columns in ``FEATURE_COLUMNS`` plus
            ``TARGET_COLUMN`` ('load').  Must have at least
            ``SEQUENCE_LENGTH`` (168) rows.

        Returns
        -------
        dict with the following keys:

        ``forecasts``
            list[dict] — one entry per 24-hour forecast step.
            Each dict: {"step": int, "predicted_load_mw": float}

        ``anomalies``
            list[dict] — one entry per input timestep evaluated.
            Each dict: {
                "timestep_index": int,
                "reconstruction_error": float,
                "is_anomaly": bool,
                "timestamp": str | None   # if df has a DatetimeIndex
            }

        ``summary``
            dict — {
                "total_timesteps_evaluated": int,
                "anomaly_count": int,
                "anomaly_rate_pct": float,
                "p99_threshold": float,
                "forecast_horizon_hours": int,
            }

        Raises
        ------
        InputValidationError
            If df is missing required columns or is too short.
        ModelLoadError
            If any model/scaler file cannot be loaded.
        """
        logger.info("predict() called — input shape: %s", df.shape)

        # 1. Validate ---------------------------------------------------
        _validate_dataframe(df)

        # 2. Scale features ---------------------------------------------
        lstm_feature_scaled = self._scale_lstm_features(df)
        ae_feature_scaled   = self._scale_ae_features(df)

        # 3. Build sequences --------------------------------------------
        lstm_sequences = _build_sequences(
            lstm_feature_scaled,
            seq_len=SEQUENCE_LENGTH,
        )   # shape: (n_windows, 168, 16)

        ae_sequences = _build_sequences(
            ae_feature_scaled,
            seq_len=AE_SEQUENCE_LENGTH,
        )   # shape: (n_windows, 168, 17)

        logger.debug(
            "Sequences built — LSTM: %s  AE: %s",
            lstm_sequences.shape,
            ae_sequences.shape,
        )

        # 4. LSTM forward pass — 24-hour forecasts ----------------------
        lstm_scaled_preds = self._lstm_model.predict(
            lstm_sequences, verbose=0
        )   # shape: (n_windows, 24)

        # 5. Autoencoder forward pass — reconstruction errors -----------
        ae_reconstructions = self._ae_model.predict(
            ae_sequences, verbose=0
        )   # shape: (n_windows, 168, 17)

        reconstruction_errors = _mean_squared_error_per_window(
            ae_sequences, ae_reconstructions
        )   # shape: (n_windows,)

        # 6. Apply P99 threshold — anomaly flags ------------------------
        anomaly_flags = reconstruction_errors > self._ae_threshold

        # 7. Inverse-transform LSTM predictions → megawatts ------------
        # Use only the last window's 24-step prediction as the "next
        # 24 hours" forecast; all windows are returned in anomalies.
        lstm_mw_preds = self._inverse_transform_forecasts(lstm_scaled_preds)

        # 8. Assemble structured output ---------------------------------
        result = _assemble_result(
            df                   = df,
            lstm_mw_preds        = lstm_mw_preds,
            reconstruction_errors= reconstruction_errors,
            anomaly_flags        = anomaly_flags,
            p99_threshold        = self._ae_threshold,
        )

        logger.info(
            "predict() done — anomalies: %d/%d  (%.1f%%)",
            result["summary"]["anomaly_count"],
            result["summary"]["total_timesteps_evaluated"],
            result["summary"]["anomaly_rate_pct"],
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scale_lstm_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale the 16 LSTM input features using the LSTM feature scaler."""
        X = df[FEATURE_COLUMNS].values   # (T, 16)
        return self._lstm_feature_scaler.transform(X)

    def _scale_ae_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale the 17 AE features (including load) using the AE scaler."""
        X = df[AE_FEATURE_COLUMNS].values   # (T, 17)
        return self._ae_feature_scaler.transform(X)

    def _inverse_transform_forecasts(
        self, scaled_preds: np.ndarray
    ) -> np.ndarray:
        """
        Inverse-transform LSTM output from scaled space back to MW.

        scaled_preds shape: (n_windows, 24)
        Returns shape:      (n_windows, 24)
        """
        n_windows, horizon = scaled_preds.shape
        flat = scaled_preds.reshape(-1, 1)           # (n_windows*24, 1)
        mw   = self._lstm_target_scaler.inverse_transform(flat)
        return mw.reshape(n_windows, horizon)        # (n_windows, 24)


# ---------------------------------------------------------------------------
# Module-level pure functions  (no class state — easy to test in isolation)
# ---------------------------------------------------------------------------

def _load_keras_model(path: Path, label: str):
    """Load a .keras model file; raise ModelLoadError on failure."""
    try:
        import tensorflow as tf   # deferred import — keeps module importable
        logger.info("Loading %s model from %s", label, path)
        model = tf.keras.models.load_model(str(path))
        logger.info("%s model loaded — params: {:,}".format(model.count_params()), label)
        return model
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load {label} model from '{path}': {exc}"
        ) from exc


def _load_pickle(path: Path, label: str):
    """Load a pickle file; raise ModelLoadError on failure."""
    try:
        logger.info("Loading %s from %s", label, path)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        return obj
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load {label} from '{path}': {exc}"
        ) from exc


def _load_ae_threshold(path: Path) -> float:
    """
    Load the P99 anomaly threshold from threshold.json.

    Expected structure:
        {
            "threshold":          0.04276900365948677,
            "max_train":          0.04073238745331764,
            "buffer_multiplier":  1.05
        }

    Returns the "threshold" value (the operative P99 value used at inference).
    """
    try:
        logger.info("Loading AE threshold from %s", path)
        with open(path) as fh:
            data = json.load(fh)
        threshold = float(data["threshold"])
        logger.info("AE P99 threshold: %.6f", threshold)
        return threshold
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load AE threshold from '{path}': {exc}"
        ) from exc


def _validate_dataframe(df: pd.DataFrame) -> None:
    """
    Ensure the input DataFrame has all required columns and enough rows.

    Raises
    ------
    InputValidationError
    """
    # Check all LSTM feature columns are present
    missing_lstm = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_lstm:
        raise InputValidationError(
            f"Input DataFrame is missing LSTM feature columns: {missing_lstm}"
        )

    # Check all AE feature columns (superset — includes 'load')
    missing_ae = [c for c in AE_FEATURE_COLUMNS if c not in df.columns]
    if missing_ae:
        raise InputValidationError(
            f"Input DataFrame is missing AE feature columns: {missing_ae}"
        )

    # Need at least one full sequence window
    min_rows = SEQUENCE_LENGTH   # 168
    if len(df) < min_rows:
        raise InputValidationError(
            f"Input DataFrame has {len(df)} rows; need at least "
            f"{min_rows} (= SEQUENCE_LENGTH) rows to form one window."
        )

    # Warn (don't crash) on NaNs — models will produce NaN outputs silently
    nan_cols = [c for c in AE_FEATURE_COLUMNS if df[c].isna().any()]
    if nan_cols:
        logger.warning(
            "Input DataFrame contains NaN values in columns: %s. "
            "Consider imputing before calling predict().",
            nan_cols,
        )


def _build_sequences(
    data: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Create sliding-window sequences from a 2D feature array.

    Parameters
    ----------
    data    : np.ndarray, shape (T, n_features)
    seq_len : int — window size (168 for both models)

    Returns
    -------
    np.ndarray, shape (n_windows, seq_len, n_features)
        where n_windows = T - seq_len + 1
    """
    T, n_features = data.shape
    n_windows = T - seq_len + 1

    if n_windows <= 0:
        raise InputValidationError(
            f"Cannot build sequences: data has {T} timesteps but "
            f"seq_len={seq_len} requires at least {seq_len} rows."
        )

    # Pre-allocate for efficiency
    sequences = np.empty((n_windows, seq_len, n_features), dtype=np.float32)
    for i in range(n_windows):
        sequences[i] = data[i : i + seq_len]

    return sequences


def _mean_squared_error_per_window(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Compute mean squared reconstruction error per window.

    Parameters
    ----------
    original      : np.ndarray, shape (n_windows, seq_len, n_features)
    reconstructed : np.ndarray, shape (n_windows, seq_len, n_features)

    Returns
    -------
    np.ndarray, shape (n_windows,)  — one MSE scalar per window
    """
    diff = original - reconstructed          # (n_windows, seq_len, n_features)
    return np.mean(diff ** 2, axis=(1, 2))  # average over time + features


def _assemble_result(
    df: pd.DataFrame,
    lstm_mw_preds: np.ndarray,
    reconstruction_errors: np.ndarray,
    anomaly_flags: np.ndarray,
    p99_threshold: float,
) -> dict[str, Any]:
    """
    Package model outputs into the canonical return dictionary.

    ``forecasts``  — the LAST window's 24-step prediction.
                     (Most callers want "next 24 hours" from the most
                      recent data; all windows are exposed in anomalies.)

    ``anomalies``  — one entry per sequence window evaluated.
                     Each window maps to the LAST timestep of that window
                     in the original DataFrame (the "present" for that window).
    """
    n_windows = len(reconstruction_errors)

    # ---- Forecasts: use the final window only ----
    last_window_preds = lstm_mw_preds[-1]   # shape: (24,)
    forecasts = [
        {
            "step": step + 1,
            "predicted_load_mw": round(float(mw), 4),
        }
        for step, mw in enumerate(last_window_preds)
    ]

    # ---- Anomalies: one entry per evaluated window ----
    # Window i covers df rows [i : i + SEQUENCE_LENGTH].
    # We tag each window by the index of its last row (the "anchor" timestep).
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    anomalies = []
    for i in range(n_windows):
        anchor_idx = i + AE_SEQUENCE_LENGTH - 1   # last row of window i

        entry: dict[str, Any] = {
            "timestep_index":       anchor_idx,
            "reconstruction_error": round(float(reconstruction_errors[i]), 8),
            "is_anomaly":           bool(anomaly_flags[i]),
        }

        if has_datetime_index:
            ts = df.index[anchor_idx]
            entry["timestamp"] = ts.isoformat()
        else:
            entry["timestamp"] = None

        anomalies.append(entry)

    # ---- Summary ----
    anomaly_count = int(anomaly_flags.sum())
    summary = {
        "total_timesteps_evaluated": n_windows,
        "anomaly_count":             anomaly_count,
        "anomaly_rate_pct":          round(anomaly_count / n_windows * 100, 2),
        "p99_threshold":             round(p99_threshold, 8),
        "forecast_horizon_hours":    FORECAST_HORIZON,
    }

    return {
        "forecasts": forecasts,
        "anomalies": anomalies,
        "summary":   summary,
    }