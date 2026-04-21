# src/models/autoencoder.py
#
# LSTM Autoencoder for anomaly detection.
# Matches the trained Kaggle notebook exactly.
#
# Key design decisions from notebook:
#   - 17 features including 'load' — autoencoder reconstructs the full signal
#   - Overlapping sequences with stride=24 for dense anomaly coverage
#   - P99 threshold — data-driven, robust to skewed error distributions
#   - Severity classification: normal / low / medium / high
#   - filter_normal_data() removes obvious outliers before training
#     so the model learns only from clean normal patterns

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Tuple, Dict, List
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers

from src.models.config import (
    SEQUENCE_LENGTH,
    AUTOENCODER_DIR,
    LOGGER_NAME,
)

# ----------------------------------------------------------------
# Autoencoder feature columns
# 17 features — includes 'load' unlike the forecasting LSTM
# The autoencoder reconstructs the FULL signal including the target
# ----------------------------------------------------------------
AE_FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
    "load",   # target included — reconstruction quality on load is key signal
]

# ----------------------------------------------------------------
# Architecture constants
# ----------------------------------------------------------------
ENCODER_UNITS_1  = 128
ENCODER_UNITS_2  = 64
BOTTLENECK_UNITS = 32
DECODER_UNITS_1  = 64
DECODER_UNITS_2  = 128
DROPOUT_RATE     = 0.2
AE_LEARNING_RATE = 1e-3

# Stride for overlapping sequence windows
# 24 = one window starting every 24 hours
# Gives ~7x more sequences than non-overlapping (stride=168)
# Dense enough to catch anomalies at day-level granularity
AE_STRIDE = 24

# P99 threshold method — top 1% of training errors are flagged
# More robust than mean+2×std for skewed reconstruction error distributions
THRESHOLD_PERCENTILE = 99

# ----------------------------------------------------------------
# Output file paths
# ----------------------------------------------------------------
AE_MODEL_FILE     = AUTOENCODER_DIR / "ae_model.keras"
AE_THRESHOLD_FILE = AUTOENCODER_DIR / "threshold.json"
AE_METRICS_FILE   = AUTOENCODER_DIR / "ae_metrics.json"
AE_HISTORY_FILE   = AUTOENCODER_DIR / "ae_history.json"
AE_SCALER_FILE    = AUTOENCODER_DIR / "ae_feature_scaler.pkl"


class AutoencoderError(Exception):
    """Raised when autoencoder operations fail."""
    pass


# ----------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------

def filter_normal_data(df, target_col: str = "load", z_threshold: float = 3.0):
    """
    Removes rows where load deviates more than z_threshold standard
    deviations from the 24-hour rolling mean.

    The autoencoder must be trained ONLY on normal data.
    If anomalies are included in training, the model learns to
    reconstruct them well too — destroying the detection signal.

    A z_threshold of 3.0 is conservative — only clear outliers are
    removed. Edge cases are kept so the model learns the full range
    of normal variation.

    Args:
        df:           DataFrame with target_col present
        target_col:   Column to compute z-scores on
        z_threshold:  Standard deviations from rolling mean to flag

    Returns:
        Filtered DataFrame with obvious anomalies removed
    """
    log = logger.bind(name=LOGGER_NAME)

    rolling_mean = df[target_col].rolling(window=24, center=True).mean()
    rolling_std  = df[target_col].rolling(window=24, center=True).std()
    z_scores     = np.abs((df[target_col] - rolling_mean) / (rolling_std + 1e-8))

    normal_mask = (z_scores <= z_threshold).fillna(True)
    df_normal   = df[normal_mask]

    removed = len(df) - len(df_normal)
    log.info(
        f"Removed {removed:,} rows ({removed/len(df):.1%}) as obvious anomalies. "
        f"{len(df_normal):,} normal rows remain."
    )
    return df_normal


def create_ae_sequences(
    X:              np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    stride:          int = AE_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping sliding window sequences for the autoencoder.

    Uses stride=24 so windows start every 24 hours.
    This gives far more training sequences than non-overlapping windows
    and ensures every hour has an anomaly score assigned to it.

    Each sequence's anomaly score is assigned to the LAST hour of
    that window — so every hour in the dataset gets exactly one score.

    Args:
        X:               Scaled feature array (n_samples, n_features)
        sequence_length: Window size in hours (168)
        stride:          Hours between window starts (24)

    Returns:
        sequences: (n_sequences, sequence_length, n_features)
        indices:   (n_sequences,) — last timestep index of each window
    """
    sequences = []
    indices   = []

    for start in range(0, len(X) - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(X[start:end])
        indices.append(end - 1)

    return (
        np.array(sequences, dtype="float32"),
        np.array(indices),
    )


# ----------------------------------------------------------------
# Model architecture
# ----------------------------------------------------------------

def build_autoencoder(
    n_features:       int,
    sequence_length:  int   = SEQUENCE_LENGTH,
    encoder_units_1:  int   = ENCODER_UNITS_1,
    encoder_units_2:  int   = ENCODER_UNITS_2,
    bottleneck_units: int   = BOTTLENECK_UNITS,
    decoder_units_1:  int   = DECODER_UNITS_1,
    decoder_units_2:  int   = DECODER_UNITS_2,
    dropout_rate:     float = DROPOUT_RATE,
    learning_rate:    float = AE_LEARNING_RATE,
) -> keras.Model:
    """
    Builds and compiles the LSTM Autoencoder.

    Architecture:
        Input(seq_len, n_features)
          Encoder:
            LSTM(128, return_sequences=True)
            Dropout(0.2)
            LSTM(64,  return_sequences=True)
            LSTM(32,  return_sequences=False)  ← bottleneck
          Bridge:
            RepeatVector(seq_len)
          Decoder:
            LSTM(64,  return_sequences=True)
            Dropout(0.2)
            LSTM(128, return_sequences=True)
          Output:
            TimeDistributed(Dense(n_features))

    The output shape is identical to the input shape.
    Loss = MAE between input and reconstruction.
    Low loss = normal pattern. High loss = anomaly.

    Args:
        n_features:       Number of input features (17)
        sequence_length:  Sequence length (168)
        encoder_units_1:  First encoder LSTM units
        encoder_units_2:  Second encoder LSTM units
        bottleneck_units: Bottleneck LSTM units (most compressed)
        decoder_units_1:  First decoder LSTM units
        decoder_units_2:  Second decoder LSTM units
        dropout_rate:     Dropout fraction
        learning_rate:    Adam learning rate

    Returns:
        Compiled Keras autoencoder model
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(
        f"Building autoencoder — n_features={n_features}, "
        f"encoder=[{encoder_units_1},{encoder_units_2},{bottleneck_units}], "
        f"decoder=[{decoder_units_1},{decoder_units_2}]"
    )

    inputs = keras.Input(shape=(sequence_length, n_features), name="ae_input")

    # Encoder — compress the sequence progressively
    x = layers.LSTM(encoder_units_1, return_sequences=True,  name="enc_lstm_1")(inputs)
    x = layers.Dropout(dropout_rate, name="enc_drop_1")(x)
    x = layers.LSTM(encoder_units_2, return_sequences=True,  name="enc_lstm_2")(x)

    # Bottleneck — single vector summarising the entire sequence
    encoded = layers.LSTM(bottleneck_units, return_sequences=False, name="bottleneck")(x)

    # Bridge — repeat bottleneck for each decoder timestep
    x = layers.RepeatVector(sequence_length, name="repeat")(encoded)

    # Decoder — reconstruct the sequence from the compressed representation
    x = layers.LSTM(decoder_units_1, return_sequences=True, name="dec_lstm_1")(x)
    x = layers.Dropout(dropout_rate, name="dec_drop_1")(x)
    x = layers.LSTM(decoder_units_2, return_sequences=True, name="dec_lstm_2")(x)

    # Output — one Dense(n_features) applied at every timestep
    outputs = layers.TimeDistributed(
        layers.Dense(n_features),
        name="reconstruction",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="smart_grid_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
        metrics=["mae"],
    )

    log.info(f"Autoencoder built — {model.count_params():,} parameters")
    return model


# ----------------------------------------------------------------
# Anomaly detection utilities
# ----------------------------------------------------------------

def compute_reconstruction_errors(
    model:      keras.Model,
    X:          np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Computes reconstruction MAE for every sequence.

    Args:
        model:      Trained autoencoder
        X:          Sequences (n_sequences, seq_len, n_features)
        batch_size: Inference batch size

    Returns:
        errors: (n_sequences,) — MAE per sequence
    """
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    # Mean over timesteps (axis=1) AND features (axis=2)
    return np.abs(X - X_pred).mean(axis=(1, 2))


def compute_threshold(
    train_errors:        np.ndarray,
    percentile:          float = THRESHOLD_PERCENTILE,
) -> Dict:
    """
    Computes the anomaly detection threshold using the P99 method.

    P99 = the value below which 99% of training reconstruction errors fall.
    Any sequence with error above P99 is flagged as an anomaly.

    Why P99 instead of mean + 2×std?
        mean + 2×std sits at ~P95 for skewed distributions.
        When test errors are systematically higher (distribution shift),
        this causes 80%+ anomaly rates.
        P99 gives more headroom and is directly interpretable:
        "flag the top 1% most unusual sequences".

    Args:
        train_errors: Reconstruction errors on normal training sequences
        percentile:   Percentile to use as threshold (default 99)

    Returns:
        Dictionary with threshold value and supporting statistics
    """
    mean      = float(train_errors.mean())
    std       = float(train_errors.std())
    threshold = float(np.percentile(train_errors, percentile))

    return {
        "threshold":         threshold,
        "method":            f"P{int(percentile)} of training reconstruction errors",
        "percentile":        percentile,
        "mean":              mean,
        "std":               std,
        "sigma_2_threshold": mean + 2.0 * std,
        "sigma_3_threshold": mean + 3.0 * std,
        "p95":               float(np.percentile(train_errors, 95)),
        "p99":               float(np.percentile(train_errors, 99)),
    }


def classify_severity(error: float, threshold: float, std: float) -> str:
    """
    Classifies a sequence's anomaly severity.

    normal: error <= threshold
    low:    threshold < error <= threshold + 1×std
    medium: threshold + 1×std < error <= threshold + 2×std
    high:   error > threshold + 2×std

    Args:
        error:     Reconstruction error for this sequence
        threshold: Anomaly threshold
        std:       Standard deviation of training errors

    Returns:
        One of: "normal", "low", "medium", "high"
    """
    if error <= threshold:
        return "normal"
    excess = (error - threshold) / std
    if excess < 1.0:
        return "low"
    elif excess < 2.0:
        return "medium"
    else:
        return "high"


def detect_anomalies(
    model:          keras.Model,
    X:              np.ndarray,
    threshold_info: Dict,
    timestamps:     np.ndarray = None,
    batch_size:     int = 256,
) -> Dict:
    """
    Runs full anomaly detection on a set of sequences.

    Args:
        model:          Trained autoencoder
        X:              Sequences (n_sequences, seq_len, n_features)
        threshold_info: From compute_threshold()
        timestamps:     Optional timestamp per sequence
        batch_size:     Inference batch size

    Returns:
        Dictionary with errors, flags, severities, counts, and timestamps
    """
    log = logger.bind(name=LOGGER_NAME)

    threshold = threshold_info["threshold"]
    std       = threshold_info["std"]

    log.info(f"Running anomaly detection — {len(X):,} sequences, threshold={threshold:.6f}")

    errors     = compute_reconstruction_errors(model, X, batch_size)
    flags      = errors > threshold
    severities = [classify_severity(e, threshold, std) for e in errors]

    n_anomalies  = int(flags.sum())
    anomaly_rate = float(flags.mean() * 100)

    severity_counts = {s: severities.count(s) for s in ["normal","low","medium","high"]}

    log.info(
        f"Detected {n_anomalies:,} anomalies ({anomaly_rate:.2f}%) — "
        f"severity: {severity_counts}"
    )

    return {
        "errors":             errors,
        "flags":              flags,
        "severities":         severities,
        "severity_counts":    severity_counts,
        "n_anomalies":        n_anomalies,
        "anomaly_rate":       anomaly_rate,
        "threshold":          threshold,
        "anomaly_timestamps": timestamps[flags] if timestamps is not None else None,
    }