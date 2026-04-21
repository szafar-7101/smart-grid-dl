# src/models/autoencoder/autoencoder.py
#
# LSTM Autoencoder for anomaly detection.
#
# Trained configuration from lstm-autoencoder.ipynb (Kaggle):
#   - 17 features including load (target reconstructed alongside inputs)
#   - Encoder: LSTM(128) → LSTM(64) → LSTM(32 bottleneck)
#   - Decoder: RepeatVector → LSTM(64) → LSTM(128) → TimeDistributed Dense
#   - Threshold: P99 of training reconstruction errors
#   - Overlapping windows, stride=24h
#   - Result: best_val_loss=0.196980

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers

# Import config from parent — one level up from autoencoder/
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.config import (
    AE_FEATURE_COLUMNS,
    AE_N_FEATURES,
    AE_ENCODER_UNITS_1,
    AE_ENCODER_UNITS_2,
    AE_BOTTLENECK_UNITS,
    AE_DECODER_UNITS_1,
    AE_DECODER_UNITS_2,
    AE_DROPOUT_RATE,
    AE_LEARNING_RATE,
    AE_SEQUENCE_LENGTH,
    AE_STRIDE,
    AE_FILTER_Z_THRESHOLD,
    AE_THRESHOLD_METHOD,
    AE_SEVERITY_LOW,
    AE_SEVERITY_MEDIUM,
    AE_MODEL_FILE,
    AE_THRESHOLD_FILE,
    AE_FEATURE_SCALER,
    LOGGER_NAME,
)


class AutoencoderError(Exception):
    """Raised when autoencoder operations fail."""
    pass


# ----------------------------------------------------------------
# Sequence creation
# ----------------------------------------------------------------

def create_ae_sequences(
    X:              np.ndarray,
    sequence_length: int = AE_SEQUENCE_LENGTH,
    stride:         int = AE_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates overlapping sequences with a stride of 24 hours.

    Why overlapping (stride=24) instead of non-overlapping (stride=168)?
        Non-overlapping windows on a 7,182-row test set gives only ~42 sequences.
        42 sequences is too few for reliable anomaly statistics — one bad week
        can dominate the entire anomaly rate.
        Stride=24 gives ~7x more sequences (one per day) while still ensuring
        consecutive windows are meaningfully different.

    Args:
        X:               Feature array, shape (n_samples, n_features)
        sequence_length: Window size in hours (168)
        stride:          Hours between window starts (24)

    Returns:
        sequences: shape (n_sequences, sequence_length, n_features)
        end_indices: index of the last hour in each window — used to
                     map anomaly scores back to timestamps
    """
    log = logger.bind(name=LOGGER_NAME)

    sequences  = []
    end_indices = []

    for start in range(0, len(X) - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(X[start:end])
        end_indices.append(end - 1)

    X_seq = np.array(sequences, dtype="float32")
    idx   = np.array(end_indices)

    log.info(
        f"Created {len(X_seq):,} sequences "
        f"(seq_len={sequence_length}, stride={stride}, "
        f"features={X.shape[1]})"
    )
    return X_seq, idx


# ----------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------

def build_autoencoder(
    n_features:       int   = AE_N_FEATURES,
    sequence_length:  int   = AE_SEQUENCE_LENGTH,
    encoder_units_1:  int   = AE_ENCODER_UNITS_1,
    encoder_units_2:  int   = AE_ENCODER_UNITS_2,
    bottleneck_units: int   = AE_BOTTLENECK_UNITS,
    decoder_units_1:  int   = AE_DECODER_UNITS_1,
    decoder_units_2:  int   = AE_DECODER_UNITS_2,
    dropout_rate:     float = AE_DROPOUT_RATE,
    learning_rate:    float = AE_LEARNING_RATE,
) -> keras.Model:
    """
    Builds and compiles the LSTM Autoencoder.

    Architecture:
        Input(168, 17)
          ENCODER:
            LSTM(128, return_sequences=True)
            Dropout(0.2)
            LSTM(64, return_sequences=True)
            LSTM(32, return_sequences=False)   ← bottleneck
          BRIDGE:
            RepeatVector(168)
          DECODER:
            LSTM(64, return_sequences=True)
            Dropout(0.2)
            LSTM(128, return_sequences=True)
          OUTPUT:
            TimeDistributed(Dense(17))

    Input = Output target — model learns to reconstruct its own input.
    Anomalies = sequences the model reconstructs poorly.

    Args:
        n_features:       17 for Panama dataset (16 features + load)
        sequence_length:  168
        encoder_units_1:  128
        encoder_units_2:  64
        bottleneck_units: 32
        decoder_units_1:  64  (mirrors encoder_units_2)
        decoder_units_2:  128 (mirrors encoder_units_1)
        dropout_rate:     0.2
        learning_rate:    1e-3

    Returns:
        Compiled Keras autoencoder
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(
        f"Building autoencoder — n_features={n_features}, "
        f"encoder=[{encoder_units_1},{encoder_units_2},{bottleneck_units}]"
    )

    inputs = keras.Input(shape=(sequence_length, n_features), name="ae_input")

    # Encoder — compress sequence to bottleneck
    x = layers.LSTM(
        encoder_units_1, return_sequences=True, name="enc_lstm_1"
    )(inputs)
    x = layers.Dropout(dropout_rate, name="enc_drop_1")(x)

    x = layers.LSTM(
        encoder_units_2, return_sequences=True, name="enc_lstm_2"
    )(x)

    # Bottleneck — single compressed vector per sequence
    encoded = layers.LSTM(
        bottleneck_units, return_sequences=False, name="bottleneck"
    )(x)

    # Bridge — repeat bottleneck for each timestep
    x = layers.RepeatVector(sequence_length, name="repeat")(encoded)

    # Decoder — expand back to full sequence
    x = layers.LSTM(
        decoder_units_1, return_sequences=True, name="dec_lstm_1"
    )(x)
    x = layers.Dropout(dropout_rate, name="dec_drop_1")(x)

    x = layers.LSTM(
        decoder_units_2, return_sequences=True, name="dec_lstm_2"
    )(x)

    # Reconstruct all features at every timestep
    outputs = layers.TimeDistributed(
        layers.Dense(n_features), name="reconstruction"
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
# Inference utilities
# ----------------------------------------------------------------

def compute_reconstruction_errors(
    model:      keras.Model,
    X:          np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Runs the autoencoder and returns per-sequence reconstruction error.

    Error = mean absolute difference between input and reconstruction,
    averaged across all timesteps and all features.
    One scalar error per sequence.

    Args:
        model:      Trained autoencoder
        X:          Sequences, shape (n_sequences, seq_len, n_features)
        batch_size: Inference batch size

    Returns:
        errors: shape (n_sequences,)
    """
    X_pred = model.predict(X, batch_size=batch_size, verbose=0)
    # Mean over timesteps (axis=1) and features (axis=2)
    return np.abs(X - X_pred).mean(axis=(1, 2))


def classify_severity(
    error:     float,
    threshold: float,
    std:       float,
) -> str:
    """
    Classifies an anomalous sequence by severity.

    Severity is based on how many extra standard deviations
    the error is above the threshold.

    Returns: "normal", "low", "medium", or "high"
    """
    if error <= threshold:
        return "normal"

    excess_sigmas = (error - threshold) / std

    if excess_sigmas < AE_SEVERITY_LOW:
        return "low"
    elif excess_sigmas < AE_SEVERITY_MEDIUM:
        return "medium"
    else:
        return "high"


def detect_anomalies(
    model:          keras.Model,
    X:              np.ndarray,
    threshold_info: Dict,
    timestamps:     Optional[np.ndarray] = None,
    batch_size:     int = 256,
) -> Dict:
    """
    Full anomaly detection pass on a set of sequences.

    Args:
        model:          Trained autoencoder
        X:              Sequences, shape (n_sequences, seq_len, n_features)
        threshold_info: Dict from threshold.json with keys:
                        threshold, mean, std, method, p95, p99
        timestamps:     Optional array of timestamps (one per sequence)
        batch_size:     Inference batch size

    Returns:
        Dictionary with:
            errors:             reconstruction error per sequence
            flags:              boolean array — True = anomaly
            severities:         list of severity strings per sequence
            n_anomalies:        count of flagged sequences
            anomaly_rate_pct:   percentage of sequences flagged
            anomaly_timestamps: timestamps of flagged sequences (if provided)
    """
    log = logger.bind(name=LOGGER_NAME)

    threshold = threshold_info["threshold"]
    std       = threshold_info["std"]

    log.info(
        f"Detecting anomalies — {len(X):,} sequences, "
        f"threshold={threshold:.6f} ({threshold_info.get('method', 'P99')})"
    )

    errors     = compute_reconstruction_errors(model, X, batch_size)
    flags      = errors > threshold
    severities = [classify_severity(e, threshold, std) for e in errors]

    n_anomalies  = int(flags.sum())
    anomaly_rate = float(flags.mean() * 100)

    log.info(
        f"Detected {n_anomalies:,} anomalies "
        f"({anomaly_rate:.2f}% of sequences)"
    )

    return {
        "errors":             errors,
        "flags":              flags,
        "severities":         severities,
        "n_anomalies":        n_anomalies,
        "anomaly_rate_pct":   anomaly_rate,
        "threshold":          threshold,
        "anomaly_timestamps": timestamps[flags] if timestamps is not None else None,
    }


# ----------------------------------------------------------------
# Load trained model and assets
# ----------------------------------------------------------------

def load_autoencoder() -> Tuple[keras.Model, Dict, object]:
    """
    Loads the trained autoencoder, threshold info, and feature scaler from disk.

    Returns:
        model:          Loaded Keras autoencoder
        threshold_info: Dictionary with threshold, mean, std, method
        scaler:         Fitted MinMaxScaler for the 17 AE features

    Raises:
        AutoencoderError: If any required file is missing
    """
    log = logger.bind(name=LOGGER_NAME)

    for path in [AE_MODEL_FILE, AE_THRESHOLD_FILE, AE_FEATURE_SCALER]:
        if not path.exists():
            raise AutoencoderError(
                f"Required file not found: {path}\n"
                f"Run the autoencoder training notebook first."
            )

    model = keras.models.load_model(str(AE_MODEL_FILE))
    log.info(f"Autoencoder loaded from {AE_MODEL_FILE}")

    with open(AE_THRESHOLD_FILE) as f:
        threshold_info = json.load(f)
    log.info(f"Threshold loaded: {threshold_info['threshold']:.6f}")

    with open(AE_FEATURE_SCALER, "rb") as f:
        scaler = pickle.load(f)
    log.info(f"Feature scaler loaded — {scaler.n_features_in_} features")

    return model, threshold_info, scaler