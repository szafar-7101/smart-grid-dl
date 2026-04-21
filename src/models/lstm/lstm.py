# src/models/lstm.py
#
# LSTM model architecture matching the trained Kaggle notebook.
# Uses Huber loss, Reshape layer, and the exact architecture
# that produced MAE=72.9 MW, RMSE=97.6 MW, sMAPE=6.00%.

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
from loguru import logger

from src.models.config import (
    SEQUENCE_LENGTH,
    FORECAST_HORIZON,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DROPOUT_1,
    DROPOUT_2,
    LEARNING_RATE,
    HUBER_DELTA,
    LOGGER_NAME,
)


class LSTMModelError(Exception):
    """Raised when model building or sequence creation fails."""
    pass


# ----------------------------------------------------------------
# Sequence creation
# ----------------------------------------------------------------

def create_sequences(
    X:                np.ndarray,
    y:                np.ndarray,
    sequence_length:  int = SEQUENCE_LENGTH,
    forecast_horizon: int = FORECAST_HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts flat arrays into overlapping sliding window sequences.

    Input:
        X: (n_samples, n_features)  e.g. (33516, 16)
        y: (n_samples,)             e.g. (33516,)

    Output:
        X_seq: (n_sequences, sequence_length, n_features)  e.g. (33324, 168, 16)
        y_seq: (n_sequences, forecast_horizon)             e.g. (33324, 24)
    """
    log = logger.bind(name=LOGGER_NAME)

    if len(X) < sequence_length + forecast_horizon:
        raise LSTMModelError(
            f"Data too short: {len(X)} rows, "
            f"need {sequence_length + forecast_horizon}"
        )
    if len(X) != len(y):
        raise LSTMModelError(f"X ({len(X)}) and y ({len(y)}) length mismatch")

    n = len(X) - sequence_length - forecast_horizon + 1
    log.info(
        f"Creating {n:,} sequences "
        f"(window={sequence_length}h, horizon={forecast_horizon}h, "
        f"features={X.shape[1]})"
    )

    Xs, ys = [], []
    for i in range(n):
        Xs.append(X[i : i + sequence_length])
        ys.append(y[i + sequence_length : i + sequence_length + forecast_horizon])

    X_seq = np.array(Xs, dtype="float32")
    y_seq = np.array(ys, dtype="float32")
    log.info(f"Sequences ready — X: {X_seq.shape}, y: {y_seq.shape}")
    return X_seq, y_seq


# ----------------------------------------------------------------
# Model builder
# ----------------------------------------------------------------

def build_lstm_model(
    n_features:       int,
    lstm_units_1:     int   = LSTM_UNITS_1,
    lstm_units_2:     int   = LSTM_UNITS_2,
    dropout_1:        float = DROPOUT_1,
    dropout_2:        float = DROPOUT_2,
    learning_rate:    float = LEARNING_RATE,
    huber_delta:      float = HUBER_DELTA,
    sequence_length:  int   = SEQUENCE_LENGTH,
    forecast_horizon: int   = FORECAST_HORIZON,
) -> keras.Model:
    """
    Builds the BiLSTM model with Huber loss.

    Architecture:
        Input → Reshape → BiLSTM(128) → Dropout(0.4)
               → LSTM(64) → Dropout(0.3) → Dense(24)

    Huber loss combines MAE and MSE:
        For errors smaller than delta: behaves like MSE (smooth gradients)
        For errors larger than delta:  behaves like MAE (robust to outliers)
    This gave significantly better convergence than plain MAE on the Panama dataset.

    Args:
        n_features:       Number of input features (16 for Panama dataset)
        lstm_units_1:     Units in Bidirectional LSTM layer 1
        lstm_units_2:     Units in LSTM layer 2
        dropout_1:        Dropout after layer 1
        dropout_2:        Dropout after layer 2
        learning_rate:    Adam optimiser learning rate
        huber_delta:      Huber loss threshold
        sequence_length:  Must match create_sequences
        forecast_horizon: Must match create_sequences

    Returns:
        Compiled Keras model ready for training
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(
        f"Building LSTM — n_features={n_features}, "
        f"units=[{lstm_units_1},{lstm_units_2}], "
        f"dropout=[{dropout_1},{dropout_2}], "
        f"lr={learning_rate}, huber_delta={huber_delta}"
    )

    inputs = keras.Input(
        shape=(sequence_length, n_features),
        name="sequence_input",
    )

    # Explicit reshape — ensures correct 3D shape throughout
    x = layers.Reshape(
        (sequence_length, n_features),
        name="reshape",
    )(inputs)

    # Bidirectional LSTM layer 1
    # Processes sequence both forward and backward
    # return_sequences=True: passes full sequence to layer 2
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units_1,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.l2(1e-4),
        ),
        name="bilstm_1",
    )(x)
    x = layers.Dropout(dropout_1, name="dropout_1")(x)

    # LSTM layer 2
    # return_sequences=False: outputs only final hidden state
    x = layers.LSTM(
        lstm_units_2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_2",
    )(x)
    x = layers.Dropout(dropout_2, name="dropout_2")(x)

    # Output: 24 hourly predictions, no activation (regression)
    outputs = layers.Dense(forecast_horizon, name="forecast_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="smart_grid_lstm")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=huber_delta),
        metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
    )

    log.info(f"Model built — {model.count_params():,} parameters")
    return model