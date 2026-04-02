# Training pipeline for the LSTM model.

# This file orchestrates:
#   1. Loading the feature data 
#   2. Creating sequences using lstm.create_sequences()
#   3. Building the model using lstm.build_lstm_model()
#   4. Setting up Keras callbacks (early stopping, LR reduction)
#   5. Running model.fit() to train
#   6. Logging everything to MLflow
#   7. Evaluating on the test set
#   8. Saving the trained model to disk

import numpy as np
import mlflow

import mlflow.tensorflow

from pathlib import Path
from typing import Dict, Any

from loguru import logger
from tensorflow import keras

from src.features.engineering import run_feature_pipeline
from src.models.lstm import build_lstm_model, create_sequences, LSTMModelError
from src.models.config import (
    LSTM_MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    SEQUENCE_LENGTH,
    FORECAST_HORIZON,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    LEARNING_RATE,
    LSTM_UNITS_LAYER1,
    LSTM_UNITS_LAYER2,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
    LOGGER_NAME,
)
from src.features.config import FEATURES_OUTPUT_FILE, SCALER_OUTPUT_FILE


# ----------------------------------------------------------------
# Custom MLflow Keras Callback
# ----------------------------------------------------------------

class MLflowCallback(keras.callbacks.Callback):
    """
    A custom Keras callback that logs metrics to MLflow after every epoch.
    What this callback does:
        After each epoch, it reads the loss and metric values from
        the 'logs' dictionary that Keras provides and logs them to
        the active MLflow run. This creates a time-series of metrics
        you can visualise in the MLflow UI.
    """

    def on_epoch_end(self, epoch: int, logs: Dict = None) -> None:
        """
        Args:
            epoch: The current epoch number (0-indexed)
            logs:  Dictionary containing loss and metric values for this epoch.
                   Keys: 'loss', 'mae', 'rmse', 'val_loss', 'val_mae', 'val_rmse'
        """
        if logs is None:
            return

        mlflow.log_metrics(
            {
                "train_loss": logs.get("loss", 0),
                "train_mae":  logs.get("mae", 0),
                "train_rmse": logs.get("rmse", 0),
                "val_loss":   logs.get("val_loss", 0),
                "val_mae":    logs.get("val_mae", 0),
                "val_rmse":   logs.get("val_rmse", 0),
            },
            step=epoch + 1,
        )


# ----------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler=None,
    y_index: int = 0,
) -> Dict[str, float]:
    """
    Computes evaluation metrics comparing predictions to ground truth.

    MAE and RMSE are computed on SCALED values — this is fine because
    they are absolute differences and scale consistently.

    MAPE is computed on INVERSE TRANSFORMED values — real kilowatts.
    Why? Because MAPE divides by the actual value. On scaled data,
    actual values are tiny fractions (like 0.03), which makes the
    percentage explode to nonsensical numbers like 900%.
    On real kW values (like 3.0), the percentage is meaningful.

    Args:
        y_true:  Actual load values in scaled space, shape (n, horizon)
        y_pred:  Predicted load values in scaled space, shape (n, horizon)
        scaler:  The fitted MinMaxScaler from Day 3. If provided,
                 MAPE is computed on inverse-transformed values.
        y_index: Column index of the target in the full feature matrix.
                 Needed to correctly inverse transform just the target column.

    Returns:
        Dictionary of metric names to float values
    """
    # Flatten both arrays — compute metrics across ALL predictions at once
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # MAE — on scaled values. Consistent and valid.
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))

    # RMSE — on scaled values. Consistent and valid.
    rmse = float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2)))

    # MAPE — must be on REAL (inverse transformed) values
    if scaler is not None:
        # The scaler was fitted on a matrix of ALL 40 features.
        # To inverse transform just the target column, we need to
        # reconstruct a full-width dummy matrix, put our values in
        # the target column, inverse transform everything, then
        # extract just the target column back out.
        #
        # Why this roundabout approach?
        # MinMaxScaler stores min/max for ALL columns together.
        # You cannot inverse transform one column in isolation —
        # you must provide all columns. We fill the other columns
        # with zeros (they will be transformed but we discard them).

        n_samples = len(y_true_flat)
        n_features = scaler.n_features_in_
        # n_features_in_ is the number of features the scaler was fitted on — 40

        # Build a zero matrix of shape (n_samples, n_features)
        dummy_true = np.zeros((n_samples, n_features))
        dummy_pred = np.zeros((n_samples, n_features))

        # Place scaled values into the target column
        dummy_true[:, y_index] = y_true_flat
        dummy_pred[:, y_index] = y_pred_flat

        # Inverse transform — converts from [0,1] back to real kW values
        real_true = scaler.inverse_transform(dummy_true)[:, y_index]
        real_pred = scaler.inverse_transform(dummy_pred)[:, y_index]

                # sMAPE — Symmetric Mean Absolute Percentage Error.
        # Divides by the AVERAGE of actual and predicted instead of just actual.
        # This prevents explosion when actual values are near zero.
        # Formula: 2 * |actual - predicted| / (|actual| + |predicted| + epsilon)
        smape = float(
            np.mean(
                2 * np.abs(real_true - real_pred) /
                (np.abs(real_true) + np.abs(real_pred) + 1e-8)
            ) * 100
        )
        mape = smape
        # We keep the variable name "mape" so nothing else needs to change,
        # but the value is now computed using the symmetric formula.

        # Also compute real-world MAE for reporting (in kilowatts)
        real_mae  = float(np.mean(np.abs(real_true - real_pred)))
        real_rmse = float(np.sqrt(np.mean((real_true - real_pred) ** 2)))

    else:
        # No scaler provided — fall back to scaled MAPE (not ideal)
        mape = float(
            np.mean(
                np.abs((y_true_flat - y_pred_flat) / (np.abs(y_true_flat) + 1e-8))
            ) * 100
        )
        real_mae  = mae
        real_rmse = rmse

    return {
        "test_mae":       mae,        # scaled — for model comparison
        "test_rmse":      rmse,       # scaled — for model comparison
        "test_mape":      mape,       # real kW — for human interpretation
        "test_real_mae":  real_mae,   # real kW — interpretable error
        "test_real_rmse": real_rmse,  # real kW — interpretable error
    }
# ----------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------

def train_lstm(
    run_name: str = "lstm_baseline",
    save_model: bool = True,
) -> Dict[str, Any]:
    """
    Runs the complete LSTM training pipeline with MLflow tracking.

    This function:
        1. Loads feature data from disk
        2. Creates sliding window sequences
        3. Builds the LSTM model
        4. Trains with callbacks
        5. Evaluates on test set
        6. Logs everything to MLflow
        7. Saves the model
    Args:
        run_name:   A human-readable name for this training run.
                    Shows up in the MLflow UI to identify experiments.
        save_model: Whether to save the trained model to disk.

    Returns:
        Dictionary containing the trained model, history, and metrics.
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("=== Starting LSTM training pipeline ===")

    # ---- Step 1: Load feature data ----
    log.info("Loading feature data from pipeline")
    feature_data = run_feature_pipeline(save=False)

    X_train = feature_data["X_train"]
    X_val   = feature_data["X_val"]
    X_test  = feature_data["X_test"]
    feature_columns = feature_data["feature_columns"]

    # Find the column index of the target variable.
    # The model needs to know which column in X contains
    # Global_active_power — that is what we predict.
    # list.index() returns the position of a value in the list.
    y_index = feature_columns.index("Global_active_power")
    log.info(
        f"Target column 'Global_active_power' is at index {y_index} "
        f"out of {len(feature_columns)} features"
    )

    # ---- Step 2: Create sequences ----
    log.info("Creating training sequences")
    X_train_seq, y_train_seq = create_sequences(X_train, y_index)
    X_val_seq,   y_val_seq   = create_sequences(X_val,   y_index)
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_index)

    log.info(
        f"Sequence shapes — "
        f"Train: {X_train_seq.shape} | "
        f"Val: {X_val_seq.shape} | "
        f"Test: {X_test_seq.shape}"
    )

    n_features = X_train_seq.shape[2]
    # shape[0] = number of sequences
    # shape[1] = sequence length (168)
    # shape[2] = number of features (40)

    # ---- Step 3: Configure MLflow ----
    # Set where MLflow saves its data.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # set_experiment creates the experiment if it does not exist, or uses the existing one if it does.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # ---- Step 4: Start MLflow run ----
    # Everything inside this 'with' block is recorded in one MLflow run.
    # When the block exits, the run is automatically closed.
    with mlflow.start_run(run_name=run_name) as run:
        log.info(f"MLflow run started — ID: {run.info.run_id}")

        # Log all hyperparameters at the start of the run.
        # mlflow.log_params() records a dictionary of settings.
        # These appear in the MLflow UI so you can compare
        # different runs side by side.
        mlflow.log_params({
            "model_type":          "BiLSTM",
            "sequence_length":     SEQUENCE_LENGTH,
            "forecast_horizon":    FORECAST_HORIZON,
            "lstm_units_layer1":   LSTM_UNITS_LAYER1,
            "lstm_units_layer2":   LSTM_UNITS_LAYER2,
            "dropout_rate_1":      DROPOUT_RATE_1,
            "dropout_rate_2":      DROPOUT_RATE_2,
            "learning_rate":       LEARNING_RATE,
            "batch_size":          BATCH_SIZE,
            "max_epochs":          MAX_EPOCHS,
            "early_stop_patience": EARLY_STOPPING_PATIENCE,
            "n_features":          n_features,
            "train_sequences":     len(X_train_seq),
        })

        # ---- Step 5: Build model ----
        model = build_lstm_model(n_features=n_features)

        # ---- Step 6: Define callbacks ----
        callbacks = [
            # EarlyStopping monitors val_loss.
            # If it does not improve for EARLY_STOPPING_PATIENCE epochs,
            # training stops. restore_best_weights=True rolls back to
            # the epoch where val_loss was lowest.
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
                # verbose=1 prints a message when early stopping triggers
            ),

            # ReduceLROnPlateau monitors val_loss.
            # If it stagnates for REDUCE_LR_PATIENCE epochs,
            # the learning rate is multiplied by REDUCE_LR_FACTOR.
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=REDUCE_LR_MIN,
                verbose=1,
            ),

            # Our custom MLflow callback — logs metrics after every epoch.
            MLflowCallback(),
        ]

        # ---- Step 7: Train the model ----
        log.info("Starting model.fit() — training begins")
        history = model.fit(
            X_train_seq,
            y_train_seq,
            # Validation data is passed separately — Keras evaluates
            # on it after every epoch but NEVER trains on it.
            validation_data=(X_val_seq, y_val_seq),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            # verbose=1 shows a progress bar for each epoch.
            # verbose=2 shows one line per epoch (cleaner for long runs).
            verbose=1,
        )
        actual_epochs = len(history.history["loss"])
        log.info(
            f"Training complete — "
            f"ran {actual_epochs} epochs "
            f"(stopped early: {actual_epochs < MAX_EPOCHS})"
        )

        # ---- Step 8: Evaluate on test set ----
        log.info("Evaluating on test set")
        y_pred = model.predict(X_test_seq, verbose=0)
        metrics = compute_evaluation_metrics(
                                        y_test_seq,
                                        y_pred,
                                        scaler=feature_data["scaler"],
                                        y_index=y_index,
                                    )

        log.info(
                f"Test results — "
                f"MAE: {metrics['test_mae']:.4f} (scaled) | "
                f"Real MAE: {metrics['test_real_mae']:.4f} kW | "
                f"RMSE: {metrics['test_rmse']:.4f} (scaled) | "
                f"MAPE: {metrics['test_mape']:.2f}% (real scale)"
            )

        # Log final test metrics to MLflow.
        mlflow.log_metrics(metrics)

        # Also log the best validation loss achieved during training.
        best_val_loss = min(history.history["val_loss"])
        mlflow.log_metric("best_val_loss", best_val_loss)

        # ---- Step 9: Save the model ----
        if save_model:
            LSTM_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(LSTM_MODEL_DIR))

            # Save as TensorFlow SavedModel format.
            # This is the production format — it saves:
            #   - the model architecture
            #   - all trained weights
            #   - the computation graph (compiled for inference)
            log.info(f"Model saved to {LSTM_MODEL_DIR}")
            mlflow.tensorflow.log_model(
                model,
                artifact_path="lstm_model",
            )
            log.info("Model logged to MLflow artifact store")

        log.info("=== LSTM training pipeline complete ===")

        return {
            "model":    model,
            "history":  history,
            "metrics":  metrics,
            "run_id":   run.info.run_id,
        }