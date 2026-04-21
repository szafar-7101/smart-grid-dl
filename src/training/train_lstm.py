# src/training/train_lstm.py
#
# LSTM training pipeline.
#
# Loads the Panama features parquet, splits chronologically,
# scales X and y with separate scalers, creates sequences,
# runs Optuna HPO, trains the final model, evaluates on test set,
# and saves everything to models/lstm/.

import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.models.config import (
    FEATURES_FILE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LSTM_DIR,
    LSTM_MODEL_FILE,
    LSTM_METRICS_FILE,
    LSTM_HISTORY_FILE,
    LSTM_PARAMS_FILE,
    LSTM_SCALER_FILE,
    SEQUENCE_LENGTH,
    FORECAST_HORIZON,
    TRAIN_RATIO,
    VAL_RATIO,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_MIN,
    N_OPTUNA_TRIALS,
    MAX_EPOCHS_HPO,
    EARLY_STOPPING_PATIENCE_HPO,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LOGGER_NAME,
)
from src.models.lstm.lstm import build_lstm_model, create_sequences


log = logger.bind(name=LOGGER_NAME)


# ----------------------------------------------------------------
# Data loading and splitting
# ----------------------------------------------------------------

def load_and_split() -> dict:
    """
    Loads panama_features.parquet, splits chronologically 70/15/15,
    scales X with a feature scaler and y with a separate target scaler.

    Returns a dictionary with everything the training loop needs:
    scaled arrays, scalers, split DataFrames, and column info.
    """
    log.info(f"Loading {FEATURES_FILE}")
    df = pd.read_parquet(FEATURES_FILE)
    log.info(f"Loaded {len(df):,} rows — columns: {list(df.columns)}")

    # Validate that expected columns are present
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in parquet: {missing}")

    # Chronological split — never random for time series
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    log.info(
        f"Split — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )

    # Feature scaler — fitted on training X only
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(
        train_df[FEATURE_COLUMNS].values
    ).astype(np.float32)
    X_val  = feature_scaler.transform(val_df[FEATURE_COLUMNS].values).astype(np.float32)
    X_test = feature_scaler.transform(test_df[FEATURE_COLUMNS].values).astype(np.float32)

    # Target scaler — fitted on training y only
    # reshape(-1, 1) because MinMaxScaler expects 2D input
    target_scaler = MinMaxScaler()
    y_train = target_scaler.fit_transform(
        train_df[TARGET_COLUMN].values.reshape(-1, 1)
    ).flatten().astype(np.float32)
    y_val  = target_scaler.transform(
        val_df[TARGET_COLUMN].values.reshape(-1, 1)
    ).flatten().astype(np.float32)
    y_test = target_scaler.transform(
        test_df[TARGET_COLUMN].values.reshape(-1, 1)
    ).flatten().astype(np.float32)

    # Save target scaler — needed at inference time to reverse predictions
    LSTM_DIR.mkdir(parents=True, exist_ok=True)
    with open(LSTM_SCALER_FILE, "wb") as f:
        pickle.dump(target_scaler, f)
    log.info(f"Target scaler saved to {LSTM_SCALER_FILE}")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "target_scaler": target_scaler,
        "n_features": len(FEATURE_COLUMNS),
    }


# ----------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_scaler) -> dict:
    """
    Computes MAE, RMSE, and sMAPE on real-world (inverse-transformed) scale.
    Both y_true and y_pred are in scaled [0,1] space on input.
    """
    real_true = target_scaler.inverse_transform(
        y_true.flatten().reshape(-1, 1)
    ).flatten()
    real_pred = target_scaler.inverse_transform(
        y_pred.flatten().reshape(-1, 1)
    ).flatten()
    real_pred = np.clip(real_pred, 0, None)

    mae   = float(np.mean(np.abs(real_true - real_pred)))
    rmse  = float(np.sqrt(np.mean((real_true - real_pred) ** 2)))
    smape = float(np.mean(
        2 * np.abs(real_true - real_pred) /
        (np.abs(real_true) + np.abs(real_pred) + 1e-8)
    ) * 100)

    return {"mae": mae, "rmse": rmse, "smape": smape}


# ----------------------------------------------------------------
# Optuna objective
# ----------------------------------------------------------------

def make_objective(X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_features):
    """
    Returns the Optuna objective function.
    Wraps training data in a closure so the function signature
    matches what Optuna expects: f(trial) → float.
    """
    def objective(trial):
        lstm_units_1 = trial.suggest_categorical("lstm_units_1", [64, 128, 256])
        lstm_units_2 = trial.suggest_categorical("lstm_units_2", [32, 64, 128])
        dropout_1    = trial.suggest_float("dropout_1", 0.1, 0.5, step=0.1)
        dropout_2    = trial.suggest_float("dropout_2", 0.1, 0.4, step=0.1)
        # Log-scale search: finer resolution near small values
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = build_lstm_model(
            n_features=n_features,
            lstm_units_1=lstm_units_1,
            lstm_units_2=lstm_units_2,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
            learning_rate=learning_rate,
        )

        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=MAX_EPOCHS_HPO,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOPPING_PATIENCE_HPO,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
            verbose=0,
        )

        best_val_mae = min(history.history["val_mae"])
        keras.backend.clear_session()
        return best_val_mae

    return objective


# ----------------------------------------------------------------
# Warmup callback
# ----------------------------------------------------------------

class WarmupLR(keras.callbacks.Callback):
    """
    Linearly ramps learning rate from lr/10 to lr over warmup_epochs.
    Prevents destructively large updates at the start of training
    before the model has a sense of the loss landscape.
    """
    def __init__(self, target_lr: float, warmup_epochs: int = 5):
        self.target_lr     = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * ((epoch + 1) / self.warmup_epochs)
            keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        elif epoch == self.warmup_epochs:
            keras.backend.set_value(
                self.model.optimizer.learning_rate, self.target_lr
            )


# ----------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------

def train(run_hpo: bool = True) -> dict:
    """
    Full training pipeline:
      1. Load and split data
      2. Create sequences
      3. Run Optuna HPO (if run_hpo=True)
      4. Train final model with best hyperparameters
      5. Evaluate on test set
      6. Save model, metrics, history to models/lstm/

    Args:
        run_hpo: Set False to skip HPO and use config defaults.
                 Useful for quick debugging runs.

    Returns:
        Dictionary with model, metrics, and best_params.
    """
    log.info("=== Starting LSTM training pipeline ===")

    # Step 1 — Load data
    data = load_and_split()
    n_features = data["n_features"]

    # Step 2 — Create sequences
    log.info("Creating sequences")
    X_train_seq, y_train_seq = create_sequences(data["X_train"], data["y_train"])
    X_val_seq,   y_val_seq   = create_sequences(data["X_val"],   data["y_val"])
    X_test_seq,  y_test_seq  = create_sequences(data["X_test"],  data["y_test"])

    log.info(
        f"Sequence shapes — "
        f"train: {X_train_seq.shape} | "
        f"val: {X_val_seq.shape} | "
        f"test: {X_test_seq.shape}"
    )

    # Step 3 — Optuna HPO
    if run_hpo:
        log.info(f"Running Optuna HPO — {N_OPTUNA_TRIALS} trials")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            make_objective(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                n_features,
            ),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
        best_params = study.best_params
        log.info(f"Best val MAE: {study.best_value:.4f}")
        log.info(f"Best params: {best_params}")
    else:
        # Use config defaults
        best_params = {
            "lstm_units_1": 128,
            "lstm_units_2": 64,
            "dropout_1":    0.3,
            "dropout_2":    0.2,
            "learning_rate": 0.0003,
            "batch_size":   32,
        }
        log.info(f"Skipping HPO — using defaults: {best_params}")

    # Save best params
    LSTM_DIR.mkdir(parents=True, exist_ok=True)
    with open(LSTM_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)

    # Step 4 — Train final model
    log.info("Training final model with best hyperparameters")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lstm_final"):
        mlflow.log_params(best_params)

        model = build_lstm_model(
            n_features=n_features,
            lstm_units_1=best_params["lstm_units_1"],
            lstm_units_2=best_params["lstm_units_2"],
            dropout_1=best_params["dropout_1"],
            dropout_2=best_params["dropout_2"],
            learning_rate=best_params["learning_rate"],
        )

        callbacks = [
            WarmupLR(target_lr=best_params["learning_rate"], warmup_epochs=5),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=REDUCE_LR_MIN,
                verbose=1,
            ),
        ]

        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=MAX_EPOCHS,
            batch_size=best_params["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )

        epochs_ran  = len(history.history["loss"])
        best_val    = min(history.history["val_loss"])
        log.info(f"Training done — {epochs_ran} epochs, best val_loss: {best_val:.4f}")

        # Step 5 — Evaluate
        y_pred = model.predict(X_test_seq, verbose=0)
        metrics = compute_metrics(y_test_seq, y_pred, data["target_scaler"])
        metrics["best_val_loss"] = best_val
        metrics["epochs_trained"] = epochs_ran

        log.info(
            f"Test results — "
            f"MAE: {metrics['mae']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f} | "
            f"sMAPE: {metrics['smape']:.2f}%"
        )

        mlflow.log_metrics({
            "test_mae":   metrics["mae"],
            "test_rmse":  metrics["rmse"],
            "test_smape": metrics["smape"],
        })

        # Step 6 — Save
        model.save(str(LSTM_MODEL_FILE))
        log.info(f"Model saved to {LSTM_MODEL_FILE}")

        with open(LSTM_METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        with open(LSTM_HISTORY_FILE, "w") as f:
            json.dump(history.history, f)

        log.info("=== Training pipeline complete ===")

    return {"model": model, "metrics": metrics, "best_params": best_params}


if __name__ == "__main__":
    results = train(run_hpo=True)
    print("\n--- Final Results ---")
    print(f"MAE:   {results['metrics']['mae']:.4f}")
    print(f"RMSE:  {results['metrics']['rmse']:.4f}")
    print(f"sMAPE: {results['metrics']['smape']:.2f}%")