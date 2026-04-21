# src/training/train_autoencoder.py

# Training pipeline for the LSTM Autoencoder.


# Run locally: python3 -m src.training.train_autoencoder

import json
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.models.config import (
    FEATURES_FILE,
    AUTOENCODER_DIR,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
    LOGGER_NAME,
)
from src.models.autoencoder.autoencoder import (
    AE_FEATURE_COLUMNS,
    AE_STRIDE,
    THRESHOLD_PERCENTILE,
    AE_MODEL_FILE,
    AE_THRESHOLD_FILE,
    AE_METRICS_FILE,
    AE_HISTORY_FILE,
    AE_SCALER_FILE,
    build_autoencoder,
    filter_normal_data,
    create_ae_sequences,
    compute_reconstruction_errors,
    compute_threshold,
    classify_severity,
)

log = logger.bind(name=LOGGER_NAME)


def train_autoencoder() -> dict:
    """
    Full autoencoder training pipeline:
      1. Load Panama features parquet
      2. Chronological 70/15/15 split
      3. Filter obvious anomalies from training data
      4. Scale features (17 columns including load)
      5. Create overlapping sequences (stride=24)
      6. Build and train the autoencoder
      7. Compute P99 threshold from training reconstruction errors
      8. Evaluate on test set
      9. Save all outputs to models/autoencoder/

    Returns:
        Dictionary with model, threshold_info, metrics
    """
    log.info("=== Starting autoencoder training pipeline ===")

    # ---- Load ----
    log.info(f"Loading {FEATURES_FILE}")
    df = pd.read_parquet(FEATURES_FILE)
    log.info(f"Loaded {len(df):,} rows — {df.shape[1]} columns")

    missing = [c for c in AE_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---- Split ----
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[train_end:val_end].copy()
    test_df  = df.iloc[val_end:].copy()

    log.info(
        f"Split — train: {len(train_df):,} | "
        f"val: {len(val_df):,} | test: {len(test_df):,}"
    )

    # ---- Filter ----
    train_normal = filter_normal_data(train_df)

    # ---- Scale ----
    feature_scaler = MinMaxScaler()
    X_train = feature_scaler.fit_transform(
        train_normal[AE_FEATURE_COLUMNS].values
    ).astype("float32")
    X_val  = feature_scaler.transform(val_df[AE_FEATURE_COLUMNS].values).astype("float32")
    X_test = feature_scaler.transform(test_df[AE_FEATURE_COLUMNS].values).astype("float32")

    n_features = X_train.shape[1]
    log.info(f"Scaled — n_features={n_features}")

    # ---- Sequences (overlapping, stride=24) ----
    X_train_seq, _ = create_ae_sequences(X_train)
    X_val_seq,   _ = create_ae_sequences(X_val)
    X_test_seq,  _ = create_ae_sequences(X_test)

    log.info(
        f"Sequences — train: {X_train_seq.shape} | "
        f"val: {X_val_seq.shape} | test: {X_test_seq.shape}"
    )

    # ---- Build ----
    model = build_autoencoder(n_features=n_features)

    # ---- Train ----
    AUTOENCODER_DIR.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train_seq, X_train_seq,   # input = target
        validation_data=(X_val_seq, X_val_seq),
        epochs=100,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    epochs_ran = len(history.history["loss"])
    best_val   = float(min(history.history["val_loss"]))
    log.info(f"Training done — {epochs_ran} epochs, best val_loss: {best_val:.6f}")

    # ---- Threshold (P99) ----
    train_errors   = compute_reconstruction_errors(model, X_train_seq)
    threshold_info = compute_threshold(train_errors, percentile=THRESHOLD_PERCENTILE)

    log.info(
        f"Threshold (P{int(THRESHOLD_PERCENTILE)}): {threshold_info['threshold']:.6f} "
        f"(mean={threshold_info['mean']:.6f}, std={threshold_info['std']:.6f})"
    )

    # ---- Evaluate ----
    test_errors  = compute_reconstruction_errors(model, X_test_seq)
    threshold    = threshold_info["threshold"]
    std_err      = threshold_info["std"]

    test_flags      = test_errors > threshold
    n_anomalies     = int(test_flags.sum())
    anomaly_rate    = float(test_flags.mean() * 100)
    severities      = [classify_severity(e, threshold, std_err) for e in test_errors]
    severity_counts = {s: severities.count(s) for s in ["normal","low","medium","high"]}

    log.info(
        f"Test — anomaly rate: {anomaly_rate:.2f}% "
        f"({n_anomalies}/{len(test_errors)} sequences) | "
        f"severity: {severity_counts}"
    )

    # ---- Save ----
    model.save(str(AE_MODEL_FILE))
    log.info(f"Model saved → {AE_MODEL_FILE}")

    with open(AE_THRESHOLD_FILE, "w") as f:
        json.dump(threshold_info, f, indent=2)

    metrics = {
        "train_mean_error":      float(train_errors.mean()),
        "train_std_error":       float(train_errors.std()),
        "threshold":             float(threshold),
        "threshold_method":      f"P{int(THRESHOLD_PERCENTILE)}",
        "test_mean_error":       float(test_errors.mean()),
        "test_max_error":        float(test_errors.max()),
        "n_test_anomalies":      n_anomalies,
        "test_anomaly_rate_pct": anomaly_rate,
        "severity_counts":       severity_counts,
        "epochs_trained":        epochs_ran,
        "best_val_loss":         best_val,
    }
    with open(AE_METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(AE_HISTORY_FILE, "w") as f:
        json.dump(history.history, f)

    with open(AE_SCALER_FILE, "wb") as f:
        pickle.dump(feature_scaler, f)

    log.info("=== Autoencoder training complete ===")

    return {"model": model, "threshold_info": threshold_info, "metrics": metrics}


if __name__ == "__main__":
    results = train_autoencoder()
    print("\n--- Autoencoder Results ---")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v}")