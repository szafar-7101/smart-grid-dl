
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ----------------------------------------------------------------
# Data paths
# ----------------------------------------------------------------
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
FEATURES_FILE = DATA_DIR / "panama_features.parquet"

# ----------------------------------------------------------------
# Model storage paths
# ----------------------------------------------------------------
MODELS_DIR       = Path(__file__).resolve().parent   # src/models/
LSTM_DIR         = MODELS_DIR / "lstm"
TRANSFORMER_DIR  = MODELS_DIR / "transformer"
AUTOENCODER_DIR  = MODELS_DIR / "autoencoder"

LSTM_MODEL_FILE   = LSTM_DIR / "lstm_model.keras"
LSTM_METRICS_FILE = LSTM_DIR / "metrics.json"
LSTM_HISTORY_FILE = LSTM_DIR / "history.json"
LSTM_PARAMS_FILE  = LSTM_DIR / "best_params.json"
LSTM_SCALER_FILE  = LSTM_DIR / "target_scaler.pkl"
LSTM_FEATURE_SCALER_FILE = LSTM_DIR / "feature_scaler.pkl"

# ----------------------------------------------------------------
# Feature definitions
# Exactly 16 features — wind_speed and precipitation removed
# after notebook experimentation showed they hurt performance
# ----------------------------------------------------------------
FEATURE_COLUMNS = [
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
]

TARGET_COLUMN = "load"

# ----------------------------------------------------------------
# Sequence parameters
# ----------------------------------------------------------------
SEQUENCE_LENGTH  = 168   # one full week of hourly data
FORECAST_HORIZON = 24    # predict next 24 hours

# ----------------------------------------------------------------
# Final trained hyperparameters
# These are the best_params found by Optuna on Kaggle
# ----------------------------------------------------------------
LSTM_UNITS_1  = 128
LSTM_UNITS_2  = 64
DROPOUT_1     = 0.4
DROPOUT_2     = 0.3
LEARNING_RATE = 5e-4
BATCH_SIZE    = 128
HUBER_DELTA   = 1.0   # threshold between MSE and MAE behaviour in Huber loss

# ----------------------------------------------------------------
# Training settings
# ----------------------------------------------------------------
MAX_EPOCHS              = 100
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE      = 5
REDUCE_LR_FACTOR        = 0.5
REDUCE_LR_MIN           = 1e-7

# ----------------------------------------------------------------
# Optuna HPO settings
# ----------------------------------------------------------------
N_OPTUNA_TRIALS             = 20
MAX_EPOCHS_HPO              = 25
EARLY_STOPPING_PATIENCE_HPO = 5

# ----------------------------------------------------------------
# Split ratios — chronological, never random
# ----------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# ----------------------------------------------------------------
# Final test results (for reference and documentation)
# ----------------------------------------------------------------
FINAL_TEST_MAE   = 72.9182   # MW
FINAL_TEST_RMSE  = 97.5648   # MW
FINAL_TEST_SMAPE = 6.00      # percent

# ================================================================
# AUTOENCODER — ANOMALY DETECTION
# Matches the final trained configuration from Kaggle notebook
# lstm-autoencoder.ipynb
# ================================================================

# ----------------------------------------------------------------
# Autoencoder file paths
# ----------------------------------------------------------------
AE_MODEL_FILE        = AUTOENCODER_DIR / "ae_model.keras"
AE_THRESHOLD_FILE    = AUTOENCODER_DIR / "threshold.json"
AE_METRICS_FILE      = AUTOENCODER_DIR / "ae_metrics.json"
AE_HISTORY_FILE      = AUTOENCODER_DIR / "ae_history.json"
AE_FEATURE_SCALER    = AUTOENCODER_DIR / "ae_feature_scaler.pkl"

# ----------------------------------------------------------------
# Autoencoder feature columns
# 17 features — all 16 input features PLUS the target column
# The autoencoder reconstructs the full signal including load
# This is different from the forecasting LSTM which excludes load from X
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
    "load",             # target included — AE reconstructs full signal
]

AE_N_FEATURES = len(AE_FEATURE_COLUMNS)   # 17

# ----------------------------------------------------------------
# Architecture hyperparameters
# These produced best_val_loss: 0.196980 after 26 epochs
# ----------------------------------------------------------------
AE_ENCODER_UNITS_1  = 128   # first encoder LSTM
AE_ENCODER_UNITS_2  = 64    # second encoder LSTM
AE_BOTTLENECK_UNITS = 32    # bottleneck — most compressed representation
AE_DECODER_UNITS_1  = 64    # first decoder LSTM (mirrors encoder_2)
AE_DECODER_UNITS_2  = 128   # second decoder LSTM (mirrors encoder_1)
AE_DROPOUT_RATE     = 0.2
AE_LEARNING_RATE    = 1e-3

# ----------------------------------------------------------------
# Sequence settings for autoencoder
# Same sequence length as LSTM but different stride
# ----------------------------------------------------------------
AE_SEQUENCE_LENGTH = 24     # 24h window — matches the trained model (ae_metrics.json)
AE_STRIDE          = 24     # stride=24 → non-overlapping windows for the AE

# ----------------------------------------------------------------
# Data filtering — remove obvious anomalies before training
# so model learns only from normal patterns
# ----------------------------------------------------------------
AE_FILTER_Z_THRESHOLD = 3.0  # rows beyond 3σ of 24h rolling mean removed
                              # removed 24 rows (0.1%) from training data

# ----------------------------------------------------------------
# Anomaly threshold settings
# ---------------------------------------------------------

# ----------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------
MLFLOW_TRACKING_URI    = str(PROJECT_ROOT / "mlruns")
MLFLOW_EXPERIMENT_NAME = "smart_grid_lstm"

LOGGER_NAME = "smart_grid.lstm"