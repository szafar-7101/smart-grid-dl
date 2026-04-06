# src/features/config.py
#
# Central configuration for the feature engineering pipeline.
# All constants, paths, and hyperparameters live here.

from pathlib import Path
from src.ingestion.config import PROJECT_ROOT, TARGET_COLUMN

# ----------------------------------------------------------------
# File paths
# ----------------------------------------------------------------

# Input: the clean hourly parquet from the ingestion pipeline
FEATURES_INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "household_power_consumption.parquet"
)

# Output: the full feature matrix — all 40 columns, all rows
FEATURES_OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "feature_matrix.parquet"

# The feature scaler — fitted on training data only, saved for inference
FEATURE_SCALER_FILE = PROJECT_ROOT / "data" / "features" / "feature_scaler.pkl"

# The target scaler — fitted only on training target column
# Separate from feature scaler so y has its own clean scale
TARGET_SCALER_FILE = PROJECT_ROOT / "data" / "features" / "target_scaler.pkl"

# ----------------------------------------------------------------
# Lag feature settings
# ----------------------------------------------------------------
# Each number is a lookback in hours.
# lag_168h (same time last week) is the most powerful single feature
# for weekly-periodic load data.
LAG_HOURS = [1, 24, 48, 168, 336]

# ----------------------------------------------------------------
# Rolling window settings
# ----------------------------------------------------------------
# Windows in hours over which to compute mean, std, max, min
ROLLING_WINDOWS = [24, 168]

# ----------------------------------------------------------------
# Cyclical encoding
# ----------------------------------------------------------------
# Features that are circular — encoded as sine/cosine pairs.
# The value is the maximum of that feature's range.
# Formula: sin(2π × value / max_value), cos(2π × value / max_value)
CYCLICAL_FEATURES = {
    "hour":        23,
    "day_of_week": 6,
    "month":       12,
    "day_of_year": 365,
}

# ----------------------------------------------------------------
# Peak hour definition
# ----------------------------------------------------------------
MORNING_PEAK_HOURS = list(range(7, 10))   # 7am, 8am, 9am
EVENING_PEAK_HOURS = list(range(17, 21))  # 5pm, 6pm, 7pm, 8pm

# ----------------------------------------------------------------
# Train / validation / test split
# ----------------------------------------------------------------
# Chronological split — never random for time series.
# Random splits cause data leakage (future rows appear in training).
TRAIN_RATIO = 0.70  # ~23,977 rows — Dec 2006 to Sep 2009
VAL_RATIO   = 0.15  # ~5,138 rows  — Sep 2009 to Apr 2010
# Test ratio = 1 - 0.70 - 0.15 = 0.15 — Apr 2010 to Nov 2010

# ----------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------
FEATURE_RANGE = (0, 1)  # MinMaxScaler target range

# ----------------------------------------------------------------
# Sequence settings (used by model layer — defined here for reference)
# ----------------------------------------------------------------
# How many past hours the model sees at once
SEQUENCE_LENGTH = 168   # one full week

# How many future hours the model predicts in one shot
FORECAST_HORIZON = 24   # next 24 hours

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
LOGGER_NAME = "smart_grid.features"