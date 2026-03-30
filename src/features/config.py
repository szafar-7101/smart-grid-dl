# src/features/config.py
#
# Configuration for the feature engineering module.
# All constants live here — nothing is hardcoded in engineering.py.

from pathlib import Path
from src.ingestion.config import PROJECT_ROOT, TARGET_COLUMN

# ----------------------------------------------------------------
# File paths
# ----------------------------------------------------------------

# Input: the clean hourly data produced by the ingestion pipeline
FEATURES_INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "power_consumption_clean.parquet"

# Output: the feature matrix ready for model input
FEATURES_OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "feature_matrix.parquet"

"""
 The scaler object is saved here so we can reuse it during inference.
 We must use the SAME scaler that was fitted on training data —
 we cannot refit it on new data at inference time.
"""
SCALER_OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "scaler.pkl"

# ----------------------------------------------------------------
# Lag feature settings
# ----------------------------------------------------------------
# These are the time offsets (in hours) we look back for lag features.
# Each number means: "what was the load N hours ago?"
LAG_HOURS = [1, 24, 48, 168, 336]


# ----------------------------------------------------------------
# Rolling window settings
# ----------------------------------------------------------------
# These define the sizes of the sliding windows for rolling statistics.
ROLLING_WINDOWS = [24, 168]


# ----------------------------------------------------------------
# Cyclical encoding settings
# ----------------------------------------------------------------
# These define which calendar features to encode as sine/cosine pairs
# and what their maximum value is (needed for the encoding formula).
CYCLICAL_FEATURES = {
    "hour":        23,   # hours go 0–23, so max is 23
    "day_of_week": 6,    # days go 0–6 (Mon=0, Sun=6)
    "month":       12,   # months go 1–12
    "day_of_year": 365,  # days in a year
}

# ----------------------------------------------------------------
# Peak hour definition
# ----------------------------------------------------------------
MORNING_PEAK_HOURS = list(range(7, 10))   # 7am, 8am, 9am
EVENING_PEAK_HOURS = list(range(17, 21))  # 5pm, 6pm, 7pm, 8pm

# ----------------------------------------------------------------
# Train/validation/test split ratios
# ----------------------------------------------------------------
# 70% train → 15% validation → 15% test
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# Test ratio is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15

# ----------------------------------------------------------------
# Scaling
# ----------------------------------------------------------------
# The range to scale all features to.
# (0, 1) is standard for neural networks.
FEATURE_RANGE = (0, 1)

# ----------------------------------------------------------------
# Sequence settings
# ----------------------------------------------------------------
# For LSTM and Transformer models, we feed sequences of past timesteps.
# This is how many hours back the model can "see" at once.
# 168 = one full week of hourly data.
SEQUENCE_LENGTH = 168

# How many hours ahead we are predicting.
# 24 = next 24 hours (one day ahead forecast)
FORECAST_HORIZON = 24

# ----------------------------------------------------------------
# Logging
# ----------------------------------------------------------------
LOGGER_NAME = "smart_grid.features"