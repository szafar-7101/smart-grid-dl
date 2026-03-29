"""
This file holds ALL the configuration for the ingestion pipeline
Nothing in pipeline.py will have harcoded values
Every setting lives here so it can be changed in one place 
"""

from pathlib import Path 

# Project Root Path 
#------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
#------------------
# These are the paths to each stage of data
# We build them relative to PROJECT_ROOT so they work on any machine
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"

# The specific raw file we want to ingest that we have placed in data/raw
RAW_DATA_FILE = RAW_DATA_DIR / "household_power_consumption.txt"

# The output file after cleaning - saved as paraquet format
# We will use it for all intermediate data.
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "household_power_consumption.parquet"

# ----------------------------------------------------------------
# Dataset-specific settings
# ----------------------------------------------------------------
# The UCI dataset uses semicolons to separate columns, not commas.
# Our loader needs to know this.
DATA_SEPARATOR = ";"

# The name of the column that contains timestamps in the raw file.
# The UCI dataset splits date and time into two separate columns.
DATE_COLUMN = "Date"
TIME_COLUMN = "Time"

# After we combine Date + Time into one column, we call it this.
DATETIME_COLUMN = "datetime"

# The column we are forecasting — global active power in kilowatts.
# This is our "load" signal — the main thing we care about.
TARGET_COLUMN = "Global_active_power"

# All the numeric columns in the dataset.
# We will try to convert all of these to float numbers.
# The UCI dataset sometimes has "?" as missing values — we handle that.
NUMERIC_COLUMNS = [
    "Global_active_power",    # Total power drawn by the household (kilowatts)
    "Global_reactive_power",  # Reactive power — relates to voltage/current phase (kilowatts)
    "Voltage",                # Household voltage (volts)
    "Global_intensity",       # Current intensity (amperes)
    "Sub_metering_1",         # Energy in kitchen (watt-hours)
    "Sub_metering_2",         # Energy in laundry room (watt-hours)
    "Sub_metering_3",         # Energy in climate control (watt-hours)
]

# ----------------------------------------------------------------
# Resampling settings
# ----------------------------------------------------------------
# The raw data is recorded every minute.
# We resample to hourly averages for two reasons:
#   1. Deep learning on minute-level data needs far more compute
#   2. Hourly patterns are what grid operators actually care about
RESAMPLE_FREQUENCY = "h"  # "h" means hourly in pandas

# ----------------------------------------------------------------
# Data quality thresholds
# ----------------------------------------------------------------
# If more than this percentage of the target column is missing,
# we raise an error rather than silently training on bad data.
MAX_MISSING_PERCENTAGE = 0.10  # 10%

# Physically impossible values — power cannot be negative,
# and anything above this is almost certainly a sensor error.
MIN_VALID_POWER = 0.0      # kilowatts — cannot be negative
MAX_VALID_POWER = 20.0     # kilowatts — physically unreasonable above this

# ----------------------------------------------------------------
# Logging settings
# ----------------------------------------------------------------
# The name of the logger used throughout the ingestion module.
# Using a named logger (instead of the root logger) means log messages
# from ingestion are clearly labelled in your log output.
LOGGER_NAME = "smart_grid.ingestion"
