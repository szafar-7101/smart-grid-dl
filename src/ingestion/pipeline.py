# This is the ingestion pipeline 

    # It does 4 things in order:
    #  1. LOAD - Read the raw CSV from the disk into a pandas DATAFRAME
    #  2. VALIDATE - Check that the data has the right columns and types, and that there are no missing values
    #  3. CLEAN - Fix missing values, convert types, remove impossible readings
    #  4. RESAMPLE - Aggregate minute-level data into hourly averages

    # Each of these is a separate function. The run_pipeline() function at the 
    # bottom calls them in order - that is the only public entry point 

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional 

from src.ingestion.config import (
    DATA_SEPARATOR,
    DATE_COLUMN,
    RAW_DATA_FILE,
    TIME_COLUMN,
    DATETIME_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    RESAMPLE_FREQUENCY,
    MAX_MISSING_PERCENTAGE,
    MIN_VALID_POWER,
    MAX_VALID_POWER,
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    LOGGER_NAME,
)

# ----------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------
class DataIngestionError(Exception):
    """
    Raised when the ingestion pipeline encounters an unrevoverable error
    """
    pass

class DataValidationError(Exception):
    """
    Raised when the data fails validation checks
    """
    pass

# ----------------------------------------------------------------
# Step 1: Load
# ----------------------------------------------------------------
def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Reads the raw UCI household power consumption data from a CSV file into a pandas DataFrame.

    Args:
        file_path (Path): The path to the raw .txt file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    Raises:
        DataIngestionError: If the file does not exist or cannot be parsed.
    """
    log = logger.bind(name=LOGGER_NAME)
    # Check if the file actually exists or not 
    if not filepath.exists():
        raise DataIngestionError(
            f"Raw data file not found at : {filepath}\n"
            f"Please place household_power_consumption.txt in the 'data/raw' directory and try again."
        )
    log.info(f"Loading raw data from {filepath}...")

    try:
        df = pd.read_csv(
            filepath,
            sep= DATA_SEPARATOR,
            low_memory=False,
            na_values=["?"]
        )
    except Exception as e:
        raise DataIngestionError(f"Failed to read file: {e}") from e
    log.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    return df


# ----------------------------------------------------------------
# Step 2: Validate
# ----------------------------------------------------------------
def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Checks that the raw DataFrame has the expected structure.
    Raises DataValidationError if anything critical is wrong.

    Args:
        df (pd.DataFrame): The raw DataFrame from load_raw_data()

    Raises:
        DataValidationError: If the DataFrame is missing expected columns or has too many missing values.
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Validating raw data")

    # Check 1 : Check if the loaded dataframe is empty or not
    if df.empty:
        raise DataValidationError("The Loaded DataFrame is empty. Please check the raw data file.")
    # Check 2 : Required Columns must be present 
    required_columns = {DATE_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    missing_columns  = required_columns - set(df.columns)
    
    if missing_columns:
        raise DataValidationError(
            f"Required columns are missing from the raw data: {missing_columns}\n"
            f"Columns found : {list(df.columns)}"
        )
    
    # Check 3 : The target column must not be entierly missing 
    target_missing_percentage = df[TARGET_COLUMN].isna().mean()
    if target_missing_percentage > MAX_MISSING_PERCENTAGE:
        raise DataValidationError(
            f"Target Column '{TARGET_COLUMN}' is {target_missing_percentage:.2%} missing"
            f" MAXIMUM Allowed is {MAX_MISSING_PERCENTAGE:.2%}.\n"
            f"Please check the raw data file for completeness."
        )
    
    log.info(
        f"Validation passed -"
        f"{target_missing_percentage:.2%} of target column is missing, which is within the allowed threshold."
    )

# ----------------------------------------------------------------
# Step 3: Clean
# ----------------------------------------------------------------
def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame into a clean , typed time series.
    Steps performed:
      - Combines Date and Time columns into a single datetime index
      - Converts all numeric columns to float
      - Replaces physically impossible values with NaN
      - Fills missing values using forward-fill then backward-fill
      - Sorts by time
    
      Args:
        df (pd.DataFrame): The raw DataFrame from load_raw_data()
    Returns:
        A clean DataFrame with a datetime index and float columns 
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Cleaning Data")

    # --- Combine Date + Time into a single datetime column ---
    # The UCI dataset has "16/12/2006" in Date and "17:24:00" in Time.
    # We combine them into "16/12/2006 17:24:00" then parse that as a datetime.
    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN] + " " + df[TIME_COLUMN],
        dayfirst=True,
        errors="coerce",
        # dayfirst=True tells pandas the format is DD/MM/YYYY (European format)
        # Without this, pandas might interpret "16/12/2006" as month 16, which fails
    )

    # Set the datetime column as the index of the DataFrame.
    # A time series DataFrame should always be indexed by time —
    # this enables time-based operations like resampling.
    df = df.set_index(DATETIME_COLUMN)

    # Drop the original Date and Time columns — we no longer need them
    df = df.drop(columns=[DATE_COLUMN, TIME_COLUMN], errors="ignore")

    # --- Convert numeric columns to float ---
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
     # --- Replace physically impossible values with NaN ---
    # A power reading below 0 or above 20kW is a sensor error.
    # We replace these with NaN so they get filled in the next step,
    # rather than corrupting our model with impossible inputs.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].where(
        (df[TARGET_COLUMN] >= MIN_VALID_POWER) &
        (df[TARGET_COLUMN] <= MAX_VALID_POWER),
        other=np.nan,
    )

    # ---- Sort by time ----
    df = df.sort_index()

  # --- Fill missing values ---
    # Forward fill: replace NaN with the most recent valid value.
    # This is appropriate for sensor data — if a reading is missing,
    # the best estimate is the last known reading.
    # limit=5 means we only forward-fill up to 5 consecutive missing values.
    # A gap longer than 5 minutes likely means a real outage, not a sensor glitch.
    df = df.ffill(limit=5)
    # Backward fill: handle any remaining NaN at the very start of the data
    # (where forward fill has nothing to look back at).
    df = df.bfill(limit=5)

    # Number of missing values in the target column after cleaning
    missing_after = df[TARGET_COLUMN].isna().sum()
    log.info(
        f"Cleaning Complete - "
        f"{missing_after:,} remaining missing values in target column after cleaning."
    )
    return df

# ----------------------------------------------------------------
# Step 4: Resample
# ----------------------------------------------------------------

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates minute-level data to hourly averages.

    The raw data has one row per minute (~2 million rows).
    After resampling, we have one row per hour (~35,000 rows).
    This is the right resolution for our forecasting models.

    Args:
        df: Clean DataFrame with a datetime index, minute resolution

    Returns:
        DataFrame with hourly resolution
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Resampling from minute-level to {RESAMPLE_FREQUENCY} frequency") 
    # Resample frequency is hourly ("h") as defined in config.py

    rows_before = len(df)
    df_resampled = df.resample(RESAMPLE_FREQUENCY).mean()

    # After resampling, any hour with no data at all becomes a row of NaN.
    # We fill these with forward fill.
    df_resampled = df_resampled.ffill()

    rows_after = len(df_resampled)
    log.info(
        f"Resampled from {rows_before:,} rows (minute) "
        f"to {rows_after:,} rows (hourly)"
    )

    return df_resampled

# ----------------------------------------------------------------
# Step 5: Save
# ----------------------------------------------------------------
def save_processed_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Saves the processed DataFrame to disk as a Parquet file.

    All intermediate data in this project is saved as Parquet.

    Args:
        df: The processed DataFrame to save
        filepath: Where to save it
    """
    log = logger.bind(name=LOGGER_NAME)

    # Create the directory if it doesn't exist yet.
    # exist_ok=True means: don't raise an error if it already exists.
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(filepath)
    # .to_parquet() saves the DataFrame including its index (our datetime index)
    # and all column types. When we load it back, everything is preserved exactly.

    log.info(f"Saved processed data to {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------
def run_pipeline(
    input_filepath: Optional[Path] = None,
    output_filepath: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Runs the complete ingestion pipeline end to end.

    This is the only function you should call from outside this module.
    It runs load → validate → clean → resample → save in order.

    Args:
        input_filepath:  Path to the raw data file.
                         Defaults to RAW_DATA_FILE from config.
        output_filepath: Path to save the processed file.
                         Defaults to PROCESSED_DATA_FILE from config.
        save:            Whether to save the result to disk.
                         Set to False during testing.

    Returns:
        The final processed DataFrame ready for feature engineering.
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("=== Starting ingestion pipeline ===")

    # Use config defaults if no paths were passed in
    input_filepath = input_filepath or RAW_DATA_FILE
    output_filepath = output_filepath or PROCESSED_DATA_FILE

    # from config — imported at the top
    from src.ingestion.config import PROCESSED_DATA_FILE, RAW_DATA_FILE

    input_filepath = input_filepath or RAW_DATA_FILE
    output_filepath = output_filepath or PROCESSED_DATA_FILE

    # Run each stage in order, passing the output of one into the next
    df_raw = load_raw_data(input_filepath)
    validate_raw_data(df_raw)
    df_clean = clean_data(df_raw)
    df_hourly = resample_to_hourly(df_clean)

    if save:
        save_processed_data(df_hourly, output_filepath)

    log.info(
        f"=== Pipeline complete — "
        f"{len(df_hourly):,} hourly rows ready for feature engineering ==="
    )

    return df_hourly
