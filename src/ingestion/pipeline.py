# Data ingestion pipeline — stage 1 of the project.
#
# Four stages run in order:
#   1. load_raw_data()       — reads the raw CSV file
#   2. validate_raw_data()   — checks the data has the right shape
#   3. clean_data()          — fixes types, removes impossible values
#   4. resample_to_hourly()  — aggregates minute data to hourly
#
# Public entry point: run_pipeline()
# Call this and you get a clean hourly DataFrame ready for features.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from src.ingestion.config import (
    DATA_SEPARATOR,
    DATE_COLUMN,
    TIME_COLUMN,
    DATETIME_COLUMN,
    TARGET_COLUMN,
    NUMERIC_COLUMNS,
    RESAMPLE_FREQUENCY,
    MAX_MISSING_PERCENTAGE,
    MIN_VALID_POWER,
    MAX_VALID_POWER,
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    PROCESSED_DATA_DIR,
    LOGGER_NAME,
)


# ----------------------------------------------------------------
# Custom exceptions
# ----------------------------------------------------------------

class DataIngestionError(Exception):
    """Raised when ingestion encounters an unrecoverable error."""
    pass


class DataValidationError(DataIngestionError):
    """Raised when data fails a quality check."""
    pass


# ----------------------------------------------------------------
# Stage 1 — Load
# ----------------------------------------------------------------

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """
    Reads the raw UCI household power consumption file into a DataFrame.

    The file is semicolon-separated and uses "?" for missing values.
    We tell pandas both of these things upfront so it handles them
    correctly during loading rather than requiring post-processing.

    Args:
        filepath: Path to the raw .txt file

    Returns:
        Raw DataFrame with all original columns as strings/NaN

    Raises:
        DataIngestionError: If the file does not exist or cannot be parsed
    """
    log = logger.bind(name=LOGGER_NAME)

    if not filepath.exists():
        raise DataIngestionError(
            f"Raw data file not found at: {filepath}\n"
            f"Please place household_power_consumption.txt in data/raw/"
        )

    log.info(f"Loading raw data from {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=DATA_SEPARATOR,
            # low_memory=False: read the whole file before inferring types
            # Without this pandas sometimes guesses wrong column types
            low_memory=False,
            # na_values: treat "?" as missing — the UCI dataset convention
            na_values=["?"],
        )
    except Exception as e:
        raise DataIngestionError(f"Failed to read file: {e}") from e

    log.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


# ----------------------------------------------------------------
# Stage 2 — Validate
# ----------------------------------------------------------------

def validate_raw_data(df: pd.DataFrame) -> None:
    """
    Checks the raw DataFrame has the expected structure.
    Raises DataValidationError if anything critical is wrong.
    Returns nothing — its only job is to raise errors early.

    Args:
        df: Raw DataFrame from load_raw_data()

    Raises:
        DataValidationError: If the data is empty, missing columns,
                             or has too many missing target values
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Validating raw data structure")

    if df.empty:
        raise DataValidationError(
            "The loaded DataFrame is empty — the file may be corrupted"
        )

    required_columns = {DATE_COLUMN, TIME_COLUMN, TARGET_COLUMN}
    missing_columns  = required_columns - set(df.columns)
    if missing_columns:
        raise DataValidationError(
            f"Required columns missing from data: {missing_columns}\n"
            f"Columns found: {list(df.columns)}"
        )

    target_missing_pct = df[TARGET_COLUMN].isna().mean()
    if target_missing_pct > MAX_MISSING_PERCENTAGE:
        raise DataValidationError(
            f"Target column '{TARGET_COLUMN}' is {target_missing_pct:.1%} missing. "
            f"Maximum allowed is {MAX_MISSING_PERCENTAGE:.1%}."
        )

    log.info(
        f"Validation passed — "
        f"{target_missing_pct:.1%} missing in target column"
    )


# ----------------------------------------------------------------
# Stage 3 — Clean
# ----------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw DataFrame into a clean typed time series.

    Steps:
      1. Combine Date + Time into a single datetime index
      2. Convert all numeric columns from strings to float
      3. Replace physically impossible readings with NaN
      4. Sort chronologically
      5. Forward-fill then backward-fill missing values

    Args:
        df: Raw DataFrame from load_raw_data()

    Returns:
        Clean DataFrame with a DatetimeIndex and float columns
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Cleaning Data")

    df = df.copy()

    # Combine "16/12/2006" + "17:24:00" into one datetime column
    df[DATETIME_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN] + " " + df[TIME_COLUMN],
        dayfirst=True,   # European date format: DD/MM/YYYY
        errors="coerce", # Unparseable values become NaT instead of crashing
    )

    df = df.set_index(DATETIME_COLUMN)
    df = df.drop(columns=[DATE_COLUMN, TIME_COLUMN], errors="ignore")

    # Convert all numeric columns from string to float
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace physically impossible power readings with NaN
    # .where(condition) keeps values where True, replaces with NaN where False
    df[TARGET_COLUMN] = df[TARGET_COLUMN].where(
        (df[TARGET_COLUMN] >= MIN_VALID_POWER) &
        (df[TARGET_COLUMN] <= MAX_VALID_POWER),
        other=np.nan,
    )

    df = df.sort_index()

    # Forward fill: replace NaN with the last known value
    # limit=5: only fill gaps of up to 5 consecutive missing minutes
    # Longer gaps likely represent real outages — leave them as NaN
    df = df.ffill(limit=5)
    df = df.bfill(limit=5)

    missing_after = df[TARGET_COLUMN].isna().sum()
    log.info(
        f"Cleaning Complete - {missing_after:,} remaining missing values "
        f"in target column after cleaning."
    )
    return df


# ----------------------------------------------------------------
# Stage 4 — Resample
# ----------------------------------------------------------------

def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates minute-level data to hourly averages.

    Raw data: ~2,075,259 rows (one per minute)
    After resampling: ~34,589 rows (one per hour)

    .resample("h") groups rows by hour.
    .mean() takes the average of each group.
    Any hour with no data at all becomes NaN — we forward-fill those.

    Args:
        df: Clean DataFrame with a DatetimeIndex at minute resolution

    Returns:
        DataFrame with hourly resolution
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Resampling from minute-level to {RESAMPLE_FREQUENCY} frequency")

    rows_before  = len(df)
    df_resampled = df.resample(RESAMPLE_FREQUENCY).mean()
    df_resampled = df_resampled.ffill()
    rows_after   = len(df_resampled)

    log.info(
        f"Resampled from {rows_before:,} rows (minute) "
        f"to {rows_after:,} rows (hourly)"
    )
    return df_resampled


# ----------------------------------------------------------------
# Stage 5 — Save
# ----------------------------------------------------------------

def save_processed_data(df: pd.DataFrame, filepath: Path) -> None:
    """
    Saves the processed DataFrame to disk as a Parquet file.

    Parquet is a compressed binary format — much faster to load than CSV
    for large files and preserves column types and the datetime index exactly.

    Args:
        df: Processed DataFrame to save
        filepath: Where to save it
    """
    log = logger.bind(name=LOGGER_NAME)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath)
    log.info(
        f"Saved processed data to {filepath} "
        f"({filepath.stat().st_size / 1024:.1f} KB)"
    )


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------

def run_pipeline(
    input_filepath: Optional[Path]  = None,
    output_filepath: Optional[Path] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Runs the complete ingestion pipeline end to end.

    This is the only function you should call from outside this module.
    It chains load → validate → clean → resample → save in order.

    Args:
        input_filepath:  Path to raw data. Defaults to RAW_DATA_FILE.
        output_filepath: Path to save result. Defaults to PROCESSED_DATA_FILE.
        save:            Whether to save to disk. Set False during testing.

    Returns:
        Clean hourly DataFrame ready for feature engineering.
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("=== Starting ingestion pipeline ===")

    input_filepath  = input_filepath  or RAW_DATA_FILE
    output_filepath = output_filepath or PROCESSED_DATA_FILE

    df_raw    = load_raw_data(input_filepath)
    validate_raw_data(df_raw)
    df_clean  = clean_data(df_raw)
    df_hourly = resample_to_hourly(df_clean)

    if save:
        save_processed_data(df_hourly, output_filepath)

    log.info(
        f"=== Pipeline complete — "
        f"{len(df_hourly):,} hourly rows ready for feature engineering ==="
    )
    return df_hourly