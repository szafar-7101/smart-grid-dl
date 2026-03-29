# tests/unit/test_ingestion.py
#
# Unit tests for the ingestion pipeline.
#
# A unit test is a small, fast, isolated check that one specific
# thing in your code behaves correctly. Each function starting with
# "test_" is one test. pytest discovers and runs all of them automatically.
#
# We do NOT use real data files in unit tests. Instead, we create
# tiny fake DataFrames that are designed to test specific behaviours.
# This makes tests fast and predictable.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.config import DATE_COLUMN, TARGET_COLUMN, TIME_COLUMN
from src.ingestion.pipeline import (
    DataIngestionError,
    DataValidationError,
    clean_data,
    load_raw_data,
    resample_to_hourly,
    run_pipeline,
    validate_raw_data,
)

# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------
# A fixture is a reusable piece of test data or setup.
# By decorating a function with @pytest.fixture, pytest will
# automatically run it and pass its return value to any test
# function that lists it as a parameter.

@pytest.fixture
def valid_raw_df() -> pd.DataFrame:
    """
    Creates a small but valid raw DataFrame that mimics the UCI dataset.
    Used by multiple tests below.
    """
    return pd.DataFrame({
        "Date": ["16/12/2006", "16/12/2006", "16/12/2006", "16/12/2006"],
        "Time": ["17:24:00", "17:25:00", "17:26:00", "17:27:00"],
        "Global_active_power": ["4.216", "5.360", "5.374", "5.388"],
        "Global_reactive_power": ["0.418", "0.436", "0.498", "0.502"],
        "Voltage": ["234.840", "233.630", "233.290", "233.740"],
        "Global_intensity": ["18.400", "23.000", "23.000", "23.000"],
        "Sub_metering_1": ["0.000", "0.000", "0.000", "0.000"],
        "Sub_metering_2": ["1.000", "1.000", "2.000", "1.000"],
        "Sub_metering_3": ["17.000", "16.000", "17.000", "17.000"],
    })


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """
    Creates a clean DataFrame with a datetime index and float values.
    Represents what clean_data() should produce.
    """
    index = pd.date_range(start="2006-12-16 17:00", periods=4, freq="h")
    return pd.DataFrame(
        {
            "Global_active_power": [4.216, 5.360, 5.374, 5.388],
            "Global_reactive_power": [0.418, 0.436, 0.498, 0.502],
            "Voltage": [234.840, 233.630, 233.290, 233.740],
            "Global_intensity": [18.4, 23.0, 23.0, 23.0],
            "Sub_metering_1": [0.0, 0.0, 0.0, 0.0],
            "Sub_metering_2": [1.0, 1.0, 2.0, 1.0],
            "Sub_metering_3": [17.0, 16.0, 17.0, 17.0],
        },
        index=index,
    )
    # We name the index to match what clean_data() produces
    clean = pd.DataFrame(...)
    clean.index.name = "datetime"
    return clean


# ----------------------------------------------------------------
# Tests for load_raw_data
# ----------------------------------------------------------------

def test_load_raises_error_for_missing_file():
    """
    If we give load_raw_data() a path that doesn't exist,
    it must raise DataIngestionError — not a confusing pandas error.
    """
    fake_path = Path("/nonexistent/path/file.txt")

    # pytest.raises() is a context manager that asserts an exception IS raised.
    # If the code inside does NOT raise the expected exception, the test fails.
    with pytest.raises(DataIngestionError):
        load_raw_data(fake_path)


# ----------------------------------------------------------------
# Tests for validate_raw_data
# ----------------------------------------------------------------

def test_validate_passes_for_valid_data(valid_raw_df):
    """
    Valid data should pass validation without raising any exception.
    """
    # If this raises anything, pytest marks the test as failed.
    validate_raw_data(valid_raw_df)


def test_validate_raises_for_empty_dataframe():
    """
    An empty DataFrame should fail validation immediately.
    """
    empty_df = pd.DataFrame()
    with pytest.raises(DataValidationError):
        validate_raw_data(empty_df)


def test_validate_raises_for_missing_columns(valid_raw_df):
    """
    If a required column is missing, validation must raise DataValidationError.
    """
    # Drop the target column to simulate a wrong file being uploaded
    df_missing_col = valid_raw_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(DataValidationError):
        validate_raw_data(df_missing_col)


def test_validate_raises_for_too_much_missing_data(valid_raw_df):
    """
    If more than 10% of the target column is missing, validation must fail.
    """
    # Set all target values to NaN — 100% missing, clearly above 10% threshold
    df_too_much_missing = valid_raw_df.copy()
    df_too_much_missing[TARGET_COLUMN] = np.nan
    with pytest.raises(DataValidationError):
        validate_raw_data(df_too_much_missing)


# ----------------------------------------------------------------
# Tests for clean_data
# ----------------------------------------------------------------

def test_clean_data_produces_datetime_index(valid_raw_df):
    """
    After cleaning, the DataFrame index must be a DatetimeIndex.
    """
    df_clean = clean_data(valid_raw_df)
    assert isinstance(df_clean.index, pd.DatetimeIndex), (
        "Expected a DatetimeIndex after cleaning but got "
        f"{type(df_clean.index)}"
    )


def test_clean_data_converts_to_float(valid_raw_df):
    """
    All numeric columns must be float dtype after cleaning.
    """
    df_clean = clean_data(valid_raw_df)
    for col in ["Global_active_power", "Voltage"]:
        assert df_clean[col].dtype == np.float64, (
            f"Column {col} should be float64 but is {df_clean[col].dtype}"
        )


def test_clean_data_removes_impossible_values():
    """
    Power readings below 0 or above 20 must be replaced with NaN then filled.
    """
    df = pd.DataFrame({
        "Date": ["16/12/2006", "16/12/2006", "16/12/2006"],
        "Time": ["17:24:00", "17:25:00", "17:26:00"],
        "Global_active_power": ["4.0", "-1.0", "25.0"],  # -1 and 25 are invalid
        "Global_reactive_power": ["0.4", "0.4", "0.4"],
        "Voltage": ["234.0", "234.0", "234.0"],
        "Global_intensity": ["18.0", "18.0", "18.0"],
        "Sub_metering_1": ["0.0", "0.0", "0.0"],
        "Sub_metering_2": ["1.0", "1.0", "1.0"],
        "Sub_metering_3": ["17.0", "17.0", "17.0"],
    })
    df_clean = clean_data(df)
    # After cleaning, all values in the target column must be within valid range
    valid_mask = df_clean[TARGET_COLUMN].notna()
    assert (df_clean.loc[valid_mask, TARGET_COLUMN] >= 0).all()
    assert (df_clean.loc[valid_mask, TARGET_COLUMN] <= 20).all()


# ----------------------------------------------------------------
# Tests for resample_to_hourly
# ----------------------------------------------------------------

def test_resample_reduces_row_count():
    """
    Resampling minute-level data to hourly must reduce the number of rows.
    We build a dedicated DataFrame here with minute-level frequency —
    multiple minutes within the same hour — so resampling genuinely compresses it.
    """
    # 10 consecutive minutes, all within the same hour (17:00 to 17:09).
    # When resampled to hourly, all 10 collapse into 1 row.
    # So we expect: 10 rows in, 1 row out — clearly a reduction.
    index = pd.date_range(start="2006-12-16 17:00", periods=10, freq="min")
    index.name = "datetime"

    df_minute_level = pd.DataFrame(
        {"Global_active_power": [4.0] * 10},
        index=index,
    )

    df_hourly = resample_to_hourly(df_minute_level)

    assert len(df_hourly) < len(df_minute_level), (
        f"Expected fewer rows after resampling but got "
        f"{len(df_hourly)} rows from {len(df_minute_level)} input rows"
    )

def test_resample_produces_hourly_frequency(clean_df):
    """
    The resampled DataFrame's index must have hourly frequency.
    """
    df_hourly = resample_to_hourly(clean_df)
    # pd.infer_freq() detects the frequency of a DatetimeIndex
    inferred_freq = pd.infer_freq(df_hourly.index)
    assert inferred_freq == "h", (
        f"Expected hourly frequency ('h') but got '{inferred_freq}'"
    )