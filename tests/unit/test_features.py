# tests/unit/test_features.py
#
# Unit tests for the feature engineering pipeline.
# Every stage gets its own set of focused tests.

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    FeatureEngineeringError,
    add_calendar_features,
    add_cyclical_features,
    add_interaction_features,
    add_lag_features,
    add_rolling_features,
    drop_nan_rows,
    scale_features,
    split_data,
)
from src.ingestion.config import TARGET_COLUMN

# ----------------------------------------------------------------
# Shared fixture
# ----------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Creates a realistic hourly DataFrame spanning 400 hours.
    400 hours gives enough data for lag_336h features to produce
    some non-NaN rows after the initial lookback window.
    """
    index = pd.date_range(
        start="2006-12-01 00:00",
        periods=400,
        freq="h",
    )
    index.name = "datetime"

    # Use a sine wave to simulate realistic periodic load values.
    # This gives us values that actually vary in a meaningful pattern
    # rather than all being the same number.
    rng = np.random.default_rng(seed=42)
    # default_rng with a seed makes the random numbers reproducible —
    # every time you run the test you get the exact same values.

    load_values = (
        3.0                                        # baseline
        + 2.0 * np.sin(np.linspace(0, 8*np.pi, 400))  # daily cycle
        + 0.3 * rng.normal(size=400)              # small random noise
    ).clip(0.1, 10.0)
    # .clip() ensures values stay within physically valid range

    return pd.DataFrame(
        {TARGET_COLUMN: load_values},
        index=index,
    )


# ----------------------------------------------------------------
# Calendar feature tests
# ----------------------------------------------------------------

def test_calendar_features_adds_expected_columns(sample_df):
    """All expected calendar columns must be present after this stage."""
    df = add_calendar_features(sample_df)
    expected = [
        "hour", "day_of_week", "month", "quarter",
        "day_of_year", "week_of_year", "is_weekend",
        "is_holiday", "is_peak_hour"
    ]
    for col in expected:
        assert col in df.columns, f"Missing expected column: {col}"


def test_hour_values_in_valid_range(sample_df):
    """Hour must always be between 0 and 23."""
    df = add_calendar_features(sample_df)
    assert df["hour"].between(0, 23).all(), "Hour values outside 0–23 range"


def test_is_weekend_is_binary(sample_df):
    """is_weekend must only contain 0 or 1."""
    df = add_calendar_features(sample_df)
    assert set(df["is_weekend"].unique()).issubset({0, 1}), (
        "is_weekend contains values other than 0 and 1"
    )


def test_is_holiday_is_binary(sample_df):
    """is_holiday must only contain 0 or 1."""
    df = add_calendar_features(sample_df)
    assert set(df["is_holiday"].unique()).issubset({0, 1})


# ----------------------------------------------------------------
# Cyclical encoding tests
# ----------------------------------------------------------------

def test_cyclical_features_creates_sin_cos_pairs(sample_df):
    """Each cyclical feature must produce both a _sin and _cos column."""
    df = add_calendar_features(sample_df)
    df = add_cyclical_features(df)
    for feature in ["hour", "day_of_week", "month", "day_of_year"]:
        assert f"{feature}_sin" in df.columns, f"Missing {feature}_sin"
        assert f"{feature}_cos" in df.columns, f"Missing {feature}_cos"


def test_cyclical_values_within_minus_one_to_one(sample_df):
    """Sine and cosine values must always be in [-1, 1]."""
    df = add_calendar_features(sample_df)
    df = add_cyclical_features(df)
    for col in [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]:
        assert df[col].between(-1.0, 1.0).all(), (
            f"Column {col} has values outside [-1, 1]"
        )


# ----------------------------------------------------------------
# Lag feature tests
# ----------------------------------------------------------------

def test_lag_features_creates_expected_columns(sample_df):
    """A lag column must be created for each configured lag value."""
    df = add_lag_features(sample_df)
    for lag in [1, 24, 48, 168, 336]:
        assert f"lag_{lag}h" in df.columns, f"Missing lag_{lag}h column"


def test_lag_1h_shifts_by_one(sample_df):
    """
    The lag_1h value at row N must equal the target value at row N-1.
    This confirms the shift direction is correct (backwards, not forwards).
    """
    df = add_lag_features(sample_df)
    # Row index 5: lag_1h should equal target at row index 4
    original_value = sample_df[TARGET_COLUMN].iloc[4]
    lag_value = df["lag_1h"].iloc[5]
    assert abs(lag_value - original_value) < 1e-10, (
        f"lag_1h at row 5 should be {original_value:.4f} but got {lag_value:.4f}"
    )


def test_lag_features_produce_nan_at_start(sample_df):
    """
    The first row must have NaN for lag_24h (and all larger lags).
    This confirms no lookback is going before the data starts.
    """
    df = add_lag_features(sample_df)
    assert pd.isna(df["lag_24h"].iloc[0]), (
        "lag_24h at row 0 should be NaN but has a value"
    )


# ----------------------------------------------------------------
# Rolling feature tests
# ----------------------------------------------------------------

def test_rolling_features_creates_expected_columns(sample_df):
    """Rolling stat columns must be created for each configured window."""
    df = add_rolling_features(sample_df)
    for window in [24, 168]:
        for stat in ["mean", "std", "max", "min"]:
            col = f"rolling_{stat}_{window}h"
            assert col in df.columns, f"Missing column: {col}"


def test_rolling_mean_no_future_leakage(sample_df):
    """
    The rolling_mean_24h at row 0 must not include the value at row 0.
    We verify this by checking that after shift(1), row 0 sees NaN.
    """
    df = add_rolling_features(sample_df)
    # With shift(1), row 0 is NaN, so rolling_mean of just NaN
    # with min_periods=1 should still not include row 0's actual value.
    # The rolling mean at row 1 should equal the target at row 0 only.
    target_row0 = sample_df[TARGET_COLUMN].iloc[0]
    rolling_mean_row1 = df["rolling_mean_24h"].iloc[1]
    assert abs(rolling_mean_row1 - target_row0) < 1e-10, (
        "Rolling mean at row 1 should only use row 0's value"
    )


# ----------------------------------------------------------------
# Split tests
# ----------------------------------------------------------------

def test_split_produces_correct_proportions(sample_df):
    """Train/val/test sizes must match configured ratios (approximately)."""
    df = add_calendar_features(sample_df)
    df = add_lag_features(df)
    df = drop_nan_rows(df)

    train, val, test = split_data(df)
    total = len(train) + len(val) + len(test)

    assert abs(len(train) / total - 0.70) < 0.02, "Train proportion not ~70%"
    assert abs(len(val)   / total - 0.15) < 0.02, "Val proportion not ~15%"


def test_split_is_chronological(sample_df):
    """The last training timestamp must be before the first validation timestamp."""
    df = add_calendar_features(sample_df)
    df = add_lag_features(df)
    df = drop_nan_rows(df)

    train, val, test = split_data(df)

    assert train.index[-1] < val.index[0], (
        "Training data overlaps with validation data — split is not chronological"
    )
    assert val.index[-1] < test.index[0], (
        "Validation data overlaps with test data"
    )


# ----------------------------------------------------------------
# Scaling tests
# ----------------------------------------------------------------

def test_scaled_values_in_zero_one_range(sample_df, tmp_path):
    """
    All scaled values must be within [0, 1].
    tmp_path is a pytest built-in fixture that provides a
    temporary directory that is automatically cleaned up after the test.
    """
    df = add_calendar_features(sample_df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_nan_rows(df)

    train, val, test = split_data(df)
    scaler_path = tmp_path / "test_scaler.pkl"

    X_train, X_val, X_test, scaler, _ = scale_features(
        train, val, test, scaler_path
    )

    # Training data — after fitting, all values should be exactly in [0,1]
    assert X_train.min() >= -1e-10, "Scaled training values below 0"
    assert X_train.max() <= 1 + 1e-10, "Scaled training values above 1"


def test_scaler_is_saved_to_disk(sample_df, tmp_path):
    """The scaler file must exist on disk after scaling completes."""
    df = add_calendar_features(sample_df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = drop_nan_rows(df)

    train, val, test = split_data(df)
    scaler_path = tmp_path / "test_scaler.pkl"

    scale_features(train, val, test, scaler_path)

    assert scaler_path.exists(), (
        f"Scaler file was not saved to {scaler_path}"
    )