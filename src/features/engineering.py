
# Feature engineering pipeline.
#
# Takes the clean hourly DataFrame from ingestion and produces
# a rich feature matrix ready for deep learning model training.
"""
 Pipeline stages:
   1. Calendar features    — time-based signals (hour, day, month, etc.)
   2. Cyclical encoding    — convert circular features to sine/cosine
   3. Lag features         — past values of the target
   4. Rolling statistics   — summaries of recent windows
   5. Interaction features — derived combinations
   6. Drop NaN rows        — remove rows created by lag lookback
   7. Train/val/test split — chronological split
   8. Scale features       — MinMax scaling to [0, 1]
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import holidays
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

# The holidays library knows the public holidays for every country.
# We use it to create an "is_holiday" flag for each row.
from src.features.config import (
    CYCLICAL_FEATURES,
    EVENING_PEAK_HOURS,
    FEATURE_RANGE,
    FEATURES_INPUT_FILE,
    FEATURES_OUTPUT_FILE,
    LAG_HOURS,
    LOGGER_NAME,
    MORNING_PEAK_HOURS,
    ROLLING_WINDOWS,
    SCALER_OUTPUT_FILE,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.ingestion.config import TARGET_COLUMN

# ----------------------------------------------------------------
# Custom exception
# ----------------------------------------------------------------

class FeatureEngineeringError(Exception):
    """Raised when feature engineering encounters an unrecoverable error."""
    pass


# ----------------------------------------------------------------
# Stage 1 — Calendar features
# ----------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the datetime index.

    Every feature here answers the question: "what time is it?"
    in increasingly fine and coarse granularities.

    Args:
        df: DataFrame with a DatetimeIndex

    Returns:
        DataFrame with new calendar columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding calendar features")

    df = df.copy()

    df["hour"] = df.index.hour
    # Integer 0–23. 0 = midnight, 12 = noon, 23 = 11pm.

    df["day_of_week"] = df.index.dayofweek
    # Integer 0–6. 0 = Monday, 6 = Sunday.

    df["month"] = df.index.month
    # Integer 1–12. 1 = January, 12 = December.

    df["quarter"] = df.index.quarter
    # Integer 1–4. Coarser seasonal signal.

    df["day_of_year"] = df.index.dayofyear
    # Integer 1–365. Captures where we are in the annual cycle.

    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    # ISO week number 1–52. Another way to capture annual seasonality.

    # is_weekend: 1 if Saturday or Sunday, 0 otherwise.
    # dayofweek >= 5 means Saturday (5) or Sunday (6).
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    # .astype(int) converts the boolean (True/False) to integer (1/0).

    # is_holiday: 1 if this date is a French public holiday, 0 otherwise.
    # The UCI dataset is from France, so we use French holidays.
    # The holidays library returns a dict-like object of {date: holiday_name}.
    # We get the years present in our data and generate holidays for those years.
    years = df.index.year.unique().tolist()
    fr_holidays = holidays.France(years=years)
    # .normalize() strips the time component from the index, giving just dates.
    # Convert the datetime index to plain date objects (no time component)
    # so they match the date objects that the holidays library produces.
    # Without this, pandas raises a FutureWarning about comparing
    # datetime64 with date objects — which we treat as an error.
    index_as_dates = df.index.normalize().to_pydatetime()
    # .to_pydatetime() converts each pandas Timestamp to a Python datetime object.
    # Then we extract just the .date() part from each one so it matches
    # what fr_holidays contains (plain date objects, not datetimes).
    index_as_dates = [dt.date() for dt in index_as_dates]
    df["is_holiday"] = pd.Series(
        [d in fr_holidays for d in index_as_dates],
        index=df.index,
        dtype=int,
    )
# Instead of .isin() which has the type mismatch problem,
# we use a plain Python list comprehension: "is this date in fr_holidays?"
# pd.Series(..., index=df.index) ensures the result aligns correctly
# with the DataFrame rows.

    # is_peak_hour: 1 if this is a typical high-demand hour, 0 otherwise.
    # Morning peak: 7–9am. Evening peak: 5–8pm.
    peak_hours = MORNING_PEAK_HOURS + EVENING_PEAK_HOURS
    df["is_peak_hour"] = df["hour"].isin(peak_hours).astype(int)

    log.info(f"Calendar features added: {['hour','day_of_week','month','quarter','day_of_year','week_of_year','is_weekend','is_holiday','is_peak_hour']}")

    return df


# ----------------------------------------------------------------
# Stage 2 — Cyclical encoding
# ----------------------------------------------------------------

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes circular calendar features as sine and cosine pairs.

    Why this matters:
        Hour 23 and hour 0 are only 1 hour apart in reality,
        but numerically they are 23 apart. A neural network sees
        them as very different. Sine/cosine encoding wraps the
        scale so that 23 and 0 end up close together.

    Formula:
        sin_feature = sin(2π × value / max_value)
        cos_feature = cos(2π × value / max_value)

    The two values together uniquely encode any position on the cycle.
    (You need both sin and cos because sin alone is ambiguous —
    sin(30°) = sin(150°), so you cannot tell which one you are at.)

    Args:
        df: DataFrame with raw calendar features already added

    Returns:
        DataFrame with sine/cosine columns added for each cyclical feature
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding cyclical encodings")

    df = df.copy()

    for feature_name, max_value in CYCLICAL_FEATURES.items():
        # Check this column exists before trying to encode it
        if feature_name not in df.columns:
            log.warning(f"Cyclical feature '{feature_name}' not found — skipping")
            continue

        # 2π is one full rotation around the circle.
        # Dividing by max_value normalises the feature to [0, 2π].
        df[f"{feature_name}_sin"] = np.sin(2 * np.pi * df[feature_name] / max_value)
        df[f"{feature_name}_cos"] = np.cos(2 * np.pi * df[feature_name] / max_value)

    log.info(f"Cyclical features added for: {list(CYCLICAL_FEATURES.keys())}")

    return df


# ----------------------------------------------------------------
# Stage 3 — Lag features
# ----------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lag features — past values of the target column.

    A lag-N feature for row at time T contains the target value
    at time T-N. This gives the model direct access to historical
    load values as input signals.

    Args:
        df: DataFrame with the target column present

    Returns:
        DataFrame with lag columns added (will have NaN in first N rows)
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Adding lag features for hours: {LAG_HOURS}")

    df = df.copy()

    for lag in LAG_HOURS:
        column_name = f"lag_{lag}h"

        # .shift(lag) moves all values DOWN by `lag` positions.
        # So the value that was at row 168 is now at row 168+168=336.
        # The first 168 rows get NaN because there is nothing to shift from.
        # This means at row T, the lag_168h column contains the value
        # from row T-168 — which is 168 hours ago. Safe, no leakage.
        df[column_name] = df[TARGET_COLUMN].shift(lag)

    log.info(f"Lag features created: {[f'lag_{l}h' for l in LAG_HOURS]}")

    return df


# ----------------------------------------------------------------
# Stage 4 — Rolling window statistics
# ----------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates rolling window statistics over the target column.

    Rolling statistics summarise recent history into single numbers.
    They give the model a sense of "what has load been like recently?"

    Args:
        df: DataFrame with target column present

    Returns:
        DataFrame with rolling statistic columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Adding rolling features for windows: {ROLLING_WINDOWS}")

    df = df.copy()

    # shift(1) moves the series back by 1 step.
    # We compute this once and reuse it for all windows.
    # This ensures no window ever includes the current row's value.
    shifted_target = df[TARGET_COLUMN].shift(1)

    for window in ROLLING_WINDOWS:
        """ 
        .rolling(window) creates a rolling view of `window` rows.
        .mean() / .std() / .max() / .min() compute statistics over that window.
        min_periods=1 means: even if the window is not fully filled yet
        (near the start of the data), compute the statistic with however
        many values are available rather than returning NaN.
        """

        df[f"rolling_mean_{window}h"] = (
            shifted_target.rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            shifted_target.rolling(window=window, min_periods=1).std()
        )
        df[f"rolling_max_{window}h"] = (
            shifted_target.rolling(window=window, min_periods=1).max()
        )
        df[f"rolling_min_{window}h"] = (
            shifted_target.rolling(window=window, min_periods=1).min()
        )

    log.info("Rolling features created")

    return df


# ----------------------------------------------------------------
# Stage 5 — Interaction features
# ----------------------------------------------------------------

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features that combine two signals into one.

    These capture relationships that neither signal expresses alone.
    For example, "how much does current load deviate from the typical
    load at this hour?" combines the current load with the hour.

    Args:
        df: DataFrame with calendar and lag features already added

    Returns:
        DataFrame with interaction columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding interaction features")

    df = df.copy()

    """--- Hourly mean profile ---
     For each hour of the day (0–23), compute the mean load across
     all historical rows that share that hour.
     This gives us a "typical daily profile" — what is load usually
     like at 9am? At midnight?
     .transform("mean") is key: it computes the group mean but
     returns a Series with the SAME index as the original DataFrame.
     So every row gets the mean of its group, not a collapsed result.
     This lets us do row-wise operations (subtraction below).
    """
    hourly_mean = df.groupby("hour")[TARGET_COLUMN].transform("mean")
    df["hourly_mean_load"] = hourly_mean

    """ --- Deviation from hourly mean ---
     How much does the current load deviate from what is typical
     at this hour? A positive value means higher than usual.
     The model can learn that unusual deviations might precede
     certain patterns.
    """
    df["load_deviation_from_hourly_mean"] = df[TARGET_COLUMN] - hourly_mean

    # --- Load range (max - min over 24h window) ---
    # How volatile has load been in the last day?
    # A high range means a lot of fluctuation — potentially unusual conditions.
    if "rolling_max_24h" in df.columns and "rolling_min_24h" in df.columns:
        df["load_range_24h"] = df["rolling_max_24h"] - df["rolling_min_24h"]

    log.info("Interaction features created")

    return df


# ----------------------------------------------------------------
# Stage 6 — Drop NaN rows
# ----------------------------------------------------------------

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows that have NaN values introduced by lag features.

    The lag_336h feature (two weeks) means the first 336 rows of the
    DataFrame have no valid lag value — they become NaN.
    We cannot train on rows with missing features, so we drop them.

    We drop NaN rows rather than filling them because:
        - Filling with 0 would introduce fake "zero load" history
        - Filling with mean would introduce fake "average" history
        - Dropping is honest — we simply start training from the first
          row that has complete information

    Args:
        df: DataFrame that may have NaN rows from lag computation

    Returns:
        DataFrame with NaN rows removed
    """
    log = logger.bind(name=LOGGER_NAME)

    rows_before = len(df)
    df = df.dropna()
    # .dropna() removes any row that has at least one NaN value
    # in any column. This is a strict policy — if any feature is
    # missing for a row, we remove the entire row.

    rows_dropped = rows_before - len(df)
    log.info(
        f"Dropped {rows_dropped:,} NaN rows "
        f"({rows_dropped / rows_before:.1%} of data) — "
        f"{len(df):,} rows remaining"
    )

    if len(df) == 0:
        raise FeatureEngineeringError(
            "All rows were dropped after NaN removal. "
            "The dataset may be too short for the configured lag features."
        )

    return df


# ----------------------------------------------------------------
# Stage 7 — Chronological train/val/test split
# ----------------------------------------------------------------

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets
    in strict chronological order.

    WHY NOT RANDOM SPLIT?
        In time series, random splitting causes data leakage.
        If you randomly assign rows to train/test, future timestamps
        end up in training and past timestamps end up in test.
        The model learns from the future, which it cannot do in production.

        Chronological split: train on the PAST, validate/test on the FUTURE.
        This is the only correct approach for time series.

    Split: 70% train | 15% validation | 15% test

    Args:
        df: Complete feature DataFrame

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    log = logger.bind(name=LOGGER_NAME)

    n = len(df)
    # Calculate the integer row indices where each split begins/ends.
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    # Everything from val_end to the end is the test set.
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    log.info(
        f"Data split — "
        f"Train: {len(train_df):,} rows ({train_df.index[0].date()} to {train_df.index[-1].date()}) | "
        f"Val: {len(val_df):,} rows ({val_df.index[0].date()} to {val_df.index[-1].date()}) | "
        f"Test: {len(test_df):,} rows ({test_df.index[0].date()} to {test_df.index[-1].date()})"
    )

    return train_df, val_df, test_df


# ----------------------------------------------------------------
# Stage 8 — Feature scaling
# ----------------------------------------------------------------

def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, list]:
    """
    Scales all features to [0, 1] using MinMaxScaler.

    CRITICAL RULE:
        Fit the scaler ONLY on training data.
        Transform all three splits using the same fitted scaler.

    Args:
        train_df:    Training DataFrame
        val_df:      Validation DataFrame
        test_df:     Test DataFrame
        scaler_path: Where to save the fitted scaler to disk

    Returns:
        Tuple of:
            - X_train: scaled training array (numpy)
            - X_val:   scaled validation array
            - X_test:  scaled test array
            - scaler:  the fitted MinMaxScaler object
            - feature_columns: list of column names (for SHAP later)
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Scaling features")

    # Get the list of all feature columns.
    # We exclude nothing — all columns are features for now.
    # The target column (Global_active_power) is included because
    # LSTM models treat it as both input and output.
    feature_columns = list(train_df.columns)

    # Create a new MinMaxScaler with our configured range (0, 1).
    scaler = MinMaxScaler(feature_range=FEATURE_RANGE)

    # FIT on training data only.
    X_train = scaler.fit_transform(train_df[feature_columns].values)
    # .values converts the DataFrame to a raw numpy array.
    # fit_transform = fit + transform in one step (only valid for train).

    # TRANSFORM (not fit) on validation and test.
    # We use the min/max from training data to scale these splits.
    # This ensures validation and test are scaled exactly as training was.
    X_val  = scaler.transform(val_df[feature_columns].values)
    X_test = scaler.transform(test_df[feature_columns].values)

    # Save the fitted scaler to disk.
    # We need to reuse this exact scaler at inference time to scale
    # new input data the same way the model was trained on.
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    # "wb" means "write binary" — pickle produces bytes, not text.
    log.info(f"Scaler saved to {scaler_path}")

    log.info(
        f"Scaling complete — "
        f"Train shape: {X_train.shape} | "
        f"Val shape: {X_val.shape} | "
        f"Test shape: {X_test.shape}"
    )

    return X_train, X_val, X_test, scaler, feature_columns


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------

def run_feature_pipeline(
    input_filepath: Path = None,
    output_filepath: Path = None,
    scaler_filepath: Path = None,
    save: bool = True,
) -> Dict:
    """
    Runs the complete feature engineering pipeline end to end.

    This is the only function you should call from outside this module.
    It orchestrates all eight stages in order and returns everything
    the training pipeline needs.

    Args:
        input_filepath:  Path to clean parquet file from ingestion.
                         Defaults to FEATURES_INPUT_FILE from config.
        output_filepath: Path to save the feature matrix.
                         Defaults to FEATURES_OUTPUT_FILE from config.
        scaler_filepath: Path to save the fitted scaler.
                         Defaults to SCALER_OUTPUT_FILE from config.
        save:            Whether to save outputs to disk.

    Returns:
        Dictionary containing:
            "X_train", "X_val", "X_test": scaled numpy arrays
            "train_df", "val_df", "test_df": unscaled DataFrames
            "scaler": fitted MinMaxScaler
            "feature_columns": list of feature names
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("=== Starting feature engineering pipeline ===")

    input_filepath  = input_filepath  or FEATURES_INPUT_FILE
    output_filepath = output_filepath or FEATURES_OUTPUT_FILE
    scaler_filepath = scaler_filepath or SCALER_OUTPUT_FILE

    # Load the processed data from ingestion stage
    if not input_filepath.exists():
        raise FeatureEngineeringError(
            f"Input file not found: {input_filepath}\n"
            f"Run the ingestion pipeline first."
        )

    log.info(f"Loading processed data from {input_filepath}")
    df = pd.read_parquet(input_filepath)
    # .read_parquet() restores the DataFrame exactly as it was saved —
    # including the datetime index and all column types.

    log.info(f"Loaded {len(df):,} rows spanning {df.index[0]} to {df.index[-1]}")

    # Run each stage in order, feeding output of one into next
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = drop_nan_rows(df)

    # Split into train/val/test
    train_df, val_df, test_df = split_data(df)

    # Scale features
    X_train, X_val, X_test, scaler, feature_columns = scale_features(
        train_df, val_df, test_df, scaler_filepath
    )

    # Save the full feature matrix to disk
    if save:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_filepath)
        log.info(f"Feature matrix saved to {output_filepath}")

    log.info(
        f"=== Feature engineering complete — "
        f"{len(feature_columns)} features, "
        f"{X_train.shape[0]:,} training samples ==="
    )

    return {
        "X_train":         X_train,
        "X_val":           X_val,
        "X_test":          X_test,
        "train_df":        train_df,
        "val_df":          val_df,
        "test_df":         test_df,
        "scaler":          scaler,
        "feature_columns": feature_columns,
    }