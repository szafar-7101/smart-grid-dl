# src/features/engineering.py
#
# Feature engineering pipeline — stage 2 of the project.
#
# Converts the clean hourly DataFrame from ingestion into a rich
# feature matrix ready for deep learning model training.
#
# KEY DESIGN DECISIONS (learned from previous iteration):
#
#   1. Global_active_power is NEVER put into X.
#      It becomes y only. Putting it in X gave the model a lazy
#      shortcut — copy the last value — instead of learning patterns.
#
#   2. split_data() is called BEFORE interaction features.
#      hourly_mean_load is computed on training rows only, then
#      mapped onto val and test. This prevents statistical leakage.
#
#   3. All rolling statistics use shift(1) before .rolling().
#      This ensures the window at row T never includes the value AT T.
#
#   4. Two separate scalers: feature_scaler for X, target_scaler for y.
#      A joint scaler polluted the loss function gradient signal.
#
# Pipeline order:
#   load → lag features → rolling features → calendar → cyclical →
#   drop NaN → split → interaction features → scale X → scale y → save

import numpy as np
import pandas as pd
import pickle
import holidays

from pathlib import Path
from typing import Tuple, Dict, Optional
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

from src.features.config import (
    FEATURES_INPUT_FILE,
    FEATURES_OUTPUT_FILE,
    FEATURE_SCALER_FILE,
    TARGET_SCALER_FILE,
    LAG_HOURS,
    ROLLING_WINDOWS,
    CYCLICAL_FEATURES,
    MORNING_PEAK_HOURS,
    EVENING_PEAK_HOURS,
    TRAIN_RATIO,
    VAL_RATIO,
    FEATURE_RANGE,
    LOGGER_NAME,
)
from src.ingestion.config import TARGET_COLUMN


# ----------------------------------------------------------------
# Custom exception
# ----------------------------------------------------------------

class FeatureEngineeringError(Exception):
    """Raised when feature engineering encounters an unrecoverable error."""
    pass


# ----------------------------------------------------------------
# Stage 1 — Lag features
# ----------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates lag features — past values of the target column.

    A lag-N feature at row T = target value at row T-N.
    .shift(N) with positive N shifts values forward (downward) in the
    DataFrame — so row T gets the value that was at row T-N.

    LEAKAGE CHECK: shift(N) with positive N always looks backwards.
    The first N rows get NaN — there is no past data for them yet.
    These NaN rows are dropped later in drop_nan_rows().

    Args:
        df: DataFrame with DatetimeIndex and TARGET_COLUMN present

    Returns:
        DataFrame with lag columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Adding lag features for hours: {LAG_HOURS}")

    df = df.copy()

    for lag in LAG_HOURS:
        # shift(lag) moves the target column DOWN by lag positions.
        # Row T now contains what was at row T-lag — i.e. lag hours ago.
        df[f"lag_{lag}h"] = df[TARGET_COLUMN].shift(lag)

    log.info(f"Lag features created: {[f'lag_{l}h' for l in LAG_HOURS]}")
    return df


# ----------------------------------------------------------------
# Stage 2 — Rolling statistics
# ----------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates rolling window statistics over the target column.

    CRITICAL — shift(1) is applied BEFORE .rolling():
        Without shift(1), the rolling window at row T includes the
        value AT row T — which is the value we are predicting.
        That is data leakage. shift(1) ensures the window ends at T-1.

    Args:
        df: DataFrame with TARGET_COLUMN present

    Returns:
        DataFrame with rolling statistic columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info(f"Adding rolling features for windows: {ROLLING_WINDOWS}")

    df = df.copy()

    # shift(1) moves the series back by 1. Row T now contains T-1's value.
    # Every rolling window computed on this shifted series is safe —
    # no window can ever include the current row's actual value.
    shifted = df[TARGET_COLUMN].shift(1)

    for window in ROLLING_WINDOWS:
        # min_periods=1: compute even if the full window is not yet filled
        # (handles the start of the dataset gracefully)
        df[f"rolling_mean_{window}h"] = (
            shifted.rolling(window=window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}h"] = (
            shifted.rolling(window=window, min_periods=1).std()
        )
        df[f"rolling_max_{window}h"] = (
            shifted.rolling(window=window, min_periods=1).max()
        )
        df[f"rolling_min_{window}h"] = (
            shifted.rolling(window=window, min_periods=1).min()
        )

    log.info("Rolling features created")
    return df


# ----------------------------------------------------------------
# Stage 3 — Calendar features
# ----------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-based features from the DatetimeIndex.

    These features encode human schedule information —
    hour of day, day of week, seasonality, holidays.
    Zero leakage risk: all information comes from the timestamp alone.

    The is_holiday check uses explicit date comparison against
    pd.Timestamp objects to avoid the FutureWarning we hit previously
    with .isin() comparing datetime64 to date objects.

    Args:
        df: DataFrame with a DatetimeIndex

    Returns:
        DataFrame with calendar columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding calendar features")

    df = df.copy()

    df["hour"]         = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    df["month"]        = df.index.month
    df["quarter"]      = df.index.quarter
    df["day_of_year"]  = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"]   = (df.index.dayofweek >= 5).astype(int)

    # Holiday detection — compare as pd.Timestamp to avoid type warnings
    years = df.index.year.unique().tolist()
    fr_holidays = holidays.France(years=years)
    # Convert holiday dates (Python date objects) to pd.Timestamp for comparison
    holiday_timestamps = {pd.Timestamp(d) for d in fr_holidays.keys()}
    # .normalize() strips the time component — gives midnight timestamps
    # Then check membership against our set of holiday timestamps
    df["is_holiday"] = df.index.normalize().isin(holiday_timestamps).astype(int)

    peak_hours = MORNING_PEAK_HOURS + EVENING_PEAK_HOURS
    df["is_peak_hour"] = df["hour"].isin(peak_hours).astype(int)

    log.info("Calendar features added")
    return df


# ----------------------------------------------------------------
# Stage 4 — Cyclical encoding
# ----------------------------------------------------------------

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes circular calendar features as sine and cosine pairs.

    Linear encoding breaks circularity:
        Hour 23 and hour 0 are 1 hour apart but numerically 23 apart.
        The model would wrongly learn they are distant.

    Sin/cos encoding fixes this:
        sin(2π × value / max) and cos(2π × value / max)
        Hour 23 and hour 0 end up close in (sin, cos) space.

    Both sin and cos are needed — sin alone is ambiguous
    (sin(30°) = sin(150°)) so you cannot uniquely identify the position.

    Args:
        df: DataFrame with raw calendar features already added

    Returns:
        DataFrame with sin/cos columns added for each cyclical feature
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding cyclical encodings")

    df = df.copy()

    for feature_name, max_value in CYCLICAL_FEATURES.items():
        if feature_name not in df.columns:
            log.warning(f"Cyclical feature '{feature_name}' not found — skipping")
            continue
        df[f"{feature_name}_sin"] = np.sin(
            2 * np.pi * df[feature_name] / max_value
        )
        df[f"{feature_name}_cos"] = np.cos(
            2 * np.pi * df[feature_name] / max_value
        )

    log.info("Cyclical features added")
    return df


# ----------------------------------------------------------------
# Stage 5 — Drop NaN rows
# ----------------------------------------------------------------

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with NaN values introduced by lag features.

    lag_336h (two weeks lookback) means the first 336 rows have no
    valid lag value. We drop them rather than filling because:
        - Filling with 0 introduces fake "zero load" history
        - Filling with mean introduces fake "average" history
        - Dropping is honest — we simply start from complete rows

    Args:
        df: DataFrame with potential NaN rows from lag computation

    Returns:
        DataFrame with NaN rows removed
    """
    log = logger.bind(name=LOGGER_NAME)

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)

    log.info(
        f"Dropped {rows_dropped:,} NaN rows "
        f"({rows_dropped / rows_before:.1%} of data) — "
        f"{len(df):,} rows remaining"
    )

    if len(df) == 0:
        raise FeatureEngineeringError(
            "All rows were dropped after NaN removal. "
            "Dataset may be too short for configured lag features."
        )

    return df


# ----------------------------------------------------------------
# Stage 6 — Chronological split
# ----------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train / validation / test in chronological order.

    WHY NOT RANDOM:
        Random splits cause data leakage in time series.
        Future timestamps would appear in training — the model learns
        from the future, which is impossible in production.

    WHY SPLIT BEFORE INTERACTION FEATURES:
        Interaction features like hourly_mean_load must be computed
        on training data only. If we compute them on the full dataset
        before splitting, val and test rows "know" future statistics.

    Split ratio: 70% train | 15% val | 15% test

    Args:
        df: Complete feature DataFrame after NaN removal

    Returns:
        Tuple of (train_df, val_df, test_df) — no row overlap
    """
    log = logger.bind(name=LOGGER_NAME)

    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

    # iloc uses integer position — not label/index value
    train_df = df.iloc[:train_end]
    val_df   = df.iloc[train_end:val_end]
    test_df  = df.iloc[val_end:]

    log.info(
        f"Data split — "
        f"Train: {len(train_df):,} rows "
        f"({train_df.index[0].date()} to {train_df.index[-1].date()}) | "
        f"Val: {len(val_df):,} rows "
        f"({val_df.index[0].date()} to {val_df.index[-1].date()}) | "
        f"Test: {len(test_df):,} rows "
        f"({test_df.index[0].date()} to {test_df.index[-1].date()})"
    )

    return train_df, val_df, test_df


# ----------------------------------------------------------------
# Stage 7 — Interaction features
# ----------------------------------------------------------------

def add_interaction_features(
    df: pd.DataFrame,
    train_hourly_means: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """
    Creates features that combine two signals.

    LEAKAGE PREVENTION:
        hourly_mean_load is the mean load for each hour of the day.
        If computed on the full dataset, val and test rows would
        contain means influenced by future data.

        Correct approach:
            - Compute means on training data only (train_hourly_means=None)
            - Pass those means to val and test calls (train_hourly_means=dict)

    Args:
        df:                  DataFrame to add features to
        train_hourly_means:  Pre-computed {hour: mean_load} from training.
                             Pass None only when processing training data.

    Returns:
        DataFrame with interaction columns added
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Adding interaction features")

    df = df.copy()

    if train_hourly_means is not None:
        # Val or test: use training means to avoid leakage
        # .map() replaces each hour integer with its mean from training
        df["hourly_mean_load"] = df["hour"].map(train_hourly_means)
    else:
        # Training: compute fresh from this data
        # .transform("mean") preserves the original index —
        # each row gets the mean of its hour group
        df["hourly_mean_load"] = df.groupby("hour")[TARGET_COLUMN].transform("mean")

    # How far is current load from what is typical at this hour?
    # Positive = higher than usual, Negative = lower than usual
    df["load_deviation"] = df[TARGET_COLUMN] - df["hourly_mean_load"]

    # How volatile has load been in the past 24h?
    # High range = unstable conditions
    if "rolling_max_24h" in df.columns and "rolling_min_24h" in df.columns:
        df["load_range_24h"] = df["rolling_max_24h"] - df["rolling_min_24h"]

    log.info("Interaction features created")
    return df


# ----------------------------------------------------------------
# Stage 8 — Scale features and target separately
# ----------------------------------------------------------------

def scale_features(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_scaler_path: Path,
    target_scaler_path:  Path,
) -> Dict:
    """
    Scales X features and y target using two separate MinMaxScalers.

    WHY TWO SCALERS:
        feature_scaler: fitted on 39 input columns (everything except target)
        target_scaler:  fitted on Global_active_power column only

        Separating them gives the model a clean, unambiguous loss signal.
        The gradient during training reflects only the target variable's
        distribution — not the joint distribution of all 40 features.

    CRITICAL RULE:
        Both scalers are FITTED on training data only.
        They are then used to TRANSFORM val and test.
        Fitting on val or test would leak future statistics.

    Args:
        train_df:            Training split DataFrame
        val_df:              Validation split DataFrame
        test_df:             Test split DataFrame
        feature_scaler_path: Where to save the feature scaler
        target_scaler_path:  Where to save the target scaler

    Returns:
        Dictionary with scaled arrays and both scaler objects
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("Scaling features and target separately")

    # ---- Identify columns ----
    # Feature columns = everything except the target
    # The target column exists in the DataFrame but will NOT enter X
    feature_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
    log.info(f"{len(feature_cols)} input features (target column excluded from X)")

    # ---- Feature scaler (for X) ----
    feature_scaler = MinMaxScaler(feature_range=FEATURE_RANGE)

    # fit_transform = fit on train, then transform train
    X_train = feature_scaler.fit_transform(train_df[feature_cols].values)
    # transform only — use the min/max learned from training
    X_val   = feature_scaler.transform(val_df[feature_cols].values)
    X_test  = feature_scaler.transform(test_df[feature_cols].values)

    # ---- Target scaler (for y) ----
    # reshape(-1, 1) converts 1D array to 2D column vector
    # MinMaxScaler requires 2D input
    target_scaler = MinMaxScaler(feature_range=FEATURE_RANGE)

    y_train_raw = train_df[TARGET_COLUMN].values.reshape(-1, 1)
    y_val_raw   = val_df[TARGET_COLUMN].values.reshape(-1, 1)
    y_test_raw  = test_df[TARGET_COLUMN].values.reshape(-1, 1)

    # Fit ONLY on training target values
    y_train = target_scaler.fit_transform(y_train_raw).flatten()
    # Transform val and test using training min/max
    y_val   = target_scaler.transform(y_val_raw).flatten()
    y_test  = target_scaler.transform(y_test_raw).flatten()

    # ---- Save both scalers ----
    for path, obj in [
        (feature_scaler_path, feature_scaler),
        (target_scaler_path,  target_scaler),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        log.info(f"Scaler saved to {path}")

    log.info(
        f"Scaling complete — "
        f"X_train: {X_train.shape} | "
        f"X_val: {X_val.shape} | "
        f"X_test: {X_test.shape}"
    )

    return {
        "X_train":         X_train,
        "X_val":           X_val,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_val":           y_val,
        "y_test":          y_test,
        "feature_cols":    feature_cols,
        "feature_scaler":  feature_scaler,
        "target_scaler":   target_scaler,
    }


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------

def run_feature_pipeline(
    input_filepath:         Optional[Path] = None,
    output_filepath:        Optional[Path] = None,
    feature_scaler_path:    Optional[Path] = None,
    target_scaler_path:     Optional[Path] = None,
    save:                   bool = True,
) -> Dict:
    """
    Runs the complete feature engineering pipeline end to end.

    Orchestrates all eight stages in the correct order.
    This is the only function external code should call.

    Args:
        input_filepath:       Clean parquet from ingestion pipeline
        output_filepath:      Where to save the feature matrix
        feature_scaler_path:  Where to save the feature scaler
        target_scaler_path:   Where to save the target scaler
        save:                 Whether to write outputs to disk

    Returns:
        Dictionary with all scaled arrays, DataFrames, and scalers
    """
    log = logger.bind(name=LOGGER_NAME)
    log.info("=== Starting feature engineering pipeline ===")

    input_filepath      = input_filepath      or FEATURES_INPUT_FILE
    output_filepath     = output_filepath     or FEATURES_OUTPUT_FILE
    feature_scaler_path = feature_scaler_path or FEATURE_SCALER_FILE
    target_scaler_path  = target_scaler_path  or TARGET_SCALER_FILE

    if not input_filepath.exists():
        raise FeatureEngineeringError(
            f"Input file not found: {input_filepath}\n"
            f"Run the ingestion pipeline first."
        )

    log.info(f"Loading processed data from {input_filepath}")
    df = pd.read_parquet(input_filepath)
    log.info(
        f"Loaded {len(df):,} rows "
        f"spanning {df.index[0]} to {df.index[-1]}"
    )

    # ---- Stages 1–4: features that have no leakage risk ----
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_cyclical_features(df)

    # ---- Stage 5: remove incomplete rows ----
    df = drop_nan_rows(df)

    # ---- Stage 6: split BEFORE interaction features ----
    train_df, val_df, test_df = split_data(df)

    # ---- Stage 7: interaction features — train stats only ----
    # Compute hourly means on training data first
    train_hourly_means = (
        train_df.groupby("hour")[TARGET_COLUMN]
        .mean()
        .to_dict()
    )
    # Apply: training computes its own means, val and test use training means
    train_df = add_interaction_features(train_df, train_hourly_means=None)
    val_df   = add_interaction_features(val_df,   train_hourly_means=train_hourly_means)
    test_df  = add_interaction_features(test_df,  train_hourly_means=train_hourly_means)

    # ---- Stage 8: scale X and y with separate scalers ----
    scaled = scale_features(
        train_df, val_df, test_df,
        feature_scaler_path, target_scaler_path,
    )

    # ---- Save the full feature matrix if requested ----
    if save:
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        # Save the full unscaled DataFrame for inspection and debugging
        full_df = pd.concat([train_df, val_df, test_df])
        full_df.to_parquet(output_filepath)
        log.info(f"Feature matrix saved to {output_filepath}")

    n_features = scaled["X_train"].shape[1]
    log.info(
        f"=== Feature engineering complete — "
        f"{n_features} input features, "
        f"{scaled['X_train'].shape[0]:,} training samples ==="
    )

    return {
        # Scaled numpy arrays ready for model training
        "X_train":        scaled["X_train"],
        "X_val":          scaled["X_val"],
        "X_test":         scaled["X_test"],
        "y_train":        scaled["y_train"],
        "y_val":          scaled["y_val"],
        "y_test":         scaled["y_test"],
        # Unscaled DataFrames for inspection
        "train_df":       train_df,
        "val_df":         val_df,
        "test_df":        test_df,
        # Scalers needed for inference and inverse transform
        "feature_scaler": scaled["feature_scaler"],
        "target_scaler":  scaled["target_scaler"],
        # Column names in the same order as X arrays
        "feature_cols":   scaled["feature_cols"],
    }