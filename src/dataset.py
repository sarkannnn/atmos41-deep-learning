"""Dataset splitting, scaling, sliding window extraction, and balanced sampling."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DAY_RADIATION_THRESHOLD,
    FEATURES,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TARGET_FEATURE,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


def split_chronologically(
    hourly_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data strictly chronologically to prevent temporal data leakage."""
    total_len = len(hourly_df)
    n_train = int(total_len * train_ratio)
    n_val = int(total_len * val_ratio)

    train_df = hourly_df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = hourly_df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = hourly_df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Normalize features using MinMaxScaler.

    Notice: The scaler is fitted SOLELY on the training set to prevent future data leakage.
    Validation and test sets are transformed using the fitted training parameters.
    """
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler strictly on training data
    x_train = scaler_x.fit_transform(train_df[FEATURES].values)
    y_train = scaler_y.fit_transform(train_df[[TARGET_FEATURE]].values)

    # Transform validation and test data using training parameters
    x_val = scaler_x.transform(val_df[FEATURES].values)
    y_val = scaler_y.transform(val_df[[TARGET_FEATURE]].values)

    x_test = scaler_x.transform(test_df[FEATURES].values)
    y_test = scaler_y.transform(test_df[[TARGET_FEATURE]].values)

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler_y


def create_sliding_windows(
    x_data: np.ndarray,
    y_data: np.ndarray,
    df_source: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    day_threshold: float = DAY_RADIATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sliding window sequences of length L predicting target step t + 1."""
    x_seq, y_seq, is_day_seq = [], [], []

    for i in range(len(x_data) - seq_len):
        x_seq.append(x_data[i : i + seq_len])
        y_seq.append(y_data[i + seq_len])
        target_solar_radiation = df_source["solar_rad_wm2"].iloc[i + seq_len]
        is_day_seq.append(target_solar_radiation > day_threshold)

    return np.array(x_seq), np.array(y_seq), np.array(is_day_seq)


def create_balanced_day_night_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    day_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 50% day and 50% night balanced dataset via random under-sampling."""
    day_indices = np.where(day_mask)[0]
    night_indices = np.where(~day_mask)[0]

    min_samples = min(len(day_indices), len(night_indices))
    selected_day = np.random.choice(day_indices, min_samples, replace=False)
    selected_night = np.random.choice(night_indices, min_samples, replace=False)

    balanced_indices = np.concatenate([selected_day, selected_night])
    np.random.shuffle(balanced_indices)

    return x_train[balanced_indices], y_train[balanced_indices]
