"""Persistence baseline model evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import SEQUENCE_LENGTH, TARGET_FEATURE


def evaluate_persistence_baseline(
    test_df: pd.DataFrame,
    is_day_test: np.ndarray,
    seq_len: int = SEQUENCE_LENGTH,
) -> dict[str, float]:
    """Evaluate naive persistence baseline where predicted T(t+1) = T(t)."""
    # Target temperatures matching the sliding window indices
    actual_temperatures = test_df[TARGET_FEATURE].iloc[seq_len:].values

    # Predicted temperature is the last observed temperature at sequence step t
    predicted_temperatures = test_df[TARGET_FEATURE].iloc[seq_len - 1 : -1].values

    mae_overall = mean_absolute_error(actual_temperatures, predicted_temperatures)
    rmse_overall = np.sqrt(mean_squared_error(actual_temperatures, predicted_temperatures))
    r2_overall = r2_score(actual_temperatures, predicted_temperatures)

    mae_day = mean_absolute_error(actual_temperatures[is_day_test], predicted_temperatures[is_day_test])
    rmse_day = np.sqrt(mean_squared_error(actual_temperatures[is_day_test], predicted_temperatures[is_day_test]))

    mae_night = mean_absolute_error(actual_temperatures[~is_day_test], predicted_temperatures[~is_day_test])
    rmse_night = np.sqrt(mean_squared_error(actual_temperatures[~is_day_test], predicted_temperatures[~is_day_test]))

    return {
        "mae": float(mae_overall),
        "rmse": float(rmse_overall),
        "r2": float(r2_overall),
        "day_mae": float(mae_day),
        "day_rmse": float(rmse_day),
        "night_mae": float(mae_night),
        "night_rmse": float(rmse_night),
        "time_sec": 0.0,
    }
