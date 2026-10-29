"""Data cleaning, duplicate removal, hourly resampling, and interpolation."""

import urllib.request
from pathlib import Path
import pandas as pd

from src.config import DATA_SOURCE_URL, FEATURES, RAW_DATA_PATH


def download_data_if_missing(file_path: Path = RAW_DATA_PATH, url: str = DATA_SOURCE_URL) -> Path:
    """Download the raw ATMOS 41 dataset if not present locally."""
    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading raw data from {url}...")
        urllib.request.urlretrieve(url, file_path)
    return file_path


def load_and_clean_raw_data(file_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Load raw dataset, standardize column names, and remove duplicate timestamps."""
    download_data_if_missing(file_path)
    raw_df = pd.read_csv(file_path, encoding="latin-1")
    raw_df.columns = ["datetime_raw", "temp_c", "pressure_kpa", "humidity_pct", "solar_rad_wm2"]

    raw_count = len(raw_df)
    duplicate_count = int(raw_df.duplicated().sum())

    clean_df = raw_df.drop_duplicates().copy()
    clean_df["datetime"] = pd.to_datetime(clean_df["datetime_raw"])
    clean_df = clean_df.sort_values("datetime").reset_index(drop=True)

    metadata = {
        "raw_count": raw_count,
        "duplicate_count": duplicate_count,
        "duplicate_percentage": round((duplicate_count / raw_count) * 100, 2),
        "cleaned_count": len(clean_df),
        "start_time": str(clean_df["datetime"].min()),
        "end_time": str(clean_df["datetime"].max()),
    }
    return clean_df, metadata


def resample_to_hourly(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Resample 15-minute observations to regular 1-hour intervals and interpolate gaps."""
    time_indexed = clean_df.set_index("datetime")
    hourly_df = time_indexed[FEATURES].resample("1h").mean()

    missing_hourly_steps = int(hourly_df["temp_c"].isnull().sum())
    interpolated_df = hourly_df.interpolate(method="time").dropna().reset_index()

    metadata = {
        "hourly_count": len(interpolated_df),
        "missing_hourly_steps": missing_hourly_steps,
        "missing_hourly_percentage": round(
            (missing_hourly_steps / (len(interpolated_df) + missing_hourly_steps)) * 100, 2
        ),
    }
    return interpolated_df, metadata
