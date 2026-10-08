"""Input validation, time-derived features, and reusable preprocessing pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SENSOR_COLUMNS = [
    "pm25",
    "pm10",
    "no2",
    "so2",
    "co",
    "o3",
    "temperature",
    "humidity",
    "wind_speed",
]
TIME_COLUMNS = ["hour", "month", "hour_sin", "hour_cos"]
NUMERIC_FEATURES = SENSOR_COLUMNS + TIME_COLUMNS
CATEGORICAL_FEATURES = ["location"]
TARGET_COLUMNS = ["aqi", "pollution_level"]
REQUIRED_COLUMNS = ["timestamp", "location", *SENSOR_COLUMNS, *TARGET_COLUMNS]


def pollution_level_from_aqi(aqi: pd.Series) -> pd.Series:
    """Map AQI scores to presentation-friendly project classes."""
    return pd.cut(
        aqi,
        bins=[-float("inf"), 50, 100, 200, float("inf")],
        labels=["Good", "Moderate", "Unhealthy", "Hazardous"],
    ).astype(str)


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create cyclic hour features so midnight and 23:00 remain close together."""
    data = frame.copy()
    if "timestamp" not in data.columns:
        raise ValueError("The dataset must contain a timestamp column.")

    timestamps = pd.to_datetime(data["timestamp"], errors="coerce")
    if timestamps.isna().any():
        bad_rows = int(timestamps.isna().sum())
        raise ValueError(f"timestamp contains {bad_rows} unparseable value(s).")

    data["hour"] = timestamps.dt.hour
    data["month"] = timestamps.dt.month
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    return data


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV before any model receives it."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(
            "CSV is missing required column(s): " + ", ".join(missing) + ". See README.md for schema."
        )

    data = add_time_features(data)
    data["location"] = data["location"].astype("string")
    data["aqi"] = pd.to_numeric(data["aqi"], errors="coerce")
    if data["aqi"].isna().any():
        raise ValueError("aqi must contain numeric, non-missing target values.")

    if data["pollution_level"].isna().any():
        data["pollution_level"] = pollution_level_from_aqi(data["aqi"])
    else:
        data["pollution_level"] = data["pollution_level"].astype(str)

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the inputs used by supervised and unsupervised models."""
    return data[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Impute sensor gaps, standardize numeric variables, and encode locations."""
    numeric_features = NUMERIC_FEATURES if numeric_features is None else numeric_features
    categorical_features = CATEGORICAL_FEATURES if categorical_features is None else categorical_features

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    category_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = [("numeric", numeric_pipeline, numeric_features)]
    if categorical_features:
        transformers.append(("location", category_pipeline, categorical_features))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
