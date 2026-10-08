"""Generate a reproducible, realistic-looking air-quality dataset for demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import pollution_level_from_aqi


CITY_PROFILES = {
    "New Delhi": {"pm25": 115, "pm10": 210, "no2": 62, "so2": 18, "co": 1.35, "o3": 42},
    "Mumbai": {"pm25": 58, "pm10": 105, "no2": 43, "so2": 12, "co": 0.85, "o3": 48},
    "Bengaluru": {"pm25": 40, "pm10": 73, "no2": 31, "so2": 9, "co": 0.62, "o3": 41},
    "Kolkata": {"pm25": 82, "pm10": 150, "no2": 51, "so2": 15, "co": 1.05, "o3": 45},
}


def _positive(values: np.ndarray, minimum: float = 0.01) -> np.ndarray:
    return np.maximum(values, minimum)


def generate_dataset(rows: int = 2000, output_path: str | Path = "data/air_quality_demo.csv", seed: int = 42) -> pd.DataFrame:
    """Create an educational data set with weather, city, hour, and season effects.

    AQI is a synthetic proxy rather than an official regulator's AQI calculation. It
    is deliberately documented as such so a final academic submission can replace it
    with an official public-data target.
    """
    if rows < 200:
        raise ValueError("Use at least 200 rows so every pollution class can be represented.")

    rng = np.random.default_rng(seed)
    cities = np.array(list(CITY_PROFILES))
    city = rng.choice(cities, size=rows, p=[0.34, 0.25, 0.20, 0.21])
    timestamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 366 * 24, size=rows), unit="h"
    )
    month = timestamps.month.to_numpy()
    hour = timestamps.hour.to_numpy()
    winter_intensity = 1 + 0.55 * np.cos(2 * np.pi * (month - 1) / 12)
    rush_hour = 1 + 0.24 * ((hour >= 7) & (hour <= 10)) + 0.18 * ((hour >= 18) & (hour <= 22))
    night_stagnation = 1 + 0.16 * ((hour <= 5) | (hour >= 23))

    def base(name: str) -> np.ndarray:
        return np.array([CITY_PROFILES[c][name] for c in city], dtype=float)

    temperature = 28 + 6 * np.sin(2 * np.pi * (month - 3) / 12) + rng.normal(0, 4, rows)
    humidity = np.clip(58 + 18 * np.sin(2 * np.pi * (month - 6) / 12) + rng.normal(0, 12, rows), 18, 98)
    wind_speed = _positive(rng.gamma(shape=2.0, scale=1.4, size=rows), 0.15)
    dispersion = np.clip(1.3 - 0.09 * wind_speed + rng.normal(0, 0.05, rows), 0.55, 1.35)

    pm25 = _positive(base("pm25") * winter_intensity * rush_hour * night_stagnation * dispersion + rng.normal(0, 18, rows))
    pm10 = _positive(base("pm10") * winter_intensity * rush_hour * dispersion + rng.normal(0, 28, rows))
    no2 = _positive(base("no2") * rush_hour * night_stagnation * dispersion + rng.normal(0, 10, rows))
    so2 = _positive(base("so2") * (0.9 + 0.1 * winter_intensity) + rng.normal(0, 3, rows))
    co = _positive(base("co") * rush_hour * night_stagnation * dispersion + rng.normal(0, 0.18, rows))
    o3 = _positive(base("o3") + 0.9 * np.maximum(temperature - 24, 0) + 0.7 * np.maximum(hour - 10, 0) - 0.12 * humidity + rng.normal(0, 8, rows))

    # Calibrated synthetic AQI proxy: pollutant burden + meteorological contribution.
    aqi = (
        0.48 * pm25
        + 0.18 * pm10
        + 0.65 * no2
        + 1.6 * so2
        + 25 * co
        + 0.20 * o3
        + 0.10 * humidity
        - 2.2 * wind_speed
        + rng.normal(0, 14, rows)
    )
    aqi = np.clip(aqi, 15, 500).round(1)

    data = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "location": city,
            "pm25": pm25.round(2),
            "pm10": pm10.round(2),
            "no2": no2.round(2),
            "so2": so2.round(2),
            "co": co.round(3),
            "o3": o3.round(2),
            "temperature": temperature.round(2),
            "humidity": humidity.round(2),
            "wind_speed": wind_speed.round(2),
            "aqi": aqi,
        }
    )
    data["pollution_level"] = pollution_level_from_aqi(data["aqi"])

    # Imitate occasional sensor outages; the pipeline should handle these safely.
    for column in ["pm25", "pm10", "no2", "so2", "co", "o3", "humidity", "wind_speed"]:
        data.loc[rng.random(rows) < 0.03, column] = np.nan

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AirSense demonstration data.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/air_quality_demo.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(rows=args.rows, output_path=args.output, seed=args.seed)
    print(f"Created {args.rows:,} rows at {args.output}")


if __name__ == "__main__":
    main()
