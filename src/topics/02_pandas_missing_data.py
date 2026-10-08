"""Topic 02: Pandas inspection of missing environmental sensor data."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from src.data_pipeline import SENSOR_COLUMNS
from src.topics.common import load_project_data, save_report


def main() -> None:
    data, _ = load_project_data()
    sensors = data[SENSOR_COLUMNS]
    filled = SimpleImputer(strategy="median").fit_transform(sensors)
    audit = pd.DataFrame(
        {
            "feature": SENSOR_COLUMNS,
            "missing_values": sensors.isna().sum().to_numpy(),
            "missing_percent": (sensors.isna().mean() * 100).to_numpy(),
            "median_used_for_imputation": pd.DataFrame(filled, columns=SENSOR_COLUMNS).median().to_numpy(),
        }
    )
    output = save_report("02_pandas_missing_data_audit.csv", audit)
    print(f"Pandas missing-data audit saved to {output}")


if __name__ == "__main__":
    main()
