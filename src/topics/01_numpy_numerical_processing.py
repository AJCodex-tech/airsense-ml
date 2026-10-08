"""Topic 01: NumPy numerical processing of environmental sensor values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_pipeline import SENSOR_COLUMNS
from src.topics.common import load_project_data, save_report


def main() -> None:
    data, _ = load_project_data()
    readings = data[SENSOR_COLUMNS].to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "feature": SENSOR_COLUMNS,
            "mean": np.nanmean(readings, axis=0),
            "median": np.nanmedian(readings, axis=0),
            "std": np.nanstd(readings, axis=0),
            "minimum": np.nanmin(readings, axis=0),
            "maximum": np.nanmax(readings, axis=0),
        }
    )
    output = save_report("01_numpy_sensor_statistics.csv", table)
    print(f"NumPy statistics saved to {output}")


if __name__ == "__main__":
    main()
