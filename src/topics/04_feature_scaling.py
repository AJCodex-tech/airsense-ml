"""Topic 04: Median imputation and StandardScaler feature scaling."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import SENSOR_COLUMNS
from src.topics.common import load_project_data, save_report


def main() -> None:
    data, _ = load_project_data()
    imputed = SimpleImputer(strategy="median").fit_transform(data[SENSOR_COLUMNS])
    scaled = StandardScaler().fit_transform(imputed)
    results = pd.DataFrame(
        {
            "feature": SENSOR_COLUMNS,
            "scaled_mean": scaled.mean(axis=0),
            "scaled_std": scaled.std(axis=0),
        }
    )
    output = save_report("04_feature_scaling_check.csv", results)
    print(f"Feature-scaling check saved to {output}")


if __name__ == "__main__":
    main()
