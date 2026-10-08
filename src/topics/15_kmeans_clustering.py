"""Topic 15: K-Means clustering of environmental conditions."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data_pipeline import build_preprocessor
from src.topics.common import load_project_data, save_report


def main() -> None:
    data, features = load_project_data()
    prepared = build_preprocessor().fit_transform(features)
    model = KMeans(n_clusters=4, n_init=20, random_state=42)
    labels = model.fit_predict(prepared)
    report = pd.DataFrame(
        {
            "cluster": range(4),
            "observations": pd.Series(labels).value_counts().sort_index().to_numpy(),
            "mean_aqi": [data.loc[labels == cluster, "aqi"].mean() for cluster in range(4)],
        }
    )
    report["silhouette_score"] = silhouette_score(prepared, labels, sample_size=min(1000, len(prepared)), random_state=42)
    output = save_report("15_kmeans_clusters.csv", report)
    print(f"K-Means clusters saved to {output}")


if __name__ == "__main__":
    main()
