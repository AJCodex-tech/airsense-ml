"""Topic 16: Agglomerative clustering, Ward linkage, and a dendrogram."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from src.data_pipeline import build_preprocessor
from src.topics.common import load_project_data, plot_path, save_report


def main() -> None:
    data, features = load_project_data()
    prepared = build_preprocessor().fit_transform(features)
    model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    labels = model.fit_predict(prepared)
    report = pd.DataFrame(
        {
            "cluster": range(4),
            "observations": pd.Series(labels).value_counts().sort_index().to_numpy(),
            "mean_aqi": [data.loc[labels == cluster, "aqi"].mean() for cluster in range(4)],
        }
    )
    report["silhouette_score"] = silhouette_score(prepared, labels, sample_size=min(1000, len(prepared)), random_state=42)
    output = save_report("16_agglomerative_linkage.csv", report)

    rng = np.random.default_rng(42)
    sample = prepared[rng.choice(len(prepared), size=min(150, len(prepared)), replace=False)]
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage(sample, method="ward"), no_labels=True, ax=ax)
    ax.set(title="Ward Linkage Dendrogram", xlabel="Sample observations", ylabel="Linkage distance")
    fig.tight_layout()
    chart = plot_path("16_agglomerative_dendrogram.png")
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    print(f"Agglomerative report saved to {output}; dendrogram saved to {chart}")


if __name__ == "__main__":
    main()
