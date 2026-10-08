"""Topic 18: Principal Component Analysis for pollutant feature reduction."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA

from src.data_pipeline import build_preprocessor
from src.topics.common import load_project_data, save_report


def main() -> None:
    _, features = load_project_data()
    preprocessor = build_preprocessor()
    prepared = preprocessor.fit_transform(features)
    pca = PCA(n_components=2, random_state=42).fit(prepared)
    report = pd.DataFrame(
        {
            "principal_component": ["PC1", "PC2"],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": pca.explained_variance_ratio_.cumsum(),
        }
    )
    output = save_report("18_pca_variance.csv", report)
    print(f"PCA variance report saved to {output}")


if __name__ == "__main__":
    main()
