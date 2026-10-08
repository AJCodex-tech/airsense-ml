"""Topic 17: Gaussian Mixture Model clustering fitted with expectation-maximization."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.data_pipeline import build_preprocessor
from src.topics.common import load_project_data, save_report


def main() -> None:
    data, features = load_project_data()
    prepared = build_preprocessor().fit_transform(features)
    model = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
    labels = model.fit_predict(prepared)
    probabilities = model.predict_proba(prepared).max(axis=1)
    report = pd.DataFrame(
        {
            "component": range(4),
            "observations": pd.Series(labels).value_counts().sort_index().to_numpy(),
            "mean_aqi": [data.loc[labels == component, "aqi"].mean() for component in range(4)],
            "mean_membership_probability": [probabilities[labels == component].mean() for component in range(4)],
        }
    )
    report["silhouette_score"] = silhouette_score(prepared, labels, sample_size=min(1000, len(prepared)), random_state=42)
    output = save_report("17_em_gaussian_mixture.csv", report)
    print(f"EM/GMM cluster report saved to {output}")


if __name__ == "__main__":
    main()
