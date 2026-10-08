"""Run every syllabus topic module in order and write separate outputs."""

from __future__ import annotations

from importlib import import_module


TOPIC_MODULES = [
    "01_numpy_numerical_processing",
    "02_pandas_missing_data",
    "03_matplotlib_visualization",
    "04_feature_scaling",
    "05_simple_linear_regression",
    "06_multiple_linear_regression",
    "07_mle_bayesian_formulation",
    "08_gaussian_naive_bayes",
    "09_logistic_regression",
    "10_id3_decision_tree",
    "11_perceptron",
    "12_mlp_backpropagation",
    "13_linear_svm",
    "14_nonlinear_svm",
    "15_kmeans_clustering",
    "16_agglomerative_linkage",
    "17_em_gaussian_mixture",
    "18_pca",
    "19_evaluation_metrics",
]


def main() -> None:
    for module_name in TOPIC_MODULES:
        print(f"\n--- Running {module_name} ---")
        import_module(f"src.topics.{module_name}").main()
    print("\nAll topic-wise demonstrations completed.")


if __name__ == "__main__":
    main()
