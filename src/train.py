"""Train AirSense models, evaluate them, and save reports, plots, and artifacts."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from src.data_pipeline import build_preprocessor, feature_frame, load_dataset


RANDOM_STATE = 42


def _ensure_directories(root: Path) -> dict[str, Path]:
    folders = {name: root / name for name in ("models", "reports", "plots")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def _multiclass_roc_auc(estimator: Pipeline, features: pd.DataFrame, target: pd.Series) -> float:
    """Compute weighted one-vs-rest AUC from probabilities or decision scores."""
    classes = np.asarray(estimator.classes_)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
        if np.ndim(scores) == 1:
            scores = np.column_stack((-scores, scores))
    else:
        return float("nan")

    scores = np.asarray(scores)
    auc_values: list[float] = []
    weights: list[int] = []
    for index, label in enumerate(classes):
        binary_target = (target.to_numpy() == label).astype(int)
        if binary_target.min() == binary_target.max():
            continue
        auc_values.append(roc_auc_score(binary_target, scores[:, index]))
        weights.append(int(binary_target.sum()))
    return float(np.average(auc_values, weights=weights)) if auc_values else float("nan")


def _save_regression_plot(actual: pd.Series, predicted: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(actual, predicted, alpha=0.55, color="#177E89", edgecolors="none")
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(limits, limits, "--", color="#D1495B", label="Perfect prediction")
    ax.set(xlabel="Actual AQI", ylabel="Predicted AQI", title="Multiple Linear Regression: AQI Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _save_model_comparison_plot(metrics: pd.DataFrame, destination: Path) -> None:
    table = metrics.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    table.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set(title="Pollution Classifier Comparison", ylabel="Score", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=28)
    ax.legend(title="Metric", ncols=3)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _save_pca_plot(reduced: np.ndarray, labels: np.ndarray, destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis", alpha=0.65, s=18)
    ax.set(title="K-Means Pollution Patterns in PCA Space", xlabel="Principal Component 1", ylabel="Principal Component 2")
    fig.colorbar(scatter, ax=ax, label="K-Means cluster")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _save_dendrogram(reduced: np.ndarray, destination: Path, seed: int = RANDOM_STATE) -> None:
    rng = np.random.default_rng(seed)
    sample_size = min(150, len(reduced))
    sample = reduced[rng.choice(len(reduced), size=sample_size, replace=False)]
    hierarchy = linkage(sample, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(hierarchy, no_labels=True, color_threshold=None, ax=ax)
    ax.set(title="Agglomerative Clustering Dendrogram (PCA sample)", xlabel="Sample observations", ylabel="Ward linkage distance")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _save_distribution_plots(data: pd.DataFrame, plots_dir: Path) -> None:
    class_order = ["Good", "Moderate", "Unhealthy", "Hazardous"]
    counts = data["pollution_level"].value_counts().reindex(class_order, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index, counts.values, color=["#46A758", "#F5A524", "#E5484D", "#8E4EC6"])
    ax.set(title="Pollution-Level Distribution", xlabel="Pollution level", ylabel="Observations")
    fig.tight_layout()
    fig.savefig(plots_dir / "pollution_level_distribution.png", dpi=170)
    plt.close(fig)

    locations = sorted(data["location"].dropna().unique())
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.boxplot([data.loc[data["location"] == place, "aqi"] for place in locations], tick_labels=locations)
    ax.set(title="AQI Distribution by Location", xlabel="Location", ylabel="AQI")
    ax.tick_params(axis="x", rotation=22)
    fig.tight_layout()
    fig.savefig(plots_dir / "aqi_by_location.png", dpi=170)
    plt.close(fig)


def _train_regression(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> tuple[Pipeline, pd.DataFrame, np.ndarray]:
    baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(numeric_features=["pm25"], categorical_features=[])),
            ("model", LinearRegression()),
        ]
    )
    multiple = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", LinearRegression())]
    )

    baseline.fit(train_features[["pm25"]], train_target)
    multiple.fit(train_features, train_target)

    rows = []
    for label, estimator, subset in [
        ("PM2.5-only Linear Regression", baseline, test_features[["pm25"]]),
        ("Multiple Linear Regression", multiple, test_features),
    ]:
        prediction = estimator.predict(subset)
        rows.append(
            {
                "model": label,
                "mae": mean_absolute_error(test_target, prediction),
                "rmse": mean_squared_error(test_target, prediction) ** 0.5,
                "r2": r2_score(test_target, prediction),
            }
        )
    return multiple, pd.DataFrame(rows), multiple.predict(test_features)


def _train_classifiers(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    model_factories = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree (entropy / ID3-style)": DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=8, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE, dual="auto"),
        "RBF SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Perceptron": Perceptron(max_iter=1500, tol=1e-3, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }
    trained: dict[str, Pipeline] = {}
    scores: list[dict[str, float | str]] = []
    for name, classifier in model_factories.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", classifier)])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipeline.fit(train_features, train_target)
        prediction = pipeline.predict(test_features)
        trained[name] = pipeline
        scores.append(
            {
                "model": name,
                "accuracy": accuracy_score(test_target, prediction),
                "precision": precision_score(test_target, prediction, average="weighted", zero_division=0),
                "recall": recall_score(test_target, prediction, average="weighted", zero_division=0),
                "f1": f1_score(test_target, prediction, average="weighted", zero_division=0),
                "roc_auc": _multiclass_roc_auc(pipeline, test_features, test_target),
            }
        )
    return trained, pd.DataFrame(scores).sort_values("f1", ascending=False).reset_index(drop=True)


def _cluster_patterns(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    preprocessor = build_preprocessor()
    prepared = preprocessor.fit_transform(features)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(prepared)
    cluster_count = 4
    kmeans = KMeans(n_clusters=cluster_count, n_init=20, random_state=RANDOM_STATE)
    agglomerative = AgglomerativeClustering(n_clusters=cluster_count, linkage="ward")
    gmm = GaussianMixture(n_components=cluster_count, covariance_type="full", random_state=RANDOM_STATE)

    labels = {
        "kmeans_cluster": kmeans.fit_predict(prepared),
        "agglomerative_cluster": agglomerative.fit_predict(prepared),
        "em_gmm_cluster": gmm.fit_predict(prepared),
    }
    sample_size = min(1000, len(prepared))
    diagnostics = {
        "kmeans_silhouette": silhouette_score(prepared, labels["kmeans_cluster"], sample_size=sample_size, random_state=RANDOM_STATE),
        "agglomerative_silhouette": silhouette_score(prepared, labels["agglomerative_cluster"], sample_size=sample_size, random_state=RANDOM_STATE),
        "gmm_silhouette": silhouette_score(prepared, labels["em_gmm_cluster"], sample_size=sample_size, random_state=RANDOM_STATE),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    artifact = {"preprocessor": preprocessor, "pca": pca, "kmeans": kmeans, "agglomerative": agglomerative, "gmm": gmm, "reduced": reduced}
    return pd.DataFrame(labels), artifact, diagnostics


def train_project(data_path: str | Path, project_root: str | Path | None = None) -> dict[str, str | float]:
    """Run the complete, repeatable AirSense training workflow."""
    data_path = Path(data_path)
    root = Path(project_root) if project_root else data_path.resolve().parent.parent
    folders = _ensure_directories(root)
    data = load_dataset(data_path)
    features = feature_frame(data)
    aqi_target = data["aqi"]
    class_target = data["pollution_level"]

    # One shared split keeps the regression comparison and classifier comparison fair.
    train_index, test_index = train_test_split(
        data.index, test_size=0.20, stratify=class_target, random_state=RANDOM_STATE
    )
    train_features, test_features = features.loc[train_index], features.loc[test_index]
    train_aqi, test_aqi = aqi_target.loc[train_index], aqi_target.loc[test_index]
    train_class, test_class = class_target.loc[train_index], class_target.loc[test_index]

    regressor, regression_metrics, regression_prediction = _train_regression(
        train_features, test_features, train_aqi, test_aqi
    )
    classifiers, classification_metrics = _train_classifiers(
        train_features, test_features, train_class, test_class
    )
    cluster_labels, cluster_artifact, cluster_diagnostics = _cluster_patterns(features)

    best_classifier_name = str(classification_metrics.iloc[0]["model"])
    best_classifier = classifiers[best_classifier_name]
    cluster_report = pd.concat(
        [data[["timestamp", "location", "aqi", "pollution_level"]].reset_index(drop=True), cluster_labels], axis=1
    )

    regression_metrics.round(4).to_csv(folders["reports"] / "regression_metrics.csv", index=False)
    classification_metrics.round(4).to_csv(folders["reports"] / "classification_metrics.csv", index=False)
    cluster_report.to_csv(folders["reports"] / "cluster_assignments.csv", index=False)
    pd.DataFrame([cluster_diagnostics]).round(4).to_csv(folders["reports"] / "clustering_diagnostics.csv", index=False)

    joblib.dump(regressor, folders["models"] / "aqi_multiple_linear_regressor.joblib")
    joblib.dump(best_classifier, folders["models"] / "best_pollution_classifier.joblib")
    joblib.dump(cluster_artifact, folders["models"] / "cluster_models.joblib")

    _save_regression_plot(test_aqi, regression_prediction, folders["plots"] / "aqi_regression.png")
    _save_model_comparison_plot(classification_metrics, folders["plots"] / "classifier_comparison.png")
    _save_pca_plot(cluster_artifact["reduced"], cluster_labels["kmeans_cluster"].to_numpy(), folders["plots"] / "pca_kmeans_clusters.png")
    _save_dendrogram(cluster_artifact["reduced"], folders["plots"] / "agglomerative_dendrogram.png")
    _save_distribution_plots(data, folders["plots"])

    multiple_row = regression_metrics.loc[regression_metrics["model"] == "Multiple Linear Regression"].iloc[0]
    summary = f"""# AirSense training summary

- Observations analysed: {len(data):,}
- Missing sensor values before preprocessing: {int(data.isna().sum().sum()):,}
- Best pollution classifier by weighted F1: **{best_classifier_name}** ({classification_metrics.iloc[0]['f1']:.3f})
- Multiple linear regression AQI R²: **{multiple_row['r2']:.3f}**
- PCA variance retained by two components: **{cluster_diagnostics['pca_explained_variance']:.1%}**
- K-Means silhouette score: **{cluster_diagnostics['kmeans_silhouette']:.3f}**

The dataset generator produces synthetic observations and a synthetic AQI proxy. Replace the CSV with official observations before presenting environmental conclusions.
"""
    (folders["reports"] / "project_summary.md").write_text(summary, encoding="utf-8")

    return {
        "best_classifier": best_classifier_name,
        "regression_r2": float(multiple_row["r2"]),
        "reports_dir": str(folders["reports"]),
        "plots_dir": str(folders["plots"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the complete AirSense ML workflow.")
    parser.add_argument("--data", type=Path, required=True, help="CSV satisfying the README schema")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = train_project(args.data, args.project_root)
    print(f"Best classifier: {result['best_classifier']}")
    print(f"Multiple regression R²: {result['regression_r2']:.3f}")
    print(f"Reports: {result['reports_dir']}")
    print(f"Plots: {result['plots_dir']}")


if __name__ == "__main__":
    main()
