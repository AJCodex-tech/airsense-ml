"""Shared data loading, splits, metrics, and output helpers for topic modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_pipeline import build_preprocessor, feature_frame, load_dataset
from src.generate_data import generate_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "air_quality_demo.csv"
TOPIC_REPORTS = ROOT / "reports" / "topics"
TOPIC_PLOTS = ROOT / "plots" / "topics"
RANDOM_STATE = 42


def ensure_dataset() -> Path:
    """Provide a reproducible demo file when a topic is run for the first time."""
    if not DATA_PATH.exists():
        generate_dataset(output_path=DATA_PATH)
    return DATA_PATH


def load_project_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_dataset(ensure_dataset())
    return data, feature_frame(data)


def regression_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data, features = load_project_data()
    return train_test_split(features, data["aqi"], test_size=0.20, random_state=RANDOM_STATE)


def classification_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data, features = load_project_data()
    return train_test_split(
        features,
        data["pollution_level"],
        test_size=0.20,
        stratify=data["pollution_level"],
        random_state=RANDOM_STATE,
    )


def classifier_pipeline(model: object) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])


def weighted_ovr_auc(estimator: Pipeline, features: pd.DataFrame, target: pd.Series) -> float:
    """ROC-AUC for a multiclass classifier, using scores where probabilities do not exist."""
    classes = np.asarray(estimator.classes_)
    if hasattr(estimator, "predict_proba"):
        scores = np.asarray(estimator.predict_proba(features))
    elif hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features))
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
    else:
        return float("nan")

    aucs, weights = [], []
    values = target.to_numpy()
    for column, label in enumerate(classes):
        binary = (values == label).astype(int)
        if binary.min() != binary.max():
            aucs.append(roc_auc_score(binary, scores[:, column]))
            weights.append(int(binary.sum()))
    return float(np.average(aucs, weights=weights)) if aucs else float("nan")


def classification_metrics(estimator: Pipeline, features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    prediction = estimator.predict(features)
    return pd.DataFrame(
        [
            {
                "accuracy": accuracy_score(target, prediction),
                "precision": precision_score(target, prediction, average="weighted", zero_division=0),
                "recall": recall_score(target, prediction, average="weighted", zero_division=0),
                "f1": f1_score(target, prediction, average="weighted", zero_division=0),
                "roc_auc": weighted_ovr_auc(estimator, features, target),
            }
        ]
    )


def save_report(name: str, table: pd.DataFrame) -> Path:
    TOPIC_REPORTS.mkdir(parents=True, exist_ok=True)
    destination = TOPIC_REPORTS / name
    table.round(4).to_csv(destination, index=False)
    return destination


def plot_path(name: str) -> Path:
    TOPIC_PLOTS.mkdir(parents=True, exist_ok=True)
    return TOPIC_PLOTS / name
