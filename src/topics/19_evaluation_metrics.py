"""Topic 19: Accuracy, precision, recall, F1, ROC, and AUC comparison."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from src.topics.common import (
    classification_metrics,
    classification_split,
    classifier_pipeline,
    plot_path,
    save_report,
)


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    candidates = {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "RBF SVM": SVC(kernel="rbf", random_state=42),
    }
    rows = []
    for name, classifier in candidates.items():
        model = classifier_pipeline(classifier).fit(train_x, train_y)
        row = classification_metrics(model, test_x, test_y)
        row.insert(0, "model", name)
        rows.append(row)
    report = pd.concat(rows, ignore_index=True)
    output = save_report("19_evaluation_metrics_comparison.csv", report)

    fig, ax = plt.subplots(figsize=(9, 5))
    report.set_index("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].plot(kind="bar", ax=ax)
    ax.set(title="Evaluation Metrics: Classifier Comparison", ylabel="Score", ylim=(0, 1.05))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    chart = plot_path("19_evaluation_metrics.png")
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    print(f"Evaluation report saved to {output}; comparison chart saved to {chart}")


if __name__ == "__main__":
    main()
