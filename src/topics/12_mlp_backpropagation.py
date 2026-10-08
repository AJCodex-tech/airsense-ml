"""Topic 12: MLP neural network, activation functions, and backpropagation."""

from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    # ReLU is the hidden-layer activation; scikit-learn trains weights by backpropagation.
    network = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=1e-4,
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model = classifier_pipeline(network).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "MLP (ReLU + backpropagation)")
    output = save_report("12_mlp_backpropagation.csv", report)
    print(f"MLP metrics saved to {output}")


if __name__ == "__main__":
    main()
