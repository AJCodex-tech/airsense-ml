"""Topic 11: Perceptron learning for pollution-level classification."""

from __future__ import annotations

from sklearn.linear_model import Perceptron

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(Perceptron(max_iter=1500, tol=1e-3, random_state=42)).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Perceptron")
    output = save_report("11_perceptron.csv", report)
    print(f"Perceptron metrics saved to {output}")


if __name__ == "__main__":
    main()
