"""Topic 09: Multiclass logistic regression for pollution-level classification."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(LogisticRegression(max_iter=2000, random_state=42)).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Logistic Regression")
    output = save_report("09_logistic_regression.csv", report)
    print(f"Logistic-regression metrics saved to {output}")


if __name__ == "__main__":
    main()
