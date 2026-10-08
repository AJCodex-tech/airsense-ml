"""Topic 08: Gaussian Naive Bayes pollution-level classification."""

from __future__ import annotations

from sklearn.naive_bayes import GaussianNB

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(GaussianNB()).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Gaussian Naive Bayes")
    output = save_report("08_gaussian_naive_bayes.csv", report)
    print(f"Naive-Bayes metrics saved to {output}")


if __name__ == "__main__":
    main()
