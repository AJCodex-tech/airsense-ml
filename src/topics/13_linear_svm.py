"""Topic 13: Linear Support Vector Machine pollution classification."""

from __future__ import annotations

from sklearn.svm import LinearSVC

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(LinearSVC(dual="auto", random_state=42)).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Linear SVM")
    output = save_report("13_linear_svm.csv", report)
    print(f"Linear-SVM metrics saved to {output}")


if __name__ == "__main__":
    main()
