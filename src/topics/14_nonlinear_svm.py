"""Topic 14: Non-linear RBF-kernel Support Vector Machine classification."""

from __future__ import annotations

from sklearn.svm import SVC

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(SVC(kernel="rbf", random_state=42)).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Non-linear SVM (RBF kernel)")
    output = save_report("14_nonlinear_svm.csv", report)
    print(f"Non-linear-SVM metrics saved to {output}")


if __name__ == "__main__":
    main()
