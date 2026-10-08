"""Topic 10: Entropy decision tree, the ID3-style classifier demonstration."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from src.topics.common import classification_metrics, classification_split, classifier_pipeline, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = classification_split()
    model = classifier_pipeline(
        DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_leaf=8, random_state=42)
    ).fit(train_x, train_y)
    report = classification_metrics(model, test_x, test_y)
    report.insert(0, "model", "Decision Tree (entropy / ID3-style)")
    output = save_report("10_id3_decision_tree.csv", report)
    print(f"ID3-style decision-tree metrics saved to {output}")


if __name__ == "__main__":
    main()
