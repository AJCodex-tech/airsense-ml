"""Topic 06: Multiple linear regression using pollutants, weather, time, and location."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.topics.common import classifier_pipeline, regression_split, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = regression_split()
    model = classifier_pipeline(LinearRegression())
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    report = pd.DataFrame(
        [{"model": "Multiple Linear Regression", "mae": mean_absolute_error(test_y, predicted), "rmse": mean_squared_error(test_y, predicted) ** 0.5, "r2": r2_score(test_y, predicted)}]
    )
    output = save_report("06_multiple_linear_regression.csv", report)
    print(f"Multiple-regression metrics saved to {output}")


if __name__ == "__main__":
    main()
