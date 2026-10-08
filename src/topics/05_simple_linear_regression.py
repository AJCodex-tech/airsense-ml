"""Topic 05: Simple linear regression using PM2.5 to predict AQI."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.data_pipeline import build_preprocessor
from src.topics.common import regression_split, save_report


def main() -> None:
    train_x, test_x, train_y, test_y = regression_split()
    model = Pipeline(
        [("preprocessor", build_preprocessor(numeric_features=["pm25"], categorical_features=[])), ("model", LinearRegression())]
    )
    model.fit(train_x[["pm25"]], train_y)
    predicted = model.predict(test_x[["pm25"]])
    report = pd.DataFrame(
        [{"model": "Simple Linear Regression (PM2.5 → AQI)", "mae": mean_absolute_error(test_y, predicted), "rmse": mean_squared_error(test_y, predicted) ** 0.5, "r2": r2_score(test_y, predicted)}]
    )
    output = save_report("05_simple_linear_regression.csv", report)
    print(f"Simple-regression metrics saved to {output}")


if __name__ == "__main__":
    main()
