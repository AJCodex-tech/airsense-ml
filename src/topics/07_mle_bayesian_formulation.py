"""Topic 07: Gaussian maximum-likelihood estimates and Bayesian posteriors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.topics.common import load_project_data, save_report


def normal_pdf(value: float, mean: float, variance: float) -> float:
    return float(np.exp(-((value - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance))


def main() -> None:
    data, _ = load_project_data()
    observed = data[["pm25", "pollution_level"]].dropna()
    parameters = observed.groupby("pollution_level")["pm25"].agg(["count", "mean", "var"]).reset_index()
    parameters["prior_probability"] = parameters["count"] / parameters["count"].sum()
    parameters["variance"] = parameters["var"].clip(lower=1e-6)
    sample_pm25 = 100.0
    likelihood = np.array([normal_pdf(sample_pm25, row.mean, row.variance) for row in parameters.itertuples()])
    posterior = likelihood * parameters["prior_probability"].to_numpy()
    parameters["posterior_for_pm25_100"] = posterior / posterior.sum()
    report = parameters[["pollution_level", "count", "mean", "variance", "prior_probability", "posterior_for_pm25_100"]]
    output = save_report("07_mle_bayesian_parameters.csv", report)
    print(f"MLE/Bayesian parameters saved to {output}")


if __name__ == "__main__":
    main()
