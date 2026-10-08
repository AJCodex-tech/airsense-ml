"""Topic 03: Matplotlib visualization of AQI observations."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.topics.common import load_project_data, plot_path


def main() -> None:
    data, _ = load_project_data()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(data["aqi"], bins=30, color="#177E89", edgecolor="white")
    axes[0].set(title="AQI distribution", xlabel="AQI", ylabel="Observations")
    locations = sorted(data["location"].unique())
    axes[1].boxplot([data.loc[data["location"] == city, "aqi"] for city in locations], tick_labels=locations)
    axes[1].set(title="AQI by location", xlabel="Location", ylabel="AQI")
    axes[1].tick_params(axis="x", rotation=22)
    fig.tight_layout()
    destination = plot_path("03_matplotlib_aqi_overview.png")
    fig.savefig(destination, dpi=170)
    plt.close(fig)
    print(f"Matplotlib chart saved to {destination}")


if __name__ == "__main__":
    main()
