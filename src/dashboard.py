"""Streamlit interface for live prediction and the complete AirSense project output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.generate_data import generate_dataset
from src.predict import make_feature_row, predict_air_quality
from src.topics.run_all_topics import main as run_all_topics
from src.train import train_project


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "air_quality_demo.csv"
REPORTS_DIR = ROOT / "reports"
PLOTS_DIR = ROOT / "plots"
TOPIC_REPORTS_DIR = REPORTS_DIR / "topics"
TOPIC_PLOTS_DIR = PLOTS_DIR / "topics"


@st.cache_resource(show_spinner="Preparing AirSense models...")
def ensure_models() -> None:
    """Make the dashboard self-starting for a first-time project run."""
    if not DATA_PATH.exists():
        generate_dataset(output_path=DATA_PATH)
    if not (ROOT / "models" / "best_pollution_classifier.joblib").exists():
        train_project(DATA_PATH, ROOT)


def display_plot_gallery(paths: list[Path]) -> None:
    """Display generated figures in a compact two-column gallery."""
    for start in range(0, len(paths), 2):
        columns = st.columns(2)
        for column, image_path in zip(columns, paths[start : start + 2]):
            column.image(
                str(image_path),
                caption=image_path.stem.replace("_", " ").title(),
                use_container_width=True,
            )


def topic_label(path: Path) -> str:
    return path.stem.split("_", 1)[-1].replace("_", " ").title()


def show_overview() -> None:
    st.subheader("Complete ML pipeline results")
    regression_path = REPORTS_DIR / "regression_metrics.csv"
    classifier_path = REPORTS_DIR / "classification_metrics.csv"
    clustering_path = REPORTS_DIR / "clustering_diagnostics.csv"
    left, right = st.columns(2)
    if regression_path.exists():
        left.markdown("#### AQI regression")
        left.dataframe(pd.read_csv(regression_path), use_container_width=True, hide_index=True)
    if clustering_path.exists():
        right.markdown("#### Clustering diagnostics")
        right.dataframe(pd.read_csv(clustering_path), use_container_width=True, hide_index=True)
    if classifier_path.exists():
        st.markdown("#### Pollution-classifier comparison")
        st.dataframe(pd.read_csv(classifier_path), use_container_width=True, hide_index=True)

    main_plots = sorted(path for path in PLOTS_DIR.glob("*.png") if path.is_file())
    if main_plots:
        st.markdown("#### Pipeline visualizations")
        display_plot_gallery(main_plots)


def show_topic_results() -> None:
    st.subheader("Syllabus topic-by-topic demonstrations")
    st.write("Every topic has its own Python file and its own generated report or chart.")
    if st.button("Run all 19 topic modules", type="primary", use_container_width=False):
        with st.spinner("Running all topic demonstrations..."):
            run_all_topics()
        st.success("All topic modules completed. The reports and charts below are refreshed.")

    reports = sorted(TOPIC_REPORTS_DIR.glob("*.csv")) if TOPIC_REPORTS_DIR.exists() else []
    topic_images = sorted(TOPIC_PLOTS_DIR.glob("*.png")) if TOPIC_PLOTS_DIR.exists() else []
    report_count, image_count, module_count = st.columns(3)
    report_count.metric("Topic reports", len(reports))
    image_count.metric("Topic visualizations", len(topic_images))
    module_count.metric("Separate modules", 19)

    if not reports and not topic_images:
        st.info("Use “Run all 19 topic modules” to generate the separate outputs.")
        return

    for report_path in reports:
        topic_number = report_path.name[:2]
        matching_images = [image for image in topic_images if image.name.startswith(topic_number)]
        with st.expander(f"{topic_number} — {topic_label(report_path)}", expanded=False):
            st.dataframe(pd.read_csv(report_path), use_container_width=True, hide_index=True)
            if matching_images:
                display_plot_gallery(matching_images)

    chart_only = [image for image in topic_images if not any(report.name.startswith(image.name[:2]) for report in reports)]
    if chart_only:
        st.markdown("#### Standalone visualization topics")
        display_plot_gallery(chart_only)


def main() -> None:
    st.set_page_config(page_title="AirSense", page_icon="🌫️", layout="wide")
    ensure_models()
    st.title("🌫️ AirSense")
    st.caption("Machine-learning AQI prediction, pollution classification, pattern analysis, and topic-wise demonstrations")

    with st.sidebar:
        st.header("Live sensor inputs")
        location = st.selectbox("Location", ["New Delhi", "Mumbai", "Bengaluru", "Kolkata"])
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 350.0, 86.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 145.0)
        no2 = st.slider("NO₂ (µg/m³)", 0.0, 250.0, 72.0)
        so2 = st.slider("SO₂ (µg/m³)", 0.0, 150.0, 18.0)
        co = st.slider("CO (mg/m³)", 0.0, 8.0, 1.4)
        o3 = st.slider("O₃ (µg/m³)", 0.0, 250.0, 52.0)
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 30.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 58.0)
        wind_speed = st.slider("Wind speed (m/s)", 0.0, 20.0, 2.8)

    prediction_tab, overview_tab, topics_tab = st.tabs(
        ["Live prediction", "Full pipeline", "All syllabus topics"]
    )
    with prediction_tab:
        features = make_feature_row(
            pm25, pm10, no2, so2, co, o3, temperature, humidity, wind_speed, location
        )
        result = predict_air_quality(features)
        first, second, third = st.columns(3)
        first.metric("Predicted AQI", f"{result['predicted_aqi']:.1f}")
        second.metric("Pollution level", result["pollution_level"])
        third.metric("Location", location)
        st.subheader("Class probability")
        probabilities = result["probabilities"]
        if probabilities:
            st.bar_chart(pd.DataFrame.from_dict(probabilities, orient="index", columns=["Probability"]))
        else:
            st.info("The selected best classifier does not expose class probabilities.")

    with overview_tab:
        show_overview()

    with topics_tab:
        show_topic_results()

    st.caption("Demo results use synthetic data and a synthetic AQI proxy. Use official observations for real environmental decisions.")


if __name__ == "__main__":
    main()
