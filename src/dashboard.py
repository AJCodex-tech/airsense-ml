"""Optional Streamlit interface for AirSense predictions and saved project results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.generate_data import generate_dataset
from src.predict import make_feature_row, predict_air_quality
from src.train import train_project


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "air_quality_demo.csv"


@st.cache_resource(show_spinner="Preparing AirSense models...")
def ensure_models() -> None:
    """Make the dashboard self-starting for a first-time project run."""
    if not DATA_PATH.exists():
        generate_dataset(output_path=DATA_PATH)
    if not (ROOT / "models" / "best_pollution_classifier.joblib").exists():
        train_project(DATA_PATH, ROOT)


def main() -> None:
    st.set_page_config(page_title="AirSense", page_icon="🌫️", layout="wide")
    ensure_models()
    st.title("🌫️ AirSense")
    st.caption("Machine-learning AQI prediction, pollution classification, and pattern analysis")

    with st.sidebar:
        st.header("Sensor inputs")
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

    st.subheader("Training outputs")
    metrics_path = ROOT / "reports" / "classification_metrics.csv"
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
    chart_path = ROOT / "plots" / "pca_kmeans_clusters.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="K-Means pollution patterns projected with PCA")

    st.caption("Demo results use synthetic data and a synthetic AQI proxy. Use official observations for real environmental decisions.")


if __name__ == "__main__":
    main()
