"""Command-line inference for trained AirSense models."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.data_pipeline import add_time_features


ROOT = Path(__file__).resolve().parents[1]


def make_feature_row(
    pm25: float,
    pm10: float,
    no2: float,
    so2: float,
    co: float,
    o3: float,
    temperature: float,
    humidity: float,
    wind_speed: float,
    location: str,
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Build one model-ready observation from environmental inputs."""
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    raw = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "location": location,
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "so2": so2,
                "co": co,
                "o3": o3,
                "temperature": temperature,
                "humidity": humidity,
                "wind_speed": wind_speed,
            }
        ]
    )
    return add_time_features(raw)


def predict_air_quality(features: pd.DataFrame, model_directory: Path = ROOT / "models") -> dict[str, object]:
    """Predict AQI and pollution class from saved project pipelines."""
    regressor_path = model_directory / "aqi_multiple_linear_regressor.joblib"
    classifier_path = model_directory / "best_pollution_classifier.joblib"
    if not regressor_path.exists() or not classifier_path.exists():
        raise FileNotFoundError("Model files are missing. Run `python main.py --action all` first.")

    regressor = joblib.load(regressor_path)
    classifier = joblib.load(classifier_path)
    predicted_aqi = float(regressor.predict(features)[0])
    predicted_class = str(classifier.predict(features)[0])
    probability: dict[str, float] = {}
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(features)[0]
        probability = {str(label): float(value) for label, value in zip(classifier.classes_, probabilities)}
    return {"predicted_aqi": predicted_aqi, "pollution_level": predicted_class, "probabilities": probability}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AQI and pollution level with AirSense.")
    parser.add_argument("--pm25", type=float, required=True)
    parser.add_argument("--pm10", type=float, required=True)
    parser.add_argument("--no2", type=float, required=True)
    parser.add_argument("--so2", type=float, required=True)
    parser.add_argument("--co", type=float, required=True)
    parser.add_argument("--o3", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--wind-speed", dest="wind_speed", type=float, required=True)
    parser.add_argument("--location", default="New Delhi")
    parser.add_argument("--timestamp", default=None, help="YYYY-MM-DD HH:MM:SS; defaults to now")
    args = parser.parse_args()

    row = make_feature_row(**vars(args))
    result = predict_air_quality(row)
    print(f"Predicted AQI: {result['predicted_aqi']:.1f}")
    print(f"Pollution level: {result['pollution_level']}")
    if result["probabilities"]:
        print("Class probabilities:")
        for label, value in sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True):
            print(f"  {label}: {value:.1%}")


if __name__ == "__main__":
    main()
