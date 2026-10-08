"""Convenient entry point for the AirSense project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.generate_data import generate_dataset
from src.train import train_project
from src.topics.run_all_topics import main as run_all_topics


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "air_quality_demo.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AirSense project tasks.")
    parser.add_argument(
        "--action",
        choices=("generate", "train", "topics", "all"),
        default="all",
        help="generate data, train existing data, run topic modules, or run both (default: all)",
    )
    parser.add_argument("--rows", type=int, default=2000, help="Demo-data row count")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV data path")
    args = parser.parse_args()

    if args.action in {"generate", "all"}:
        generate_dataset(rows=args.rows, output_path=args.data)
        print(f"Dataset written to {args.data}")

    if args.action in {"train", "all"}:
        if not args.data.exists():
            raise FileNotFoundError(
                f"{args.data} does not exist. Run with --action generate or --action all first."
            )
        result = train_project(args.data, ROOT)
        print("Training complete.")
        print(f"Best pollution classifier: {result['best_classifier']}")
        print(f"AQI regression R²: {result['regression_r2']:.3f}")

    if args.action == "topics":
        run_all_topics()


if __name__ == "__main__":
    main()
