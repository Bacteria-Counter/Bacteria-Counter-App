from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.species_classification import (
    evaluate_species_classifier,
    load_species_model,
    save_species_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--model-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_species_model(args.model_json)
    evaluation = evaluate_species_classifier(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        model=model,
        max_images=args.max_images,
    )
    save_species_evaluation(evaluation, args.output_dir)

    print("Classical AGAR species image classification evaluation")
    print(f"image_count: {evaluation.metrics['image_count']}")
    print(f"accuracy: {evaluation.metrics['accuracy']}")
    print(f"macro_f1: {evaluation.metrics['macro_f1']}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
