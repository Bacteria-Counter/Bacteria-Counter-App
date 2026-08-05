from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.count_error_analysis import analyze_count_predictions, save_count_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worst-n", default=30, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, worst = analyze_count_predictions(args.predictions_csv, worst_n=args.worst_n)
    save_count_error_analysis(summary, worst, args.output_dir)

    overall = summary["overall"]
    print("Classical AGAR detection/counting error analysis")
    print(f"image_count: {overall['image_count']}")
    print(f"true_total: {overall['true_total']}")
    print(f"predicted_total: {overall['predicted_total']}")
    print(f"mae: {overall['mae']}")
    print(f"rmse: {overall['rmse']}")
    print(f"bias: {overall['bias']}")
    print(f"zero_exact_accuracy: {overall.get('zero_exact_accuracy')}")
    print(f"Saved analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
