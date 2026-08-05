from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.agar_evaluation import (
    run_agar_count_evaluation,
    save_count_config,
)
from src.classical.detect_count import ClassicalCountConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--background-kernels", default="61,101,151")
    parser.add_argument("--min-areas", default="80,200,500")
    parser.add_argument("--min-circularities", default="0.05,0.15")
    parser.add_argument("--morph-open-kernels", default="5,7")
    parser.add_argument("--morph-close-kernels", default="3")
    parser.add_argument("--max-areas", default="10000")
    parser.add_argument("--threshold-methods", default="otsu")
    parser.add_argument("--threshold-polarities", default="dark")
    parser.add_argument("--plate-mask-options", default="false")
    parser.add_argument("--plate-margin-ratios", default="0.04")
    parser.add_argument("--peak-count-options", default="false")
    parser.add_argument("--large-component-min-areas", default="1200")
    parser.add_argument("--peak-blur-kernels", default="5")
    parser.add_argument("--peak-local-max-kernels", default="9")
    parser.add_argument("--peak-relative-thresholds", default="0.45")
    parser.add_argument("--peak-min-distances", default="2.5")
    parser.add_argument("--area-count-estimation-options", default="true")
    parser.add_argument("--area-count-scales", default="1.6")
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--best-overlays", action="store_true")
    parser.add_argument("--overlay-limit", default=40, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "sweep_results.csv"
    records = load_existing_records(results_path) if args.resume else []
    completed_ids = {int(record["config_id"]) for record in records}
    best_config = None
    best_metrics = None
    configs = list(build_grid(args))
    if records:
        best_record = min(records, key=ranking_key_from_record)
        best_config = config_from_record(best_record)
        best_metrics = metrics_from_record(best_record)

    for index, config in enumerate(configs, start=1):
        if index in completed_ids:
            print(f"{index}/{len(configs)} skipped")
            continue
        evaluation = run_agar_count_evaluation(
            dataset_dir=args.dataset_dir,
            split_csv=args.split_csv,
            config=config,
            max_images=args.max_images,
        )
        record = {
            "config_id": index,
            **flatten_metrics(evaluation.metrics),
            **config.to_dict(),
        }
        records.append(record)
        if best_metrics is None or ranking_key(evaluation.metrics) < ranking_key(best_metrics):
            best_config = config
            best_metrics = evaluation.metrics
            save_count_config(best_config, args.output_dir / "best_config.json")
            with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=2)
        write_records(records, results_path)
        print(
            f"{index}/{len(configs)} "
            f"mae={evaluation.metrics.get('mae')} "
            f"rmse={evaluation.metrics.get('rmse')} "
            f"bias={evaluation.metrics.get('bias')} "
            f"zero_exact={evaluation.metrics.get('zero_exact_accuracy')}"
        )

    write_records(records, results_path)
    if best_config is None or best_metrics is None:
        raise RuntimeError("No sweep configurations were evaluated")

    save_count_config(best_config, args.output_dir / "best_config.json")
    with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    best_output_dir = args.output_dir / "best_run"
    run_agar_count_evaluation(
        dataset_dir=args.dataset_dir,
        split_csv=args.split_csv,
        config=best_config,
        output_dir=best_output_dir,
        max_images=args.max_images,
        save_overlays=args.best_overlays,
        overlay_limit=args.overlay_limit,
    )

    print("Best classical AGAR detection/counting validation summary")
    for key, value in best_metrics.items():
        if key != "config":
            print(f"{key}: {value}")
    print(f"Saved sweep outputs to: {args.output_dir}")


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def write_records(records: list[dict], path: Path) -> None:
    frame = pd.DataFrame(records).sort_values(["mae", "rmse", "abs_bias"], ascending=[True, True, True])
    frame.to_csv(path, index=False)


def build_grid(args: argparse.Namespace):
    for values in itertools.product(
        parse_str_list(args.threshold_methods),
        parse_str_list(args.threshold_polarities),
        parse_int_list(args.background_kernels),
        parse_float_list(args.min_areas),
        parse_float_list(args.min_circularities),
        parse_int_list(args.morph_open_kernels),
        parse_int_list(args.morph_close_kernels),
        parse_float_list(args.max_areas),
        parse_bool_list(args.plate_mask_options),
        parse_float_list(args.plate_margin_ratios),
        parse_bool_list(args.peak_count_options),
        parse_float_list(args.large_component_min_areas),
        parse_int_list(args.peak_blur_kernels),
        parse_int_list(args.peak_local_max_kernels),
        parse_float_list(args.peak_relative_thresholds),
        parse_float_list(args.peak_min_distances),
        parse_bool_list(args.area_count_estimation_options),
        parse_float_list(args.area_count_scales),
    ):
        (
            threshold_method,
            threshold_polarity,
            background_kernel,
            min_area,
            min_circularity,
            open_kernel,
            close_kernel,
            max_area,
            use_plate_mask,
            plate_margin_ratio,
            use_peak_count_estimation,
            large_component_min_area,
            peak_blur_kernel,
            peak_local_max_kernel,
            peak_relative_threshold,
            peak_min_distance,
            use_area_count_estimation,
            area_count_scale,
        ) = values
        yield ClassicalCountConfig(
            threshold_method=threshold_method,
            threshold_polarity=threshold_polarity,
            background_kernel=background_kernel,
            min_area=min_area,
            min_circularity=min_circularity,
            morph_open_kernel=open_kernel,
            morph_close_kernel=close_kernel,
            max_area=max_area,
            use_plate_mask=use_plate_mask,
            plate_margin_ratio=plate_margin_ratio,
            use_peak_count_estimation=use_peak_count_estimation,
            large_component_min_area=large_component_min_area,
            peak_blur_kernel=peak_blur_kernel,
            peak_local_max_kernel=peak_local_max_kernel,
            peak_relative_threshold=peak_relative_threshold,
            peak_min_distance=peak_min_distance,
            use_area_count_estimation=use_area_count_estimation,
            area_count_scale=area_count_scale,
        )


def ranking_key(metrics: dict) -> tuple[float, float, float]:
    return (
        float(metrics.get("mae", float("inf"))),
        float(metrics.get("rmse", float("inf"))),
        abs(float(metrics.get("bias", float("inf")))),
    )


def ranking_key_from_record(record: dict) -> tuple[float, float, float]:
    return (
        float(record.get("mae", float("inf"))),
        float(record.get("rmse", float("inf"))),
        abs(float(record.get("bias", float("inf")))),
    )


def flatten_metrics(metrics: dict) -> dict:
    output = {}
    for key, value in metrics.items():
        if key == "config":
            continue
        output[key] = value
    output["abs_bias"] = abs(float(metrics.get("bias", 0.0)))
    return output


def metrics_from_record(record: dict) -> dict:
    skip_keys = set(ClassicalCountConfig().__dict__.keys()) | {"config_id", "abs_bias"}
    metrics = {key: normalize_json_value(value) for key, value in record.items() if key not in skip_keys}
    metrics["config"] = config_from_record(record).to_dict()
    return metrics


def config_from_record(record: dict) -> ClassicalCountConfig:
    return ClassicalCountConfig(
        gaussian_kernel=int(record.get("gaussian_kernel", 5)),
        clahe_clip_limit=float(record.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid_size=int(record.get("clahe_tile_grid_size", 8)),
        background_kernel=int(record["background_kernel"]),
        threshold_method=str(record["threshold_method"]),
        threshold_polarity=str(record["threshold_polarity"]),
        adaptive_block_size=int(record.get("adaptive_block_size", 51)),
        adaptive_c=float(record.get("adaptive_c", 2.0)),
        morph_open_kernel=int(record["morph_open_kernel"]),
        morph_close_kernel=int(record["morph_close_kernel"]),
        min_area=float(record["min_area"]),
        max_area=float(record.get("max_area", 10000.0)),
        min_circularity=float(record["min_circularity"]),
        use_plate_mask=bool(record.get("use_plate_mask", False)),
        plate_margin_ratio=float(record.get("plate_margin_ratio", 0.04)),
        use_peak_count_estimation=bool(record.get("use_peak_count_estimation", False)),
        large_component_min_area=float(record.get("large_component_min_area", 1200.0)),
        peak_blur_kernel=int(record.get("peak_blur_kernel", 5)),
        peak_local_max_kernel=int(record.get("peak_local_max_kernel", 9)),
        peak_relative_threshold=float(record.get("peak_relative_threshold", 0.45)),
        peak_min_distance=float(record.get("peak_min_distance", 2.5)),
        use_area_count_estimation=bool(record.get("use_area_count_estimation", True)),
        area_count_scale=float(record.get("area_count_scale", 1.6)),
    )


def normalize_json_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_bool_list(value: str) -> list[bool]:
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    return [mapping[item.strip().lower()] for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
