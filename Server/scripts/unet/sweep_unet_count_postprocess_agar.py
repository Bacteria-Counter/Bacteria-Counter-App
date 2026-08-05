from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classical.detect_count import CountPrediction, evaluate_count_predictions, read_image
from scripts.unet.evaluate_unet_count_agar import count_components, render_segmentation_overlay, resize_mask_to_image, safe_output_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--model-pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--min-component-areas", default="4,8,12,16,24,32")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--max-images", default=0, type=int)
    parser.add_argument("--best-overlays", action="store_true")
    parser.add_argument("--overlay-limit", default=40, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for U-Net evaluation. Install torch in Colab before running this script.") from exc

    from src.unet.torch_backend import CountSegmentationDataset, UNetSegmenter, choose_device

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    if args.max_images > 0:
        manifest = manifest.head(args.max_images).copy()
    dataset = CountSegmentationDataset(
        manifest,
        dataset_dir=args.dataset_dir,
        manifest_root=args.manifest_csv.parent,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    checkpoint = torch.load(args.model_pt, map_location="cpu")
    model = UNetSegmenter(base_channels=int(checkpoint.get("base_channels", 32)))
    model.load_state_dict(checkpoint["state_dict"])
    device = choose_device(args.device)
    model.to(device)
    model.eval()
    cached_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            target_masks = batch["mask"].cpu().numpy()
            for local_index in range(len(batch["image_path"])):
                cached_rows.append(
                    {
                        "image_path": str(batch["image_path"][local_index]),
                        "true_count": int(batch["true_count"][local_index]),
                        "category": str(batch["category"][local_index]),
                        "primary_class": str(batch["primary_class"][local_index]),
                        "prob_mask": probs[local_index, 0].astype(np.float32),
                        "target_mask": target_masks[local_index, 0].astype(np.uint8),
                    }
                )
    records = []
    best_result = None
    configs = list(build_grid(args))
    for index, (threshold, min_component_area) in enumerate(configs, start=1):
        result = evaluate_cached_rows(cached_rows, threshold, min_component_area)
        record = {
            "config_id": index,
            "threshold": threshold,
            "min_component_area": min_component_area,
            **flatten_metrics(result["metrics"]),
        }
        records.append(record)
        write_records(records, args.output_dir / "sweep_results.csv")
        if best_result is None or ranking_key(result["metrics"]) < ranking_key(best_result["metrics"]):
            best_result = result
            with (args.output_dir / "best_config.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "threshold": threshold,
                        "min_component_area": min_component_area,
                    },
                    f,
                    indent=2,
                )
            with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, indent=2)
        print(
            f"{index}/{len(configs)} "
            f"threshold={threshold} "
            f"min_component_area={min_component_area} "
            f"mae={result['metrics']['mae']} "
            f"rmse={result['metrics']['rmse']} "
            f"bias={result['metrics']['bias']}"
        )
    if best_result is None:
        raise RuntimeError("No sweep configurations were evaluated")
    best_config = min(records, key=ranking_key_from_record)
    save_best_run(
        args.output_dir,
        args.dataset_dir,
        best_result,
        threshold=float(best_config["threshold"]),
        min_component_area=int(best_config["min_component_area"]),
        save_overlays=args.best_overlays,
        overlay_limit=args.overlay_limit,
    )
    print("Best UNet AGAR counting validation summary")
    print(f"threshold: {best_config['threshold']}")
    print(f"min_component_area: {best_config['min_component_area']}")
    for key, value in best_result["metrics"].items():
        print(f"{key}: {value}")
    print(f"Saved sweep outputs to: {args.output_dir}")


def build_grid(args: argparse.Namespace) -> list[tuple[float, int]]:
    return list(itertools.product(parse_float_list(args.thresholds), parse_int_list(args.min_component_areas)))


def evaluate_cached_rows(cached_rows: list[dict], threshold: float, min_component_area: int) -> dict:
    predictions = []
    segmentation_rows = []
    pred_masks = []
    for row in cached_rows:
        pred_mask = ((row["prob_mask"] >= threshold).astype(np.uint8)) * 255
        predicted_count = count_components(pred_mask, min_component_area)
        predictions.append(
            CountPrediction(
                image_path=row["image_path"],
                true_count=row["true_count"],
                predicted_count=predicted_count,
                category=row["category"],
                primary_class=row["primary_class"],
                error=predicted_count - row["true_count"],
                abs_error=abs(predicted_count - row["true_count"]),
                squared_error=(predicted_count - row["true_count"]) ** 2,
            )
        )
        segmentation_metrics = segmentation_metrics_from_binary(pred_mask > 0, row["target_mask"] > 0)
        segmentation_rows.append(
            {
                "image_path": row["image_path"],
                "true_count": row["true_count"],
                "predicted_count": predicted_count,
                "dice": segmentation_metrics["dice"],
                "iou": segmentation_metrics["iou"],
                "pixel_accuracy": segmentation_metrics["pixel_accuracy"],
            }
        )
        pred_masks.append(pred_mask)
    count_metrics = evaluate_count_predictions(predictions, None)
    segmentation_frame = pd.DataFrame(segmentation_rows)
    metrics = {
        **{key: value for key, value in count_metrics.metrics.items() if key != "config"},
        "segmentation_dice_mean": float(segmentation_frame["dice"].mean()),
        "segmentation_iou_mean": float(segmentation_frame["iou"].mean()),
        "segmentation_pixel_accuracy_mean": float(segmentation_frame["pixel_accuracy"].mean()),
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "segmentation_frame": segmentation_frame,
        "pred_masks": pred_masks,
    }


def segmentation_metrics_from_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(np.float32)
    target = target_mask.astype(np.float32)
    intersection = float((pred * target).sum())
    pred_sum = float(pred.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum - intersection
    dice = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    accuracy = float((pred_mask == target_mask).mean())
    return {"dice": float(dice), "iou": float(iou), "pixel_accuracy": accuracy}


def save_best_run(
    output_dir: Path,
    dataset_dir: Path,
    result: dict,
    threshold: float,
    min_component_area: int,
    save_overlays: bool,
    overlay_limit: int,
) -> None:
    best_output_dir = output_dir / "best_run"
    best_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([prediction.__dict__ for prediction in result["predictions"]]).to_csv(best_output_dir / "predictions.csv", index=False)
    result["segmentation_frame"].to_csv(best_output_dir / "segmentation_predictions.csv", index=False)
    with (best_output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)
    with (best_output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, "min_component_area": min_component_area}, f, indent=2)
    if not save_overlays:
        return
    overlay_dir = best_output_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for index, (prediction, pred_mask) in enumerate(zip(result["predictions"], result["pred_masks"])):
        if index >= overlay_limit:
            break
        source_image = read_image(dataset_dir / prediction.image_path)
        resized_mask = resize_mask_to_image(pred_mask, source_image)
        overlay = render_segmentation_overlay(source_image, resized_mask)
        cv2.imwrite(str(overlay_dir / safe_output_name(prediction.image_path, ".jpg")), overlay)
        cv2.imwrite(str(overlay_dir / safe_output_name(prediction.image_path, "_mask.png")), resized_mask)


def write_records(records: list[dict], path: Path) -> None:
    frame = pd.DataFrame(records).sort_values(["mae", "rmse", "abs_bias"], ascending=[True, True, True])
    frame.to_csv(path, index=False)


def flatten_metrics(metrics: dict) -> dict:
    output = dict(metrics)
    output["abs_bias"] = abs(float(metrics.get("bias", 0.0)))
    return output


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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    main()
