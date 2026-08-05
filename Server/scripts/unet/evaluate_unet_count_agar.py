from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--model-pt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--min-component-area", default=24, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--overlay-limit", default=30, type=int)
    parser.add_argument("--max-images", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for U-Net evaluation. Install torch in Colab before running this script.") from exc

    from src.unet.torch_backend import CountSegmentationDataset, UNetSegmenter, choose_device, segmentation_metrics_from_logits

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
    overlay_dir = None
    if args.save_overlays:
        overlay_dir = args.output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    segmentation_rows = []
    sample_index = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            target_masks = masks.cpu().numpy()
            metrics = segmentation_metrics_from_logits(logits, masks)
            for local_index in range(len(batch["image_path"])):
                image_relative = str(batch["image_path"][local_index])
                true_count = int(batch["true_count"][local_index])
                category = str(batch["category"][local_index])
                primary_class = str(batch["primary_class"][local_index])
                pred_mask = (probs[local_index, 0] >= args.threshold).astype(np.uint8) * 255
                predicted_count = count_components(pred_mask, args.min_component_area)
                predictions.append(
                    CountPrediction(
                        image_path=image_relative,
                        true_count=true_count,
                        predicted_count=predicted_count,
                        category=category,
                        primary_class=primary_class,
                        error=predicted_count - true_count,
                        abs_error=abs(predicted_count - true_count),
                        squared_error=(predicted_count - true_count) ** 2,
                    )
                )
                segmentation_rows.append(
                    {
                        "image_path": image_relative,
                        "true_count": true_count,
                        "predicted_count": predicted_count,
                        "dice": metrics["dice"],
                        "iou": metrics["iou"],
                        "pixel_accuracy": metrics["pixel_accuracy"],
                    }
                )
                if overlay_dir is not None and sample_index < args.overlay_limit:
                    source_image = read_image(args.dataset_dir / image_relative)
                    resized_mask = resize_mask_to_image(pred_mask, source_image)
                    overlay = render_segmentation_overlay(source_image, resized_mask)
                    cv2.imwrite(str(overlay_dir / safe_output_name(image_relative, ".jpg")), overlay)
                    cv2.imwrite(str(overlay_dir / safe_output_name(image_relative, "_mask.png")), resized_mask)
                sample_index += 1
    count_metrics = evaluate_count_predictions(predictions, None)
    segmentation_frame = pd.DataFrame(segmentation_rows)
    metrics = {
        **{key: value for key, value in count_metrics.metrics.items() if key != "config"},
        "segmentation_dice_mean": float(segmentation_frame["dice"].mean()),
        "segmentation_iou_mean": float(segmentation_frame["iou"].mean()),
        "segmentation_pixel_accuracy_mean": float(segmentation_frame["pixel_accuracy"].mean()),
    }
    pd.DataFrame([prediction.__dict__ for prediction in predictions]).to_csv(args.output_dir / "predictions.csv", index=False)
    segmentation_frame.to_csv(args.output_dir / "segmentation_predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("UNet AGAR counting evaluation summary")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved predictions and metrics to: {args.output_dir}")


def count_components(mask: np.ndarray, min_component_area: int) -> int:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    count = 0
    for index in range(1, component_count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_component_area):
            count += 1
    return count


def render_segmentation_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    colored = np.zeros_like(overlay)
    colored[:, :, 1] = mask
    return cv2.addWeighted(overlay, 0.8, colored, 0.35, 0.0)


def resize_mask_to_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    return cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)


def safe_output_name(path: str, suffix: str) -> str:
    source = Path(path)
    stem = "__".join(source.with_suffix("").parts)
    return f"{stem}{suffix}"


if __name__ == "__main__":
    main()
