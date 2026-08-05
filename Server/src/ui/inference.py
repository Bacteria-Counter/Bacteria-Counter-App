from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from src.classical.agar_evaluation import load_count_config
from src.classical.detect_count import ClassicalCountConfig, count_colonies, read_image, render_count_overlay
from src.classical.species_classification import extract_species_features, load_species_model, load_species_model_bytes


@dataclass(frozen=True)
class CountingInferenceResult:
    predicted_count: int
    overlay_bgr: np.ndarray
    mask: np.ndarray
    config: dict[str, Any]


@dataclass(frozen=True)
class ClassificationInferenceResult:
    predicted_class: str
    classifier_type: str
    score_name: str
    score_value: float
    config: dict[str, Any]
    accepted: bool
    rejection_reason: str
    rejection_distance_threshold: float | None


@dataclass(frozen=True)
class EmptyPlateGateResult:
    rejected: bool
    reason: str
    estimated_count: int
    foreground_ratio: float
    center_saturated_fraction: float
    center_s_std: float
    center_lab_a_std: float
    center_lab_b_std: float
    center_edge_density: float


@dataclass(frozen=True)
class SpeciesModelPathStatus:
    path: str
    exists: bool
    ready: bool
    is_colab_path: bool
    message: str
    local_candidates: tuple[str, ...]


@dataclass(frozen=True)
class UNetModelPathStatus:
    path: str
    exists: bool
    ready: bool
    is_colab_path: bool
    message: str
    local_candidates: tuple[str, ...]


@dataclass(frozen=True)
class YOLODetection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    area: float


@dataclass(frozen=True)
class YOLOCountingInferenceResult:
    predicted_count: int
    overlay_bgr: np.ndarray
    detections: tuple[YOLODetection, ...]
    config: dict[str, Any]


@dataclass(frozen=True)
class YOLOSpeciesInferenceResult:
    total_detections: int
    overlay_bgr: np.ndarray
    detections: tuple[YOLODetection, ...]
    class_counts: dict[str, int]
    config: dict[str, Any]


CLASSICAL_COUNT_PRESETS = {
    "Config 23 baseline": ClassicalCountConfig(
        threshold_method="otsu",
        threshold_polarity="dark",
        background_kernel=101,
        min_area=500,
        max_area=10000.0,
        min_circularity=0.15,
        morph_open_kernel=5,
        morph_close_kernel=3,
    ),
    "Conservative dense alternative": ClassicalCountConfig(
        threshold_method="otsu",
        threshold_polarity="dark",
        background_kernel=101,
        min_area=500,
        max_area=50000.0,
        min_circularity=0.15,
        morph_open_kernel=5,
        morph_close_kernel=3,
        use_peak_count_estimation=True,
        large_component_min_area=6000.0,
        peak_blur_kernel=5,
        peak_local_max_kernel=7,
        peak_relative_threshold=0.75,
        peak_min_distance=3.5,
        use_area_count_estimation=True,
        area_count_scale=2.8,
    ),
}


CLASSICAL_SPECIES_MODEL_PRESETS = {
    "Latest sweep best run": "/content/drive/MyDrive/bacterial_colony_detection/outputs/classical_species_image/val_sweep_001/best_run/model.json",
}


UNET_COUNT_MODEL_PRESETS = {
    "Locked test best run": "/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_count/train_run_001/model.pt",
}


UNET_SPECIES_MODEL_PRESETS = {
    "Locked best checkpoint": "/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_species_image/train_run_001/model.pt",
}


YOLO_COUNT_MODEL_PRESETS = {
    "Locked best checkpoint": "/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_count/train_run_001/weights/best.pt",
}


YOLO_SPECIES_MODEL_PRESETS = {
    "Locked best checkpoint": "/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_species/train_run_001/weights/best.pt",
}


EMPTY_PLATE_GATE_CONFIG = ClassicalCountConfig(
    threshold_method="otsu",
    threshold_polarity="dark",
    background_kernel=101,
    min_area=180.0,
    max_area=9000.0,
    min_circularity=0.62,
    morph_open_kernel=5,
    morph_close_kernel=3,
    use_plate_mask=True,
    plate_margin_ratio=0.08,
)


def load_image_input(image_path: str = "", uploaded_bytes: bytes | None = None) -> tuple[np.ndarray, str]:
    if uploaded_bytes:
        return decode_uploaded_image(uploaded_bytes), "uploaded image"
    if image_path.strip():
        return read_image(Path(image_path.strip())), image_path.strip()
    raise ValueError("Provide either an image path or an uploaded image")


def decode_uploaded_image(uploaded_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode uploaded image")
    return image


def count_with_classical_model(
    image: np.ndarray,
    preset_name: str = "Config 23 baseline",
    config_json_path: str = "",
) -> CountingInferenceResult:
    config = load_count_config(Path(config_json_path.strip())) if config_json_path.strip() else preset_count_config(preset_name)
    predicted_count, mask, contours = count_colonies(image, config)
    overlay = render_count_overlay(image, predicted_count, contours)
    return CountingInferenceResult(
        predicted_count=int(predicted_count),
        overlay_bgr=overlay,
        mask=mask,
        config=config.to_dict(),
    )


def count_with_unet_model(
    image: np.ndarray,
    model_pt_path: str = "",
    model_pt_bytes: bytes | None = None,
    threshold: float = 0.5,
    min_component_area: int = 2,
) -> CountingInferenceResult:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for local U-Net inference. Install torch in the local environment before using this model family.") from exc

    from src.unet.torch_backend import UNetSegmenter

    checkpoint = load_unet_count_checkpoint(model_pt_path=model_pt_path, model_pt_bytes=model_pt_bytes)
    image_size = int(checkpoint.get("image_size", 256))
    base_channels = int(checkpoint.get("base_channels", 32))
    model = UNetSegmenter(base_channels=base_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_tensor = prepare_unet_image_tensor(image, image_size)
    with torch.no_grad():
        logits = model(image_tensor)
        prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask = ((prob_mask >= float(threshold)).astype(np.uint8)) * 255
    predicted_count = count_connected_components(mask, int(min_component_area))
    resized_mask = resize_binary_mask(mask, image.shape[:2])
    overlay = render_segmentation_overlay(image, resized_mask)
    return CountingInferenceResult(
        predicted_count=int(predicted_count),
        overlay_bgr=overlay,
        mask=resized_mask,
        config={
            "model_type": str(checkpoint.get("model_type", "unet_count_segmenter")),
            "image_size": image_size,
            "base_channels": base_channels,
            "threshold": float(threshold),
            "min_component_area": int(min_component_area),
        },
    )


def classify_with_classical_species_model(
    image: np.ndarray,
    model_json_path: str = "",
    model_json_bytes: bytes | None = None,
    reject_empty_unsupported: bool = False,
) -> ClassificationInferenceResult:
    if model_json_bytes is not None:
        upload_status = inspect_uploaded_species_model(model_json_bytes)
        if not upload_status.ready:
            raise ValueError(upload_status.message)
        model = load_species_model_bytes(model_json_bytes)
    else:
        status = inspect_species_model_path(model_json_path)
        if not status.ready:
            raise FileNotFoundError(status.message)
        model = load_species_model(Path(status.path))
    features = extract_species_features(image, model.config)
    predicted_class, distance = model.predict_features(features)
    accepted = True
    rejection_reason = ""
    if reject_empty_unsupported:
        empty_gate = evaluate_empty_plate_gate(image, features)
        if empty_gate.rejected:
            accepted = False
            predicted_class = "Empty / unsupported"
            rejection_reason = empty_gate.reason
    return ClassificationInferenceResult(
        predicted_class=predicted_class,
        classifier_type=model.classifier_type,
        score_name="Distance",
        score_value=float(distance),
        config=model.config.to_dict(),
        accepted=accepted,
        rejection_reason=rejection_reason,
        rejection_distance_threshold=None,
    )


def classify_with_unet_species_model(
    image: np.ndarray,
    model_pt_path: str = "",
    model_pt_bytes: bytes | None = None,
    reject_empty_unsupported: bool = False,
) -> ClassificationInferenceResult:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for local U-Net inference. Install torch in the local environment before using this model family.") from exc

    from src.classical.species_classification import SpeciesFeatureConfig
    from src.unet.torch_backend import UNetEncoderClassifier

    checkpoint = load_unet_species_checkpoint(model_pt_path=model_pt_path, model_pt_bytes=model_pt_bytes)
    class_names = [str(name) for name in checkpoint["class_names"]]
    image_size = int(checkpoint.get("image_size", 256))
    base_channels = int(checkpoint.get("base_channels", 32))
    dropout = float(checkpoint.get("dropout", 0.2))
    model = UNetEncoderClassifier(
        class_count=len(class_names),
        base_channels=base_channels,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_tensor = prepare_unet_image_tensor(image, image_size)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])
    accepted = True
    rejection_reason = ""
    if reject_empty_unsupported:
        empty_gate = evaluate_empty_plate_gate(
            image,
            extract_species_features(image, SpeciesFeatureConfig(image_size=image_size)),
        )
        if empty_gate.rejected:
            accepted = False
            predicted_class = "Empty / unsupported"
            rejection_reason = empty_gate.reason
    return ClassificationInferenceResult(
        predicted_class=predicted_class,
        classifier_type="unet_encoder_classifier",
        score_name="Confidence",
        score_value=confidence,
        config={
            "model_type": str(checkpoint.get("model_type", "unet_species_image_classifier")),
            "image_size": image_size,
            "base_channels": base_channels,
            "dropout": dropout,
            "class_names": ", ".join(class_names),
        },
        accepted=accepted,
        rejection_reason=rejection_reason,
        rejection_distance_threshold=None,
    )


def count_with_yolo_model(
    image: np.ndarray,
    model_pt_path: str = "",
    model_pt_bytes: bytes | None = None,
    confidence_threshold: float = 0.45,
    image_size: int = 1536,
    iou_threshold: float = 0.6,
    max_det: int = 1000,
) -> YOLOCountingInferenceResult:
    detections = run_yolo_detections(
        image=image,
        model_pt_path=model_pt_path,
        model_pt_bytes=model_pt_bytes,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )
    overlay = render_yolo_overlay(image, detections)
    return YOLOCountingInferenceResult(
        predicted_count=len(detections),
        overlay_bgr=overlay,
        detections=detections,
        config={
            "model_type": "yolo_count_detector",
            "image_size": int(image_size),
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
            "max_det": int(max_det),
        },
    )


def classify_with_yolo_species_model(
    image: np.ndarray,
    model_pt_path: str = "",
    model_pt_bytes: bytes | None = None,
    confidence_threshold: float = 0.25,
    image_size: int = 1536,
    iou_threshold: float = 0.6,
    max_det: int = 1000,
) -> YOLOSpeciesInferenceResult:
    detections = run_yolo_detections(
        image=image,
        model_pt_path=model_pt_path,
        model_pt_bytes=model_pt_bytes,
        confidence_threshold=confidence_threshold,
        image_size=image_size,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )
    class_counts: dict[str, int] = {}
    for detection in detections:
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
    overlay = render_yolo_overlay(image, detections)
    return YOLOSpeciesInferenceResult(
        total_detections=len(detections),
        overlay_bgr=overlay,
        detections=detections,
        class_counts=dict(sorted(class_counts.items())),
        config={
            "model_type": "yolo_species_detector",
            "image_size": int(image_size),
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
            "max_det": int(max_det),
            "class_names": ", ".join(sorted(class_counts)) if class_counts else "",
        },
    )


def preset_count_config(preset_name: str) -> ClassicalCountConfig:
    if preset_name not in CLASSICAL_COUNT_PRESETS:
        raise ValueError(f"Unknown counting preset: {preset_name}")
    return CLASSICAL_COUNT_PRESETS[preset_name]


def inspect_species_model_path(model_json_path: str, search_root: Path | None = None) -> SpeciesModelPathStatus:
    raw_path = model_json_path.strip()
    if not raw_path:
        return SpeciesModelPathStatus(
            path="",
            exists=False,
            ready=False,
            is_colab_path=False,
            message="Provide a local species model JSON path.",
            local_candidates=find_local_species_model_candidates(search_root),
        )

    path = Path(raw_path)
    exists = path.exists()
    is_colab = raw_path.startswith("/content/")
    local_candidates = find_local_species_model_candidates(search_root)
    if exists:
        return SpeciesModelPathStatus(
            path=str(path),
            exists=True,
            ready=True,
            is_colab_path=is_colab,
            message=f"Local model is ready: {path}",
            local_candidates=local_candidates,
        )
    if is_colab:
        return SpeciesModelPathStatus(
            path=str(path),
            exists=False,
            ready=False,
            is_colab_path=True,
            message=(
                "This preset points to a Colab or Google Drive path under /content/. "
                "The local Streamlit app cannot read that location. Use a local model.json path."
            ),
            local_candidates=local_candidates,
        )
    return SpeciesModelPathStatus(
        path=str(path),
        exists=False,
        ready=False,
        is_colab_path=False,
        message=f"Model file not found: {path}",
        local_candidates=local_candidates,
    )


def inspect_uploaded_species_model(model_json_bytes: bytes | None, uploaded_name: str = "") -> SpeciesModelPathStatus:
    display_name = uploaded_name or "uploaded model.json"
    if model_json_bytes is None:
        return SpeciesModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message="Upload a species model JSON or provide a local path.",
            local_candidates=(),
        )
    try:
        load_species_model_bytes(model_json_bytes)
    except Exception as exc:
        return SpeciesModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message=f"Uploaded species model is invalid: {exc}",
            local_candidates=(),
        )
    return SpeciesModelPathStatus(
        path=display_name,
        exists=True,
        ready=True,
        is_colab_path=False,
        message=f"Uploaded model is ready: {display_name}",
        local_candidates=(),
    )


def inspect_unet_count_model_path(model_pt_path: str, search_root: Path | None = None) -> UNetModelPathStatus:
    return inspect_unet_model_path(
        model_pt_path=model_pt_path,
        search_root=search_root,
        label="U-Net counting",
        missing_message="Provide a local U-Net counting model .pt path.",
        candidate_fn=find_local_unet_count_model_candidates,
    )


def inspect_unet_species_model_path(model_pt_path: str, search_root: Path | None = None) -> UNetModelPathStatus:
    return inspect_unet_model_path(
        model_pt_path=model_pt_path,
        search_root=search_root,
        label="U-Net species",
        missing_message="Provide a local U-Net species model .pt path.",
        candidate_fn=find_local_unet_species_model_candidates,
    )


def inspect_yolo_count_model_path(model_pt_path: str, search_root: Path | None = None) -> UNetModelPathStatus:
    return inspect_unet_model_path(
        model_pt_path=model_pt_path,
        search_root=search_root,
        label="YOLO counting",
        missing_message="Provide a local YOLO counting model .pt path.",
        candidate_fn=find_local_yolo_count_model_candidates,
    )


def inspect_yolo_species_model_path(model_pt_path: str, search_root: Path | None = None) -> UNetModelPathStatus:
    return inspect_unet_model_path(
        model_pt_path=model_pt_path,
        search_root=search_root,
        label="YOLO species",
        missing_message="Provide a local YOLO species model .pt path.",
        candidate_fn=find_local_yolo_species_model_candidates,
    )


def inspect_unet_model_path(
    model_pt_path: str,
    search_root: Path | None,
    label: str,
    missing_message: str,
    candidate_fn,
) -> UNetModelPathStatus:
    raw_path = model_pt_path.strip()
    if not raw_path:
        return UNetModelPathStatus(
            path="",
            exists=False,
            ready=False,
            is_colab_path=False,
            message=missing_message,
            local_candidates=candidate_fn(search_root),
        )
    path = Path(raw_path)
    exists = path.exists()
    is_colab = raw_path.startswith("/content/")
    local_candidates = candidate_fn(search_root)
    if exists:
        return UNetModelPathStatus(
            path=str(path),
            exists=True,
            ready=True,
            is_colab_path=is_colab,
            message=f"Local {label} checkpoint is ready: {path}",
            local_candidates=local_candidates,
        )
    if is_colab:
        return UNetModelPathStatus(
            path=str(path),
            exists=False,
            ready=False,
            is_colab_path=True,
            message=(
                "This preset points to a Colab or Google Drive path under /content/. "
                "The local Streamlit app cannot read that location. Use a local model.pt path or upload the checkpoint."
            ),
            local_candidates=local_candidates,
        )
    return UNetModelPathStatus(
        path=str(path),
        exists=False,
        ready=False,
        is_colab_path=False,
        message=f"{label} checkpoint not found: {path}",
        local_candidates=local_candidates,
    )


def inspect_uploaded_unet_count_model(model_pt_bytes: bytes | None, uploaded_name: str = "") -> UNetModelPathStatus:
    return inspect_uploaded_unet_model(
        model_pt_bytes=model_pt_bytes,
        uploaded_name=uploaded_name,
        empty_message="Upload a U-Net counting model checkpoint or provide a local path.",
        ready_label="Uploaded U-Net counting checkpoint is ready",
        invalid_label="Uploaded U-Net counting checkpoint is invalid",
        validator=load_unet_count_checkpoint,
    )


def inspect_uploaded_unet_species_model(model_pt_bytes: bytes | None, uploaded_name: str = "") -> UNetModelPathStatus:
    return inspect_uploaded_unet_model(
        model_pt_bytes=model_pt_bytes,
        uploaded_name=uploaded_name,
        empty_message="Upload a U-Net species model checkpoint or provide a local path.",
        ready_label="Uploaded U-Net species checkpoint is ready",
        invalid_label="Uploaded U-Net species checkpoint is invalid",
        validator=load_unet_species_checkpoint,
    )


def inspect_uploaded_yolo_count_model(model_pt_bytes: bytes | None, uploaded_name: str = "") -> UNetModelPathStatus:
    return inspect_uploaded_unet_model(
        model_pt_bytes=model_pt_bytes,
        uploaded_name=uploaded_name,
        empty_message="Upload a YOLO counting model checkpoint or provide a local path.",
        ready_label="Uploaded YOLO counting checkpoint is ready",
        invalid_label="Uploaded YOLO counting checkpoint is invalid",
        validator=load_yolo_model_for_validation,
    )


def inspect_uploaded_yolo_species_model(model_pt_bytes: bytes | None, uploaded_name: str = "") -> UNetModelPathStatus:
    return inspect_uploaded_unet_model(
        model_pt_bytes=model_pt_bytes,
        uploaded_name=uploaded_name,
        empty_message="Upload a YOLO species model checkpoint or provide a local path.",
        ready_label="Uploaded YOLO species checkpoint is ready",
        invalid_label="Uploaded YOLO species checkpoint is invalid",
        validator=load_yolo_model_for_validation,
    )


def inspect_uploaded_unet_model(
    model_pt_bytes: bytes | None,
    uploaded_name: str,
    empty_message: str,
    ready_label: str,
    invalid_label: str,
    validator,
) -> UNetModelPathStatus:
    display_name = uploaded_name or "uploaded model.pt"
    if model_pt_bytes is None:
        return UNetModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message=empty_message,
            local_candidates=(),
        )
    try:
        validator(model_pt_bytes=model_pt_bytes)
    except Exception as exc:
        return UNetModelPathStatus(
            path=display_name,
            exists=False,
            ready=False,
            is_colab_path=False,
            message=f"{invalid_label}: {exc}",
            local_candidates=(),
        )
    return UNetModelPathStatus(
        path=display_name,
        exists=True,
        ready=True,
        is_colab_path=False,
        message=f"{ready_label}: {display_name}",
        local_candidates=(),
    )


def find_local_species_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("model.json"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        if not any("species" in part for part in relative_parts[:-1]):
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def find_local_unet_count_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("model.pt"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        if "unet_count" not in "/".join(relative_parts):
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def find_local_unet_species_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("model.pt"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        if "unet_species" not in "/".join(relative_parts):
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def find_local_yolo_count_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("*.pt"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        joined = "/".join(relative_parts)
        if "yolo_count" not in joined:
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def find_local_yolo_species_model_candidates(search_root: Path | None = None) -> tuple[str, ...]:
    if search_root is None or not search_root.exists():
        return ()
    candidates = []
    for path in search_root.rglob("*.pt"):
        try:
            relative_parts = [part.lower() for part in path.relative_to(search_root).parts]
        except ValueError:
            relative_parts = [part.lower() for part in path.parts]
        joined = "/".join(relative_parts)
        if "yolo_species" not in joined:
            continue
        candidates.append(str(path))
    return tuple(sorted(candidates)[:8])


def evaluate_empty_plate_gate(
    image: np.ndarray,
    features: dict[str, float],
    config: ClassicalCountConfig | None = None,
) -> EmptyPlateGateResult:
    gate_config = config or EMPTY_PLATE_GATE_CONFIG
    estimated_count, mask, _ = count_colonies(image, gate_config)
    foreground_ratio = float((mask > 0).mean())
    center_saturated_fraction = float(features.get("feature_center_saturated_fraction", 0.0))
    center_s_std = float(features.get("feature_center_s_std", 0.0))
    center_lab_a_std = float(features.get("feature_center_lab_a_std", 0.0))
    center_lab_b_std = float(features.get("feature_center_lab_b_std", 0.0))
    center_edge_density = float(features.get("feature_center_edge_density", 0.0))
    rejected = (
        estimated_count <= 1
        and foreground_ratio < 0.0024
        and center_saturated_fraction < 0.03
        and center_s_std < 14.0
        and center_lab_a_std < 4.0
        and center_lab_b_std < 4.0
        and center_edge_density < 0.03
    )
    reason = ""
    if rejected:
        reason = "The image has no colony-like evidence inside the plate and is treated as empty or unsupported."
    return EmptyPlateGateResult(
        rejected=rejected,
        reason=reason,
        estimated_count=int(estimated_count),
        foreground_ratio=foreground_ratio,
        center_saturated_fraction=center_saturated_fraction,
        center_s_std=center_s_std,
        center_lab_a_std=center_lab_a_std,
        center_lab_b_std=center_lab_b_std,
        center_edge_density=center_edge_density,
    )


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


def config_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    return [{"Parameter": key, "Value": str(value)} for key, value in config.items()]


def load_unet_count_checkpoint(model_pt_path: str = "", model_pt_bytes: bytes | None = None) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for local U-Net inference. Install torch in the local environment before using this model family.") from exc
    if model_pt_bytes is not None:
        checkpoint = torch.load(io.BytesIO(model_pt_bytes), map_location="cpu")
    else:
        status = inspect_unet_count_model_path(model_pt_path)
        if not status.ready:
            raise FileNotFoundError(status.message)
        checkpoint = torch.load(Path(status.path), map_location="cpu")
    validate_unet_count_checkpoint(checkpoint)
    return checkpoint


def load_unet_species_checkpoint(model_pt_path: str = "", model_pt_bytes: bytes | None = None) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for local U-Net inference. Install torch in the local environment before using this model family.") from exc
    if model_pt_bytes is not None:
        checkpoint = torch.load(io.BytesIO(model_pt_bytes), map_location="cpu")
    else:
        status = inspect_unet_species_model_path(model_pt_path)
        if not status.ready:
            raise FileNotFoundError(status.message)
        checkpoint = torch.load(Path(status.path), map_location="cpu")
    validate_unet_species_checkpoint(checkpoint)
    return checkpoint


def validate_unet_count_checkpoint(checkpoint: dict[str, Any]) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing state_dict.")
    model_type = str(checkpoint.get("model_type", ""))
    if model_type and model_type != "unet_count_segmenter":
        raise ValueError(f"Checkpoint model_type is not a U-Net counting model: {model_type}")


def validate_unet_species_checkpoint(checkpoint: dict[str, Any]) -> None:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")
    if "state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing state_dict.")
    if "class_names" not in checkpoint:
        raise ValueError("Checkpoint is missing class_names.")
    model_type = str(checkpoint.get("model_type", ""))
    if model_type and model_type != "unet_species_image_classifier":
        raise ValueError(f"Checkpoint model_type is not a U-Net species model: {model_type}")


def prepare_unet_image_tensor(image: np.ndarray, image_size: int):
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for local U-Net inference. Install torch in the local environment before using this model family.") from exc
    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def resize_binary_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def count_connected_components(mask: np.ndarray, min_component_area: int) -> int:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
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


def load_yolo_model_for_validation(model_pt_path: str = "", model_pt_bytes: bytes | None = None):
    model, cleanup_path = load_yolo_model(model_pt_path=model_pt_path, model_pt_bytes=model_pt_bytes)
    if cleanup_path is not None and cleanup_path.exists():
        cleanup_path.unlink()
    return model


def load_yolo_model(model_pt_path: str = "", model_pt_bytes: bytes | None = None):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required for local YOLO inference. Install ultralytics in the local environment before using this model family.") from exc
    cleanup_path: Path | None = None
    if model_pt_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
            handle.write(model_pt_bytes)
            cleanup_path = Path(handle.name)
        model_path = cleanup_path
    else:
        status = inspect_yolo_count_model_path(model_pt_path)
        if not status.ready:
            status = inspect_yolo_species_model_path(model_pt_path)
        if not status.ready:
            raise FileNotFoundError(status.message)
        model_path = Path(status.path)
    try:
        model = YOLO(str(model_path))
    except Exception:
        if cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()
        raise
    return model, cleanup_path


def run_yolo_detections(
    image: np.ndarray,
    model_pt_path: str = "",
    model_pt_bytes: bytes | None = None,
    confidence_threshold: float = 0.25,
    image_size: int = 1536,
    iou_threshold: float = 0.6,
    max_det: int = 1000,
) -> tuple[YOLODetection, ...]:
    model, cleanup_path = load_yolo_model(model_pt_path=model_pt_path, model_pt_bytes=model_pt_bytes)
    try:
        results = model.predict(
            source=image,
            conf=float(confidence_threshold),
            imgsz=int(image_size),
            iou=float(iou_threshold),
            max_det=int(max_det),
            verbose=False,
            save=False,
        )
    finally:
        if cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()
    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()
    names = getattr(result, "names", getattr(model, "names", {}))
    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    detections = []
    for index, confidence in enumerate(confidences):
        box = xyxy[index]
        class_id = int(class_ids[index])
        class_name = resolve_yolo_class_name(names, class_id)
        detections.append(
            YOLODetection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                area=max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])),
            )
        )
    return tuple(detections)


def resolve_yolo_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def render_yolo_overlay(image: np.ndarray, detections: tuple[YOLODetection, ...]) -> np.ndarray:
    overlay = image.copy()
    for detection in detections:
        color = yolo_class_color(detection.class_id)
        start = (int(round(detection.x1)), int(round(detection.y1)))
        end = (int(round(detection.x2)), int(round(detection.y2)))
        cv2.rectangle(overlay, start, end, color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_origin = (start[0], max(18, start[1] - 6))
        cv2.putText(overlay, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return overlay


def yolo_class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (33, 150, 243),
        (76, 175, 80),
        (255, 193, 7),
        (156, 39, 176),
        (244, 67, 54),
        (0, 188, 212),
        (255, 87, 34),
    )
    return palette[class_id % len(palette)]


def detection_rows(detections: tuple[YOLODetection, ...]) -> list[dict[str, str]]:
    rows = []
    for detection in detections:
        rows.append(
            {
                "Class": detection.class_name,
                "Confidence": f"{detection.confidence:.3f}",
                "Box": f"({detection.x1:.1f}, {detection.y1:.1f}) - ({detection.x2:.1f}, {detection.y2:.1f})",
                "Area": f"{detection.area:.1f}",
            }
        )
    return rows
