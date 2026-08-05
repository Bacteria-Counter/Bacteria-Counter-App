from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ClassicalCountConfig:
    gaussian_kernel: int = 5
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    background_kernel: int = 61
    threshold_method: str = "otsu"
    threshold_polarity: str = "dark"
    adaptive_block_size: int = 51
    adaptive_c: float = 2.0
    morph_open_kernel: int = 3
    morph_close_kernel: int = 3
    min_area: float = 12.0
    max_area: float = 10000.0
    min_circularity: float = 0.05
    use_plate_mask: bool = False
    plate_margin_ratio: float = 0.04
    use_peak_count_estimation: bool = False
    large_component_min_area: float = 1200.0
    peak_blur_kernel: int = 5
    peak_local_max_kernel: int = 9
    peak_relative_threshold: float = 0.45
    peak_min_distance: float = 2.5
    use_area_count_estimation: bool = True
    area_count_scale: float = 1.6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CountPrediction:
    image_path: str
    true_count: int
    predicted_count: int
    category: str
    primary_class: str
    error: int
    abs_error: int
    squared_error: int


@dataclass(frozen=True)
class CountEvaluation:
    metrics: dict[str, Any]
    predictions: list[CountPrediction]


def count_colonies(image: np.ndarray, config: ClassicalCountConfig | None = None) -> tuple[int, np.ndarray, list[np.ndarray]]:
    active_config = config or ClassicalCountConfig()
    prepared = preprocess_image(image, active_config)
    binary = threshold_image(prepared, active_config)
    cleaned = clean_mask(binary, active_config)
    masked = apply_plate_mask(cleaned, active_config)
    mask, contours = filter_components(masked, active_config)
    count = estimate_colony_count(mask, contours, active_config)
    return count, mask, contours


def preprocess_image(image: np.ndarray, config: ClassicalCountConfig) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_kernel = odd_kernel(config.gaussian_kernel)
    if blur_kernel > 1:
        gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    tile_grid = max(1, int(config.clahe_tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(config.clahe_clip_limit), tileGridSize=(tile_grid, tile_grid))
    equalized = clahe.apply(gray)

    background_kernel = odd_kernel(config.background_kernel)
    if background_kernel <= 1:
        normalized = equalized
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_kernel, background_kernel))
        if config.threshold_polarity == "bright":
            background = cv2.morphologyEx(equalized, cv2.MORPH_OPEN, kernel)
            normalized = cv2.subtract(equalized, background)
        else:
            background = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)
            normalized = cv2.subtract(background, equalized)

    return normalize_uint8(normalized)


def threshold_image(image: np.ndarray, config: ClassicalCountConfig) -> np.ndarray:
    method = config.threshold_method.lower()
    if method == "adaptive":
        block_size = odd_kernel(config.adaptive_block_size)
        block_size = max(3, block_size)
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            float(config.adaptive_c),
        )
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
    raise ValueError(f"Unsupported threshold method: {config.threshold_method}")


def clean_mask(mask: np.ndarray, config: ClassicalCountConfig) -> np.ndarray:
    output = mask.copy()
    open_kernel = odd_kernel(config.morph_open_kernel)
    close_kernel = odd_kernel(config.morph_close_kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    return output


def filter_components(mask: np.ndarray, config: ClassicalCountConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    output = np.zeros(mask.shape, dtype=np.uint8)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.min_area or area > config.max_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 0.0 if perimeter <= 0 else (4.0 * pi * area) / (perimeter * perimeter)
        if circularity < config.min_circularity:
            continue
        kept.append(contour)

    kept.sort(key=lambda contour: cv2.boundingRect(contour)[0:2])
    if kept:
        cv2.drawContours(output, kept, -1, 255, thickness=cv2.FILLED)
    return output, kept


def apply_plate_mask(mask: np.ndarray, config: ClassicalCountConfig) -> np.ndarray:
    if not config.use_plate_mask:
        return mask
    height, width = mask.shape[:2]
    plate = np.zeros((height, width), dtype=np.uint8)
    radius = max(1, int(round(min(height, width) * 0.5 * max(0.1, 1.0 - config.plate_margin_ratio))))
    cv2.circle(plate, (width // 2, height // 2), radius, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(mask, plate)


def estimate_colony_count(mask: np.ndarray, contours: list[np.ndarray], config: ClassicalCountConfig) -> int:
    if not contours:
        return 0
    reference_area = estimate_reference_area(contours, config)
    total = 0
    for contour in contours:
        total += estimate_component_count(mask, contour, reference_area, config)
    return int(total)


def estimate_reference_area(contours: list[np.ndarray], config: ClassicalCountConfig) -> float:
    areas = sorted(float(cv2.contourArea(contour)) for contour in contours)
    if not areas:
        return float(max(config.min_area, 1.0))
    sample_size = max(1, len(areas) // 2)
    reference = float(np.median(np.array(areas[:sample_size], dtype=np.float64)))
    return max(float(config.min_area), reference)


def estimate_component_count(
    mask: np.ndarray,
    contour: np.ndarray,
    reference_area: float,
    config: ClassicalCountConfig,
) -> int:
    area = float(cv2.contourArea(contour))
    if not config.use_peak_count_estimation:
        return 1
    split_threshold = float(config.large_component_min_area)
    if area < split_threshold:
        return 1
    peak_estimate, distance_reference_area = analyze_distance_map(mask, contour, config)
    effective_reference_area = reference_area
    if distance_reference_area > 0:
        effective_reference_area = min(reference_area, distance_reference_area)
    estimates = [1, peak_estimate]
    if config.use_area_count_estimation:
        scaled_area = max(effective_reference_area * float(config.area_count_scale), 1.0)
        estimates.append(int(max(1, round(area / scaled_area))))
    return int(max(estimates))


def analyze_distance_map(mask: np.ndarray, contour: np.ndarray, config: ClassicalCountConfig) -> tuple[int, float]:
    x, y, width, height = cv2.boundingRect(contour)
    pad = max(2, odd_kernel(config.peak_local_max_kernel))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + width + pad)
    y1 = min(mask.shape[0], y + height + pad)
    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = contour.copy()
    shifted[:, :, 0] -= x0
    shifted[:, :, 1] -= y0
    cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=cv2.FILLED)
    distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
    blur_kernel = odd_kernel(config.peak_blur_kernel)
    if blur_kernel > 1:
        distance = cv2.GaussianBlur(distance, (blur_kernel, blur_kernel), 0)
    max_distance = float(distance.max())
    if max_distance <= 0:
        return 0, 0.0
    local_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (odd_kernel(config.peak_local_max_kernel), odd_kernel(config.peak_local_max_kernel)),
    )
    dilated = cv2.dilate(distance, local_kernel)
    threshold_values = [
        float(config.peak_relative_threshold),
        min(0.95, float(config.peak_relative_threshold) + 0.15),
        min(0.98, float(config.peak_relative_threshold) + 0.30),
    ]
    peak_count = 0
    for threshold_value in threshold_values:
        threshold = max(float(config.peak_min_distance), max_distance * threshold_value)
        seed_mask = ((distance >= threshold) & (roi_mask > 0)).astype(np.uint8)
        seed_components, _ = cv2.connectedComponents(seed_mask)
        peak_count = max(peak_count, int(seed_components - 1))
    maxima = (distance >= max(float(config.peak_min_distance), max_distance * float(config.peak_relative_threshold)))
    maxima &= distance >= (dilated - 1e-6)
    maxima &= roi_mask > 0
    maxima_components, _ = cv2.connectedComponents(maxima.astype(np.uint8))
    peak_count = max(peak_count, int(maxima_components - 1))
    single_colony_area = float(pi * max_distance * max_distance)
    return max(0, peak_count), single_colony_area


def render_count_overlay(image: np.ndarray, count: int, contours: list[np.ndarray]) -> np.ndarray:
    output = image.copy()
    cv2.drawContours(output, contours, -1, (0, 0, 255), 2)
    text = f"predicted_count={count}"
    cv2.putText(output, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 6)
    cv2.putText(output, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return output


def evaluate_count_predictions(predictions: list[CountPrediction], config: ClassicalCountConfig | None = None) -> CountEvaluation:
    count = len(predictions)
    if count == 0:
        metrics = {"image_count": 0}
        if config is not None:
            metrics["config"] = config.to_dict()
        return CountEvaluation(metrics=metrics, predictions=predictions)

    true_counts = np.array([prediction.true_count for prediction in predictions], dtype=float)
    predicted_counts = np.array([prediction.predicted_count for prediction in predictions], dtype=float)
    errors = predicted_counts - true_counts
    absolute_errors = np.abs(errors)
    nonzero = true_counts > 0
    zero = true_counts == 0
    metrics = {
        "image_count": int(count),
        "true_total": int(true_counts.sum()),
        "predicted_total": int(predicted_counts.sum()),
        "mae": float(absolute_errors.mean()),
        "rmse": float(sqrt(np.square(errors).mean())),
        "bias": float(errors.mean()),
        "exact_match_accuracy": float((absolute_errors == 0).mean()),
        "within_5_accuracy": float((absolute_errors <= 5).mean()),
        "zero_image_count": int(zero.sum()),
        "nonzero_image_count": int(nonzero.sum()),
    }
    if nonzero.any():
        metrics["mape_nonzero_percent"] = float((absolute_errors[nonzero] / true_counts[nonzero]).mean() * 100.0)
    if zero.any():
        metrics["zero_exact_accuracy"] = float((predicted_counts[zero] == 0).mean())
    if config is not None:
        metrics["config"] = config.to_dict()
    return CountEvaluation(metrics=metrics, predictions=predictions)


def read_image(path: Path) -> np.ndarray:
    tried = []
    for candidate in candidate_image_paths(path):
        tried.append(candidate)
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    attempted = ", ".join(str(candidate) for candidate in tried)
    raise FileNotFoundError(f"Could not read image: {path}. Tried: {attempted}")


def candidate_image_paths(path: Path) -> list[Path]:
    candidates = [path]
    suffixes = []
    for extension in IMAGE_EXTENSIONS:
        suffixes.append(extension)
        suffixes.append(extension.upper())
    for suffix in suffixes:
        candidate = path.with_suffix(suffix)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
    return normalized.astype(np.uint8)


def odd_kernel(value: int) -> int:
    kernel = max(1, int(value))
    return kernel if kernel % 2 == 1 else kernel + 1
