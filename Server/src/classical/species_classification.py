from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.classical.detect_count import read_image


DEFAULT_CLASSIFIER_TYPES = (
    "centroid_l2",
    "centroid_cosine",
    "knn_3",
    "knn_5",
    "knn_9",
    "knn_5_weighted",
    "knn_9_weighted",
)


@dataclass(frozen=True)
class SpeciesFeatureConfig:
    image_size: int = 256
    center_crop_ratio: float = 0.72
    hue_bins: int = 16
    saturation_bins: int = 8
    value_bins: int = 8
    gray_bins: int = 16
    radial_bins: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeciesPrediction:
    image_path: str
    true_class: str
    predicted_class: str
    distance: float
    correct: bool
    category: str


@dataclass(frozen=True)
class SpeciesEvaluation:
    metrics: dict[str, Any]
    predictions: list[SpeciesPrediction]


class ClassicalSpeciesClassifier:
    def __init__(
        self,
        classes: list[str],
        feature_names: list[str],
        mean: list[float],
        scale: list[float],
        config: SpeciesFeatureConfig,
        classifier_type: str,
        centroids: dict[str, list[float]] | None = None,
        train_vectors: list[list[float]] | None = None,
        train_labels: list[str] | None = None,
    ) -> None:
        self.classes = classes
        self.feature_names = feature_names
        self.mean = np.array(mean, dtype=np.float64)
        self.scale = np.array(scale, dtype=np.float64)
        self.config = config
        self.classifier_type = classifier_type
        self.centroids = {
            label: np.array(values, dtype=np.float64) for label, values in (centroids or {}).items()
        }
        self.train_vectors = np.array(train_vectors or [], dtype=np.float64)
        self.train_labels = np.array(train_labels or [], dtype=object)

    def predict_features(self, features: dict[str, float]) -> tuple[str, float]:
        vector = vectorize_features(features, self.feature_names)
        normalized = (vector - self.mean) / self.scale
        if self.classifier_type == "centroid_l2":
            return predict_centroid_l2(normalized, self.centroids)
        if self.classifier_type == "centroid_cosine":
            return predict_centroid_cosine(normalized, self.centroids)
        if self.classifier_type.startswith("knn_"):
            return predict_knn(normalized, self.train_vectors, self.train_labels, self.classifier_type)
        raise ValueError(f"Unsupported classifier type: {self.classifier_type}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "classical_species_classifier",
            "classifier_type": self.classifier_type,
            "classes": self.classes,
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "centroids": {label: centroid.tolist() for label, centroid in self.centroids.items()},
            "train_vectors": self.train_vectors.tolist(),
            "train_labels": self.train_labels.tolist(),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassicalSpeciesClassifier":
        return cls(
            classes=[str(label) for label in data["classes"]],
            feature_names=[str(name) for name in data["feature_names"]],
            mean=[float(value) for value in data["mean"]],
            scale=[float(value) for value in data["scale"]],
            config=SpeciesFeatureConfig(**data.get("config", {})),
            classifier_type=str(data.get("classifier_type", "centroid_l2")),
            centroids={str(label): [float(value) for value in values] for label, values in data.get("centroids", {}).items()},
            train_vectors=[[float(value) for value in row] for row in data.get("train_vectors", [])],
            train_labels=[str(label) for label in data.get("train_labels", [])],
        )


NearestCentroidSpeciesClassifier = ClassicalSpeciesClassifier


def train_species_classifier(
    dataset_dir: Path,
    train_csv: Path,
    config: SpeciesFeatureConfig | None = None,
    max_images: int = 0,
    classifier_type: str = "centroid_l2",
) -> tuple[ClassicalSpeciesClassifier, pd.DataFrame]:
    active_config = config or SpeciesFeatureConfig()
    rows = load_species_split(train_csv, max_images=max_images)
    feature_rows = extract_feature_rows(dataset_dir, rows, active_config)
    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        raise ValueError(f"No species-image rows found in {train_csv}")
    return fit_species_classifier(feature_frame, active_config, classifier_type), feature_frame


def select_species_classifier(
    dataset_dir: Path,
    train_csv: Path,
    val_csv: Path,
    config: SpeciesFeatureConfig | None = None,
    max_train_images: int = 0,
    max_val_images: int = 0,
    classifier_types: tuple[str, ...] = DEFAULT_CLASSIFIER_TYPES,
) -> tuple[ClassicalSpeciesClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_config = config or SpeciesFeatureConfig()
    train_rows = load_species_split(train_csv, max_images=max_train_images)
    val_rows = load_species_split(val_csv, max_images=max_val_images)
    train_frame = pd.DataFrame(extract_feature_rows(dataset_dir, train_rows, active_config))
    val_frame = pd.DataFrame(extract_feature_rows(dataset_dir, val_rows, active_config))
    if train_frame.empty:
        raise ValueError(f"No species-image rows found in {train_csv}")
    if val_frame.empty:
        raise ValueError(f"No species-image rows found in {val_csv}")

    candidates = []
    best_model = None
    best_metrics = None
    for classifier_type in classifier_types:
        model = fit_species_classifier(train_frame, active_config, classifier_type)
        evaluation = evaluate_species_classifier_from_features(val_frame, model)
        row = {
            "classifier_type": classifier_type,
            "accuracy": evaluation.metrics["accuracy"],
            "macro_f1": evaluation.metrics["macro_f1"],
            "macro_precision": evaluation.metrics["macro_precision"],
            "macro_recall": evaluation.metrics["macro_recall"],
        }
        candidates.append(row)
        if best_metrics is None or candidate_ranking(evaluation.metrics) > candidate_ranking(best_metrics):
            best_model = model
            best_metrics = evaluation.metrics

    if best_model is None:
        raise RuntimeError("Could not select a species classifier")
    selection = pd.DataFrame(candidates).sort_values(
        ["macro_f1", "accuracy", "macro_precision"],
        ascending=[False, False, False],
    )
    return best_model, train_frame, val_frame, selection


def fit_species_classifier(
    feature_frame: pd.DataFrame,
    config: SpeciesFeatureConfig,
    classifier_type: str,
) -> ClassicalSpeciesClassifier:
    labels = sorted(feature_frame["true_class"].unique().tolist())
    feature_names = [column for column in feature_frame.columns if column.startswith("feature_")]
    matrix = feature_frame[feature_names].to_numpy(dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    normalized = (matrix - mean) / scale
    centroids = {}
    label_array = feature_frame["true_class"].to_numpy()
    for label in labels:
        centroids[label] = normalized[label_array == label].mean(axis=0).tolist()
    return ClassicalSpeciesClassifier(
        classes=labels,
        feature_names=feature_names,
        mean=mean.tolist(),
        scale=scale.tolist(),
        config=config,
        classifier_type=classifier_type,
        centroids=centroids,
        train_vectors=normalized.tolist(),
        train_labels=label_array.tolist(),
    )


def evaluate_species_classifier(
    dataset_dir: Path,
    split_csv: Path,
    model: ClassicalSpeciesClassifier,
    max_images: int = 0,
) -> SpeciesEvaluation:
    rows = load_species_split(split_csv, max_images=max_images)
    feature_frame = pd.DataFrame(extract_feature_rows(dataset_dir, rows, model.config))
    return evaluate_species_classifier_from_features(feature_frame, model)


def evaluate_species_classifier_from_features(
    feature_frame: pd.DataFrame,
    model: ClassicalSpeciesClassifier,
) -> SpeciesEvaluation:
    predictions = []
    for row in feature_frame.to_dict("records"):
        features = {name: float(row[name]) for name in model.feature_names}
        predicted_class, distance = model.predict_features(features)
        true_class = str(row["true_class"])
        predictions.append(
            SpeciesPrediction(
                image_path=str(row["image_path"]),
                true_class=true_class,
                predicted_class=predicted_class,
                distance=float(distance),
                correct=predicted_class == true_class,
                category=str(row.get("category", "")),
            )
        )
    metrics = classification_metrics(predictions, model.classes)
    metrics["classifier_type"] = model.classifier_type
    return SpeciesEvaluation(metrics=metrics, predictions=predictions)


def extract_species_features(image: np.ndarray, config: SpeciesFeatureConfig | None = None) -> dict[str, float]:
    active_config = config or SpeciesFeatureConfig()
    resized = resize_square(image, active_config.image_size)
    center = crop_center(resized, active_config.center_crop_ratio)
    features = {}
    append_view_features(features, "global", resized, active_config)
    append_view_features(features, "center", center, active_config)
    append_cross_channel_features(features, resized)
    append_radial_features(features, resized, active_config.radial_bins)
    return {f"feature_{key}": float(value) for key, value in features.items()}


def extract_feature_rows(dataset_dir: Path, rows: pd.DataFrame, config: SpeciesFeatureConfig) -> list[dict[str, Any]]:
    feature_rows = []
    for row in rows.to_dict("records"):
        image_path = dataset_dir / str(row["image_path"])
        image = read_image(image_path)
        features = extract_species_features(image, config)
        feature_rows.append(
            {
                "image_path": str(row["image_path"]),
                "true_class": str(row["primary_class"]),
                "category": str(row.get("category", "")),
                **features,
            }
        )
    return feature_rows


def load_species_split(split_csv: Path, max_images: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(split_csv)
    if "species_image_eligible" in frame.columns:
        frame = frame[frame["species_image_eligible"].map(is_truthy)].copy()
    frame = frame[frame["primary_class"].fillna("").astype(str) != ""].copy()
    if max_images > 0:
        frame = frame.head(max_images).copy()
    return frame


def classification_metrics(predictions: list[SpeciesPrediction], classes: list[str]) -> dict[str, Any]:
    if not predictions:
        return {"image_count": 0}
    total = len(predictions)
    correct = sum(1 for prediction in predictions if prediction.correct)
    per_class = {}
    recalls = []
    precisions = []
    f1_scores = []
    for label in classes:
        true_positive = sum(
            1 for prediction in predictions if prediction.true_class == label and prediction.predicted_class == label
        )
        false_positive = sum(
            1 for prediction in predictions if prediction.true_class != label and prediction.predicted_class == label
        )
        false_negative = sum(
            1 for prediction in predictions if prediction.true_class == label and prediction.predicted_class != label
        )
        support = true_positive + false_negative
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if support == 0 else true_positive / support
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        if support > 0:
            recalls.append(recall)
            precisions.append(precision)
            f1_scores.append(f1)
    return {
        "image_count": int(total),
        "accuracy": float(correct / total),
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "classes": classes,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(predictions, classes),
    }


def confusion_matrix(predictions: list[SpeciesPrediction], classes: list[str]) -> dict[str, dict[str, int]]:
    matrix = {true_label: {predicted_label: 0 for predicted_label in classes} for true_label in classes}
    for prediction in predictions:
        if prediction.true_class in matrix and prediction.predicted_class in matrix[prediction.true_class]:
            matrix[prediction.true_class][prediction.predicted_class] += 1
    return matrix


def save_species_model(model: ClassicalSpeciesClassifier, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)


def load_species_model(path: Path) -> ClassicalSpeciesClassifier:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ClassicalSpeciesClassifier.from_dict(data)


def load_species_model_bytes(model_bytes: bytes) -> ClassicalSpeciesClassifier:
    data = json.loads(model_bytes.decode("utf-8"))
    return ClassicalSpeciesClassifier.from_dict(data)


def save_species_evaluation(evaluation: SpeciesEvaluation, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(prediction) for prediction in evaluation.predictions]).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(evaluation.metrics, f, indent=2)


def resize_square(image: np.ndarray, image_size: int) -> np.ndarray:
    size = max(16, int(image_size))
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def crop_center(image: np.ndarray, ratio: float) -> np.ndarray:
    active_ratio = min(max(float(ratio), 0.2), 1.0)
    height, width = image.shape[:2]
    crop_height = max(8, int(round(height * active_ratio)))
    crop_width = max(8, int(round(width * active_ratio)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def append_view_features(features: dict[str, float], prefix: str, image: np.ndarray, config: SpeciesFeatureConfig) -> None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    append_channel_stats(features, f"{prefix}_b", image[:, :, 0])
    append_channel_stats(features, f"{prefix}_g", image[:, :, 1])
    append_channel_stats(features, f"{prefix}_r", image[:, :, 2])
    append_channel_stats(features, f"{prefix}_h", hsv[:, :, 0])
    append_channel_stats(features, f"{prefix}_s", hsv[:, :, 1])
    append_channel_stats(features, f"{prefix}_v", hsv[:, :, 2])
    append_channel_stats(features, f"{prefix}_lab_l", lab[:, :, 0])
    append_channel_stats(features, f"{prefix}_lab_a", lab[:, :, 1])
    append_channel_stats(features, f"{prefix}_lab_b", lab[:, :, 2])
    append_histogram(features, f"{prefix}_hue", hsv[:, :, 0], config.hue_bins, 0, 180)
    append_histogram(features, f"{prefix}_saturation", hsv[:, :, 1], config.saturation_bins, 0, 256)
    append_histogram(features, f"{prefix}_value", hsv[:, :, 2], config.value_bins, 0, 256)
    append_histogram(features, f"{prefix}_gray", gray, config.gray_bins, 0, 256)
    append_texture_features(features, prefix, gray, hsv)


def append_cross_channel_features(features: dict[str, float], image: np.ndarray) -> None:
    float_image = image.astype(np.float32) + 1.0
    channel_sum = float_image.sum(axis=2)
    b_norm = float_image[:, :, 0] / channel_sum
    g_norm = float_image[:, :, 1] / channel_sum
    r_norm = float_image[:, :, 2] / channel_sum
    features["chromaticity_b_mean"] = float(b_norm.mean())
    features["chromaticity_g_mean"] = float(g_norm.mean())
    features["chromaticity_r_mean"] = float(r_norm.mean())
    features["rg_ratio_mean"] = float((float_image[:, :, 2] / float_image[:, :, 1]).mean())
    features["rb_ratio_mean"] = float((float_image[:, :, 2] / float_image[:, :, 0]).mean())
    features["gb_ratio_mean"] = float((float_image[:, :, 1] / float_image[:, :, 0]).mean())


def append_radial_features(features: dict[str, float], image: np.ndarray, radial_bins: int) -> None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    height, width = gray.shape
    yy, xx = np.indices((height, width))
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    normalized_radius = radius / max(radius.max(), 1.0)
    edges = np.linspace(0.0, 1.0, max(2, int(radial_bins)) + 1)
    for index in range(len(edges) - 1):
        mask = (normalized_radius >= edges[index]) & (normalized_radius < edges[index + 1])
        if not np.any(mask):
            continue
        features[f"radial_gray_mean_{index}"] = float(gray[mask].mean())
        features[f"radial_gray_std_{index}"] = float(gray[mask].std())
        features[f"radial_s_mean_{index}"] = float(hsv[:, :, 1][mask].mean())
        features[f"radial_v_mean_{index}"] = float(hsv[:, :, 2][mask].mean())


def append_channel_stats(features: dict[str, float], prefix: str, channel: np.ndarray) -> None:
    values = channel.astype(np.float32)
    features[f"{prefix}_mean"] = float(values.mean())
    features[f"{prefix}_std"] = float(values.std())
    features[f"{prefix}_p10"] = float(np.percentile(values, 10))
    features[f"{prefix}_p50"] = float(np.percentile(values, 50))
    features[f"{prefix}_p90"] = float(np.percentile(values, 90))


def append_histogram(
    features: dict[str, float],
    prefix: str,
    channel: np.ndarray,
    bins: int,
    minimum: float,
    maximum: float,
) -> None:
    hist, _ = np.histogram(channel, bins=max(1, int(bins)), range=(minimum, maximum))
    total = max(1, int(hist.sum()))
    for index, value in enumerate(hist.astype(np.float64) / total):
        features[f"{prefix}_hist_{index:02d}"] = float(value)


def append_texture_features(features: dict[str, float], prefix: str, gray: np.ndarray, hsv: np.ndarray) -> None:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges = cv2.Canny(gray, 60, 160)
    features[f"{prefix}_laplacian_var"] = float(laplacian.var())
    features[f"{prefix}_sobel_mean"] = float(magnitude.mean())
    features[f"{prefix}_sobel_std"] = float(magnitude.std())
    features[f"{prefix}_edge_density"] = float((edges > 0).mean())
    features[f"{prefix}_dark_fraction"] = float((gray < 80).mean())
    features[f"{prefix}_bright_fraction"] = float((gray > 200).mean())
    features[f"{prefix}_saturated_fraction"] = float((hsv[:, :, 1] > 80).mean())


def vectorize_features(features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    return np.array([float(features[name]) for name in feature_names], dtype=np.float64)


def predict_centroid_l2(vector: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[str, float]:
    distances = {label: float(np.linalg.norm(vector - centroid)) for label, centroid in centroids.items()}
    predicted = min(distances, key=distances.get)
    return predicted, distances[predicted]


def predict_centroid_cosine(vector: np.ndarray, centroids: dict[str, np.ndarray]) -> tuple[str, float]:
    vector_norm = max(float(np.linalg.norm(vector)), 1e-8)
    distances = {}
    for label, centroid in centroids.items():
        centroid_norm = max(float(np.linalg.norm(centroid)), 1e-8)
        similarity = float(np.dot(vector, centroid) / (vector_norm * centroid_norm))
        distances[label] = 1.0 - similarity
    predicted = min(distances, key=distances.get)
    return predicted, distances[predicted]


def predict_knn(
    vector: np.ndarray,
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    classifier_type: str,
) -> tuple[str, float]:
    parts = classifier_type.split("_")
    k = int(parts[1])
    weighted = len(parts) > 2 and parts[2] == "weighted"
    distances = np.linalg.norm(train_vectors - vector, axis=1)
    order = np.argsort(distances)[:k]
    neighbor_labels = train_labels[order]
    neighbor_distances = distances[order]
    scores: dict[str, float] = {}
    for label, distance in zip(neighbor_labels, neighbor_distances):
        score = 1.0 / max(float(distance), 1e-8) if weighted else 1.0
        scores[str(label)] = scores.get(str(label), 0.0) + score
    best_score = max(scores.values())
    best_labels = sorted(label for label, score in scores.items() if score == best_score)
    predicted = best_labels[0]
    predicted_distances = neighbor_distances[neighbor_labels == predicted]
    return predicted, float(predicted_distances.mean())


def candidate_ranking(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        float(metrics["accuracy"]),
        float(metrics["macro_precision"]),
    )


def save_species_selection(selection: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "model_selection.csv", index=False)


def is_truthy(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}
