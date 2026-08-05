from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from src.classical.species_classification import (
    SpeciesFeatureConfig,
    extract_species_features,
    fit_species_classifier,
    save_species_model,
)
from src.ui.inference import (
    CLASSICAL_COUNT_PRESETS,
    CLASSICAL_SPECIES_MODEL_PRESETS,
    UNET_COUNT_MODEL_PRESETS,
    UNET_SPECIES_MODEL_PRESETS,
    YOLO_COUNT_MODEL_PRESETS,
    YOLO_SPECIES_MODEL_PRESETS,
    classify_with_classical_species_model,
    classify_with_unet_species_model,
    classify_with_yolo_species_model,
    config_rows,
    count_with_classical_model,
    count_with_unet_model,
    count_with_yolo_model,
    detection_rows,
    decode_uploaded_image,
    evaluate_empty_plate_gate,
    find_local_species_model_candidates,
    find_local_unet_count_model_candidates,
    find_local_unet_species_model_candidates,
    find_local_yolo_count_model_candidates,
    find_local_yolo_species_model_candidates,
    inspect_species_model_path,
    inspect_unet_count_model_path,
    inspect_unet_species_model_path,
    inspect_uploaded_species_model,
    inspect_uploaded_unet_count_model,
    inspect_uploaded_unet_species_model,
    inspect_uploaded_yolo_count_model,
    inspect_uploaded_yolo_species_model,
    inspect_yolo_count_model_path,
    inspect_yolo_species_model_path,
)


def test_decode_uploaded_image_roundtrip() -> None:
    image = np.full((32, 32, 3), (20, 80, 180), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)

    assert ok
    decoded = decode_uploaded_image(encoded.tobytes())
    assert decoded.shape == image.shape


def test_count_with_classical_model_uses_presets() -> None:
    image = np.full((220, 260, 3), 230, dtype=np.uint8)
    for center in [(60, 70), (130, 80), (190, 150)]:
        cv2.circle(image, center, 14, (45, 45, 45), -1)

    result = count_with_classical_model(image, preset_name="Config 23 baseline")

    assert "min_area" in result.config
    assert result.config["min_area"] == CLASSICAL_COUNT_PRESETS["Config 23 baseline"].min_area
    assert result.overlay_bgr.shape == image.shape


def test_classify_with_classical_species_model_predicts_saved_model(tmp_path: Path) -> None:
    rows = [
        {
            "image_path": "a.png",
            "true_class": "A",
            "category": "countable",
            "feature_global_b_mean": 1.0,
            "feature_global_hue_hist_00": 0.8,
        },
        {
            "image_path": "b.png",
            "true_class": "B",
            "category": "countable",
            "feature_global_b_mean": 9.0,
            "feature_global_hue_hist_00": 0.2,
        },
    ]
    import pandas as pd

    feature_frame = pd.DataFrame(rows)
    config = SpeciesFeatureConfig()
    model = fit_species_classifier(feature_frame, config, "centroid_l2")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    image = np.full((64, 64, 3), (1, 1, 240), dtype=np.uint8)
    result = classify_with_classical_species_model(image, str(model_path))

    assert result.classifier_type == "centroid_l2"
    assert result.predicted_class in {"A", "B"}
    assert result.score_name == "Distance"


def test_classify_with_classical_species_model_accepts_uploaded_model_bytes(tmp_path: Path) -> None:
    rows = [
        {
            "image_path": "a.png",
            "true_class": "A",
            "category": "countable",
            "feature_global_b_mean": 1.0,
            "feature_global_hue_hist_00": 0.8,
        },
        {
            "image_path": "b.png",
            "true_class": "B",
            "category": "countable",
            "feature_global_b_mean": 9.0,
            "feature_global_hue_hist_00": 0.2,
        },
    ]
    import pandas as pd

    feature_frame = pd.DataFrame(rows)
    config = SpeciesFeatureConfig()
    model = fit_species_classifier(feature_frame, config, "centroid_l2")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    image = np.full((64, 64, 3), (1, 1, 240), dtype=np.uint8)
    model_bytes = model_path.read_bytes()
    result = classify_with_classical_species_model(image, model_json_bytes=model_bytes)

    assert result.classifier_type == "centroid_l2"
    assert result.predicted_class in {"A", "B"}
    assert result.score_name == "Distance"


def test_classify_with_classical_species_model_can_reject_empty_like_input(tmp_path: Path) -> None:
    config = SpeciesFeatureConfig()
    train_rows = []
    for index, bgr, label in [
        (0, (25, 40, 220), "A"),
        (1, (35, 55, 210), "A"),
        (2, (40, 60, 200), "A"),
        (3, (220, 60, 30), "B"),
        (4, (210, 70, 40), "B"),
        (5, (200, 80, 50), "B"),
    ]:
        image = np.full((96, 96, 3), 235, dtype=np.uint8)
        cv2.circle(image, (48, 48), 14 + index % 3, bgr, -1)
        features = extract_species_features(image, config)
        train_rows.append(
            {
                "image_path": f"{label}_{index}.png",
                "true_class": label,
                "category": "countable",
                **features,
            }
        )

    import pandas as pd

    model = fit_species_classifier(pd.DataFrame(train_rows), config, "knn_3")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    empty_like = np.full((96, 96, 3), 240, dtype=np.uint8)
    result = classify_with_classical_species_model(
        empty_like,
        model_json_path=str(model_path),
        reject_empty_unsupported=True,
    )

    assert not result.accepted
    assert result.predicted_class == "Empty / unsupported"
    assert result.rejection_reason


def test_evaluate_empty_plate_gate_rejects_empty_like_plate() -> None:
    config = SpeciesFeatureConfig()
    empty_like = np.full((96, 96, 3), 240, dtype=np.uint8)
    features = extract_species_features(empty_like, config)

    gate = evaluate_empty_plate_gate(empty_like, features)

    assert gate.rejected
    assert gate.estimated_count == 0


def test_evaluate_empty_plate_gate_keeps_clear_colony_plate() -> None:
    config = SpeciesFeatureConfig()
    colony_like = np.full((128, 128, 3), 235, dtype=np.uint8)
    cv2.circle(colony_like, (44, 48), 12, (40, 50, 170), -1)
    cv2.circle(colony_like, (84, 76), 11, (45, 60, 165), -1)
    features = extract_species_features(colony_like, config)

    gate = evaluate_empty_plate_gate(colony_like, features)

    assert not gate.rejected


def test_evaluate_empty_plate_gate_keeps_dense_colony_plate() -> None:
    config = SpeciesFeatureConfig()
    dense_plate = np.full((160, 160, 3), 232, dtype=np.uint8)
    for y in range(28, 132, 18):
        for x in range(28, 132, 18):
            cv2.circle(dense_plate, (x, y), 7, (58, 78, 180), -1)
    features = extract_species_features(dense_plate, config)

    gate = evaluate_empty_plate_gate(dense_plate, features)

    assert not gate.rejected


def test_config_rows_are_displayable() -> None:
    rows = config_rows({"alpha": 1, "beta": True})

    assert rows == [{"Parameter": "alpha", "Value": "1"}, {"Parameter": "beta", "Value": "True"}]


def test_species_model_preset_points_to_latest_sweep_best_run() -> None:
    assert CLASSICAL_SPECIES_MODEL_PRESETS["Latest sweep best run"].endswith(
        "outputs/classical_species_image/val_sweep_001/best_run/model.json"
    )


def test_inspect_species_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_species_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/classical_species_image/val_sweep_001/best_run/model.json"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_species_model_path_marks_existing_local_model_ready(tmp_path: Path) -> None:
    model_path = tmp_path / "species_model.json"
    model_path.write_text("{}", encoding="utf-8")

    status = inspect_species_model_path(str(model_path), tmp_path)

    assert status.ready
    assert status.exists
    assert status.path == str(model_path)


def test_inspect_uploaded_species_model_marks_valid_upload_ready(tmp_path: Path) -> None:
    rows = [
        {
            "image_path": "a.png",
            "true_class": "A",
            "category": "countable",
            "feature_global_b_mean": 1.0,
            "feature_global_hue_hist_00": 0.8,
        },
        {
            "image_path": "b.png",
            "true_class": "B",
            "category": "countable",
            "feature_global_b_mean": 9.0,
            "feature_global_hue_hist_00": 0.2,
        },
    ]
    import pandas as pd

    feature_frame = pd.DataFrame(rows)
    config = SpeciesFeatureConfig()
    model = fit_species_classifier(feature_frame, config, "centroid_l2")
    model_path = tmp_path / "model.json"
    save_species_model(model, model_path)

    status = inspect_uploaded_species_model(model_path.read_bytes(), "model.json")

    assert status.ready
    assert status.exists
    assert status.path == "model.json"


def test_inspect_uploaded_species_model_marks_invalid_upload_not_ready() -> None:
    status = inspect_uploaded_species_model(b"{bad json", "broken.json")

    assert not status.ready
    assert "invalid" in status.message.lower()


def test_unet_count_model_preset_points_to_locked_checkpoint() -> None:
    assert UNET_COUNT_MODEL_PRESETS["Locked test best run"].endswith(
        "outputs/unet_count/train_run_001/model.pt"
    )


def test_unet_species_model_preset_points_to_locked_checkpoint() -> None:
    assert UNET_SPECIES_MODEL_PRESETS["Locked best checkpoint"].endswith(
        "outputs/unet_species_image/train_run_001/model.pt"
    )


def test_yolo_count_model_preset_points_to_locked_checkpoint() -> None:
    assert YOLO_COUNT_MODEL_PRESETS["Locked best checkpoint"].endswith(
        "outputs/yolo_count/train_run_001/weights/best.pt"
    )


def test_yolo_species_model_preset_points_to_locked_checkpoint() -> None:
    assert YOLO_SPECIES_MODEL_PRESETS["Locked best checkpoint"].endswith(
        "outputs/yolo_species/train_run_001/weights/best.pt"
    )


def test_inspect_unet_count_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_unet_count_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_count/train_run_001/model.pt"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_unet_species_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_unet_species_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/unet_species_image/train_run_001/model.pt"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_yolo_count_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_yolo_count_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_count/train_run_001/weights/best.pt"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_yolo_species_model_path_marks_colab_path_unavailable() -> None:
    status = inspect_yolo_species_model_path(
        "/content/drive/MyDrive/bacterial_colony_detection/outputs/yolo_species/train_run_001/weights/best.pt"
    )

    assert not status.ready
    assert status.is_colab_path
    assert "local Streamlit app cannot read that location" in status.message


def test_inspect_unet_count_model_path_marks_existing_local_checkpoint_ready(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")

    status = inspect_unet_count_model_path(str(model_path), tmp_path)

    assert status.ready
    assert status.exists
    assert status.path == str(model_path)


def test_inspect_unet_species_model_path_marks_existing_local_checkpoint_ready(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")

    status = inspect_unet_species_model_path(str(model_path), tmp_path)

    assert status.ready
    assert status.exists
    assert status.path == str(model_path)


def test_find_local_unet_count_model_candidates_filters_to_unet_count(tmp_path: Path) -> None:
    wanted = tmp_path / "outputs" / "unet_count" / "train_run_001"
    wanted.mkdir(parents=True, exist_ok=True)
    unwanted = tmp_path / "outputs" / "classical_species"
    unwanted.mkdir(parents=True, exist_ok=True)
    (wanted / "model.pt").write_bytes(b"ok")
    (unwanted / "model.pt").write_bytes(b"skip")

    candidates = find_local_unet_count_model_candidates(tmp_path)

    assert candidates == (str(wanted / "model.pt"),)


def test_find_local_unet_species_model_candidates_filters_to_unet_species(tmp_path: Path) -> None:
    wanted = tmp_path / "outputs" / "unet_species_image" / "train_run_001"
    wanted.mkdir(parents=True, exist_ok=True)
    unwanted = tmp_path / "outputs" / "unet_count"
    unwanted.mkdir(parents=True, exist_ok=True)
    (wanted / "model.pt").write_bytes(b"ok")
    (unwanted / "model.pt").write_bytes(b"skip")

    candidates = find_local_unet_species_model_candidates(tmp_path)

    assert candidates == (str(wanted / "model.pt"),)


def test_find_local_yolo_count_model_candidates_filters_to_yolo_count(tmp_path: Path) -> None:
    wanted = tmp_path / "outputs" / "yolo_count" / "train_run_001" / "weights"
    wanted.mkdir(parents=True, exist_ok=True)
    unwanted = tmp_path / "outputs" / "unet_count"
    unwanted.mkdir(parents=True, exist_ok=True)
    (wanted / "best.pt").write_bytes(b"ok")
    (unwanted / "model.pt").write_bytes(b"skip")

    candidates = find_local_yolo_count_model_candidates(tmp_path)

    assert candidates == (str(wanted / "best.pt"),)


def test_find_local_yolo_species_model_candidates_filters_to_yolo_species(tmp_path: Path) -> None:
    wanted = tmp_path / "outputs" / "yolo_species" / "train_run_001" / "weights"
    wanted.mkdir(parents=True, exist_ok=True)
    unwanted = tmp_path / "outputs" / "unet_species_image"
    unwanted.mkdir(parents=True, exist_ok=True)
    (wanted / "best.pt").write_bytes(b"ok")
    (unwanted / "model.pt").write_bytes(b"skip")

    candidates = find_local_yolo_species_model_candidates(tmp_path)

    assert candidates == (str(wanted / "best.pt"),)


def test_inspect_uploaded_unet_count_model_marks_invalid_upload_not_ready() -> None:
    status = inspect_uploaded_unet_count_model(b"not a checkpoint", "broken.pt")

    assert not status.ready
    assert "invalid" in status.message.lower()


def test_inspect_uploaded_unet_species_model_marks_invalid_upload_not_ready() -> None:
    status = inspect_uploaded_unet_species_model(b"not a checkpoint", "broken.pt")

    assert not status.ready
    assert "invalid" in status.message.lower()


def test_inspect_uploaded_yolo_count_model_marks_valid_upload_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validator(*, model_pt_bytes=None, model_pt_path=""):
        return object()

    monkeypatch.setattr("src.ui.inference.load_yolo_model_for_validation", fake_validator)

    status = inspect_uploaded_yolo_count_model(b"checkpoint", "best.pt")

    assert status.ready
    assert status.path == "best.pt"


def test_inspect_uploaded_yolo_species_model_marks_invalid_upload_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_validator(*, model_pt_bytes=None, model_pt_path=""):
        raise ValueError("bad checkpoint")

    monkeypatch.setattr("src.ui.inference.load_yolo_model_for_validation", fake_validator)

    status = inspect_uploaded_yolo_species_model(b"broken", "broken.pt")

    assert not status.ready
    assert "invalid" in status.message.lower()


def test_count_with_unet_model_accepts_uploaded_checkpoint_bytes() -> None:
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    from src.unet.torch_backend import UNetSegmenter

    model = UNetSegmenter(base_channels=8)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(10.0)
    checkpoint = {
        "model_type": "unet_count_segmenter",
        "image_size": 64,
        "base_channels": 8,
        "state_dict": model.state_dict(),
    }
    import io

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    image = np.full((96, 112, 3), 180, dtype=np.uint8)
    result = count_with_unet_model(
        image,
        model_pt_bytes=buffer.getvalue(),
        threshold=0.5,
        min_component_area=2,
    )

    assert result.predicted_count == 1
    assert result.overlay_bgr.shape == image.shape
    assert result.mask.shape == image.shape[:2]
    assert result.config["threshold"] == 0.5


def test_classify_with_unet_species_model_accepts_uploaded_checkpoint_bytes() -> None:
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    from src.unet.torch_backend import UNetEncoderClassifier

    model = UNetEncoderClassifier(class_count=2, base_channels=8, dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head[2].bias.copy_(torch.tensor([0.0, 2.0]))
    checkpoint = {
        "model_type": "unet_species_image_classifier",
        "image_size": 64,
        "base_channels": 8,
        "dropout": 0.0,
        "class_names": ["A", "B"],
        "state_dict": model.state_dict(),
    }
    import io

    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    image = np.full((96, 112, 3), (30, 40, 220), dtype=np.uint8)
    result = classify_with_unet_species_model(
        image,
        model_pt_bytes=buffer.getvalue(),
        reject_empty_unsupported=False,
    )

    assert result.predicted_class == "B"
    assert result.classifier_type == "unet_encoder_classifier"
    assert result.score_name == "Confidence"
    assert 0.5 < result.score_value < 1.0


def test_count_with_yolo_model_uses_detection_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui.inference import YOLODetection

    def fake_run_yolo_detections(**kwargs):
        assert kwargs["confidence_threshold"] == 0.45
        assert kwargs["image_size"] == 1536
        return (
            YOLODetection(0, "colony", 0.91, 10.0, 12.0, 42.0, 44.0, 1024.0),
            YOLODetection(0, "colony", 0.83, 60.0, 52.0, 90.0, 86.0, 1020.0),
        )

    monkeypatch.setattr("src.ui.inference.run_yolo_detections", fake_run_yolo_detections)
    image = np.full((96, 112, 3), 180, dtype=np.uint8)

    result = count_with_yolo_model(image, model_pt_path="/tmp/model.pt", confidence_threshold=0.45, image_size=1536)

    assert result.predicted_count == 2
    assert result.overlay_bgr.shape == image.shape
    assert result.config["confidence_threshold"] == 0.45


def test_classify_with_yolo_species_model_aggregates_class_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui.inference import YOLODetection

    def fake_run_yolo_detections(**kwargs):
        return (
            YOLODetection(1, "E.coli", 0.88, 10.0, 10.0, 24.0, 26.0, 224.0),
            YOLODetection(2, "S.aureus", 0.81, 30.0, 20.0, 44.0, 36.0, 224.0),
            YOLODetection(1, "E.coli", 0.79, 50.0, 30.0, 68.0, 48.0, 324.0),
        )

    monkeypatch.setattr("src.ui.inference.run_yolo_detections", fake_run_yolo_detections)
    image = np.full((96, 112, 3), 180, dtype=np.uint8)

    result = classify_with_yolo_species_model(image, model_pt_path="/tmp/model.pt")

    assert result.total_detections == 3
    assert result.class_counts == {"E.coli": 2, "S.aureus": 1}
    assert result.overlay_bgr.shape == image.shape


def test_detection_rows_are_displayable() -> None:
    from src.ui.inference import YOLODetection

    rows = detection_rows(
        (
            YOLODetection(0, "colony", 0.91, 1.0, 2.0, 10.0, 12.0, 90.0),
            YOLODetection(1, "E.coli", 0.82, 3.0, 4.0, 15.0, 18.0, 168.0),
        )
    )

    assert rows[0]["Class"] == "colony"
    assert rows[1]["Confidence"] == "0.820"


def test_find_local_species_model_candidates_filters_model_paths(tmp_path: Path) -> None:
    species_model = tmp_path / "outputs" / "classical_species_image" / "best_run" / "model.json"
    generic_model = tmp_path / "outputs" / "other" / "model.json"
    species_model.parent.mkdir(parents=True)
    generic_model.parent.mkdir(parents=True)
    species_model.write_text("{}", encoding="utf-8")
    generic_model.write_text("{}", encoding="utf-8")

    candidates = find_local_species_model_candidates(tmp_path)

    assert candidates == (str(species_model),)
