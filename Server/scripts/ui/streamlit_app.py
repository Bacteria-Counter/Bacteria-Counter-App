from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ui.inference import (
    CLASSICAL_COUNT_PRESETS,
    CLASSICAL_SPECIES_MODEL_PRESETS,
    UNET_COUNT_MODEL_PRESETS,
    UNET_SPECIES_MODEL_PRESETS,
    YOLO_COUNT_MODEL_PRESETS,
    YOLO_SPECIES_MODEL_PRESETS,
    bgr_to_rgb,
    classify_with_classical_species_model,
    classify_with_yolo_species_model,
    classify_with_unet_species_model,
    config_rows,
    count_with_classical_model,
    count_with_yolo_model,
    count_with_unet_model,
    detection_rows,
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
    load_image_input,
    mask_to_rgb,
)


st.set_page_config(page_title="Bacterial Colony Model Tester", page_icon="BC", layout="wide")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
        }
        .stApp {
            background: #eef2ef;
            color: #16211c;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.6rem;
            padding-bottom: 2.2rem;
        }
        section[data-testid="stSidebar"] {
            background: #16211c;
            border-right: 1px solid #2b3c33;
        }
        section[data-testid="stSidebar"] * {
            color: #edf5ef;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #223128;
            border-color: #40584b;
            color: #f7fbf8;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: #dce8e0;
        }
        div[data-testid="stFileUploader"] section {
            background: #223128;
            border: 1px dashed #5d7468;
            border-radius: 8px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid #d9e3dd;
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(23, 33, 28, 0.055);
        }
        h1, h2, h3 {
            color: #16211c;
            letter-spacing: 0;
        }
        h1 {
            font-size: 1.9rem;
            line-height: 1.12;
            margin-bottom: 0.15rem;
        }
        h2 {
            font-size: 1.12rem;
        }
        h3 {
            font-size: 1rem;
        }
        .app-shell {
            background: #ffffff;
            border: 1px solid #d9e3dd;
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 26px rgba(23, 33, 28, 0.06);
            margin-bottom: 1rem;
        }
        .app-title {
            color: #16211c;
            font-size: 1.9rem;
            line-height: 1.12;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.85rem;
        }
        .phase-line {
            color: #5d6c64;
            font-size: 0.94rem;
            margin-bottom: 1rem;
        }
        .status-card {
            background: #f4f7f4;
            border: 1px solid #dce6df;
            border-left: 4px solid #2f7d5b;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            min-height: 5.15rem;
        }
        .status-label {
            color: #64736b;
            font-size: 0.78rem;
            text-transform: uppercase;
            font-weight: 700;
        }
        .status-value {
            color: #16211c;
            font-size: 1.04rem;
            font-weight: 700;
            margin-top: 0.22rem;
        }
        .status-detail {
            color: #65746c;
            font-size: 0.86rem;
            margin-top: 0.14rem;
        }
        .result-title {
            color: #16211c;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-muted {
            color: #697b72;
            font-size: 0.9rem;
        }
        .empty-state {
            background: #f5f7f6;
            border: 1px dashed #c4d1ca;
            border-radius: 8px;
            color: #5e6f66;
            padding: 1.1rem;
            min-height: 3.5rem;
        }
        div.stButton > button {
            border-radius: 6px;
            font-weight: 700;
            background: #2f7d5b;
            border-color: #2f7d5b;
            color: #ffffff;
        }
        div.stButton > button:hover {
            background: #25694c;
            border-color: #25694c;
            color: #ffffff;
        }
        div[data-testid="stMetric"] {
            background: #f6f8f7;
            border: 1px solid #dfe7e2;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
        }
        div[data-testid="stMetric"] label {
            color: #5f7068;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] * {
            color: #16211c;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"],
        div[data-testid="stMetric"] div[data-testid="stMetricLabel"] * {
            color: #5f7068;
        }
        div[data-baseweb="select"] > div,
        input,
        textarea {
            border-radius: 7px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()
    task, model_family, uploaded_file, image_path = render_sidebar()
    render_header(task, model_family)
    image, source_name = resolve_input(uploaded_file, image_path)

    if model_family == "Classical":
        if task == "Counting":
            render_classical_counting(image, source_name)
        else:
            render_classical_classification(image, source_name)
        return

    if model_family == "U-Net" and task == "Counting":
        render_unet_counting(image, source_name)
        return

    if model_family == "U-Net" and task == "Classification":
        render_unet_classification(image, source_name)
        return

    if model_family == "YOLO" and task == "Counting":
        render_yolo_counting(image, source_name)
        return

    if model_family == "YOLO" and task == "Classification":
        render_yolo_classification(image, source_name)
        return

    if model_family != "Classical":
        render_unavailable_family(task, model_family, image, source_name)
        return


def render_sidebar():
    with st.sidebar:
        st.header("Run Setup")
        task = st.radio("Task", options=["Counting", "Classification"], horizontal=False)
        model_family = st.selectbox("Model family", options=["Classical", "YOLO", "U-Net"], index=0)
        st.divider()
        image_source = st.radio("Image source", options=["Upload", "Path"], horizontal=True)
        uploaded_file = None
        image_path = ""
        if image_source == "Upload":
            uploaded_file = st.file_uploader("Plate image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        else:
            image_path = st.text_input("Image path", value="")
    return task, model_family, uploaded_file, image_path


def render_header(task: str, model_family: str) -> None:
    statuses = (
        ("Active task", task, model_family),
        ("YOLO count", "Locked", "test MAE 1.11"),
        ("YOLO species", "Locked", "test mAP50-95 0.694"),
        ("Classical count", "Locked", "test MAE 33.53"),
        ("Classical species", "Locked", "test F1 0.6036"),
        ("U-Net count", "Locked", "test MAE 11.33"),
        ("U-Net species", "Locked", "test F1 0.7160"),
    )
    cards = "\n".join(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-detail">{detail}</div>
        </div>
        """
        for label, value, detail in statuses
    )
    st.markdown(
        f"""
        <div class="app-shell">
            <div class="app-title">Bacterial Colony Model Tester</div>
            <div class="phase-line">Single-image workbench for the AGAR-primary locked classical, YOLO, and U-Net models.</div>
            <div class="status-grid">{cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_input(uploaded_file, image_path: str):
    try:
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        return load_image_input(image_path=image_path, uploaded_bytes=uploaded_bytes)
    except ValueError:
        return None, ""
    except Exception as exc:
        st.error(str(exc))
        return None, ""


def render_unavailable_family(task: str, model_family: str, image, source_name: str) -> None:
    left, right = st.columns([0.9, 1.1])
    with left:
        with st.container(border=True):
            st.subheader(f"{model_family} {task}")
            st.warning("Backend pending for this phase.")
            st.metric("Implementation status", "Not connected")
            st.metric("Current available family", "Classical")
    with right:
        render_input_panel(image, source_name)


def render_classical_counting(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("Classical Counting")
            preset_name = st.selectbox("Counting preset", options=list(CLASSICAL_COUNT_PRESETS.keys()) + ["Config JSON path"])
            config_json_path = st.text_input("Config JSON path", value="") if preset_name == "Config JSON path" else ""
            run = st.button("Run counting", type="primary", use_container_width=True)
            st.caption("Locked baseline is config 23 unless a saved config JSON is selected.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Counting Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No counting run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running counting.")
                return
            try:
                result = count_with_classical_model(image=image, preset_name=preset_name, config_json_path=config_json_path)
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Predicted count", result.predicted_count)
            metric_b.metric("Image source", source_name)
            view_a, view_b = st.columns(2)
            view_a.image(bgr_to_rgb(result.overlay_bgr), caption="Overlay", use_container_width=True)
            view_b.image(mask_to_rgb(result.mask), caption="Binary mask", use_container_width=True)
            render_config_table(result.config)


def render_classical_classification(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("Classical Classification")
            preset_names = list(CLASSICAL_SPECIES_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("Species model", options=preset_names, key="species_model_preset")
            preset_path = CLASSICAL_SPECIES_MODEL_PRESETS.get(preset_name, "")
            current_preset_state = st.session_state.get("species_model_preset_previous")
            if current_preset_state != preset_name:
                if preset_name != "Manual path":
                    st.session_state["species_model_json_path"] = preset_path
                elif "species_model_json_path" not in st.session_state:
                    st.session_state["species_model_json_path"] = ""
                st.session_state["species_model_preset_previous"] = preset_name
            uploaded_model_file = st.file_uploader("Model JSON upload", type=["json"], key="species_model_json_upload")
            model_json_path = st.text_input("Model JSON path", key="species_model_json_path")
            uploaded_model_bytes = uploaded_model_file.getvalue() if uploaded_model_file is not None else None
            uploaded_model_name = uploaded_model_file.name if uploaded_model_file is not None else ""
            model_status = inspect_species_model_path(model_json_path, ROOT)
            upload_status = inspect_uploaded_species_model(uploaded_model_bytes, uploaded_model_name)
            reject_empty_unsupported = st.checkbox(
                "Reject likely empty plates",
                value=True,
                help="Keep species classification for countable and uncountable plates, but avoid forcing a species label on likely empty plates.",
            )
            render_model_source_status(upload_status, model_status)
            run = st.button(
                "Run classification",
                type="primary",
                use_container_width=True,
                disabled=(not upload_status.ready and not model_status.ready) or image is None,
            )
            st.caption("Uploaded model JSON overrides the path field. Use a local path only when you want the app to load a model from disk.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Classification Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No classification run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running classification.")
                return
            try:
                result = classify_with_classical_species_model(
                    image=image,
                    model_json_path=model_json_path,
                    model_json_bytes=uploaded_model_bytes,
                    reject_empty_unsupported=reject_empty_unsupported,
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Predicted class", result.predicted_class)
            metric_b.metric("Classifier", result.classifier_type)
            metric_c.metric(result.score_name, f"{result.score_value:.3f}")
            if not result.accepted:
                st.warning(result.rejection_reason)
            st.caption(f"Image source: {source_name}")
            render_config_table(result.config)


def render_unet_counting(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("U-Net Counting")
            preset_names = list(UNET_COUNT_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("U-Net checkpoint", options=preset_names, key="unet_count_model_preset")
            preset_path = UNET_COUNT_MODEL_PRESETS.get(preset_name, "")
            current_preset_state = st.session_state.get("unet_count_model_preset_previous")
            if current_preset_state != preset_name:
                if preset_name != "Manual path":
                    st.session_state["unet_count_model_pt_path"] = preset_path
                elif "unet_count_model_pt_path" not in st.session_state:
                    st.session_state["unet_count_model_pt_path"] = ""
                st.session_state["unet_count_model_preset_previous"] = preset_name
            uploaded_model_file = st.file_uploader("Model checkpoint upload", type=["pt"], key="unet_count_model_pt_upload")
            model_pt_path = st.text_input("Model checkpoint path", key="unet_count_model_pt_path")
            uploaded_model_bytes = uploaded_model_file.getvalue() if uploaded_model_file is not None else None
            uploaded_model_name = uploaded_model_file.name if uploaded_model_file is not None else ""
            upload_status = inspect_uploaded_unet_count_model(uploaded_model_bytes, uploaded_model_name)
            path_status = inspect_unet_count_model_path(model_pt_path, ROOT)
            threshold = st.number_input("Segmentation threshold", min_value=0.05, max_value=0.95, value=0.5, step=0.05)
            min_component_area = st.number_input("Minimum component area", min_value=1, max_value=128, value=2, step=1)
            render_unet_count_model_source_status(upload_status, path_status)
            run = st.button(
                "Run U-Net counting",
                type="primary",
                use_container_width=True,
                disabled=(not upload_status.ready and not path_status.ready) or image is None,
            )
            st.caption("Locked test configuration uses threshold 0.50 and minimum component area 2. Uploaded checkpoint overrides the path field.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Counting Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No U-Net counting run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running counting.")
                return
            try:
                result = count_with_unet_model(
                    image=image,
                    model_pt_path=model_pt_path,
                    model_pt_bytes=uploaded_model_bytes,
                    threshold=float(threshold),
                    min_component_area=int(min_component_area),
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b = st.columns(2)
            metric_a.metric("Predicted count", result.predicted_count)
            metric_b.metric("Image source", source_name)
            view_a, view_b = st.columns(2)
            view_a.image(bgr_to_rgb(result.overlay_bgr), caption="Overlay", use_container_width=True)
            view_b.image(mask_to_rgb(result.mask), caption="Predicted mask", use_container_width=True)
            render_config_table(result.config)


def render_unet_classification(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("U-Net Classification")
            preset_names = list(UNET_SPECIES_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("U-Net checkpoint", options=preset_names, key="unet_species_model_preset")
            preset_path = UNET_SPECIES_MODEL_PRESETS.get(preset_name, "")
            current_preset_state = st.session_state.get("unet_species_model_preset_previous")
            if current_preset_state != preset_name:
                if preset_name != "Manual path":
                    st.session_state["unet_species_model_pt_path"] = preset_path
                elif "unet_species_model_pt_path" not in st.session_state:
                    st.session_state["unet_species_model_pt_path"] = ""
                st.session_state["unet_species_model_preset_previous"] = preset_name
            uploaded_model_file = st.file_uploader("Model checkpoint upload", type=["pt"], key="unet_species_model_pt_upload")
            model_pt_path = st.text_input("Model checkpoint path", key="unet_species_model_pt_path")
            uploaded_model_bytes = uploaded_model_file.getvalue() if uploaded_model_file is not None else None
            uploaded_model_name = uploaded_model_file.name if uploaded_model_file is not None else ""
            upload_status = inspect_uploaded_unet_species_model(uploaded_model_bytes, uploaded_model_name)
            path_status = inspect_unet_species_model_path(model_pt_path, ROOT)
            reject_empty_unsupported = st.checkbox(
                "Reject likely empty plates",
                value=True,
                key="unet_species_reject_empty",
                help="Keep species classification for countable and uncountable plates, but avoid forcing a species label on likely empty plates.",
            )
            render_unet_model_source_status(
                upload_status,
                path_status,
                empty_message="Upload a U-Net species model checkpoint or provide a local path.",
                candidate_label="Detected local U-Net species checkpoint candidates:",
            )
            run = st.button(
                "Run U-Net classification",
                type="primary",
                use_container_width=True,
                disabled=(not upload_status.ready and not path_status.ready) or image is None,
            )
            st.caption("Uploaded checkpoint overrides the path field. The locked U-Net species model is the current best image-level classifier in the project.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Classification Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No U-Net classification run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running classification.")
                return
            try:
                result = classify_with_unet_species_model(
                    image=image,
                    model_pt_path=model_pt_path,
                    model_pt_bytes=uploaded_model_bytes,
                    reject_empty_unsupported=reject_empty_unsupported,
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Predicted class", result.predicted_class)
            metric_b.metric("Model", result.classifier_type)
            metric_c.metric(result.score_name, f"{result.score_value:.3f}")
            if not result.accepted:
                st.warning(result.rejection_reason)
            st.caption(f"Image source: {source_name}")
            render_config_table(result.config)


def render_yolo_counting(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("YOLO Counting")
            preset_names = list(YOLO_COUNT_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("YOLO checkpoint", options=preset_names, key="yolo_count_model_preset")
            preset_path = YOLO_COUNT_MODEL_PRESETS.get(preset_name, "")
            current_preset_state = st.session_state.get("yolo_count_model_preset_previous")
            if current_preset_state != preset_name:
                if preset_name != "Manual path":
                    st.session_state["yolo_count_model_pt_path"] = preset_path
                elif "yolo_count_model_pt_path" not in st.session_state:
                    st.session_state["yolo_count_model_pt_path"] = ""
                st.session_state["yolo_count_model_preset_previous"] = preset_name
            uploaded_model_file = st.file_uploader("Model checkpoint upload", type=["pt"], key="yolo_count_model_pt_upload")
            model_pt_path = st.text_input("Model checkpoint path", key="yolo_count_model_pt_path")
            uploaded_model_bytes = uploaded_model_file.getvalue() if uploaded_model_file is not None else None
            uploaded_model_name = uploaded_model_file.name if uploaded_model_file is not None else ""
            upload_status = inspect_uploaded_yolo_count_model(uploaded_model_bytes, uploaded_model_name)
            path_status = inspect_yolo_count_model_path(model_pt_path, ROOT)
            confidence_threshold = st.number_input("Confidence threshold", min_value=0.05, max_value=0.95, value=0.45, step=0.05)
            image_size = st.number_input("Inference image size", min_value=640, max_value=2048, value=1536, step=64)
            render_unet_model_source_status(
                upload_status,
                path_status,
                empty_message="Upload a YOLO counting model checkpoint or provide a local path.",
                candidate_label="Detected local YOLO counting checkpoint candidates:",
            )
            run = st.button(
                "Run YOLO counting",
                type="primary",
                use_container_width=True,
                disabled=(not upload_status.ready and not path_status.ready) or image is None,
            )
            st.caption("Locked test configuration uses confidence threshold 0.45. Uploaded checkpoint overrides the path field.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Counting Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No YOLO counting run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running counting.")
                return
            try:
                result = count_with_yolo_model(
                    image=image,
                    model_pt_path=model_pt_path,
                    model_pt_bytes=uploaded_model_bytes,
                    confidence_threshold=float(confidence_threshold),
                    image_size=int(image_size),
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Predicted count", result.predicted_count)
            metric_b.metric("Detections kept", len(result.detections))
            metric_c.metric("Image source", source_name)
            st.image(bgr_to_rgb(result.overlay_bgr), caption="Detection overlay", use_container_width=True)
            if result.detections:
                st.dataframe(pd.DataFrame(detection_rows(result.detections)), use_container_width=True, hide_index=True)
            else:
                st.info("No detections passed the selected confidence threshold.")
            render_config_table(result.config)


def render_yolo_classification(image, source_name: str) -> None:
    control_col, result_col = st.columns([0.9, 1.35])
    with control_col:
        with st.container(border=True):
            st.subheader("YOLO Classification")
            preset_names = list(YOLO_SPECIES_MODEL_PRESETS.keys()) + ["Manual path"]
            preset_name = st.selectbox("YOLO checkpoint", options=preset_names, key="yolo_species_model_preset")
            preset_path = YOLO_SPECIES_MODEL_PRESETS.get(preset_name, "")
            current_preset_state = st.session_state.get("yolo_species_model_preset_previous")
            if current_preset_state != preset_name:
                if preset_name != "Manual path":
                    st.session_state["yolo_species_model_pt_path"] = preset_path
                elif "yolo_species_model_pt_path" not in st.session_state:
                    st.session_state["yolo_species_model_pt_path"] = ""
                st.session_state["yolo_species_model_preset_previous"] = preset_name
            uploaded_model_file = st.file_uploader("Model checkpoint upload", type=["pt"], key="yolo_species_model_pt_upload")
            model_pt_path = st.text_input("Model checkpoint path", key="yolo_species_model_pt_path")
            uploaded_model_bytes = uploaded_model_file.getvalue() if uploaded_model_file is not None else None
            uploaded_model_name = uploaded_model_file.name if uploaded_model_file is not None else ""
            upload_status = inspect_uploaded_yolo_species_model(uploaded_model_bytes, uploaded_model_name)
            path_status = inspect_yolo_species_model_path(model_pt_path, ROOT)
            confidence_threshold = st.number_input("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05, key="yolo_species_conf_threshold")
            image_size = st.number_input("Inference image size", min_value=640, max_value=2048, value=1536, step=64, key="yolo_species_image_size")
            render_unet_model_source_status(
                upload_status,
                path_status,
                empty_message="Upload a YOLO species model checkpoint or provide a local path.",
                candidate_label="Detected local YOLO species checkpoint candidates:",
            )
            run = st.button(
                "Run YOLO classification",
                type="primary",
                use_container_width=True,
                disabled=(not upload_status.ready and not path_status.ready) or image is None,
            )
            st.caption("This is the locked object-level species detector for countable plates, not the image-level species classifier used for uncountable plates.")
    with result_col:
        render_input_panel(image, source_name)
        with st.container(border=True):
            st.markdown('<div class="result-title">Classification Result</div>', unsafe_allow_html=True)
            if not run:
                st.markdown('<div class="result-muted">No YOLO classification run in this session.</div>', unsafe_allow_html=True)
                return
            if image is None:
                st.error("Provide an image before running classification.")
                return
            try:
                result = classify_with_yolo_species_model(
                    image=image,
                    model_pt_path=model_pt_path,
                    model_pt_bytes=uploaded_model_bytes,
                    confidence_threshold=float(confidence_threshold),
                    image_size=int(image_size),
                )
            except Exception as exc:
                st.error(str(exc))
                return
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Detected colonies", result.total_detections)
            metric_b.metric("Species classes", len(result.class_counts))
            metric_c.metric("Image source", source_name)
            st.image(bgr_to_rgb(result.overlay_bgr), caption="Species detection overlay", use_container_width=True)
            if result.class_counts:
                st.dataframe(
                    pd.DataFrame(
                        [{"Species": name, "Detections": count} for name, count in result.class_counts.items()]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No species detections passed the selected confidence threshold.")
            if result.detections:
                st.dataframe(pd.DataFrame(detection_rows(result.detections)), use_container_width=True, hide_index=True)
            render_config_table(result.config)


def render_input_panel(image, source_name: str) -> None:
    with st.container(border=True):
        st.subheader("Input Image")
        if image is None:
            st.markdown('<div class="empty-state">Upload a plate image or provide a local image path.</div>', unsafe_allow_html=True)
            return
        st.image(bgr_to_rgb(image), caption=source_name, use_container_width=True)


def render_config_table(config: dict) -> None:
    st.dataframe(pd.DataFrame(config_rows(config)), use_container_width=True, hide_index=True)


def render_model_source_status(upload_status, path_status) -> None:
    if upload_status.ready:
        st.success(upload_status.message)
        return
    if upload_status.path and upload_status.message != "Upload a species model JSON or provide a local path.":
        st.error(upload_status.message)
        return
    if path_status.ready:
        st.success(path_status.message)
        return
    if path_status.is_colab_path:
        st.warning(path_status.message)
    else:
        st.info(path_status.message)
    if path_status.local_candidates:
        st.caption("Detected local species model candidates:")
        for candidate in path_status.local_candidates:
            st.code(candidate, language="text")


def render_unet_count_model_source_status(upload_status, path_status) -> None:
    render_unet_model_source_status(
        upload_status,
        path_status,
        empty_message="Upload a U-Net counting model checkpoint or provide a local path.",
        candidate_label="Detected local U-Net counting checkpoint candidates:",
    )


def render_unet_model_source_status(upload_status, path_status, empty_message: str, candidate_label: str) -> None:
    if upload_status.ready:
        st.success(upload_status.message)
        return
    if upload_status.path and upload_status.message != empty_message:
        st.error(upload_status.message)
        return
    if path_status.ready:
        st.success(path_status.message)
        return
    if path_status.is_colab_path:
        st.warning(path_status.message)
    else:
        st.info(path_status.message)
    if path_status.local_candidates:
        st.caption(candidate_label)
        for candidate in path_status.local_candidates:
            st.code(candidate, language="text")


if __name__ == "__main__":
    main()
