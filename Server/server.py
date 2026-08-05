"""Local inference server for the Bacteria-Counter-App (Swift/macOS).

Exposes every validated (and experimental-but-safety-checked) pipeline over
HTTP so the native app can call them without needing Python/torch bundled
into the app itself, and so the user can judge for themselves on real
photos which is most robust -- ground truth and internet-photo testing are
proxies, real deployment is the real test.

  - "yolo_old": the original YOLOv8n colony counter trained on AGAR only
    (validated MAE 1.31 on the 700-image AGAR ground-truth benchmark,
    conf>=0.4). Best on AGAR-style plates, fails on the user's own lab
    photos (different visual domain).
  - "yolo_new": the same checkpoint fine-tuned on SAM Copy-Paste synthetic
    data (real AGAR colonies composited onto the user's real empty-dish
    backgrounds) plus real AGAR data. Adds real-domain robustness (0/36
    false positives on real empty backgrounds) at a modest AGAR-domain
    accuracy cost in the "vague" background category (MAE 5.61 -> 8.72 on
    the same 18-image stratified sample). This is the default/recommended
    pipeline.
  - "sam":  CLAHE contrast enhancement + FastSAM zero-shot segmentation +
    dish-boundary/shape filtering (see fastsam_colony_count.py). Useful for
    domains neither YOLO checkpoint has seen, but not held to the same
    validated accuracy as YOLO (MAE 35.11 on the same AGAR sample).
  - "dog_blend", "clahe", "lab_ab": yolo_new further fine-tuned on the same
    data with a preprocessing technique baked into training (not just
    bolted onto inference, which was shown to cause a train/inference
    mismatch penalty). All three pass the real-empty-background safety
    check (0-2/36 false positives). AGAR ground-truth MAE: dog_blend 8.78
    (basically unchanged from yolo_new), clahe 7.33 (modest broad
    improvement), lab_ab 2.94 (by far the best AGAR number, huge gain on
    "vague" backgrounds). BUT: on a broader set of real/internet photos,
    all three detect noticeably fewer colonies than yolo_new, lab_ab most
    severely (its color-isolation transform discards luminance/edges, so
    it can badly under-detect pale, low-color-contrast colonies -- e.g.
    137->7 on one real dense pale-colony plate, even after its full
    70-epoch training completed). Included here anyway, deliberately, so
    real-world testing on your own photos can be the tiebreaker instead of
    ground-truth proxies alone.

A SAM-proposes/YOLO-verifies ensemble was also tried and re-validated after
fine-tuning, but performed worse than yolo_new alone in both cases (MAE 39.44
and 32.22 vs 8.72) — deliberately not exposed here.

  - "gsam2": Colony Grounded SAM2 (Korporaal et al. 2026) -- zero-shot
    Grounding DINO fine-tuned to the microbiology domain, no training on our
    own data at all. Genuinely decent AGAR ground-truth numbers (MAE 6.89,
    best-in-class on "vague" backgrounds specifically), BUT it FAILED the
    real-empty-background safety check outright: 36/36 real empty photos
    from your own lab produced false detections (up to 9 per photo,
    hallucinated from paper texture/creases). No other model in this app
    fails that check anywhere near that badly. Included anyway, at your
    explicit request, so you can judge robustness yourself on real photos
    rather than take this checkpoint's word for it -- treat any count from
    this one with real skepticism, especially on backgrounds it hasn't
    seen. Runs as a subprocess into the separate ColonyGroundedSam2 /
    Grounded-SAM-2 checkout (different venv, different dependencies from
    everything else here) rather than being loaded in-process.

  - "csrnet": density-map regression (CSRNet, VGG16 frontend + dilated-conv
    backend) instead of box detection -- counts by summing a predicted
    heatmap, no NMS, so no double-counting from overlapping boxes in
    principle. Genuinely different paradigm from every other option here.
    Training was interrupted partway through (60 planned epochs, stopped
    early, still noisy epoch-to-epoch on its own validation metric when it
    stopped) -- this is NOT a finished model. On our AGAR ground truth:
    MAE 6.26 (better than yolo_new's 8.72). Real-empty-background safety
    check: 0/36 -- the cleanest of every model in this app. BUT on a
    broader set of real photos it noticeably OVER-counts, sometimes badly
    (137 -> 475 on one dense pale-colony plate) -- the predicted heatmap
    lights up in roughly the right places but spreads too much density per
    colony rather than one clean blob, consistent with unfinished/unstable
    training rather than a fundamental flaw. No per-colony location is
    produced at all (a heatmap isn't discrete detections), so this option
    always returns an empty detections list -- the app will show a count
    with no overlay circles for it. Training may resume later; this
    checkpoint is a snapshot, not a final answer.

Run with:  .venv/bin/python server.py
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import FastSAM, YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastsam_colony_count import count_colonies_fastsam
from csrnet_model import CSRNet

app = FastAPI(title="Bacteria Counter Inference Server")

YOLO_MODEL_PATHS = {
    "yolo_old": "models_trained/YOLO/counter/best.pt",
    "yolo_new": "colony_finetuned_best.pt",
    "dog_blend": "dog_blend_best.pt",
    "clahe": "clahe_best.pt",
    "lab_ab": "lab_ab_best.pt",
}
YOLO_CONF = 0.4
YOLO_IMGSZ = 1536

SAM_IMGSZ = 3840
SAM_MIN_AREA_FRAC = 0.000005
SAM_MAX_AREA_FRAC = 0.02

CSRNET_CKPT_PATH = "csrnet_best.pt"
CSRNET_WORKING_SIZE = 768
CSRNET_IMAGENET_MEAN = [0.485, 0.456, 0.406]
CSRNET_IMAGENET_STD = [0.229, 0.224, 0.225]

print("Loading YOLO counter models...")
yolo_models = {key: YOLO(path) for key, path in YOLO_MODEL_PATHS.items()}
print("Loading FastSAM model...")
sam_model = FastSAM("FastSAM-s.pt")
print("Loading CSRNet model...")
csrnet_model = CSRNet(load_weights=True)
csrnet_model.load_state_dict(torch.load(CSRNET_CKPT_PATH, map_location="cpu", weights_only=True))
csrnet_model.eval()
csrnet_transform = T.Compose([T.ToTensor(), T.Normalize(CSRNET_IMAGENET_MEAN, CSRNET_IMAGENET_STD)])
print("Models loaded. Server ready.")


def make_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def make_dog_blend(img_bgr: np.ndarray, sigma1=2, sigma2=12, strength=1.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1 - g2
    out = img_bgr.astype(np.float32) + (strength * dog)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_lab_ab(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


# Preprocessing baked into each model's own training -- must match at
# inference too, or the model sees a distribution it was never trained on.
YOLO_PREPROCESS = {
    "yolo_old": None,
    "yolo_new": None,
    "dog_blend": make_dog_blend,
    "clahe": make_clahe,
    "lab_ab": make_lab_ab,
}


def run_yolo(image_path: str, model_key: str) -> dict:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    preprocess = YOLO_PREPROCESS[model_key]
    source = preprocess(img) if preprocess is not None else image_path
    res = yolo_models[model_key].predict(source=source, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
                                          max_det=1000, device="cpu", verbose=False)[0]
    boxes = res.boxes
    count = 0 if boxes is None else len(boxes)
    confidence = 0.0
    detections = []
    if count > 0:
        confidence = float(boxes.conf.mean().item() * 100)
        for (x1, y1, x2, y2) in boxes.xyxy.cpu().tolist():
            detections.append({
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "radius": max(x2 - x1, y2 - y1) / 2.0,
            })
    return {"count": count, "confidence": round(confidence, 1), "detections": detections,
            "imageWidth": w, "imageHeight": h}


def run_sam(image_path: str) -> dict:
    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]
    clahe_bgr = make_clahe(img_bgr)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, clahe_bgr)
        clahe_path = tmp.name
    count, kept, _dish = count_colonies_fastsam(
        sam_model, clahe_path, imgsz=SAM_IMGSZ,
        min_area_frac=SAM_MIN_AREA_FRAC, max_area_frac=SAM_MAX_AREA_FRAC,
    )
    confidence = 0.0
    detections = []
    if kept:
        confidence = float(np.mean([k["circularity"] for k in kept]) * 100)
        for k in kept:
            contours, _ = cv2.findContours(k["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            detections.append({"cx": float(cx), "cy": float(cy), "radius": float(radius)})
    return {"count": count, "confidence": round(confidence, 1), "detections": detections,
            "imageWidth": w, "imageHeight": h}


# Defaults to this machine's checkout; teammates should set these two env
# vars instead of editing this file (see Server/README.md's gsam2 section).
GSAM2_PYTHON = os.environ.get(
    "GSAM2_PYTHON",
    "/Users/satriabaladewaharahap/bacteriaserius/repos/Grounded-SAM-2/.venv/bin/python",
)
GSAM2_SCRIPT = os.environ.get(
    "GSAM2_SCRIPT",
    "/Users/satriabaladewaharahap/bacteriaserius/repos/ColonyGroundedSam2/infer_count_only.py",
)
GSAM2_RESULT_PREFIX = "GSAM2_RESULT_JSON:"


def run_gsam2(image_path: str) -> dict:
    proc = subprocess.run(
        [GSAM2_PYTHON, GSAM2_SCRIPT, image_path],
        capture_output=True, text=True, timeout=120,
    )
    result_line = next(
        (line for line in proc.stdout.splitlines() if line.startswith(GSAM2_RESULT_PREFIX)), None
    )
    if result_line is None:
        raise RuntimeError(f"Grounded SAM2 subprocess produced no result "
                            f"(exit={proc.returncode}): {proc.stderr[-500:]}")
    parsed = json.loads(result_line[len(GSAM2_RESULT_PREFIX):])
    detections = []
    confidences = parsed.get("confidences", [])
    for i, (x1, y1, x2, y2) in enumerate(parsed["boxes"]):
        detections.append({
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "radius": max(x2 - x1, y2 - y1) / 2.0,
        })
    confidence = float(np.mean(confidences) * 100) if confidences else 0.0
    return {"count": parsed["count"], "confidence": round(confidence, 1), "detections": detections,
            "imageWidth": parsed["imageWidth"], "imageHeight": parsed["imageHeight"]}


def run_csrnet(image_path: str) -> dict:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (CSRNET_WORKING_SIZE, CSRNET_WORKING_SIZE), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_t = csrnet_transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        density = csrnet_model(img_t)
    count = float(density.sum().item())

    # No discrete per-colony locations to draw as circles -- instead, render
    # the density map itself as a color heatmap blended over the original
    # photo, so there's still something honest to look at (this is exactly
    # what the model actually produces, not a fabricated detection).
    density_map = density[0, 0].numpy()
    heat = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_full_res = cv2.resize(heat_color, (w, h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(img, 0.55, heat_full_res, 0.45, 0)
    ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    heatmap_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

    return {"count": round(count), "confidence": 0.0, "detections": [],
            "imageWidth": w, "imageHeight": h, "heatmapImage": heatmap_b64}


@app.get("/health")
def health():
    return {"status": "ok"}


VALID_MODELS = ("yolo_old", "yolo_new", "sam", "dog_blend", "clahe", "lab_ab", "gsam2", "csrnet")


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), model: str = Form(...)):
    if model not in VALID_MODELS:
        return JSONResponse(status_code=400, content={"error": f"model must be one of {VALID_MODELS}"})

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    try:
        if model == "sam":
            result = run_sam(image_path)
        elif model == "gsam2":
            result = run_gsam2(image_path)
        elif model == "csrnet":
            result = run_csrnet(image_path)
        else:
            result = run_yolo(image_path, model)
    except Exception as exc:  # surface a readable error to the app instead of a bare 500
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return {
        "totalColonies": result["count"],
        "averageConfidence": result["confidence"],
        "modelUsed": model,
        "detections": result["detections"],
        "imageWidth": result["imageWidth"],
        "imageHeight": result["imageHeight"],
        "heatmapImage": result.get("heatmapImage"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8721, log_level="info")
