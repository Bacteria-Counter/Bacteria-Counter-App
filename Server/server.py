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
  - "mac1": yolo_new fine-tuned further on external colony datasets (the
    Mac Studio run of August 2026). The most accurate YOLO here on bright
    plates -- on the 42-image PCA holdout, MAE 8.12 against yolo_new's 9.62,
    and clearly better on dense plates (23.4 vs 35.6). Like every YOLO
    variant it keeps a perfect empty-plate record (0/34, zero false
    detections), which no SAM variant matches -- sam_tuned produces 48. So
    this is the option to reach for when a false positive costs more than a
    miss: sterility checks, negative controls, anything reported as "no
    growth". It is not the accuracy leader overall (sam_tuned reaches 3.86)
    and it heavily over-counts dense lab plates -- 437 on the plate where
    280 was confirmed correct by eye.

  - "sam":  CLAHE contrast enhancement + FastSAM zero-shot segmentation +
    dish-boundary/shape filtering (see fastsam_colony_count.py). Useful for
    domains neither YOLO checkpoint has seen, but not held to the same
    validated accuracy as YOLO (MAE 35.11 on the same AGAR sample).
    Settings frozen at their original values so older results stay
    reproducible -- the August 2026 tuning is exposed as the two options
    below instead of being folded in here.

  - "sam_tuned": "sam" with its filter parameters tuned against a new
    127-image bright-background benchmark (PCA), the first benchmark in this
    project that resembles the Ciputra lab rather than AGAR. On a 42-image
    holdout never used for tuning, measured through the production path:
    MAE 5.36 -> 3.86, 2.9s -> 1.3s per photo, false positives on 34 empty
    plates 62 -> 48. Best general accuracy of any option on bright plates.
    Its blind spot is pinpoint colonies -- on the sparse lab plate it finds
    22 where roughly 104 are visible.

  - "sam_micro": "sam_tuned" plus a second pass at imgsz 4480 when the
    plate's median colony is under 1.5% of the dish diameter. Built for
    pinpoint colonies: 22 -> 104 on the sparse lab plate, both counts
    confirmed by eye against the photo. Identical to sam_tuned on ordinary
    plates (the second pass simply does not trigger) and on empty plates,
    where escalation would be dangerous -- an earlier count-based trigger
    was rejected precisely because empty plates always count low and so
    always escalated, doubling their false positives. Costs ~8s instead of
    ~1.3s on the plates where it does trigger.
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
    MAE 6.26 (better than yolo_new's 8.72), and 9.25 on the 42-image PCA
    holdout. Real-empty-background safety check: 0/36 -- the cleanest of
    every model in this app. This checkpoint is a snapshot, not a final
    answer.

    Shows a heatmap rather than circles, and that is a deliberate limit of
    the method, not an omission. Drawing circles from density peaks was
    tried and looked visibly misaligned on real plates: the density map is
    96x96 for a 3120x4160 photo, so each cell covers 32x43 real pixels and
    every peak snaps to that grid. Circles would assert a per-colony
    position this model does not have. Raising the working size above 768
    was also tried and made the count worse (9.68 at 1024, 10.22 at 1280),
    so 768 stays.

All YOLO variants (yolo_old/yolo_new/dog_blend/clahe/lab_ab) use an adaptive
inference imgsz (see adaptive_imgsz()) instead of a fixed 1536 -- it's
computed per-photo from the detected dish radius so small/faint colonies in
photos where the dish is framed smaller (e.g. a phone shot further back)
don't get shrunk past the point of detection by a one-size-fits-all resize.
Verified against the AGAR ground truth + real lab photos: no accuracy
regression, meaningfully better detection on real phone photos.

Run with:  .venv/bin/python server.py
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import FastSAM, YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastsam_colony_count import count_colonies_fastsam, find_dish_circle
from csrnet_model import CSRNet
from cfu_calculator import (
    DEFAULT_DISH_AREA_CM2,
    PlateReading,
    assess_plate_countability,
    calculate_cfu,
)

app = FastAPI(title="Bacteria Counter Inference Server")

YOLO_MODEL_PATHS = {
    "yolo_old": "models_trained/YOLO/counter/best.pt",
    "yolo_new": "colony_finetuned_best.pt",
    "dog_blend": "dog_blend_best.pt",
    "clahe": "clahe_best.pt",
    "lab_ab": "lab_ab_best.pt",
    "mac1": "mac1_best.pt",
    "mac2": "mac2_best.pt",
}
YOLO_CONF = 0.4
YOLO_IMGSZ = 1536  # fallback only -- used when the dish can't be detected

# The AGAR training photos show the dish radius at ~581px once YOLO's own
# letterbox resize has been applied at imgsz=1536 (measured: dish radius
# ~1510px on ~3990px-long-side training photos, 1510 * 1536/3990 ~= 581).
# Real phone photos often frame the dish much smaller in the raw frame (e.g.
# ~945px radius on a 4160px-tall photo), so resizing them to the same fixed
# 1536 shrinks small/faint colonies past the point the model can see them.
# Instead, pick imgsz per-photo so the dish (and therefore each colony)
# lands at roughly that same ~581px reference scale the model was trained
# on, regardless of how the photo was framed or what resolution it came in
# at. Verified against the 18-image AGAR ground truth + real lab photos:
# zero regression on AGAR/empty-background safety, meaningfully more
# detections on real phone photos with small/distant colonies.
DISH_RADIUS_REFERENCE_PX = 581
ADAPTIVE_IMGSZ_MIN = 1280
ADAPTIVE_IMGSZ_MAX = 3200


def adaptive_imgsz(img_bgr: np.ndarray) -> int:
    h, w = img_bgr.shape[:2]
    dish = find_dish_circle(img_bgr)
    if dish is None:
        return YOLO_IMGSZ
    _, _, dish_radius = dish
    long_side = max(h, w)
    required = DISH_RADIUS_REFERENCE_PX * long_side / dish_radius
    required = int(round(required / 32) * 32)  # YOLO requires multiples of 32
    return max(ADAPTIVE_IMGSZ_MIN, min(ADAPTIVE_IMGSZ_MAX, required))

# "sam" -- ORIGINAL settings, deliberately left untouched. Everything tuned in
# August 2026 is exposed as the separate "sam_tuned"/"sam_micro" options below
# rather than folded in here, so a result produced with this model last month
# can still be reproduced today.
SAM_IMGSZ = 3840
SAM_MIN_AREA_FRAC = 0.000005
SAM_MAX_AREA_FRAC = 0.02

# "sam_tuned" -- tuned 2026-08-12 against the PCA bright-background benchmark,
# on a 42-image holdout never seen during tuning, and measured through this
# production path rather than the tuning harness: MAE 5.36 -> 3.86, 2.9s ->
# 1.3s per photo, false positives on 34 empty plates 62 -> 48.
#
# Three settings reverse "sam"'s defaults, each for a measured reason:
#   - imgsz adaptive rather than a fixed 3840. Bigger is not better on ordinary
#     plates: fixed 3840 and 3200 both scored worse, because the model stops
#     recognising a colony once it fills far more of the frame than the scale
#     it was tuned around.
#   - dish margin 1.00; cropping the outer 6% discarded real colonies at the rim.
#   - square-cell filter off. Written for grid-patterned backgrounds, on bright
#     plates it rejects genuine colonies.
#
# conf stays at 0.2, not the 0.3 the sweep preferred. The sweep scored cached
# masks generated at conf=0.05 and filtered upward, which is not how inference
# runs -- generating at the threshold changes what NMS suppresses. Re-measured
# here, 0.2 beat 0.3 on the holdout and avoided collapsing the sparse lab photo
# from 22 colonies to 8.
SAM_TUNED_CONF = 0.2
SAM_TUNED_DISH_MARGIN_RATIO = 1.0
SAM_TUNED_MIN_AREA_FRAC = 0.0000001
SAM_TUNED_MAX_AREA_FRAC = 0.005
SAM_TUNED_DROP_SQUARE_LIKE = False
# Plates are labelled -- pen or a printed sticker -- and the writing sits
# inside the dish where every geometric filter treats it as a colony. See
# looks_like_marking(): colour distance catches ink of any hue, darkness
# catches black marker and printed labels. "sam" keeps its original behaviour
# and is deliberately not given this.
SAM_TUNED_REJECT_MARKINGS = True

# "sam_micro" -- sam_tuned plus a second pass at much higher resolution for
# plates whose colonies are pinpoint. Both halves were confirmed on the lab
# photos by the microbiologist: on the sparse plate the extra detections at
# 4480 are real colonies (24 -> 104), on the dense plate the lower resolution
# is the correct one (280, not 171).
#
# The trigger is colony SIZE, not colony count. A count-based rule was tried
# first and rejected: empty plates always count low, so they always escalated
# and their false positives doubled. Size separates the cases cleanly -- median
# colony diameter over dish diameter puts the sparse plate at 1.08% and both
# dense plates above 2.4%.
#
# The minimum count is a second guard rather than the trigger: below it there
# are too few detections for a median to mean anything, and an empty plate
# (1-2 spurious detections) can never reach it. On the 42-image holdout and 34
# empty plates: MAE 3.86 -> 3.83, false positives unchanged at 48, firing on
# 1 of 42 benchmark images and 0 of 34 empty plates.
SAM_ESCALATE_IMGSZ = 4480
SAM_ESCALATE_MIN_COUNT = 8
SAM_ESCALATE_MAX_COLONY_PCT = 1.5

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
    "mac1": None,
    "mac2": None,
    "dog_blend": make_dog_blend,
    "clahe": make_clahe,
    "lab_ab": make_lab_ab,
}


def run_yolo(image_path: str, model_key: str) -> dict:
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    imgsz = adaptive_imgsz(img)
    preprocess = YOLO_PREPROCESS[model_key]
    source = preprocess(img) if preprocess is not None else image_path
    res = yolo_models[model_key].predict(source=source, imgsz=imgsz, conf=YOLO_CONF,
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


def run_sam(image_path: str, variant: str = "sam") -> dict:
    """FastSAM counting. `variant` selects one of three exposed models:
    "sam" (original settings), "sam_tuned", or "sam_micro"."""
    img_bgr = cv2.imread(image_path)
    h, w = img_bgr.shape[:2]
    clahe_bgr = make_clahe(img_bgr)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, clahe_bgr)
        clahe_path = tmp.name
    # Measured on the original frame, not the CLAHE copy -- CLAHE changes
    # contrast, and the dish circle it keys off must come from the same pixels
    # the caller sees.
    if variant == "sam":
        count, kept, _dish = count_colonies_fastsam(
            sam_model, clahe_path, imgsz=SAM_IMGSZ,
            min_area_frac=SAM_MIN_AREA_FRAC, max_area_frac=SAM_MAX_AREA_FRAC,
        )
        return _sam_response(count, kept, w, h)

    sam_args = dict(
        conf=SAM_TUNED_CONF, min_circularity=0.75,
        dish_margin_ratio=SAM_TUNED_DISH_MARGIN_RATIO,
        min_area_frac=SAM_TUNED_MIN_AREA_FRAC,
        max_area_frac=SAM_TUNED_MAX_AREA_FRAC,
        drop_square_like=SAM_TUNED_DROP_SQUARE_LIKE,
        reject_markings=SAM_TUNED_REJECT_MARKINGS, color_ref_bgr=img_bgr,
    )
    count, kept, _dish = count_colonies_fastsam(
        sam_model, clahe_path, imgsz=adaptive_imgsz(img_bgr), **sam_args)

    if (variant == "sam_micro" and kept and _dish is not None
            and count >= SAM_ESCALATE_MIN_COUNT):
        median_area = float(np.median([k["area"] for k in kept]))
        colony_pct = (median_area / np.pi) ** 0.5 / _dish[2] * 100
        if colony_pct < SAM_ESCALATE_MAX_COLONY_PCT:
            print(f"[sam_micro] koloni {colony_pct:.2f}% diameter cawan -- "
                  f"ulangi pada imgsz {SAM_ESCALATE_IMGSZ}", flush=True)
            count, kept, _dish = count_colonies_fastsam(
                sam_model, clahe_path, imgsz=SAM_ESCALATE_IMGSZ, **sam_args)
    return _sam_response(count, kept, w, h)


def _sam_response(count: int, kept: list, w: int, h: int) -> dict:
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

    # Deliberately a heatmap and not circles. Circles were tried, drawn from
    # peaks in the density map, and looked plainly misaligned against the
    # colonies on the plate. The reason is structural rather than a tuning
    # miss: for a 3120x4160 photo the density map is only 96x96, so one cell
    # spans 32x43 real pixels and every peak snaps to that coarse grid -- and
    # the square 768 resize distorts a portrait photo on top of that. A circle
    # claims "the colony is here"; this model only supports "there is roughly
    # this much colony around here". The heatmap states exactly that, and is
    # what the model actually predicts rather than an inference drawn from it.
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


class PlateInput(BaseModel):
    """One petri dish in a CFU calculation. `dilution` is the fraction (0.01
    for a 1:100 dilution). `status` is "ok", "spreading", "lab_accident" or
    "tntc" -- the latter three are the microbiologist's judgement call at
    counting time, not something the detector can decide."""
    dilution: float
    count: Optional[int] = None
    status: str = "ok"
    spreadFraction: Optional[float] = None


class CFURequest(BaseModel):
    plates: list[PlateInput]
    dishAreaCm2: float = DEFAULT_DISH_AREA_CM2
    method: str = "pour"          # "pour" (1 mL) or "spread" (0.1 mL)
    unit: str = "CFU/ml"          # or "CFU/g"


@app.post("/calculate-cfu")
def calculate_cfu_endpoint(request: CFURequest):
    """Turn per-plate colony counts into a reportable CFU figure using the
    APHA 2002 rules (see cfu_calculator.py). This is separate from /analyze:
    /analyze counts one photo, this combines several counted plates across
    dilutions into the number the lab actually reports."""
    try:
        readings = [
            PlateReading(
                dilution=p.dilution,
                count=p.count,
                status=p.status,
                spread_fraction=p.spreadFraction,
            )
            for p in request.plates
        ]
        result = calculate_cfu(
            readings,
            dish_area_cm2=request.dishAreaCm2,
            method=request.method,
            unit=request.unit,
        )
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return {
        "value": result.value,
        "display": result.display,
        "regulations": result.regulations,
        "estimated": result.estimated,
        "bounded": result.bounded,
        "detail": result.detail,
    }


VALID_MODELS = ("yolo_old", "yolo_new", "mac1", "mac2", "sam", "sam_tuned", "sam_micro",
                "dog_blend", "clahe", "lab_ab", "gsam2", "csrnet")


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), model: str = Form(...)):
    if model not in VALID_MODELS:
        return JSONResponse(status_code=400, content={"error": f"model must be one of {VALID_MODELS}"})

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    # Log what actually arrived. The app's CAPTURE panel reads
    # output.maxPhotoDimensions at session-start time, which Continuity Camera
    # renegotiates afterward -- so that display can be stale. This line reports
    # the real photo, which is what any resolution claim should rest on.
    _probe = cv2.imread(image_path)
    if _probe is not None:
        print(f"[analyze] model={model} foto diterima: {_probe.shape[1]}x{_probe.shape[0]} px",
              flush=True)

    try:
        if model in ("sam", "sam_tuned", "sam_micro"):
            result = run_sam(image_path, model)
        elif model == "gsam2":
            result = run_gsam2(image_path)
        elif model == "csrnet":
            result = run_csrnet(image_path)
        else:
            result = run_yolo(image_path, model)
    except Exception as exc:  # surface a readable error to the app instead of a bare 500
        return JSONResponse(status_code=500, content={"error": str(exc)})

    # APHA 2002 reliability labelling -- the count itself is untouched, this
    # just says how far it can be trusted (25-250 countable, below/above that
    # is estimate-only, past ~100 colonies/cm^2 it isn't estimable at all).
    countability = assess_plate_countability(result["count"])

    return {
        "totalColonies": result["count"],
        "averageConfidence": result["confidence"],
        "modelUsed": model,
        "detections": result["detections"],
        "imageWidth": result["imageWidth"],
        "imageHeight": result["imageHeight"],
        "countability": countability,
        "heatmapImage": result.get("heatmapImage"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8721, log_level="info")
