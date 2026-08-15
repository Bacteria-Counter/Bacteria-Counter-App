"""Step 2a verification: Core ML against PyTorch across every benchmark we have.

The 12-image sample that motivated this showed MAE 3.00 against 3.17 with only
3 of 12 counts identical, the difference traced to Core ML exports accepting
only square input while ultralytics letterboxes PyTorch input to a stride
multiple. That is a structural difference, not a bug, so the question is
whether it is small enough to accept -- and twelve images is not enough to
answer it.

Three populations, because each can fail differently:
  holdout 42   -- accuracy against ground truth
  empty 34     -- false positives, where a regression would be most damaging
  lab photos   -- the actual target domain, including the sam_micro escalation

Reported per population: MAE for each path, how often the counts agree exactly,
and the signed difference, since a systematic bias would matter far more than
noise of the same magnitude.
"""
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = ROOT / "Bacteria-Counter-App/Server"
# Overridable so the quantised directory can be measured through the exact
# same harness -- a separate copy of this script would be a second variable.
MODELS = Path(os.environ.get("AGARSCOPE_MODELS",
                             os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius") + "/coreml_models"))
ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SERVER))

import coremltools as ct
import fastsam_colony_count as FC
from coreml_fastsam import run_coreml
from ultralytics import FastSAM

SAM_SIZES = [1280, 1920, 2560, 3200, 4480]
FILTERS = dict(min_circularity=0.75, dish_margin_ratio=1.0, min_area_frac=1e-7,
               max_area_frac=0.005, drop_square_like=False)
WORK_MAX = 3200
ESCALATE_IMGSZ, ESCALATE_MIN_COUNT, ESCALATE_MAX_PCT = 4480, 8, 1.5

_cache = {}


def coreml_for(size):
    key = min(SAM_SIZES, key=lambda s: abs(s - size))
    if key not in _cache:
        _cache[key] = ct.models.MLModel(str(MODELS / f"fastsam_{key}.mlpackage"))
    return _cache[key], key


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    if max(img.shape[:2]) > WORK_MAX:
        s = WORK_MAX / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)),
                         interpolation=cv2.INTER_AREA)
    return img


def adaptive(img):
    h, w = img.shape[:2]
    d = FC.find_dish_circle(img)
    if d is None:
        return 1536, None
    return max(1280, min(3200, int(round((581 * max(h, w) / d[2]) / 32) * 32))), d


def filter_masks(masks, img, dish):
    """The filter block from count_colonies_fastsam, applied to either source."""
    H, W = img.shape[:2]
    img_area = H * W
    kept = []
    for mr in masks:
        if mr.shape != (H, W):
            mr = cv2.resize(mr, (W, H), interpolation=cv2.INTER_NEAREST)
        circ, area, square = FC.mask_circularity(mr)
        if circ < FILTERS["min_circularity"]:
            continue
        if FILTERS["drop_square_like"] and square:
            continue
        af = area / img_area
        if not (FILTERS["min_area_frac"] <= af <= FILTERS["max_area_frac"]):
            continue
        ys, xs = np.where(mr > 0)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        if dish is not None:
            dx, dy, dr = dish
            if ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5 > dr * FILTERS["dish_margin_ratio"]:
                continue
        kept.append({"mask": mr, "area": area})
    if kept and dish is not None:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        inner = np.zeros((H, W), np.uint8)
        cv2.circle(inner, (int(dish[0]), int(dish[1])), int(dish[2] * 0.75), 255, -1)
        bg = tuple(float(np.median(lab[:, :, i][inner > 0])) for i in range(3))
        feats = [FC.marking_features(lab, k["mask"], bg) for k in kept]
        flagged = FC.find_markings(feats)
        kept = [k for i, k in enumerate(kept) if i not in flagged]
    return kept


def masks_pytorch(model, jpeg_path, imgsz):
    r = model(jpeg_path, device="cpu", retina_masks=True, imgsz=imgsz, conf=0.2,
              iou=0.7, max_det=3000, verbose=False)[0]
    return [] if r.masks is None else [m.astype(np.uint8)
                                       for m in r.masks.data.cpu().numpy()]


def count_both(model, img, escalate=False):
    """Return (pytorch_count, coreml_count) through identical filtering."""
    imgsz, dish = adaptive(img)
    # Both paths must read the SAME pixels. run_sam writes its CLAHE copy to a
    # temporary JPEG and lets ultralytics read it back, so the PyTorch path has
    # always seen JPEG-compressed input. Handing the Core ML path the raw array
    # instead made it look 17.6% worse on the sparse plate -- that gap was this
    # harness comparing two preprocessing chains, not two model formats.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as t:
        cv2.imwrite(t.name, clahe(img))
        jpeg_path = t.name
    ci = cv2.imread(jpeg_path)
    kp = filter_masks(masks_pytorch(model, jpeg_path, imgsz), img, dish)
    ml, key = coreml_for(imgsz)
    kc = filter_masks(run_coreml(ml, ci, key), img, dish)

    if escalate:
        for kept, is_pt in ((kp, True), (kc, False)):
            if not kept or dish is None or len(kept) < ESCALATE_MIN_COUNT:
                continue
            med = float(np.median([k["area"] for k in kept]))
            pct = (med / np.pi) ** 0.5 / dish[2] * 100
            if pct >= ESCALATE_MAX_PCT:
                continue
            if is_pt:
                kp = filter_masks(masks_pytorch(model, jpeg_path, ESCALATE_IMGSZ), img, dish)
            else:
                m2, k2 = coreml_for(ESCALATE_IMGSZ)
                kc = filter_masks(run_coreml(m2, ci, k2), img, dish)
    Path(jpeg_path).unlink(missing_ok=True)
    return len(kp), len(kc)


def report(name, rows, truths=None):
    pt = [r[0] for r in rows]
    cm = [r[1] for r in rows]
    same = sum(1 for a, b in rows if a == b)
    diff = [b - a for a, b in rows]
    print(f"\n=== {name} (n={len(rows)}) ===")
    if truths:
        mp = sum(abs(a - t) for (a, _), t in zip(rows, truths)) / len(rows)
        mc = sum(abs(b - t) for (_, b), t in zip(rows, truths)) / len(rows)
        print(f"  MAE  PyTorch {mp:.2f} | CoreML {mc:.2f} | selisih {mc-mp:+.2f}")
    print(f"  total PyTorch {sum(pt)} | CoreML {sum(cm)} "
          f"({(sum(cm)-sum(pt))/max(1,sum(pt))*100:+.1f}%)")
    print(f"  hitungan identik {same}/{len(rows)}")
    print(f"  beda: rata-rata {np.mean(diff):+.2f}, median {np.median(diff):+.1f}, "
          f"maks |{max(abs(d) for d in diff)}|")
    return {"n": len(rows), "same": same, "mean_diff": float(np.mean(diff)),
            "max_abs_diff": int(max(abs(d) for d in diff)),
            "total_pt": int(sum(pt)), "total_cm": int(sum(cm))}


def main():
    model = FastSAM(str(SERVER / "FastSAM-s.pt"))
    split = json.load(open(HERE / "pca_split.json"))
    bench = {e["base"]: e for e in json.load(open(HERE / "pca_benchmark.json"))}
    out = {}

    rows, truths = [], []
    for i, b in enumerate(split["holdout"], 1):
        img = load(HERE / "pca_images" / f"{b}.png")
        if img is None:
            continue
        rows.append(count_both(model, img))
        truths.append(bench[b]["truth"])
        if i % 10 == 0:
            print(f"  holdout {i}/42", flush=True)
    out["holdout"] = report("HOLDOUT 42 (akurasi)", rows, truths)

    plates = [p for p in sorted((ROOT / "kaggle_finetune_package/real_empty_backgrounds").iterdir())
              if p.suffix.lower() in (".jpg", ".jpeg", ".png")
              and p.stem not in ("IMG_1702", "IMG_1703")]
    rows = []
    for i, p in enumerate(plates, 1):
        img = load(p)
        if img is None:
            continue
        rows.append(count_both(model, img))
        if i % 10 == 0:
            print(f"  kosong {i}/{len(plates)}", flush=True)
    out["empty"] = report("CAWAN KOSONG 34 (deteksi palsu)", rows)

    LAB = [("padat 1", "WhatsApp Image 2026-08-05 at 15.56.55.jpeg"),
           ("padat 2", "WhatsApp Image 2026-08-05 at 15.56.56 (1).jpeg"),
           ("SEPI", "WhatsApp Image 2026-08-05 at 15.56.56.jpeg")]
    rows = []
    print("\n=== FOTO LAB (dengan eskalasi sam_micro) ===")
    for n, f in LAB:
        img = load(Path.home() / "Downloads" / f)
        a, b = count_both(model, img, escalate=True)
        rows.append((a, b))
        print(f"  {n:<9} PyTorch {a:>4} | CoreML {b:>4}  ({b-a:+d})", flush=True)
    out["lab"] = {"rows": rows}

    json.dump(out, open(HERE / "verify_2a_result.json", "w"), indent=1)
    print("\nVERIFIKASI 2A SELESAI")


if __name__ == "__main__":
    main()
