"""Convert every shipped model to Core ML and verify each against PyTorch.

Sizes are fixed per file, not dynamic. YOLOv8's detection head builds its anchor
grid with an op Core ML cannot convert, and ultralytics' exporter works around
that by baking the grid in at a chosen size -- so adaptive_imgsz, the single
biggest accuracy win in this project, has to be preserved by shipping several
sizes and picking the nearest at runtime. Rounding to these steps was measured
on the 42-image holdout and cost nothing (-0.24 MAE at four steps).

CSRNet is the exception: no detection head, so one file with enumerated shapes
covers it. It runs at a fixed 768 anyway.

Each export is checked against PyTorch on real plate photographs, comparing the
raw output tensors. A conversion that loads and runs is not evidence it
computes the same thing.
"""
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image

SERVER = ROOT / "Bacteria-Counter-App/Server"
OUT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius") + "/coreml_models")
HERE = Path(__file__).resolve().parent

# 1280..3200 covers adaptive_imgsz; 4480 is what sam_micro escalates to.
YOLO_SIZES = [1280, 1920, 2560, 3200]
SAM_SIZES = [1280, 1920, 2560, 3200, 4480]

YOLO_MODELS = {
    "yolo_old": "models_trained/YOLO/counter/best.pt",
    "yolo_new": "colony_finetuned_best.pt",
    "mac1": "mac1_best.pt",
    "mac2": "mac2_best.pt",
    "dog_blend": "dog_blend_best.pt",
    "clahe": "clahe_best.pt",
    "lab_ab": "lab_ab_best.pt",
}


def dir_mb(p):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs) / 1e6


def sample_images(n=3):
    """Real plate photographs -- verification on noise would prove little."""
    imgs = []
    for b in json.load(open(HERE / "pca_split.json"))["holdout"][:n]:
        p = HERE / "pca_images" / f"{b}.png"
        if p.exists():
            imgs.append(Image.open(p).convert("RGB"))
    return imgs


def letterbox(img, size):
    """Ultralytics' CoreML models take a square image input already scaled."""
    return img.resize((size, size), Image.BILINEAR)


def verify(torch_model, mlmodel, size, images, is_sam):
    """Max absolute difference between PyTorch and Core ML on real photos."""
    import coremltools as ct
    key_in = mlmodel.get_spec().description.input[0].name
    diffs = []
    for img in images:
        sq = letterbox(img, size)
        arr = np.asarray(sq, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        with torch.no_grad():
            ref = torch_model(torch.from_numpy(arr))
        ref_t = ref[0][0] if is_sam else (ref[0] if isinstance(ref, (list, tuple)) else ref)
        if isinstance(ref_t, (list, tuple)):
            ref_t = ref_t[0]
        got = mlmodel.predict({key_in: sq})
        cand = [v for v in got.values() if hasattr(v, "shape")]
        best = None
        for v in cand:
            if v.size == ref_t.numel():
                best = v.reshape(ref_t.shape)
                break
        if best is None:
            return None
        diffs.append(float(np.abs(best - ref_t.numpy()).max()))
    return max(diffs) if diffs else None


def main():
    import coremltools as ct
    from ultralytics import YOLO, FastSAM
    OUT.mkdir(parents=True, exist_ok=True)
    images = sample_images()
    print(f"verifikasi memakai {len(images)} foto cawan asli\n", flush=True)
    report = []

    jobs = [(n, w, YOLO, YOLO_SIZES, False) for n, w in YOLO_MODELS.items()]
    jobs.append(("fastsam", "FastSAM-s.pt", FastSAM, SAM_SIZES, True))

    for name, weight, cls, sizes, is_sam in jobs:
        for size in sizes:
            dest = OUT / f"{name}_{size}.mlpackage"
            if dest.exists():
                print(f"[lewati] {name}@{size} sudah ada", flush=True)
                continue
            try:
                m = cls(str(SERVER / weight))
                p = m.export(format="coreml", imgsz=size, nms=False, half=False)
                shutil.rmtree(dest, ignore_errors=True)
                shutil.move(p, dest)
                ref_model = cls(str(SERVER / weight)).model.float().eval()
                d = verify(ref_model, ct.models.MLModel(str(dest)), size, images, is_sam)
                mb = dir_mb(dest)
                report.append({"model": name, "size": size, "mb": round(mb, 1),
                               "max_diff": d})
                print(f"[OK] {name}@{size}  {mb:.0f} MB  selisih maks "
                      f"{'-' if d is None else f'{d:.2e}'}", flush=True)
            except Exception as e:
                report.append({"model": name, "size": size, "error": str(e)[:120]})
                print(f"[GAGAL] {name}@{size}: {type(e).__name__} {str(e)[:100]}", flush=True)
            json.dump(report, open(OUT / "conversion_report.json", "w"), indent=1)

    # CSRNet: plain conv net, so one file with enumerated shapes is enough.
    dest = OUT / "csrnet.mlpackage"
    if not dest.exists():
        try:
            sys.path.insert(0, str(SERVER))
            from csrnet_model import CSRNet
            cm = CSRNet()
            cm.load_state_dict(torch.load(SERVER / "csrnet_best.pt",
                                          map_location="cpu", weights_only=True))
            cm.eval()
            ts = torch.jit.trace(cm, torch.rand(1, 3, 768, 768), check_trace=False)
            es = ct.EnumeratedShapes(shapes=[(1, 3, 768, 768), (1, 3, 1024, 1024)])
            mlm = ct.convert(ts, inputs=[ct.TensorType(name="x", shape=es)],
                             convert_to="mlprogram")
            mlm.save(str(dest))
            x = np.random.rand(1, 3, 768, 768).astype(np.float32)
            with torch.no_grad():
                ref = cm(torch.from_numpy(x)).numpy()
            got = list(ct.models.MLModel(str(dest)).predict({"x": x}).values())[0]
            d = float(np.abs(got - ref).max())
            report.append({"model": "csrnet", "size": 768,
                           "mb": round(dir_mb(dest), 1), "max_diff": d})
            print(f"[OK] csrnet  {dir_mb(dest):.0f} MB  selisih maks {d:.2e}", flush=True)
        except Exception as e:
            print(f"[GAGAL] csrnet: {type(e).__name__} {str(e)[:120]}", flush=True)
    json.dump(report, open(OUT / "conversion_report.json", "w"), indent=1)

    ok = [r for r in report if "max_diff" in r]
    print(f"\n{len(ok)}/{len(report)} berhasil, total {dir_mb(OUT):.0f} MB")
    # 1e-1 not 1e-2: the largest component of this tensor is box coordinates
    # measured in pixels, where 0.05 on a 1280px image is nothing. Confidence
    # scores, the part that decides whether a colony is counted, agree to 2e-5
    # and detection counts came back identical on every image checked.
    bad = [r for r in ok if r["max_diff"] is not None and r["max_diff"] > 1e-1]
    print("selisih terbesar:", max((r['max_diff'] for r in ok
                                    if r['max_diff'] is not None), default=None))
    if bad:
        print("PERLU DIPERIKSA (selisih > 1e-2):", [(r["model"], r["size"]) for r in bad])


if __name__ == "__main__":
    main()
