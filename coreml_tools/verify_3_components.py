"""Step 3 component checks: each Swift preprocessing stage against OpenCV.

Deliberately run BEFORE any count comparison. A count is a single number at the
end of a long chain, and this project has twice traced a "model" discrepancy to
a preprocessing one -- so each transform is asked its own question first, where
the answer is a pixel difference and not an aggregate.

Usage:  python verify_3_components.py [image.png]
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("AGARSCOPE_ROOT", "/Users/satriabaladewaharahap/bacteriaserius"))
SWIFT = ROOT / "Bacteria-Counter-App/AgarScopeKit/.build/release/agarscope"
DEFAULT_IMG = ROOT / "bench_data/verify_set/h_E.coli-0330-100-1.png"


def make_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def make_dog_blend(img_bgr, sigma1=2, sigma2=12, strength=1.0):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma2)
    dog = g1 - g2
    out = img_bgr.astype(np.float32) + (strength * dog)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def make_lab_ab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a_stretch = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
    b_stretch = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.merge([a_stretch, b_stretch, np.full_like(a, 128)])


def compare(name, swift_rgb, cv_bgr):
    """Both sides as RGB. Ultralytics flips BGR->RGB before the model, so RGB
    is the representation the model actually sees and the one worth matching."""
    ref = cv_bgr[:, :, ::-1].astype(np.int32)
    got = swift_rgb.astype(np.int32)
    if ref.shape != got.shape:
        print(f"  {name:<10} BENTUK BEDA swift {got.shape} vs opencv {ref.shape}")
        return
    d = np.abs(ref - got)
    pct = (d > 0).mean() * 100
    print(f"  {name:<10} beda {pct:6.2f}% byte, rata2 {d.mean():.4f}, maks {d.max()}")


def dump(kind, img_path, shape):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
        out = t.name
    r = subprocess.run([str(SWIFT), "--dump-prep", kind, str(img_path), out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"swift --dump-prep {kind} gagal: {r.stderr[-300:]}")
    data = np.fromfile(out, np.uint8).reshape(shape)
    Path(out).unlink(missing_ok=True)
    return data


def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMG
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    print(f"gambar {img_path.name} {w}x{h}\n")

    print("praproses YOLO (Swift vs OpenCV):")
    compare("clahe", dump("clahe", img_path, (h, w, 3)), make_clahe(img))
    compare("dog_blend", dump("dog_blend", img_path, (h, w, 3)), make_dog_blend(img))
    compare("lab_ab", dump("lab_ab", img_path, (h, w, 3)), make_lab_ab(img))

    # make_clahe is a round trip, so a whole-transform difference cannot say
    # which half caused it. Three separate questions, each with one variable:
    # the forward conversion, CLAHE given identical input, and the inverse.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_raw, a_ch, b_ch = cv2.split(lab)

    print("\nisolasi, satu variabel per baris:")
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
        l_path = t.name
    l_raw.tofile(l_path)
    r = subprocess.run([str(SWIFT), "--lchan", str(img_path), l_path],
                       capture_output=True, text=True)
    print("  L maju     " + r.stdout.strip().splitlines()[-1].replace("beda vs OpenCV: ", ""))

    clahe_ref = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_raw)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
        ref_path = t.name
    clahe_ref.tofile(ref_path)
    r = subprocess.run([str(SWIFT), "--clahe-bin", l_path, str(w), str(h), ref_path],
                       capture_output=True, text=True)
    print("  CLAHE      " + r.stdout.strip().splitlines()[-1].replace("beda: ", "")
          + "   (L identik di kedua sisi)")
    for p in (l_path, ref_path):
        Path(p).unlink(missing_ok=True)

    l = clahe_ref
    lab_adj = cv2.merge([l, a_ch, b_ch])
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as t:
        lab_path, out_path = t.name, t.name + ".out"
    lab_adj.tofile(lab_path)
    r = subprocess.run([str(SWIFT), "--dump-lab2rgb", lab_path, str(w), str(h), out_path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"swift --dump-lab2rgb gagal: {r.stderr[-300:]}")
    got = np.fromfile(out_path, np.uint8).reshape(h, w, 3)
    compare("lab2rgb", got, cv2.cvtColor(lab_adj, cv2.COLOR_LAB2BGR))
    for p in (lab_path, out_path):
        Path(p).unlink(missing_ok=True)

    print("\nresize INTER_AREA CSRNet (Swift vs OpenCV):")
    ref = cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA)
    compare("area 768", dump("area", img_path, (768, 768, 3)), ref)


if __name__ == "__main__":
    main()
