import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from ultralytics import FastSAM


def split_merged_mask(mask_u8, reference_area, min_distance_px=15):
    """Estimate how many colonies are actually inside a mask that's much
    larger than a typical single colony, via distance-transform peak counting
    on the mask's own real shape (not a bounding box heuristic)."""
    dist = ndi.distance_transform_edt(mask_u8 > 0)
    max_d = float(dist.max())
    if max_d <= 0:
        return 1
    coords = peak_local_max(dist, min_distance=min_distance_px, labels=mask_u8 > 0)
    if len(coords) <= 1:
        return 1
    peak_mask = np.zeros(dist.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    _, n_peaks = ndi.label(peak_mask)
    # sanity cap: never estimate more sub-colonies than area ratio plausibly allows
    max_plausible = max(1, round(mask_u8.sum() / max(1.0, reference_area * 0.6)))
    return max(1, min(n_peaks, max_plausible))


def find_dish_circle(img_bgr):
    """Detect the petri dish boundary via Hough circle transform on edges."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1200 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)))
    small = cv2.GaussianBlur(small, (9, 9), 2)
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=small.shape[0] // 2,
        param1=60, param2=60,
        minRadius=int(small.shape[1] * 0.25), maxRadius=int(small.shape[1] * 0.5),
    )
    if circles is None:
        return None
    c = circles[0][0]
    return (c[0] / scale, c[1] / scale, c[2] / scale)


def mask_circularity(mask_u8):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, True
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter <= 0:
        return 0.0, area, True
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # reject grid-cell squares: approximate the contour polygon and check vertex
    # count / rectangularity. Real colonies approximate to many-sided (round)
    # polygons; grid cells collapse to ~4 corners even after rounding.
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    rect_area = cv2.minAreaRect(c)[1]
    rect_fill = area / max(1.0, rect_area[0] * rect_area[1])
    is_square_like = len(approx) <= 5 and rect_fill > 0.85
    return circularity, area, is_square_like


def count_colonies_fastsam(model, img_path, imgsz=1536, conf=0.2, iou=0.7, max_det=2000,
                            min_circularity=0.75, dish_margin_ratio=0.94,
                            min_area_frac=0.00002, max_area_frac=0.02):
    img_bgr = cv2.imread(img_path)
    H, W = img_bgr.shape[:2]
    dish = find_dish_circle(img_bgr)

    res = model(img_path, device="cpu", retina_masks=True, imgsz=imgsz, conf=conf, iou=iou,
                max_det=max_det, verbose=False)[0]
    if res.masks is None:
        return 0, [], dish

    masks = res.masks.data.cpu().numpy()
    img_area = H * W
    kept = []
    for m in masks:
        mask_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        circularity, area, is_square_like = mask_circularity(mask_resized)
        area_frac = area / img_area
        if circularity < min_circularity:
            continue
        if is_square_like:
            continue
        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        ys, xs = np.where(mask_resized > 0)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        if dish is not None:
            dx, dy, dr = dish
            dist = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
            if dist > dr * dish_margin_ratio:
                continue
        k = {"mask": mask_resized, "circularity": circularity, "area": area, "centroid": (cx, cy), "sub_count": 1}
        kept.append(k)

    return len(kept), kept, dish


if __name__ == "__main__":
    import sys
    model = FastSAM("FastSAM-s.pt")
    for p in sys.argv[1:]:
        count, kept, dish = count_colonies_fastsam(model, p)
        print(p, "-> count:", count, " dish:", dish)
