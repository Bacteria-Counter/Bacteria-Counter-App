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


def marking_features(lab_img, mask_u8, background_lab):
    """Colour distance and darkness of one detection, relative to the agar.

    The 75th percentile, not the median: pen strokes and printed characters
    are thin, so a mask straddling one still holds a majority of ordinary agar
    pixels and its median stays innocent.
    """
    sel = mask_u8 > 0
    if not sel.any():
        return 0.0, 0.0
    bg_l, bg_a, bg_b = background_lab
    px = lab_img[sel].astype(np.float32)
    chroma = np.sqrt((px[:, 1] - bg_a) ** 2 + (px[:, 2] - bg_b) ** 2)
    return (float(np.percentile(chroma, 75)),
            float(np.percentile(bg_l - px[:, 0], 75)))


def find_markings(features, z=20.0, min_detections=12):
    """Pick out detections that are pen or label text rather than colonies.

    Judged against the OTHER detections on the same plate, not against fixed
    thresholds. Fixed thresholds were tried first and were badly wrong: tuned
    on a lab plate whose colonies sit at chroma ~1 against blue marker at 12.2,
    they then deleted 99% of the PCA benchmark's colonies, which are themselves
    chroma 24 on their own agar. A colony on one medium can be more chromatic
    than ink on another, so no global cut exists.

    What does hold across both is that markings are OUTLIERS among their own
    plate's detections -- ink is unlike the colonies beside it even when it is
    not unlike colonies elsewhere. Median absolute deviation is used rather
    than mean/stdev so that a handful of marking detections cannot drag the
    centre out to meet themselves.

    z is deliberately conservative. Swept on a lab plate with real marker and
    again with synthetic label text: z=6 threw away 29 of 104 real colonies,
    z=15 still threw away 3, z=20 costs 2. Since this filter has already been
    wrong once in a way that destroyed a whole benchmark, it is set to catch
    the obvious markings and leave the ambiguous ones alone -- an uncounted
    ink blob is a smaller error than a deleted colony. On the PCA benchmark it
    rejects nothing at any z tested, so the benchmark is untouched either way.

    Below min_detections the spread is not measurable and nothing is rejected:
    an almost-empty plate offers no population to be an outlier from, and
    wrongly dropping its one real colony matters far more than keeping a
    stray ink blob.
    """
    if len(features) < min_detections:
        return set()
    arr = np.asarray(features, dtype=np.float32)
    flagged = set()
    for col in (0, 1):
        v = arr[:, col]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad <= 1e-6:
            continue
        # 1.4826 scales MAD to a standard-deviation equivalent for normal data.
        cut = med + z * 1.4826 * mad
        flagged.update(int(i) for i in np.nonzero(v > cut)[0])
    return flagged


def split_by_concavity(mask_u8, reference_area, depth_ratio=0.22, min_solidity=0.93):
    """Count colonies in a mask by finding the concave notches where two round
    colonies overlap.

    Two overlapping discs meet at two points, and the outline dips inward at
    each -- the "neck" a microbiologist reads to call a clump two colonies
    rather than one. A single colony, however oblong, has no such notch. So the
    notches are counted, not the mask's area or its distance-transform peaks:
    area alone cannot tell a large colony from two small ones, and distance
    peaks fire on any elongated shape.

    This is the concave-chord method from the overlapping-convex-objects
    literature: deviation of the contour from its own convex hull. Each merge
    contributes a pair of opposing notches, so n colonies leave 2(n-1) of them
    along a chain.

    Replaces split_merged_mask(), whose distance-transform peaks were unusable
    at both ends of their sensitivity range -- at min_distance=10 it split real
    single colonies, at 20 it fired almost never.
    """
    # CHAIN_APPROX_NONE, not _SIMPLE: the compressed form drops the very
    # boundary points a notch is made of, so defects come back too shallow.
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 1
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area <= 0 or len(c) < 5:
        return 1

    hull_pts = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull_pts)
    # A convex blob is one colony no matter how big: bail out before counting
    # notches, so natural size variation is never mistaken for a merge.
    if hull_area <= 0 or area / hull_area >= min_solidity:
        return 1

    hull_idx = cv2.convexHull(c, returnPoints=False)
    if len(hull_idx) < 3:
        return 1
    # convexityDefects needs the hull indices monotonic, which convexHull does
    # not guarantee; try as-returned, then sorted, before giving up.
    defects = None
    for idx in (hull_idx, np.sort(hull_idx.flatten())[::-1].reshape(-1, 1)):
        try:
            defects = cv2.convexityDefects(c, idx)
            break
        except cv2.error:
            continue
    if defects is None:
        return 1

    # Scale the notch-depth threshold to the typical colony radius, so the same
    # setting works on a plate of small colonies and one of large ones.
    r_ref = max(1.0, (reference_area / np.pi) ** 0.5)
    deep = sum(1 for d in defects.reshape(-1, 4) if d[3] / 256.0 > depth_ratio * r_ref)
    if deep < 2:
        return 1

    # The notches establish THAT the blob is merged; area establishes HOW MANY.
    # Counting notches alone cannot do the second job: OpenCV reports two deep
    # defects for a chain of three colonies just as it does for two, so the
    # notch count saturates. Area alone cannot do the first job either -- it
    # reads one big colony as two. Each measure covers the other's blind spot.
    return max(2, round(area / max(1.0, reference_area)))


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
                            min_area_frac=0.00002, max_area_frac=0.02,
                            drop_square_like=True, split_colonies=False,
                            reject_markings=False, color_ref_bgr=None):
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

    # Reference colour for the ink filter: the agar's own median, taken from
    # well inside the dish so the rim and the bench never skew it.
    # Colour is judged on the UNPROCESSED frame when one is supplied: callers
    # pass a CLAHE copy for detection, and CLAHE rewrites local luminance, so
    # a darkness threshold calibrated on real pixels does not survive it.
    lab_img = background_lab = None
    if reject_markings and dish is not None:
        ref = color_ref_bgr if color_ref_bgr is not None else img_bgr
        if ref.shape[:2] != (H, W):
            ref = cv2.resize(ref, (W, H), interpolation=cv2.INTER_AREA)
        lab_img = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
        inner = np.zeros((H, W), np.uint8)
        cv2.circle(inner, (int(dish[0]), int(dish[1])), int(dish[2] * 0.75), 255, -1)
        background_lab = tuple(float(np.median(lab_img[:, :, i][inner > 0]))
                               for i in range(3))
    for m in masks:
        mask_resized = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        circularity, area, is_square_like = mask_circularity(mask_resized)
        area_frac = area / img_area
        if circularity < min_circularity:
            continue
        # The square-cell rejection was written for grid-patterned backgrounds.
        # On bright plates it also discards real colonies, so it is now optional
        # -- default True keeps the previous behaviour for every existing caller.
        if drop_square_like and is_square_like:
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

    # Markings are removed only after every mask is known: the test is
    # relative to the plate's own population of detections, so it cannot be
    # made one mask at a time.
    if lab_img is not None and kept:
        feats = [marking_features(lab_img, k["mask"], background_lab) for k in kept]
        flagged = find_markings(feats)
        if flagged:
            kept = [k for i, k in enumerate(kept) if i not in flagged]

    if split_colonies and kept:
        # The reference is this plate's own median colony, so the notch-depth
        # and area thresholds scale to whatever the colonies here happen to
        # measure -- a plate of pinpoint colonies and a plate of large ones
        # both get sensible splits without retuning.
        reference_area = float(np.median([k["area"] for k in kept]))
        for k in kept:
            k["sub_count"] = split_by_concavity(k["mask"], reference_area)

    return sum(k["sub_count"] for k in kept), kept, dish


if __name__ == "__main__":
    import sys
    model = FastSAM("FastSAM-s.pt")
    for p in sys.argv[1:]:
        count, kept, dish = count_colonies_fastsam(model, p)
        print(p, "-> count:", count, " dish:", dish)
