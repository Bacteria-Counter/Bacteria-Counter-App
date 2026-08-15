"""YOLO and CSRNet inference through Core ML, in Python.

Step 3's equivalent of coreml_fastsam.py: write out in Python exactly what the
Swift port must do, so the two can be compared image by image against the same
pixels. The Swift side is then measured against something that has itself been
checked, rather than against an assumption about what ultralytics does.

Both paths here are deliberately the Core ML path, not the PyTorch one. Step 2
already established that Core-ML-vs-PyTorch is a separate question with its own
answer (0.05 colonies on FastSAM), and mixing the two comparisons is how this
project has repeatedly produced confident wrong diagnoses -- one variable at a
time.
"""
import numpy as np
import torch
from PIL import Image
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

YOLO_SIZES = [1280, 1920, 2560, 3200]
YOLO_CONF = 0.4
YOLO_IOU = 0.7
YOLO_MAX_DET = 1000

CSRNET_WORKING_SIZE = 768
CSRNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
CSRNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def nearest_size(target, sizes=YOLO_SIZES):
    return min(sizes, key=lambda s: abs(s - target))


def run_yolo_coreml(mlmodel, img_bgr, size):
    """Return boxes in ORIGINAL image coordinates, as run_yolo() would."""
    padded = LetterBox((size, size), auto=False)(image=img_bgr)
    key_in = mlmodel.get_spec().description.input[0].name
    out = mlmodel.predict({key_in: Image.fromarray(padded[:, :, ::-1])})

    arrays = [v for v in out.values() if hasattr(v, "shape")]
    # (1, 5, N): 4 box + 1 class score for the single "colony" class.
    det = next(a for a in arrays if a.ndim == 3 and a.shape[1] == 5)
    pred = non_max_suppression(torch.from_numpy(det), YOLO_CONF, YOLO_IOU, nc=1,
                               max_det=YOLO_MAX_DET)[0]
    if pred.shape[0] == 0:
        return np.zeros((0, 5), np.float32)
    boxes = ops.scale_boxes((size, size), pred[:, :4].clone(), img_bgr.shape[:2])
    return np.concatenate([boxes.numpy(), pred[:, 4:5].numpy()], axis=1)


def run_csrnet_coreml(mlmodel, img_bgr):
    """Return the raw density map; the count is its sum."""
    import cv2
    small = cv2.resize(img_bgr, (CSRNET_WORKING_SIZE, CSRNET_WORKING_SIZE),
                       interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - CSRNET_MEAN) / CSRNET_STD).transpose(2, 0, 1)[None].astype(np.float32)
    key_in = mlmodel.get_spec().description.input[0].name
    out = mlmodel.predict({key_in: x})
    arr = next(v for v in out.values() if hasattr(v, "ndim") and v.ndim == 4)
    return arr[0, 0]
