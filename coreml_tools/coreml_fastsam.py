"""FastSAM inference through Core ML, producing the same masks as the PyTorch path.

This is step 2a of the migration: the post-processing that ultralytics performs
today gets written out explicitly, in Python, where it can be compared against
ultralytics image by image. Only once it matches does it become the
specification for the Swift port -- translating unproven logic would leave two
unknowns stacked on top of each other.

One difference from the PyTorch path is unavoidable. Ultralytics letterboxes a
PyTorch model's input to a stride multiple, feeding e.g. 1952x2560 for a
2560x1922 photo, whereas an exported Core ML model accepts only the square size
it was built at. LetterBox scales the content by the same factor either way, so
the colonies land at identical size and only the amount of grey padding
differs -- padding that produces no detections. That reasoning is checked
against real counts rather than trusted.
"""
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


def letterbox_square(img_bgr, size):
    """Scale to fit `size` and pad to an exact square, as the export expects."""
    lb = LetterBox((size, size), auto=False)
    return lb(image=img_bgr)


def run_coreml(mlmodel, img_bgr, size, conf=0.2, iou=0.7, max_det=3000):
    """Return masks at the original image resolution, as ultralytics would.

    Mirrors the ultralytics segmentation path step for step: letterbox, NMS over
    the raw predictions, mask assembly from the 32 prototype planes weighted by
    each detection's coefficients, then scaling back to the source frame.
    """
    padded = letterbox_square(img_bgr, size)
    rgb = padded[:, :, ::-1]
    from PIL import Image
    key_in = mlmodel.get_spec().description.input[0].name
    out = mlmodel.predict({key_in: Image.fromarray(rgb)})

    arrays = [v for v in out.values() if hasattr(v, "shape")]
    # (1, 37, N) predictions: 4 box + 1 score + 32 mask coefficients.
    det = next(a for a in arrays if a.ndim == 3 and a.shape[1] == 37)
    # (1, 32, H/4, W/4) prototype masks.
    proto = next(a for a in arrays if a.ndim == 4 and a.shape[1] == 32)

    pred = non_max_suppression(torch.from_numpy(det), conf, iou, nc=1,
                               max_det=max_det)[0]
    if pred.shape[0] == 0:
        return []

    proto_t = torch.from_numpy(proto)[0]
    masks = ops.process_mask(proto_t, pred[:, 6:], pred[:, :4],
                             padded.shape[:2], upsample=True)
    # Undo the letterbox: crop the padding away, then resize to the source.
    masks = ops.scale_masks(masks[None], img_bgr.shape[:2])[0]
    return (masks > 0.5).cpu().numpy().astype(np.uint8)
