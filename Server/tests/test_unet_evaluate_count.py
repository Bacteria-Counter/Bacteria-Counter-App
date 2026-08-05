from __future__ import annotations

import numpy as np

from scripts.unet.evaluate_unet_count_agar import render_segmentation_overlay, resize_mask_to_image


def test_resize_mask_to_image_matches_source_resolution() -> None:
    image = np.zeros((400, 320, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:128, 64:128] = 255

    resized = resize_mask_to_image(mask, image)

    assert resized.shape == image.shape[:2]
    assert resized.dtype == np.uint8
    assert int(resized.max()) == 255


def test_render_segmentation_overlay_accepts_resized_mask() -> None:
    image = np.full((300, 280, 3), 120, dtype=np.uint8)
    mask = np.zeros((300, 280), dtype=np.uint8)
    mask[40:120, 90:170] = 255

    overlay = render_segmentation_overlay(image, mask)

    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert int(overlay[:, :, 1].max()) > int(image[:, :, 1].max())
