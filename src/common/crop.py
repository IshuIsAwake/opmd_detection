"""
crop.py — THE shared crop logic.

Highest-risk spot in the project. Mistake #2 (eval looked fine, production was
~0%) happened because training crops and serving crops were produced by
different code. There is exactly ONE crop function and it lives here, imported
by:
  - src/classifier/build_data.py   (Step 3 — builds the classifier dataset)
  - src/pipeline.py                (Step 5 — serves predictions)

Padding is PROPORTIONAL: a fixed pixel pad is meaningless across images that
range from tiny to multi-thousand-px. Each side is padded by ``pad_frac`` of
the box's own width/height, with a small pixel floor.
"""

from __future__ import annotations

import numpy as np

from config import CROP_PAD_FRAC, CROP_PAD_MIN_PX


def crop_with_padding(
    image: np.ndarray,
    box_xyxy: tuple[float, float, float, float],
    pad_frac: float = CROP_PAD_FRAC,
    pad_min_px: int = CROP_PAD_MIN_PX,
) -> np.ndarray:
    """
    Crop ``image`` to ``box_xyxy`` expanded by proportional padding, clamped to
    image bounds.

    Args:
        image:      HxWxC array (channel order irrelevant — spatial slice only).
        box_xyxy:   (x1, y1, x2, y2) in pixels.
        pad_frac:   Padding per side as a fraction of box width (x) / height
                    (y). Project default CROP_PAD_FRAC so train and serve never
                    diverge. Pass 0.0 to crop exactly the box.
        pad_min_px: Minimum padding per side in pixels (context floor for tiny
                    boxes).

    Returns:
        Cropped sub-image (a copy). Never empty: collapses to ≥1x1.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(pad_min_px, int(round(pad_frac * bw)))
    pad_y = max(pad_min_px, int(round(pad_frac * bh)))

    x1 = int(round(x1)) - pad_x
    y1 = int(round(y1)) - pad_y
    x2 = int(round(x2)) + pad_x
    y2 = int(round(y2)) + pad_y

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return image[y1:y2, x1:x2].copy()


def center_box(image: np.ndarray, frac: float) -> tuple[int, int, int, int]:
    """
    The central region covering ``frac`` of width and height — used by the
    no-detection fallback. Returned as a box so it flows through the SAME
    crop_with_padding() path as real detections (pad_frac=0).
    """
    h, w = image.shape[:2]
    cw, ch = w * frac, h * frac
    cx, cy = w / 2.0, h / 2.0
    return (
        int(cx - cw / 2),
        int(cy - ch / 2),
        int(cx + cw / 2),
        int(cy + ch / 2),
    )
