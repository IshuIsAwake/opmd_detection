"""
geometry.py — Pure box math. No I/O, no model, no config.

Kept tiny and dependency-free so it can be unit-tested or reused anywhere.
"""

from __future__ import annotations


def yolo_to_pixel(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """YOLO normalised (cx, cy, w, h) → clamped pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iop(pred: tuple[float, float, float, float],
        gt: tuple[float, float, float, float]) -> float:
    """
    Intersection-over-Prediction: fraction of the *prediction* box that lies
    inside the GT box. Robust to loose/oversized annotations — a tight correct
    prediction fully inside a giant GT scores ~1.0 even when IoU is tiny.
    """
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gt

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    return inter / pred_area if pred_area > 0 else 0.0
