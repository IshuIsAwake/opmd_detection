"""
crops.py — Single source of truth for box → crop geometry.

Two callers feed off this:
  * materialise.py — pre-generates GT crops at pad_frac ∈ {0.0, 0.2, 0.4} into a
    flat pool dir, then per-fold trees symlink into that pool.
  * phase2_pipeline.py — emits YOLO-detector crops at serve time, same geometry.

The pad_frac semantics MUST match predict_with_old_classifier.pad_and_crop —
that script is the §12 baseline our Phase 2 numbers compete with. Reproduced
verbatim here so the two are byte-equivalent on the same bbox + image.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def pad_and_crop(bgr: np.ndarray, xyxy: tuple[float, float, float, float],
                 pad_frac: float) -> np.ndarray | None:
    """Expand bbox by pad_frac * (w, h) on each side, clip to image, return BGR
    crop. Returns None if the padded box clips to a zero-area rectangle."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    px = pad_frac * w
    py = pad_frac * h
    nx1 = int(max(0, round(x1 - px)))
    ny1 = int(max(0, round(y1 - py)))
    nx2 = int(min(W, round(x2 + px)))
    ny2 = int(min(H, round(y2 + py)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return bgr[ny1:ny2, nx1:nx2].copy()


def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO axis-aligned label parser. Returns [(class_id, cx, cy, w, h), ...]
    in NORMALISED coords. Mirrors datasets._read_box_label so the polygon-eating
    landmine flagged in CLAUDE.md cannot bite us — these labels are the same
    axis-aligned originals YOLO trains on, never the polygon Roboflow exports."""
    if not label_path.exists():
        return []
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        try:
            rows.append((int(float(p[0])), *(float(v) for v in p[1:])))
        except ValueError:
            continue
    return rows


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Normalised YOLO (cx, cy, w, h) → pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def crop_from_gt_box(bgr: np.ndarray, box_norm: tuple[float, float, float, float],
                     pad_frac: float) -> np.ndarray | None:
    """Convenience wrapper: normalised GT box → padded BGR crop."""
    H, W = bgr.shape[:2]
    xyxy = yolo_to_xyxy(*box_norm, img_w=W, img_h=H)
    return pad_and_crop(bgr, xyxy, pad_frac)
