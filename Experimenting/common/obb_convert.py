"""
obb_convert.py — Roboflow label → single-class YOLO-OBB label. PURE-ish.

Every expert is a yolov8-obb model so the four are directly comparable. The
Roboflow sources do not all ship the same label kind, so each is reformatted —
never hand-drawn:

  * "obb"  (OPMD-SEG-obb): already `cls x1 y1 x2 y2 x3 y3 x4 y4` → passthrough.
  * "box"  (Leukoplakia.v2): `cls cx cy w h` → 4 axis-aligned corners
            (zero-rotation OBB). Reformatting an existing box, not redrawing.
  * "poly" (OSMF DETECTION): `cls x1 y1 …` polygon → cv2.minAreaRect → the
            *minimum-area oriented rectangle* of the polygon. This is the
            tight oriented box the experiment is meant to test, derived from
            the annotation, not invented.

minAreaRect runs in NORMALISED coordinate space (we do not read every image to
get pixel dims at build time). The 4 returned corners still contain every
polygon vertex (affine maps preserve containment), which is what matters for a
baseline; the exact aspect of the rectangle is mildly distorted vs pixel space.
This caveat is documented in the README.
"""

from __future__ import annotations

import cv2
import numpy as np

_DEGENERATE = 1e-4


def parse_lines(text: str) -> list[tuple[int, list[float]]]:
    """Parse a Roboflow label into [(class_id, [coords...]), ...].

    Accepts box (4 coords), obb (8 coords) and polygon (>=6, even) lines.
    Short/odd/garbage lines are skipped — raw exports are known to be messy.
    """
    rows: list[tuple[int, list[float]]] = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(coords) < 4 or len(coords) % 2 != 0:
            continue
        rows.append((cid, coords))
    return rows


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def box_to_obb(cx: float, cy: float, w: float, h: float) -> list[float] | None:
    """Axis-aligned YOLO box → 4 corners (TL, TR, BR, BL), zero rotation."""
    if w <= _DEGENERATE or h <= _DEGENERATE:
        return None
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    corners = [x1, y1, x2, y1, x2, y2, x1, y2]
    return [_clamp01(v) for v in corners]


def obb_passthrough(coords: list[float]) -> list[float] | None:
    """An 8-coord OBB line → clamped corners (degenerate area → None)."""
    if len(coords) != 8:
        return None
    pts = np.array(coords, dtype=np.float64).reshape(4, 2)
    if cv2.contourArea(pts.astype(np.float32)) <= _DEGENERATE:
        return None
    return [_clamp01(v) for v in coords]


def poly_to_obb(coords: list[float]) -> list[float] | None:
    """Polygon (>=6 even coords, normalised) → min-area oriented rectangle."""
    pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or cv2.contourArea(pts) <= _DEGENERATE:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(pts))      # (4, 2)
    return [_clamp01(float(v)) for v in box.reshape(-1)]


def to_obb(coords: list[float], kind: str) -> list[float] | None:
    """Dispatch one parsed annotation's coords to 8 normalised OBB corners."""
    if kind == "box":
        if len(coords) != 4:
            return None
        return box_to_obb(*coords)
    if kind == "obb":
        return obb_passthrough(coords)
    if kind == "poly":
        return poly_to_obb(coords)
    raise ValueError(f"unknown label kind {kind!r}")


def format_obb_line(corners: list[float], cls: int = 0) -> str:
    """8 corners → a single-class YOLO-OBB label line."""
    return f"{cls} " + " ".join(f"{v:.6f}" for v in corners)


def to_axis_box(coords: list[float], kind: str) -> tuple[float, ...] | None:
    """Any label kind → axis-aligned YOLO (cx, cy, w, h), normalised/clamped.

    For an axis-aligned detect model (exp7). box → passthrough; obb/poly →
    min/max enclosing rectangle. Degenerate → None.
    """
    if kind == "box":
        if len(coords) != 4:
            return None
        cx, cy, w, h = coords
    else:
        xs = [_clamp01(v) for v in coords[0::2]]
        ys = [_clamp01(v) for v in coords[1::2]]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    cx, cy = _clamp01(cx), _clamp01(cy)
    w, h = _clamp01(w), _clamp01(h)
    if w <= _DEGENERATE or h <= _DEGENERATE:
        return None
    return cx, cy, w, h
