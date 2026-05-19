"""
convert_roboflow.py — Roboflow → binary-lesion YOLO conversion. PURE.

No cv2, no torch, no ultralytics, no config side effects beyond reading paths
— so it is trivially unit-testable. Responsibilities:

  * parse YOLO label lines that are EITHER a real box (4 coords) OR a polygon
    segmentation (≥6 coords). `read_yolo_label()` is deliberately NOT reused:
    it hard-drops any line with len(parts)!=5, silently eating every polygon.
  * polygon → tight axis-aligned bbox (min/max of the points).
  * collapse every disease class to single class 0 ('lesion').
  * namespace every output stem (`opmdseg__`, `osmf__`, `leukov2__`) so
    Roboflow stems can never silently overwrite pool/ or each other.
  * emit disease-provenance sidecar entries (recorded now, used by a FUTURE
    deliberate classifier phase — NOT wired into the classifier in this run).

`non-osmf` (OSMF DETECTION class 0) is NOT a lesion class: per the agreed
design those boxes are dropped and an image left with zero lesion boxes
becomes a background NEGATIVE (empty label) — folded into treatment train.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import config

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
_DEGENERATE = 1e-4


@dataclass(frozen=True)
class RoboflowSource:
    name: str                                  # sidecar source_dataset id
    dir_name: str                              # folder under data/additional
    prefix: str                                # filename namespace
    # class_id -> (original_class_name, mapped_disease | None)
    class_map: dict[int, tuple[str, str | None]]
    lesion_classes: frozenset[int]             # ids that become a lesion box

    @property
    def root(self) -> Path:
        return config.ADDITIONAL_DIR / self.dir_name


# Class ids/names verified from each export's data.yaml.
DATASETS: tuple[RoboflowSource, ...] = (
    RoboflowSource(
        name="Leukoplakia.v2",
        dir_name="Leukoplakia.v2i.yolov8",
        prefix="leukov2",
        class_map={0: ("Leukoplakia", "Leukoplakia")},
        lesion_classes=frozenset({0}),
    ),
    RoboflowSource(
        name="OPMD-SEG.v1",
        dir_name="OPMD-SEG.v1i.yolov8",
        prefix="opmdseg",
        class_map={
            0: ("Leukoplakia", "Leukoplakia"),
            1: ("Oral Lichen Planus", "Lichen_Planus"),
            2: ("erythroplakia", "Erythroplakia"),
        },
        lesion_classes=frozenset({0, 1, 2}),
    ),
    RoboflowSource(
        name="OSMF-DETECTION.v1",
        dir_name="OSMF DETECTION.v1i.yolov8",
        prefix="osmf",
        # class 0 = non-osmf → NOT a lesion (image becomes a negative)
        class_map={0: ("non-osmf", None), 1: ("osmf", "OSMF")},
        lesion_classes=frozenset({1}),
    ),
)

SPLITS = ("train", "valid", "test")


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_label_text(text: str) -> list[tuple[int, list[float]]]:
    """
    Parse a Roboflow YOLO label into [(class_id, [coords...]), ...].

    Accepts BOTH a box line (`cls cx cy w h` → 4 coords) and a polygon line
    (`cls x1 y1 x2 y2 …` → ≥6 coords). Malformed/short lines are skipped.
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


def coords_to_bbox(coords: list[float]) -> tuple[float, float, float, float] | None:
    """
    Normalized coords → normalized (cx, cy, w, h), clamped to [0, 1].

    4 coords  → already (cx, cy, w, h); just clamp.
    ≥6 coords → polygon: xs=coords[0::2], ys=coords[1::2]; bbox = min/max.
    Degenerate boxes (w ≤ 1e-4 or h ≤ 1e-4) → None (caller drops the line).
    """
    if len(coords) == 4:
        cx, cy, w, h = coords
    else:
        xs = [min(1.0, max(0.0, v)) for v in coords[0::2]]
        ys = [min(1.0, max(0.0, v)) for v in coords[1::2]]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1

    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    w = min(1.0, max(0.0, w))
    h = min(1.0, max(0.0, h))
    if w <= _DEGENERATE or h <= _DEGENERATE:
        return None
    return cx, cy, w, h


# ── conversion ────────────────────────────────────────────────────────────────

@dataclass
class ConvertedImage:
    source: str                 # RoboflowSource.name
    src_image: Path             # original Roboflow image (read-only)
    dst_stem: str               # namespaced stem (no extension)
    dst_name: str               # namespaced filename (with extension)
    label_text: str             # binary single-class label ("" = negative)
    is_negative: bool           # zero lesion boxes after conversion
    split: str                  # train | valid | test (Roboflow split)
    sidecar: list[dict]         # provenance entries (positive boxes only)


def _image_for(label_path: Path, images_dir: Path) -> Path | None:
    for ext in _IMG_EXTS:
        cand = images_dir / f"{label_path.stem}{ext}"
        if cand.exists():
            return cand
    return None


def convert_split(source: RoboflowSource, split: str) -> list[ConvertedImage]:
    """Convert one Roboflow split of one source. Empty list if split absent."""
    labels_dir = source.root / split / "labels"
    images_dir = source.root / split / "images"
    if not labels_dir.is_dir() or not images_dir.is_dir():
        return []

    out: list[ConvertedImage] = []
    for lbl in sorted(labels_dir.glob("*.txt"), key=lambda p: p.stem):
        img = _image_for(lbl, images_dir)
        if img is None:
            continue

        lines: list[str] = []
        sidecar: list[dict] = []
        for cid, coords in parse_label_text(lbl.read_text()):
            if cid not in source.lesion_classes:
                continue
            box = coords_to_bbox(coords)
            if box is None:
                continue
            cx, cy, w, h = box
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            orig, mapped = source.class_map.get(cid, (str(cid), None))
            sidecar.append(
                {
                    "box_index": len(lines) - 1,
                    "source_dataset": source.name,
                    "original_class": orig,
                    "mapped_class": mapped,
                }
            )

        dst_stem = f"{source.prefix}__{img.stem}"
        dst_name = f"{dst_stem}{img.suffix}"
        for entry in sidecar:
            entry["generated_image"] = dst_name
        out.append(
            ConvertedImage(
                source=source.name,
                src_image=img,
                dst_stem=dst_stem,
                dst_name=dst_name,
                label_text="\n".join(lines),
                is_negative=not lines,
                split=split,
                sidecar=sidecar,
            )
        )
    return out


def valid_holdout_stems(
    records: list[ConvertedImage], frac: float, seed: int
) -> set[str]:
    """
    Deterministically pick a `frac` slice of a source's *valid* records to
    divert into web_holdout (kept out of training). Seeded + sorted → the
    same images every run, identical across both arms.
    """
    stems = sorted(r.dst_stem for r in records)
    if not stems or frac <= 0.0:
        return set()
    k = max(1, round(len(stems) * frac))
    rng = random.Random(seed)
    return set(rng.sample(stems, min(k, len(stems))))
