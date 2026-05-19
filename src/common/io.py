"""
io.py — Image discovery and YOLO-label parsing helpers.

Small, shared, side-effect-light. Anything that touches the raw data layout
goes through here so the rest of the codebase never globs by hand.
"""

from __future__ import annotations

from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images(directory: Path) -> list[Path]:
    """All image files in ``directory``, sorted by stem for determinism."""
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(directory.glob(f"*{ext}"))
    # De-dupe (case-insensitive globs can double up on some filesystems).
    uniq = {p.resolve(): p for p in files}
    return sorted(uniq.values(), key=lambda p: p.stem)


def read_yolo_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a YOLO ``.txt`` into [(class_id, cx, cy, w, h), ...].

    Missing or empty file → empty list (a valid background image). Malformed
    lines are skipped rather than raising — raw labels are known to be messy.
    """
    if not label_path.exists():
        return []

    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            continue
        rows.append((cid, cx, cy, w, h))
    return rows


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    """Matching label path for an image (same stem, ``.txt`` in labels_dir)."""
    return labels_dir / f"{image_path.stem}.txt"
