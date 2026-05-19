"""
settings.py — Static paths + training knobs for the Experimenting baseline.

This is a *separate* experiment tree from the main pipeline. It reads the
original data and the Roboflow exports read-only and writes everything it
generates under Experimenting/ only. Nothing here touches the main project's
config.py, artifacts/, or data/new_data/.

Training knobs are env-overridable so a run can be made shorter without
editing code, but the DEFAULTS are deliberately plain — this is a baseline,
not a tuned model. We do not optimise any metric.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────────────────────

# Experimenting/common/settings.py  →  parents[1] = Experimenting/
EXP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "data"

# Original annotated lesion data (READ-ONLY).
POOL_IMAGES = DATA_ROOT / "pool" / "images"
POOL_LABELS = DATA_ROOT / "pool" / "labels"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TEST_LABELS = DATA_ROOT / "test" / "labels"

# Healthy mouths — no labels, used only as negatives in the expert test set.
# list_images here is non-recursive, so BOTH dirs are enumerated explicitly.
NORMAL_DIRS = (DATA_ROOT / "Normal", DATA_ROOT / "Normal" / "NON CANCER")

# Roboflow exports (READ-ONLY).
ADDITIONAL_DIR = DATA_ROOT / "additional"

# Everything this experiment generates lives here (safe to delete & rebuild).
DATASETS_ROOT = EXP_ROOT / "_datasets"     # generated YOLO trees + data.yaml
RESULTS_ROOT = EXP_ROOT / "results"        # per-experiment metrics + weights

# ── Original class mapping (label class_id is reliable; filename is not) ───────

ORIG_ID_TO_CLASS = {
    0: "Leukoplakia",
    1: "Erythroplakia",
    2: "OSMF",
    3: "Lichen_Planus",
    4: "NH_Ulcers",
}
ORIG_CLASS_NAMES = [ORIG_ID_TO_CLASS[i] for i in range(len(ORIG_ID_TO_CLASS))]

# ── Split ─────────────────────────────────────────────────────────────────────

SEED = 42
VAL_FRACTION = 0.15        # train/val split fraction for every experiment

# ── Training knobs (env-overridable; plain defaults — NOT tuned) ──────────────

EPOCHS = int(os.environ.get("EXP_EPOCHS", "100"))
IMGSZ = int(os.environ.get("EXP_IMGSZ", "640"))
BATCH = int(os.environ.get("EXP_BATCH", "8"))     # safe for an RTX 3050 6 GB
# Ultralytics' AMP self-check downloads a helper model that 404s/corrupts in
# this environment; the project standard is to disable it. Not a tuning knob.
AMP = False
DEVICE = os.environ.get("EXP_DEVICE", "0")        # "0" = first GPU, "cpu" to force

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def list_images(directory: Path) -> list[Path]:
    """All images directly in ``directory`` (non-recursive), sorted by stem."""
    files: list[Path] = []
    for ext in IMG_EXTS:
        files.extend(directory.glob(f"*{ext}"))
    uniq = {p.resolve(): p for p in files}
    return sorted(uniq.values(), key=lambda p: p.name)
