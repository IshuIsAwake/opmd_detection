"""
settings.py — Paths + knobs for the classifier experiments.

Mirrors Experimenting/common/settings.py in spirit: env-overridable training
knobs, deterministic seed, plain defaults (NOT tuned). Anything generated
lives under classifier_experiments/_datasets/ or /results/ — both gitignored
and safe to delete and rebuild.
"""

from __future__ import annotations

import os
from pathlib import Path

# classifier_experiments/common/settings.py → parents[1] = classifier_experiments/
CLF_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = CLF_ROOT.parent                       # Experimenting/
PROJECT_ROOT = EXP_ROOT.parent                   # Oral_cancer/

# Reach the detector experiment's shared kfold split file + per-fold weights.
DET_SPLITS_PATH = EXP_ROOT / "_datasets" / "kfold5_splits.json"
DET_RESULTS_ROOT = EXP_ROOT / "results" / "kfold5_geom_no_color_binary"

# Disease class mapping — duplicated from Experimenting/common/settings.py
# rather than imported, because the parent tree has its own ``common`` package
# which would shadow ours under sys.path. These constants change rarely; if
# you edit them upstream, mirror here.
ORIG_ID_TO_CLASS = {
    0: "Leukoplakia",
    1: "Erythroplakia",
    2: "OSMF",
    3: "Lichen_Planus",
    4: "NH_Ulcers",
}
ORIG_CLASS_NAMES = [ORIG_ID_TO_CLASS[i] for i in range(len(ORIG_ID_TO_CLASS))]
SEED = 42
IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ── Classifier roots (writes only) ───────────────────────────────────────────
DATASETS_ROOT = CLF_ROOT / "_datasets"
RESULTS_ROOT = CLF_ROOT / "results"

# ── Backbone ─────────────────────────────────────────────────────────────────
# DINOv2-S patch size is 14 → input side must be a multiple of 14.
# HANDOFF chose 224 (= 14×16). CLS embed dim is 384.
DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_NAME = "dinov2_vits14"
DINOV2_EMBED = 384
INPUT_SIZE = 224

# ── Training knobs (env-overridable; defaults are deliberate, not tuned) ─────
EPOCHS = int(os.environ.get("CLF_EPOCHS", "50"))
BATCH = int(os.environ.get("CLF_BATCH", "32"))
LR = float(os.environ.get("CLF_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("CLF_WD", "1e-4"))
EARLY_STOP_PATIENCE = int(os.environ.get("CLF_PATIENCE", "10"))
DROPOUT = float(os.environ.get("CLF_DROPOUT", "0.1"))
HEAD_HIDDEN = int(os.environ.get("CLF_HIDDEN", "256"))
DEVICE = os.environ.get("CLF_DEVICE", "cuda")
NUM_WORKERS = int(os.environ.get("CLF_WORKERS", "2"))

# ── Pad fractions covered in Round 1 ─────────────────────────────────────────
GT_PADS = (0.0, 0.2, 0.4)

# Adopted detector operating point (CLAUDE.md).
DETECTOR_CONF = 0.10

# CLAUDE.md landmine: fold 0 is globally less aggressive. Phase 1 runs all
# folds (no detector in the loop); Phase 2 excludes fold 0.
ALL_FOLDS = (0, 1, 2, 3, 4)
PHASE2_FOLDS = (1, 2, 3, 4)
