"""
config.py — Single source of truth for paths and hyper-parameters.

No logic beyond reading env overrides. Run-scoped paths (weights, reports)
live in src/common/run_dir.py, not here — this file only holds the static
roots and knobs.

Detector model/imgsz/batch are env-overridable so the three planned detector
variants can be run without editing this file:

    DET_MODEL=yolov8n.pt DET_IMGSZ=1280 DET_BATCH=4 python scripts/02_train_detector.py
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Roots ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "data"
POOL_IMAGES = DATA_ROOT / "pool" / "images"
POOL_LABELS = DATA_ROOT / "pool" / "labels"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TEST_LABELS = DATA_ROOT / "test" / "labels"

# Negatives: list_images() is NON-recursive, so BOTH dirs must be enumerated
# explicitly — data/Normal/ (120) AND data/Normal/NON CANCER/ (450) ≈ 570.
NORMAL_DIR = DATA_ROOT / "Normal"
NORMAL_NONCANCER_DIR = DATA_ROOT / "Normal" / "NON CANCER"
NEGATIVE_DIRS = (NORMAL_DIR, NORMAL_NONCANCER_DIR)

# Read-only Roboflow exports (use the POLYGON yolov8 trees, never -obb).
ADDITIONAL_DIR = DATA_ROOT / "additional"

# Everything generated lives under here (safe to delete and rebuild).
ARTIFACTS = PROJECT_ROOT / "artifacts"

# Legacy single-dataset roots — kept so any out-of-pipeline importer still
# resolves. The arm-aware pipeline does NOT write here; use the helpers below.
DETECTOR_DATASET = ARTIFACTS / "detector_dataset"
SPLITS_JSON = ARTIFACTS / "splits.json"
DATA_YAML = DETECTOR_DATASET / "data.yaml"

# ── New-data root: every generated artifact of THIS experiment ────────────────
# Originals (pool/, test/, Normal/, additional/) are read-only. Everything the
# detector-data experiment produces lives under here.
NEW_DATA_ROOT = DATA_ROOT / "new_data"

ARMS = ("original_only", "plus_roboflow")

# web_holdout/ is a secondary, EVAL-ONLY higher-N detector signal (Roboflow
# test splits + a seeded slice of Roboflow valid). Never trained/tuned on.
WEB_HOLDOUT_DIR = NEW_DATA_ROOT / "web_holdout"
WEB_HOLDOUT_VALID_FRAC = float(os.environ.get("WEB_HOLDOUT_VALID_FRAC", "0.30"))

# Pool train/val split is computed once and reused byte-identically by BOTH
# arms (and Step 3) — the only allowed cross-arm difference is the extra
# Roboflow images appended to the treatment train split.
POOL_SPLIT_JSON = NEW_DATA_ROOT / "pool_split.json"

DISEASE_SIDECAR = NEW_DATA_ROOT / "disease_sidecar.json"
DEDUP_REPORT = NEW_DATA_ROOT / "dedup_report.json"

# pHash dedup: exclude any Roboflow image within this Hamming distance of a
# locked test/ image; pool/ near-dupes are reported only (not excluded).
PHASH_HAMMING_THRESH = int(os.environ.get("PHASH_HAMMING_THRESH", "5"))


def det_dataset_dir(arm: str) -> Path:
    """Per-arm YOLO detector dataset root under new_data/."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return NEW_DATA_ROOT / f"det_{arm}"


def data_yaml_for(arm: str) -> Path:
    return det_dataset_dir(arm) / "data.yaml"


def splits_json_for(arm: str) -> Path:
    """Per-arm pool train/val stem lists (pool-only — Roboflow never here)."""
    return det_dataset_dir(arm) / "splits.json"

# Per-run outputs go in artifacts/runs/<ts>_<tag>/ — see src/common/run_dir.py.
RUNS_ROOT = ARTIFACTS / "runs"
LATEST_LINK = ARTIFACTS / "latest"          # symlink → the run the demo serves
CURRENT_RUN_FILE = ARTIFACTS / "CURRENT_RUN"  # text pointer for the step chain

# ── Class mapping (label class_id is reliable; filename is not) ────────────────

YOLO_ID_TO_CLASS = {
    0: "Leukoplakia",
    1: "Erythroplakia",
    2: "OSMF",
    3: "Lichen_Planus",
    4: "NH_Ulcers",
}
CLASS_NAMES = [YOLO_ID_TO_CLASS[i] for i in range(len(YOLO_ID_TO_CLASS))]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ── Split ─────────────────────────────────────────────────────────────────────

VAL_FRACTION = 0.15
SPLIT_SEED = 42

# ── Shared crop (used IDENTICALLY in Step 3 and pipeline.py) ───────────────────
# Proportional padding: fixed pixels are wrong across wildly different image
# sizes. Pad each side by CROP_PAD_FRAC of the box's own width/height, with a
# small pixel floor so tiny boxes still get some context.
CROP_PAD_FRAC = float(os.environ.get("CROP_PAD_FRAC", "0.20"))
CROP_PAD_MIN_PX = int(os.environ.get("CROP_PAD_MIN_PX", "8"))

# ── Detector→GT matching (Step 3 + recall metric) ─────────────────────────────
# Annotator boxes are huge/loose. A correct tight prediction sits *inside* the
# GT, so IoU is tiny but containment (intersection ÷ prediction area) is ~1.
# Match on containment; IoU is still computed and reported as a secondary view.
MATCH_IOP_THRESH = float(os.environ.get("MATCH_IOP_THRESH", "0.70"))
IOU_SECONDARY = 0.50          # reported only, never gates anything

# ── No-detection behaviour ────────────────────────────────────────────────────
# "healthy"      → spec-compliant: "looks fine, no visit" (default)
# "center_crop"  → recall-max: classify the central CENTER_CROP_FRAC of the
#                  image instead (flagged low-confidence in the UI)
NO_DETECTION_FALLBACK = os.environ.get("NO_DETECTION_FALLBACK", "healthy")
CENTER_CROP_FRAC = float(os.environ.get("CENTER_CROP_FRAC", "0.40"))

# ── Detector (YOLOv8) — env-overridable for the variant sweep ──────────────────

DETECTOR_MODEL = os.environ.get("DET_MODEL", "yolov8n.pt")
DET_IMGSZ = int(os.environ.get("DET_IMGSZ", "640"))
DET_BATCH = int(os.environ.get("DET_BATCH", "8"))
DET_EPOCHS = 100
DET_PATIENCE = 30
# Ultralytics' AMP self-check downloads a helper model that 404s / corrupts in
# this environment. YOLOv8 is small, so skip it.
DET_AMP = False
DET_CONF = 0.20               # fallback only; the real value is swept & saved

# ── Classifier (EfficientNet-B2) ──────────────────────────────────────────────

CLF_MODEL = "efficientnet_b2"
CLF_IMG_SIZE = 260
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLF_BATCH = 16
CLF_FREEZE_EPOCHS = 5
CLF_FREEZE_LR = 1e-3
CLF_FINETUNE_EPOCHS = 25
CLF_FINETUNE_LR = 2e-4
CLF_WEIGHT_DECAY = 1e-4
CLF_NUM_WORKERS = 4

# ── Smoke-test overrides ──────────────────────────────────────────────────────

SMOKE = os.environ.get("ORAL_SMOKE", "0") == "1"
if SMOKE:
    DET_EPOCHS = 2
    DET_PATIENCE = 0
    CLF_FREEZE_EPOCHS = 1
    CLF_FINETUNE_EPOCHS = 1


def detector_tag() -> str:
    """Short, filesystem-safe tag identifying the detector config of a run."""
    stem = Path(DETECTOR_MODEL).stem            # e.g. yolov8n
    return f"{stem}_{DET_IMGSZ}"
