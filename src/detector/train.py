"""
Step 2 — Train the binary lesion detector (YOLOv8).

Starts a NEW run (artifacts/runs/<ts>_<tag>/). Model / imgsz / batch come from
config (env-overridable) so the planned variants run without code edits:

    DET_MODEL=yolov8n.pt DET_IMGSZ=1280 DET_BATCH=4 python scripts/02_train_detector.py

Light geometric aug only. NO HSV/colour distortion — colour is diagnostic.
Mosaic off (keep augmentation close to serve-time inputs). Ultralytics MLflow
logging is disabled so nothing lands at the project root with hash names.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics import settings as ul_settings

import config
from src.common import run_dir


def train(arm: str, name: str | None = None) -> tuple[Path, Path]:
    """Train ONE arm into a fresh run. Returns (run_dir, detector_weights)."""
    if arm not in config.ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {config.ARMS}")

    ul_settings.update({"mlflow": False})  # no runs/mlflow/<hash> at root

    run = run_dir.new_run(config.detector_tag(), name)
    det_out = run / "detector"

    data_yaml = config.data_yaml_for(arm)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} missing — run scripts/01 --arm {arm} first."
        )

    model = YOLO(config.DETECTOR_MODEL)
    model.train(
        data=str(data_yaml),
        imgsz=config.DET_IMGSZ,
        epochs=config.DET_EPOCHS,
        patience=config.DET_PATIENCE,
        batch=config.DET_BATCH,
        amp=config.DET_AMP,
        project=str(det_out),
        name="ultralytics",
        exist_ok=True,
        fliplr=0.5,
        flipud=0.0,
        scale=0.10,
        translate=0.10,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        mosaic=0.0,
        mixup=0.0,
        verbose=True,
    )

    best = det_out / "ultralytics" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected trained weights at {best}")
    shutil.copy2(best, run_dir.detector_weights(run))

    run_dir.update_manifest(
        run,
        "config",
        {
            "arm": arm,
            "detector_model": config.DETECTOR_MODEL,
            "det_imgsz": config.DET_IMGSZ,
            "det_batch": config.DET_BATCH,
            "det_epochs": config.DET_EPOCHS,
            "match_rule": f"IoP>={config.MATCH_IOP_THRESH}",
            "crop_pad_frac": config.CROP_PAD_FRAC,
            "no_detection_fallback": config.NO_DETECTION_FALLBACK,
            "smoke": config.SMOKE,
        },
    )
    return run, run_dir.detector_weights(run)
