"""
train_eval.py — One call per experiment: train (stock settings) then evaluate.

Deliberately dumb: take the dataset the builder produced, train the given
YOLO checkpoint with default-ish args (no metric is tuned), then run the
honest custom evaluation. Detect models also get a stock Ultralytics val()
for cross-reference; OBB models skip it (their test labels are axis boxes).

Everything lands under Experimenting/results/<name>/.
"""

from __future__ import annotations

import json
from pathlib import Path

from common import settings
from common.metrics import evaluate


def run(spec, model_weights: str) -> dict:
    from ultralytics import YOLO

    results_dir = settings.RESULTS_ROOT / spec.name
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "run.json").write_text(json.dumps({
        "experiment": spec.name,
        "task": spec.task,
        "model_weights": model_weights,
        "class_names": spec.class_names,
        "data_yaml": str(spec.data_yaml),
        "epochs": settings.EPOCHS,
        "imgsz": settings.IMGSZ,
        "batch": settings.BATCH,
        "amp": settings.AMP,
        "seed": settings.SEED,
        "device": settings.DEVICE,
    }, indent=2))

    # ── train ────────────────────────────────────────────────────────────────
    model = YOLO(model_weights)
    model.train(
        data=str(spec.data_yaml),
        epochs=settings.EPOCHS,
        imgsz=settings.IMGSZ,
        batch=settings.BATCH,
        device=settings.DEVICE,
        amp=settings.AMP,
        seed=settings.SEED,
        project=str(results_dir),
        name="train",
        exist_ok=True,
        verbose=True,
    )
    best = results_dir / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training produced no weights at {best}")

    trained = YOLO(str(best))

    # ── stock val() for detect models only (cross-reference mAP) ─────────────
    if spec.task == "detect":
        try:
            m = trained.val(
                data=str(spec.data_yaml), split="test",
                device=settings.DEVICE, imgsz=settings.IMGSZ,
                project=str(results_dir), name="val", exist_ok=True,
                verbose=False,
            )
            (results_dir / "val_stock.json").write_text(json.dumps({
                "map50": float(m.box.map50),
                "map50_95": float(m.box.map),
                "precision": float(m.box.mp),
                "recall": float(m.box.mr),
            }, indent=2))
        except Exception as e:        # noqa: BLE001 — cross-ref only, never fatal
            (results_dir / "val_stock.json").write_text(
                json.dumps({"error": str(e)}, indent=2))

    # ── honest custom evaluation ─────────────────────────────────────────────
    report = evaluate(trained, spec, results_dir)
    print(f"\n[{spec.name}] done → {results_dir}/metrics.txt")
    print((results_dir / "metrics.txt").read_text())
    return report
