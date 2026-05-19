"""
Final evaluation — the ONLY number that counts.

Every locked test/ image runs through the full pipeline (detect → crop →
classify). Reports end-to-end accuracy, per-class recall, confusion matrix.
A no-detection with no fallback counts as a miss (its own column) — in the
product that means the patient was told they were fine.

Writes into the current run's eval/ dir and updates run.json.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

import config
from src.common import run_dir
from src.common.io import label_path_for, list_images, read_yolo_label
from src.detector.evaluate import web_holdout_detector_metrics
from src.pipeline import OralLesionPipeline

NO_DETECTION = config.NUM_CLASSES  # synthetic extra column


def _gt_class(image_path) -> int:
    rows = read_yolo_label(label_path_for(image_path, config.TEST_LABELS))
    counts = Counter(cid for cid, *_ in rows if cid in config.YOLO_ID_TO_CLASS)
    if not counts:
        return -1
    top = max(counts.values())
    return min(c for c, n in counts.items() if n == top)


def evaluate() -> dict:
    run = run_dir.current_run()
    pipe = OralLesionPipeline(run=run)
    images = list_images(config.TEST_IMAGES)

    y_true: list[int] = []
    y_pred: list[int] = []
    n_fallback = 0
    edir = run_dir.eval_dir(run)
    samples_dir = edir / "samples"

    for i, img in enumerate(images):
        gt = _gt_class(img)
        if gt < 0:
            continue
        res = pipe.analyze(img)
        if res.disease is not None:
            pred = config.CLASS_TO_IDX[res.disease]
            if res.used_fallback:
                n_fallback += 1
        else:
            pred = NO_DETECTION
        y_true.append(gt)
        y_pred.append(pred)

        if i < 12 and res.boxed_image is not None:  # a few qualitative samples
            import cv2

            tag = "OK" if pred == gt else "WRONG"
            cv2.imwrite(
                str(samples_dir / f"{img.stem}_{tag}.jpg"),
                cv2.cvtColor(res.boxed_image, cv2.COLOR_RGB2BGR),
            )

    n = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = correct / n if n else 0.0

    cm = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES + 1), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_recall = {}
    for c in range(config.NUM_CLASSES):
        total_c = int(cm[c].sum())
        per_class_recall[config.CLASS_NAMES[c]] = (
            round(cm[c, c] / total_c, 4) if total_c else None
        )

    # Secondary, higher-N detector-only signal (eval only — never tuned on).
    wh = web_holdout_detector_metrics(run_dir.detector_weights(run))

    report = {
        "run": run.name,
        "arm": run_dir.run_arm(run),
        "test_images_evaluated": n,
        "end_to_end_accuracy": round(accuracy, 4),
        "fallback_used_count": n_fallback,
        "per_class_recall": per_class_recall,
        "web_holdout_detector": wh,
        "confusion_matrix": {
            "rows_true": config.CLASS_NAMES,
            "cols_pred": config.CLASS_NAMES + ["NoDetection"],
            "matrix": cm.tolist(),
        },
        "detector_conf": pipe.conf,
        "match_rule": f"IoP>={config.MATCH_IOP_THRESH}",
        "crop_pad_frac": config.CROP_PAD_FRAC,
        "no_detection_fallback": config.NO_DETECTION_FALLBACK,
    }

    (edir / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    _save_confusion_png(cm, edir / "confusion_matrix.png")
    run_dir.update_manifest(run, "eval", report)
    return report


def _save_confusion_png(cm: np.ndarray, out_path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    cols = config.CLASS_NAMES + ["NoDet"]
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(config.NUM_CLASSES), config.CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Locked test/ — full pipeline")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
