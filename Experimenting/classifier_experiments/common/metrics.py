"""
metrics.py — Phase 1 (matched train/eval) and Phase 2 (pipeline) metrics.

Phase 1: each crop is an independent example.
  - micro accuracy, macro accuracy, per-class recall, confusion matrix.

Phase 2: each image's boxes aggregate by mean-softmax → argmax (matches §12).
  - caught rate (image-level from detector)
  - conditional disease accuracy (of caught, % correct)
  - system-level accuracy (correct over all positives, missed = wrong)
  - negative false-alarm rate (detector firings on negatives, classifier
    output ignored for this number — matches HANDOFF Phase 2 metrics list)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import settings


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def phase1_report(y_true: list[int], y_pred: list[int]) -> dict:
    n_classes = len(settings.ORIG_ID_TO_CLASS)
    y_true_a = np.array(y_true, dtype=int)
    y_pred_a = np.array(y_pred, dtype=int)
    n = len(y_true_a)
    micro = float((y_true_a == y_pred_a).mean()) if n else 0.0

    conf = np.zeros((n_classes, n_classes), dtype=int)
    per_class_n = np.zeros(n_classes, dtype=int)
    per_class_correct = np.zeros(n_classes, dtype=int)
    for t, p in zip(y_true_a, y_pred_a):
        conf[t, p] += 1
        per_class_n[t] += 1
        if t == p:
            per_class_correct[t] += 1
    per_class_recall = np.where(
        per_class_n > 0, per_class_correct / np.maximum(per_class_n, 1), 0.0)
    represented = per_class_n > 0
    macro = float(per_class_recall[represented].mean()) if represented.any() else 0.0

    return {
        "n": int(n),
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "per_class_recall": {
            settings.ORIG_ID_TO_CLASS[c]: float(per_class_recall[c])
            for c in range(n_classes)
        },
        "per_class_n": {settings.ORIG_ID_TO_CLASS[c]: int(per_class_n[c])
                        for c in range(n_classes)},
        "confusion_matrix": {
            "rows_gt": list(settings.ORIG_CLASS_NAMES),
            "cols_pred": list(settings.ORIG_CLASS_NAMES),
            "matrix": conf.tolist(),
        },
    }


def format_phase1(report: dict) -> str:
    lines = [
        f"n        : {report['n']}",
        f"micro acc: {report['micro_accuracy']:.4f}",
        f"macro acc: {report['macro_accuracy']:.4f}",
        "per-class recall:",
    ]
    for name, r in report["per_class_recall"].items():
        n = report["per_class_n"][name]
        lines.append(f"  {name:14s} {r:.4f}  (n={n})")
    lines.append("confusion matrix (rows=GT, cols=pred):")
    names = report["confusion_matrix"]["rows_gt"]
    header = "             " + "  ".join(f"{n[:6]:>6s}" for n in names)
    lines.append(header)
    for r, row in enumerate(report["confusion_matrix"]["matrix"]):
        lines.append(f"  {names[r][:10]:10s} " + "  ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def phase2_report(per_image: list[dict], n_negatives: int, n_fa: int) -> dict:
    """per_image rows are positives only, each: {gt: int, caught: bool,
    pred: int | None (None if missed)}."""
    n_pos = len(per_image)
    caught = [r for r in per_image if r["caught"]]
    correct = [r for r in caught if r["pred"] == r["gt"]]

    n_caught = len(caught)
    n_correct = len(correct)
    cond_acc = (n_correct / n_caught) if n_caught else 0.0
    sys_acc = (n_correct / n_pos) if n_pos else 0.0
    catch_rate = (n_caught / n_pos) if n_pos else 0.0
    fa_rate = (n_fa / n_negatives) if n_negatives else 0.0

    # Confusion matrix on caught positives (rows GT 0..4, cols pred 0..4).
    n_classes = len(settings.ORIG_ID_TO_CLASS)
    conf = np.zeros((n_classes, n_classes), dtype=int)
    per_class_n = np.zeros(n_classes, dtype=int)
    per_class_correct = np.zeros(n_classes, dtype=int)
    for r in caught:
        conf[r["gt"], r["pred"]] += 1
        per_class_n[r["gt"]] += 1
        if r["gt"] == r["pred"]:
            per_class_correct[r["gt"]] += 1
    per_class_recall = np.where(
        per_class_n > 0, per_class_correct / np.maximum(per_class_n, 1), 0.0)

    return {
        "n_positives": n_pos,
        "caught": n_caught,
        "missed": n_pos - n_caught,
        "caught_correct": n_correct,
        "catch_rate": catch_rate,
        "conditional_disease_accuracy": cond_acc,
        "system_accuracy": sys_acc,
        "n_negatives": n_negatives,
        "n_negative_false_alarms": n_fa,
        "negative_false_alarm_rate": fa_rate,
        "per_class_recall_on_caught": {
            settings.ORIG_ID_TO_CLASS[c]: float(per_class_recall[c])
            for c in range(n_classes)
        },
        "per_class_n_on_caught": {
            settings.ORIG_ID_TO_CLASS[c]: int(per_class_n[c])
            for c in range(n_classes)
        },
        "confusion_matrix_caught": {
            "rows_gt": list(settings.ORIG_CLASS_NAMES),
            "cols_pred": list(settings.ORIG_CLASS_NAMES),
            "matrix": conf.tolist(),
        },
    }


def format_phase2(report: dict) -> str:
    lines = [
        f"positives n         : {report['n_positives']}",
        f"caught              : {report['caught']} / {report['n_positives']}  ({report['catch_rate']:.1%})",
        f"  caught + correct  : {report['caught_correct']}",
        f"missed              : {report['missed']}",
        f"conditional acc     : {report['conditional_disease_accuracy']:.4f}  ← headline",
        f"system-level acc    : {report['system_accuracy']:.4f}",
        f"negatives n         : {report['n_negatives']}",
        f"neg FA              : {report['n_negative_false_alarms']} / {report['n_negatives']}  ({report['negative_false_alarm_rate']:.1%})",
        "per-class recall on caught:",
    ]
    for name, r in report["per_class_recall_on_caught"].items():
        n = report["per_class_n_on_caught"][name]
        lines.append(f"  {name:14s} {r:.4f}  (n={n})")
    lines.append("confusion matrix on caught (rows=GT, cols=pred):")
    names = report["confusion_matrix_caught"]["rows_gt"]
    lines.append("             " + "  ".join(f"{n[:6]:>6s}" for n in names))
    for r, row in enumerate(report["confusion_matrix_caught"]["matrix"]):
        lines.append(f"  {names[r][:10]:10s} " + "  ".join(f"{v:>6d}" for v in row))
    return "\n".join(lines)


# ── Aggregation across folds ─────────────────────────────────────────────────

def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
