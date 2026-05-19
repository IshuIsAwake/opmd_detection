"""
Detector evaluation — recall first, mAP irrelevant.

A missed lesion is far worse than a spurious box (the classifier filters false
positives downstream). We measure *lesion recall*: of all GT boxes, how many
are covered by at least one prediction, sweeping low confidences.

Matching uses CONTAINMENT (IoP) because annotator boxes are huge/loose — a
correct tight prediction has near-zero IoU but ~1.0 IoP. IoU-based recall is
still reported as a (pessimistic) secondary view.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from src.common.geometry import iop, iou_xyxy, yolo_to_pixel
from src.common.io import label_path_for, list_images, read_yolo_label


def _gt_pixel_boxes(image_path: Path, labels_dir: Path) -> list[tuple]:
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    return [
        yolo_to_pixel(cx, cy, bw, bh, w, h)
        for _, cx, cy, bw, bh in read_yolo_label(label_path_for(image_path, labels_dir))
    ]


def _recall_at_conf(model, images, labels_dir, conf) -> dict:
    total_gt = 0
    matched_iop = 0          # primary criterion
    matched_iou = 0          # secondary, reported only
    imgs_with_gt = 0
    imgs_any_hit = 0

    for img_path in images:
        gts = _gt_pixel_boxes(img_path, labels_dir)
        if not gts:
            continue
        imgs_with_gt += 1
        total_gt += len(gts)

        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        preds = (
            [tuple(b) for b in result.boxes.xyxy.cpu().numpy()]
            if result.boxes is not None and len(result.boxes)
            else []
        )

        hit = False
        for gt in gts:
            if any(iop(p, gt) >= config.MATCH_IOP_THRESH for p in preds):
                matched_iop += 1
                hit = True
            if any(iou_xyxy(p, gt) >= config.IOU_SECONDARY for p in preds):
                matched_iou += 1
        if hit:
            imgs_any_hit += 1

    return {
        "conf": round(conf, 3),
        "box_recall": round(matched_iop / total_gt, 4) if total_gt else 0.0,
        "box_recall_iou": round(matched_iou / total_gt, 4) if total_gt else 0.0,
        "image_recall": round(imgs_any_hit / imgs_with_gt, 4) if imgs_with_gt else 0.0,
        "total_gt": total_gt,
    }


# Probes very low confidences too: recall-first product, and an undertrained
# model (smoke run) only emits low-conf boxes.
DEFAULT_CONF_GRID = (0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def sweep(
    weights: Path,
    images_dir: Path,
    labels_dir: Path,
    conf_grid: tuple[float, ...] = DEFAULT_CONF_GRID,
) -> list[dict]:
    """Recall across a grid of low confidences. Model loaded once."""
    model = YOLO(str(weights))
    images = list_images(images_dir)
    return [_recall_at_conf(model, images, labels_dir, c) for c in conf_grid]


def web_holdout_detector_metrics(weights: Path) -> dict | None:
    """
    Detector-only metrics on data/new_data/web_holdout/ — a higher-N,
    tight-box signal vs the noisy 37. Eval only; never trained/tuned on.
    Returns {map50, recall} or None if web_holdout is absent.
    """
    data_yaml = config.WEB_HOLDOUT_DIR / "data.yaml"
    if not data_yaml.exists():
        return None
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml), split="val", verbose=False,
        save_json=False, plots=False,
    )
    return {
        "map50": round(float(m.box.map50), 4),
        "recall": round(float(m.box.mr), 4),
        "images": int(getattr(m, "seen", 0)) or None,
    }


def recommend_conf(rows: list[dict], min_recall: float = 0.90) -> float:
    """
    Highest confidence still clearing ``min_recall`` box-recall (fewer FPs for
    the classifier). Else the conf with best recall. If the model emits nothing
    anywhere (degenerate), the lowest probed conf so downstream still gets data.
    """
    ok = [r for r in rows if r["box_recall"] >= min_recall]
    if ok:
        return max(ok, key=lambda r: r["conf"])["conf"]
    best = max(rows, key=lambda r: r["box_recall"])
    if best["box_recall"] == 0.0:
        return min(r["conf"] for r in rows)
    return best["conf"]
