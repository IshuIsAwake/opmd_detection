"""
Step 3 — Build classifier data from the TRAINED detector. (Critical step.)

The fix for mistake #2: the classifier learns from the crops the detector
actually emits, not from human annotations.

Matching uses CONTAINMENT (IoP), not IoU. Annotator boxes here are huge/loose;
a correct tight prediction sits inside the GT, scoring near-zero IoU but ~1.0
IoP. Gating on IoU≥0.5 was discarding good crops and starving classes — this
is what IoP fixes.

  for each pool image (split from splits.json):
    run this run's detector at the chosen low conf
    for each predicted box:
      best GT by IoP
      IoP >= MATCH_IOP_THRESH -> crop (SHARED fn), label = matched GT class
      else                    -> false positive, discarded (baseline)

Output: <run>/classifier/data/<split>/<Disease>/{stem}_d{n}.jpg
No Normal class — 'healthy' is decided upstream by the detector finding nothing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

import config
from src.common import run_dir
from src.common.crop import crop_with_padding          # the ONE shared crop fn
from src.common.geometry import iop, yolo_to_pixel
from src.common.io import label_path_for, read_yolo_label


def _gt_boxes_with_class(image_path: Path, img_w: int, img_h: int) -> list[tuple]:
    out = []
    for cid, cx, cy, bw, bh in read_yolo_label(
        label_path_for(image_path, config.POOL_LABELS)
    ):
        if cid in config.YOLO_ID_TO_CLASS:
            out.append((cid, yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)))
    return out


def _fresh_tree(data_dir: Path) -> None:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    for split in ("train", "val"):
        for cls in config.CLASS_NAMES:
            (data_dir / split / cls).mkdir(parents=True, exist_ok=True)


def build(conf: float) -> dict:
    """Emit detector-produced crops into the current run. Returns counts."""
    run = run_dir.current_run()
    arm = run_dir.run_arm(run)
    weights = run_dir.detector_weights(run)
    if not weights.exists():
        raise FileNotFoundError(f"No detector weights at {weights} — run Step 2.")
    splits_json = config.splits_json_for(arm)
    if not splits_json.exists():
        raise FileNotFoundError(
            f"{splits_json} missing — run scripts/01 --arm {arm}."
        )

    splits = json.loads(splits_json.read_text())
    stem_to_split = {s: "train" for s in splits["train"]}
    stem_to_split.update({s: "val" for s in splits["val"]})

    data_dir = run_dir.classifier_data_dir(run)
    _fresh_tree(data_dir)
    model = YOLO(str(weights))

    counts = {
        "train": {c: 0 for c in config.CLASS_NAMES},
        "val": {c: 0 for c in config.CLASS_NAMES},
    }
    n_fp = 0
    n_no_det = 0

    for stem, split in stem_to_split.items():
        candidates = list(config.POOL_IMAGES.glob(f"{stem}.*"))
        if not candidates:
            continue
        img_path = candidates[0]
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        gts = _gt_boxes_with_class(img_path, w, h)
        result = model.predict(str(img_path), conf=conf, verbose=False)[0]
        preds = (
            [tuple(b) for b in result.boxes.xyxy.cpu().numpy()]
            if result.boxes is not None and len(result.boxes)
            else []
        )
        if not preds:
            n_no_det += 1
            continue

        for n, pbox in enumerate(preds):
            best_iop, best_cid = 0.0, None
            for cid, gbox in gts:
                score = iop(pbox, gbox)
                if score > best_iop:
                    best_iop, best_cid = score, cid

            if best_cid is None or best_iop < config.MATCH_IOP_THRESH:
                n_fp += 1
                continue

            cls_name = config.YOLO_ID_TO_CLASS[best_cid]
            crop = crop_with_padding(image, pbox)
            out_path = data_dir / split / cls_name / f"{stem}_d{n}.jpg"
            cv2.imwrite(str(out_path), crop)
            counts[split][cls_name] += 1

    stats = {
        "match_rule": f"IoP>={config.MATCH_IOP_THRESH}",
        "detector_conf": conf,
        "per_split_class": counts,
        "false_positives_discarded": n_fp,
        "images_with_no_detection": n_no_det,
    }
    run_dir.update_manifest(run, "classifier_data", stats)
    return stats
