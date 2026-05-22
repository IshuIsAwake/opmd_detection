"""
phase2_pipeline.py — The real-question evaluation: trained classifier ←
detector-emitted crops.

For each Phase 2 fold k ∈ {1,2,3,4} (CLAUDE.md: do not ship fold 0):
  * Load that fold's detector best.pt at conf=0.10 (the adopted operating point).
  * Run YOLO on every positive image in fold k's test slice and every negative
    in fold k's test slice. Collect boxes.
  * For each train_pad ∈ {0.0, 0.2, 0.4} (loads the matching exp2{a,b,c} head):
      for each serve_pad ∈ {0.0, 0.2, 0.4}:
          - Pad detector boxes by serve_pad, crop, classify each crop.
          - Aggregate per image: mean softmax across that image's boxes → argmax.
          - Compare argmax to GT disease (positives) or count as FA (negatives).
  → 3 × 3 matrix of conditional disease accuracy + system accuracy.

Also runs arm #1 (whole-image) on each fold's full test positives — no
detector path, no pad axis. Reported alongside as the "no detector" reference.

Negative FA is geometry-free (detector-only) so it is identical across all
9 cells in the 3×3 matrix for a given fold.

Two backbones supported via ``--backbone``. Results land at
``results/phase2/`` (dinov2, the Round 1 layout) or ``results/phase2_b0/``
(b0, Round 2), so the two never overwrite each other.

Usage:
    eval "$(conda shell.bash hook)" && conda activate ai_env
    # Round 1 (DINOv2, already done):
    python Experimenting/classifier_experiments/phase2_pipeline.py --backbone dinov2
    # Round 2 (B0):
    python Experimenting/classifier_experiments/phase2_pipeline.py --backbone b0
    # Single fold:
    python Experimenting/classifier_experiments/phase2_pipeline.py --backbone b0 --fold 1
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.box_ops import merge_boxes
from common.crops import pad_and_crop
from common.dataset import eval_transform
from common.folds import load_split
from common.metrics import (format_phase1, format_phase2, mean_std,
                            phase1_report, phase2_report, write_json)

GT_PADS = settings.GT_PADS                 # train-time pads (one head per pad)


# ── Backbone-dispatch factories ──────────────────────────────────────────────

def _factories(backbone_tag: str):
    """Returns (model_factory, results_suffix). Imports lazily so this module
    is callable on a machine missing one backbone's deps."""
    if backbone_tag == "dinov2":
        from common.model import DinoV2Classifier, _load_dinov2
        shared_bb = _load_dinov2()       # one load, reused across folds
        return (lambda: DinoV2Classifier(backbone=shared_bb), "")
    elif backbone_tag in ("b0", "b0_aug"):
        # b0_aug uses the same architecture as b0; only the trained weights
        # (different aug recipe) and the results-dir suffix differ.
        from common.model_b0 import EfficientNetB0Classifier
        return (lambda: EfficientNetB0Classifier(pretrained=False),
                f"_{backbone_tag}")
    else:
        raise ValueError(f"unknown backbone: {backbone_tag}")


def _arm_results_root(arm_name: str, backbone_tag: str) -> Path:
    suffix = "" if backbone_tag == "dinov2" else f"_{backbone_tag}"
    return settings.RESULTS_ROOT / f"{arm_name}{suffix}"


def _phase2_root(backbone_tag: str, merged: bool = False,
                 tta: bool = False) -> Path:
    suffix = "" if backbone_tag == "dinov2" else f"_{backbone_tag}"
    if merged:
        suffix += "_merged"
    if tta:
        suffix += "_tta"
    return settings.RESULTS_ROOT / f"phase2{suffix}"


# ── Detector / classifier weight discovery ───────────────────────────────────

def _det_weights(fold_idx: int) -> Path:
    p = settings.DET_RESULTS_ROOT / f"fold_{fold_idx}" / "train" / "weights" / "best.pt"
    if not p.exists():
        raise FileNotFoundError(
            f"detector weights missing for fold {fold_idx}: {p}")
    return p


def _classifier_weights(arm_name: str, backbone_tag: str, fold_idx: int) -> Path:
    """best.pt for fresh runs; legacy head.pt for the Round-1 DINOv2 runs."""
    root = _arm_results_root(arm_name, backbone_tag) / f"fold_{fold_idx}"
    for fname in ("best.pt", "head.pt"):
        cand = root / fname
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"classifier weights missing for {arm_name}/fold {fold_idx} "
        f"under {root} (tried best.pt, head.pt)")


def _load_classifier(arm_name: str, backbone_tag: str, fold_idx: int,
                     factory, device: torch.device) -> nn.Module:
    weights = _classifier_weights(arm_name, backbone_tag, fold_idx)
    model = factory().to(device)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    model.load_trainable_state(state)
    model.eval()
    return model


def _load_detector(fold_idx: int):
    from ultralytics import YOLO
    return YOLO(str(_det_weights(fold_idx)))


# ── Detector inference helpers ───────────────────────────────────────────────

@dataclass
class ImageResult:
    image_key: str
    image_path: Path
    boxes: list[tuple[float, float, float, float, float]]   # xyxy + conf
    bgr: np.ndarray | None


def _detect_image(detector, image_path: Path, conf: float) -> list[tuple]:
    """Returns [(x1, y1, x2, y2, conf), ...] in pixel coords."""
    res = detector.predict(source=str(image_path), conf=conf, verbose=False)
    out: list[tuple] = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    return out


# ── Classifier inference ─────────────────────────────────────────────────────

_TFM = eval_transform()

# 4-view TTA: identity + hflip + ±10° rotation. Matches the geometric range
# the strong-aug training already saw, so each view stays in-distribution.
# Larger rotations / vflip skipped — vflip changes oral anatomy semantics
# (upper vs lower lip), and >10° rotation already gets hit by training aug.
_TTA_VIEW_NAMES = ("identity", "hflip", "rot_+10", "rot_-10")


def _tta_views(pil: Image.Image) -> list[Image.Image]:
    return [
        pil,
        TF.hflip(pil),
        TF.rotate(pil, 10),
        TF.rotate(pil, -10),
    ]


def _classify_crops(model: nn.Module, crops_bgr: list[np.ndarray],
                    device: torch.device,
                    use_tta: bool = False) -> np.ndarray:
    """Returns mean softmax across the input crops (shape [5]).

    With ``use_tta=True``, each crop is expanded into 4 augmented views
    (see _TTA_VIEW_NAMES); the per-crop probability is the mean across
    those views, then per-image probability is the mean across crops as
    before. Adds 4× forward-pass cost per crop, no retraining."""
    if not crops_bgr:
        return np.zeros(len(settings.ORIG_ID_TO_CLASS), dtype=np.float32)
    probs_acc = []
    model.eval()
    with torch.no_grad():
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            views = _tta_views(pil) if use_tta else [pil]
            x = torch.stack([_TFM(v) for v in views]).to(device)  # [V,3,H,W]
            logits = model(x)
            # Average probabilities across views, not logits — averaging
            # post-softmax respects the boundedness of the simplex.
            probs = F.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
            probs_acc.append(probs)
    return np.mean(probs_acc, axis=0)


# ── Phase 2 per-fold runner ──────────────────────────────────────────────────

def run_fold(fold_idx: int, conf: float, device: torch.device,
             backbone_tag: str, factory,
             train_pads: tuple[float, ...] = GT_PADS,
             merge: bool = False, merge_iou: float = 0.3,
             use_tta: bool = False) -> dict:
    phase2_root = _phase2_root(backbone_tag, merged=merge, tta=use_tta)
    print(f"\n══ Phase 2 ({backbone_tag}) — fold {fold_idx} ══")
    split = load_split()
    pos_meta = split["positives"]
    test_keys = list(split["positives_by_fold"][fold_idx])
    neg_paths = [Path(p) for p in split["negatives_by_fold"][fold_idx][:len(test_keys)]]

    print(f"   positives: {len(test_keys)}   negatives: {len(neg_paths)}")
    print(f"   detector:  {_det_weights(fold_idx)}")

    detector = _load_detector(fold_idx)

    def _maybe_merge(boxes):
        """Apply union-merge if requested; preserve the raw-tuple output
        shape so downstream code (pad_and_crop, ImageResult) is unchanged."""
        if not merge or not boxes:
            return boxes, len(boxes)
        raw_n = len(boxes)
        merged = merge_boxes(boxes, iou_thresh=merge_iou)
        out = [(m.xyxy[0], m.xyxy[1], m.xyxy[2], m.xyxy[3], m.conf)
               for m in merged]
        return out, raw_n

    pos_results: list[ImageResult] = []
    pos_box_counts = {"raw": 0, "after_merge": 0, "images_with_merging": 0}
    for ikey in test_keys:
        img_path = Path(pos_meta[ikey]["image"])
        raw = _detect_image(detector, img_path, conf=conf)
        boxes, raw_n = _maybe_merge(raw)
        pos_box_counts["raw"] += raw_n
        pos_box_counts["after_merge"] += len(boxes)
        if raw_n != len(boxes):
            pos_box_counts["images_with_merging"] += 1
        bgr = cv2.imread(str(img_path)) if boxes else None
        pos_results.append(ImageResult(ikey, img_path, boxes, bgr))

    n_fa = 0
    fa_paths: list[str] = []
    for np_ in neg_paths:
        # Neg FA is a detector-only property; merging never turns a firing
        # into a silent (or vice versa). Skip the merge step on negatives.
        boxes = _detect_image(detector, np_, conf=conf)
        if boxes:
            n_fa += 1
            fa_paths.append(str(np_))

    print(f"   detector fired on {sum(1 for r in pos_results if r.boxes)}/"
          f"{len(pos_results)} positives, {n_fa}/{len(neg_paths)} negatives "
          f"(@ conf={conf})")
    if merge:
        print(f"   merge: {pos_box_counts['raw']} raw boxes → "
              f"{pos_box_counts['after_merge']} clusters across positives "
              f"({pos_box_counts['images_with_merging']} images affected, "
              f"iou_thresh={merge_iou})")

    # Free the detector before loading classifiers (6 GB GPU is tight).
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_results: dict[str, dict] = {}
    for tp in train_pads:
        arm = f"gt_pad_{tp:.2f}"
        model = _load_classifier(arm, backbone_tag, fold_idx, factory, device)
        for sp in GT_PADS:
            cell = f"train_pad_{tp:.2f}__serve_pad_{sp:.2f}"
            per_image: list[dict] = []
            for r in pos_results:
                gt = int(pos_meta[r.image_key]["disease"])
                if not r.boxes:
                    per_image.append({"image_key": r.image_key, "gt": gt,
                                      "caught": False, "pred": None,
                                      "n_boxes": 0})
                    continue
                crops = [pad_and_crop(r.bgr, b[:4], sp) for b in r.boxes]
                crops = [c for c in crops if c is not None]
                if not crops:
                    per_image.append({"image_key": r.image_key, "gt": gt,
                                      "caught": False, "pred": None,
                                      "n_boxes": 0})
                    continue
                mean_probs = _classify_crops(model, crops, device,
                                             use_tta=use_tta)
                pred = int(np.argmax(mean_probs))
                per_image.append({"image_key": r.image_key, "gt": gt,
                                  "caught": True, "pred": pred,
                                  "n_boxes": len(crops),
                                  "mean_probs": mean_probs.tolist()})

            rep = phase2_report(per_image, n_negatives=len(neg_paths), n_fa=n_fa)
            fold_results[cell] = rep
            print(f"   [{cell}]  cond_acc={rep['conditional_disease_accuracy']:.4f}  "
                  f"sys_acc={rep['system_accuracy']:.4f}  "
                  f"caught={rep['caught']}/{rep['n_positives']}  "
                  f"FA={rep['n_negative_false_alarms']}/{rep['n_negatives']}")

            out_dir = phase2_root / f"fold_{fold_idx}" / cell
            out_dir.mkdir(parents=True, exist_ok=True)
            write_json(out_dir / "metrics.json", rep)
            (out_dir / "metrics.txt").write_text(format_phase2(rep) + "\n")
            with open(out_dir / "per_image.jsonl", "w") as f:
                for row in per_image:
                    f.write(json.dumps(row) + "\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for r in pos_results:
        r.bgr = None

    summary = {
        "fold": fold_idx,
        "backbone": backbone_tag,
        "conf": conf,
        "merge": merge,
        "merge_iou": merge_iou if merge else None,
        "tta": use_tta,
        "tta_views": list(_TTA_VIEW_NAMES) if use_tta else None,
        "pos_box_counts": pos_box_counts,
        "n_positives": len(pos_results),
        "n_negatives": len(neg_paths),
        "detector_fires_pos": sum(1 for r in pos_results if r.boxes),
        "n_negative_false_alarms": n_fa,
        "false_alarm_paths": fa_paths,
        "cells": fold_results,
    }
    write_json(phase2_root / f"fold_{fold_idx}" / "summary.json", summary)
    return summary


# ── Whole-image arm Phase 2 (no detector) ────────────────────────────────────

def run_fold_whole_image(fold_idx: int, device: torch.device,
                         backbone_tag: str, factory,
                         merged: bool = False,
                         use_tta: bool = False) -> dict:
    """Arm #1 has no pipeline path: score the trained whole-image model on
    the fold's full positive test images."""
    split = load_split()
    pos_meta = split["positives"]
    test_keys = list(split["positives_by_fold"][fold_idx])

    model = _load_classifier("whole_image", backbone_tag, fold_idx,
                             factory, device)

    y_true: list[int] = []
    y_pred: list[int] = []
    model.eval()
    tfm = eval_transform()
    with torch.no_grad():
        for ikey in test_keys:
            img_path = Path(pos_meta[ikey]["image"])
            pil = Image.open(img_path).convert("RGB")
            views = _tta_views(pil) if use_tta else [pil]
            x = torch.stack([tfm(v) for v in views]).to(device)
            probs = F.softmax(model(x), dim=1).mean(dim=0)
            pred = int(probs.argmax().item())
            y_true.append(int(pos_meta[ikey]["disease"]))
            y_pred.append(pred)

    rep = phase1_report(y_true, y_pred)
    out_dir = (_phase2_root(backbone_tag, merged=merged, tta=use_tta)
               / f"fold_{fold_idx}" / "whole_image_no_detector")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", rep)
    (out_dir / "metrics.txt").write_text(format_phase1(rep) + "\n")
    print(f"   [whole_image_no_detector]  micro={rep['micro_accuracy']:.4f}  "
          f"macro={rep['macro_accuracy']:.4f}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rep


# ── Cross-fold summary ───────────────────────────────────────────────────────

def write_phase2_summary(fold_indices: list[int], backbone_tag: str,
                         merged: bool = False, tta: bool = False) -> Path:
    out_root = _phase2_root(backbone_tag, merged=merged, tta=tta)
    rows = []
    for k in fold_indices:
        fp = out_root / f"fold_{k}" / "summary.json"
        if fp.exists():
            rows.append(json.loads(fp.read_text()))

    cells = sorted({c for r in rows for c in r["cells"]})
    cell_summary = {}
    for c in cells:
        cond = [r["cells"][c]["conditional_disease_accuracy"] for r in rows
                if c in r["cells"]]
        sys_ = [r["cells"][c]["system_accuracy"] for r in rows
                if c in r["cells"]]
        catch = [r["cells"][c]["catch_rate"] for r in rows if c in r["cells"]]
        fa = [r["cells"][c]["negative_false_alarm_rate"] for r in rows
              if c in r["cells"]]
        cell_summary[c] = {
            "conditional_disease_accuracy": {"mean": mean_std(cond)[0],
                                             "std": mean_std(cond)[1],
                                             "per_fold": cond},
            "system_accuracy": {"mean": mean_std(sys_)[0],
                                "std": mean_std(sys_)[1], "per_fold": sys_},
            "catch_rate": {"mean": mean_std(catch)[0],
                           "std": mean_std(catch)[1], "per_fold": catch},
            "negative_false_alarm_rate": {"mean": mean_std(fa)[0],
                                          "std": mean_std(fa)[1],
                                          "per_fold": fa},
        }

    summary = {"backbone": backbone_tag,
               "folds": [r["fold"] for r in rows],
               "cells": cell_summary}
    write_json(out_root / "summary.json", summary)

    lines = [
        f"backbone: {backbone_tag}",
        f"merge:    {merged}",
        f"tta:      {tta}",
        f"folds: {summary['folds']}",
        "",
        f"{'train_pad':>10s}  {'serve_pad':>10s}  "
        f"{'cond_acc':>15s}  {'sys_acc':>15s}  "
        f"{'catch':>13s}  {'neg_FA':>13s}",
    ]
    for tp in GT_PADS:
        for sp in GT_PADS:
            c = f"train_pad_{tp:.2f}__serve_pad_{sp:.2f}"
            agg = cell_summary.get(c)
            if not agg:
                continue
            ca = agg["conditional_disease_accuracy"]
            sa = agg["system_accuracy"]
            cr = agg["catch_rate"]
            fa = agg["negative_false_alarm_rate"]
            lines.append(
                f"  {tp:>8.2f}    {sp:>8.2f}    "
                f"{ca['mean']:.3f} ± {ca['std']:.3f}    "
                f"{sa['mean']:.3f} ± {sa['std']:.3f}    "
                f"{cr['mean']:.3f}±{cr['std']:.3f}  "
                f"{fa['mean']:.3f}±{fa['std']:.3f}"
            )
    (out_root / "summary.txt").write_text("\n".join(lines) + "\n")
    return out_root / "summary.txt"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["dinov2", "b0", "b0_aug"],
                    default="b0",
                    help="which classifier backbone to evaluate")
    ap.add_argument("--train-pads", type=float, nargs="+", default=None,
                    help="subset of train_pads to evaluate (default: all of "
                         f"{settings.GT_PADS}). Use e.g. --train-pads 0.4 when "
                         "only one arm has been trained.")
    ap.add_argument("--fold", type=int, default=None,
                    help="run one fold (1..4). Default: all Phase 2 folds.")
    ap.add_argument("--conf", type=float, default=settings.DETECTOR_CONF)
    ap.add_argument("--skip-whole", action="store_true",
                    help="skip the whole-image-no-detector reference path")
    ap.add_argument("--merge-boxes", action="store_true",
                    help="union-merge overlapping detector boxes before "
                         "cropping. Output goes to a parallel results/ dir "
                         "with suffix '_merged' so the un-merged numbers are "
                         "preserved for direct comparison.")
    ap.add_argument("--merge-iou", type=float, default=0.3,
                    help="IoU threshold for the box merge (default 0.3).")
    ap.add_argument("--tta", action="store_true",
                    help="4-view test-time augmentation (identity, hflip, "
                         "rot ±10°). Mean softmax across views per crop. "
                         "Leak-free. Output goes to a parallel results/ dir "
                         "with suffix '_tta'.")
    args = ap.parse_args()

    if args.fold is not None and args.fold not in settings.PHASE2_FOLDS:
        raise SystemExit(
            f"fold {args.fold} not in Phase 2 set {settings.PHASE2_FOLDS} "
            "(fold 0 is excluded by CLAUDE.md landmine).")
    folds = [args.fold] if args.fold is not None else list(settings.PHASE2_FOLDS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:   {device}")
    print(f"backbone: {args.backbone}")
    print(f"conf:     {args.conf}")
    print(f"folds:    {folds}")
    print(f"merge:    {args.merge_boxes} "
          + (f"(iou_thresh={args.merge_iou})" if args.merge_boxes else ""))
    print(f"tta:      {args.tta}"
          + (f"  ({len(_TTA_VIEW_NAMES)} views)" if args.tta else ""))

    factory, _suffix = _factories(args.backbone)
    train_pads = tuple(args.train_pads) if args.train_pads else GT_PADS
    if not all(tp in GT_PADS for tp in train_pads):
        raise SystemExit(
            f"--train-pads {args.train_pads} ⊄ {list(GT_PADS)}")
    print(f"train_pads: {train_pads}")

    # Single-arm runs (e.g. --train-pads 0.4) have no whole-image counterpart;
    # auto-skip rather than crash on a missing whole_image_<tag>/ tree.
    skip_whole = args.skip_whole or train_pads != GT_PADS

    for k in folds:
        run_fold(k, conf=args.conf, device=device,
                 backbone_tag=args.backbone, factory=factory,
                 train_pads=train_pads,
                 merge=args.merge_boxes, merge_iou=args.merge_iou,
                 use_tta=args.tta)
        if not skip_whole:
            run_fold_whole_image(k, device=device,
                                 backbone_tag=args.backbone, factory=factory,
                                 merged=args.merge_boxes,
                                 use_tta=args.tta)

    if args.fold is None:
        out = write_phase2_summary(list(settings.PHASE2_FOLDS), args.backbone,
                                   merged=args.merge_boxes, tta=args.tta)
        print(f"\n→ {out}\n")
        print(out.read_text())


if __name__ == "__main__":
    main()
