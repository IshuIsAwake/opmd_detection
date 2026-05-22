"""
folds.py — Resolve the shared kfold5 split into per-fold train/val/test KEY lists.

The detector experiments build splits.json over IMAGES; the classifier trains
on BOXES (522 boxes across 362 images — multi-box images contribute every
box). HANDOFF: "All 522 GT boxes ... are training examples". So the
image→fold assignment comes from splits.json; the box-level expansion lives
here.

Returned key per box:  "<image_key>__b{box_index}"  e.g. "pool__Leukoplakia_1__b0"
Disease per box       = the per-image disease from splits.json (verified to
                        match the per-row class in every multi-box label).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from common import settings
from common.crops import read_yolo_boxes

INNER_VAL_FRAC = 0.15


@dataclass
class BoxEntry:
    key: str                     # "<image_key>__b<i>" (one per box) OR "<image_key>" (whole-image)
    image_key: str               # the splits.json positive key
    image: Path
    disease: int                 # 0..4
    box_idx: int | None          # None for whole-image arm
    box_norm: tuple[float, float, float, float] | None  # (cx, cy, w, h) normalised; None for whole-image


@dataclass
class FoldEntries:
    train: list[BoxEntry]
    val: list[BoxEntry]
    test: list[BoxEntry]


def load_split() -> dict:
    if not settings.DET_SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"detector kfold5 splits not found at {settings.DET_SPLITS_PATH}. "
            "Build it first by running the detector kfold5 experiment, or copy "
            "Experimenting/_datasets/kfold5_splits.json into place.")
    return json.loads(settings.DET_SPLITS_PATH.read_text())


def _stratified_inner_split(items: list[BoxEntry], seed: int
                            ) -> tuple[list[BoxEntry], list[BoxEntry]]:
    """Inner train/val split on the fold's non-test positives, stratified by
    disease. Box-level: a multi-box image's boxes can land on different sides
    of the inner split — fine for the classifier (each box is its own example,
    they share a class label) and saves us from constructing image-level
    stratification machinery just for the inner split."""
    by_class: dict[int, list[BoxEntry]] = {}
    for it in items:
        by_class.setdefault(it.disease, []).append(it)
    rng = random.Random(seed)
    train: list[BoxEntry] = []
    val: list[BoxEntry] = []
    for c in sorted(by_class):
        bucket = sorted(by_class[c], key=lambda e: e.key)
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * INNER_VAL_FRAC)) if len(bucket) > 1 else 0
        val += bucket[:n_val]
        train += bucket[n_val:]
    return train, val


def _expand_boxes(image_keys: list[str], positives: dict) -> list[BoxEntry]:
    """Expand a list of image keys → one BoxEntry per GT box (522 total
    across the whole dataset)."""
    out: list[BoxEntry] = []
    for ikey in image_keys:
        meta = positives[ikey]
        img = Path(meta["image"])
        lbl = Path(meta["label"])
        disease = int(meta["disease"])
        rows = read_yolo_boxes(lbl)
        if not rows:
            # Should not happen for a positive — surface it loudly.
            raise RuntimeError(f"positive {ikey} has no GT boxes at {lbl}")
        for i, (cid, cx, cy, w, h) in enumerate(rows):
            # Sanity: per-row class must match the image's disease label.
            if cid != disease:
                raise RuntimeError(
                    f"{ikey} box {i} class_id={cid} != image disease={disease}")
            out.append(BoxEntry(
                key=f"{ikey}__b{i}",
                image_key=ikey,
                image=img,
                disease=disease,
                box_idx=i,
                box_norm=(cx, cy, w, h),
            ))
    return out


def _whole_image_entries(image_keys: list[str], positives: dict) -> list[BoxEntry]:
    """One entry per image (no box expansion) — for the whole-image arm."""
    out: list[BoxEntry] = []
    for ikey in image_keys:
        meta = positives[ikey]
        out.append(BoxEntry(
            key=ikey,
            image_key=ikey,
            image=Path(meta["image"]),
            disease=int(meta["disease"]),
            box_idx=None,
            box_norm=None,
        ))
    return out


def fold_entries(fold_idx: int, mode: str) -> FoldEntries:
    """Materialise one fold's train/val/test entry lists.

    mode = "box"          → one entry per GT box (for the GT-crop arms)
    mode = "whole_image"  → one entry per image (for the whole-image arm)
    """
    if mode not in {"box", "whole_image"}:
        raise ValueError(f"unknown mode: {mode}")

    split = load_split()
    pos_meta = split["positives"]
    test_keys = list(split["positives_by_fold"][fold_idx])
    other_keys = [k for i, fold in enumerate(split["positives_by_fold"])
                  if i != fold_idx for k in fold]

    expand = _expand_boxes if mode == "box" else _whole_image_entries
    test = expand(test_keys, pos_meta)
    pool = expand(other_keys, pos_meta)

    train, val = _stratified_inner_split(pool, seed=split["seed"] + 1000 + fold_idx)
    return FoldEntries(train=train, val=val, test=test)
