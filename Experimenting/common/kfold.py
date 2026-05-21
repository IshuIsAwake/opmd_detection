"""
kfold.py — Deterministic K-fold CV split builder (default K=10; K=5 also used).

Combines pool + locked-test positives into one 362-image set, stratified by
disease class (the label class_id is reliable, the filename is not). The
~570 resolution-normalised Normal images are shuffled once and K-way
interleaved into negative folds. Every experiment that reads the SAME
splits.json sees byte-identical folds, so cross-config comparisons (e.g.
geom_no_color vs heavy_no_color) are controlled.

For every fold k the resulting layout is:

    test_pos  = positives_by_fold[k]                 (~36 imgs, blackbox)
    test_neg  = first len(test_pos) of negatives_by_fold[k]   (1:1, fair)
    pool_pos  = the other 9 folds' positives
    pool_neg  = the other 9 folds' negatives + leftover from fold k's neg slice
    inner 85/15 stratified split of pool_pos → train_pos, val_pos
    inner 85/15 split of pool_neg            → train_neg, val_neg

The inner val is used by Ultralytics for early stopping; the test slice is
NEVER seen during train or val of its fold — that is what makes it a true
black-box test (the user's explicit requirement).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from common import negatives, settings

K = 10                 # default for legacy callers (exp9_kfold10_*)
INNER_VAL_FRAC = 0.15


@dataclass
class FoldSet:
    train_pos: list[tuple[Path, Path, str]]    # (image, label, stem_key)
    val_pos:   list[tuple[Path, Path, str]]
    test_pos:  list[tuple[Path, Path, str]]
    train_neg: list[Path]
    val_neg:   list[Path]
    test_neg:  list[Path]


@dataclass
class _Pos:
    key: str
    image: Path
    label: Path
    disease: int


def _read_first_class(label: Path) -> int:
    if not label.exists():
        return -1
    for line in label.read_text().strip().splitlines():
        p = line.split()
        if len(p) == 5:
            try:
                return int(float(p[0]))
            except ValueError:
                continue
    return -1


def _all_positives() -> list[_Pos]:
    """pool + locked-test positives. Stems are namespaced ("pool__"/"test__")
    because a raw collision between the two dirs would silently overwrite the
    other's symlink/label downstream."""
    out: list[_Pos] = []
    for img in settings.list_images(settings.POOL_IMAGES):
        lbl = settings.POOL_LABELS / f"{img.stem}.txt"
        out.append(_Pos(f"pool__{img.stem}", img, lbl, _read_first_class(lbl)))
    for img in settings.list_images(settings.TEST_IMAGES):
        lbl = settings.TEST_LABELS / f"{img.stem}.txt"
        out.append(_Pos(f"test__{img.stem}", img, lbl, _read_first_class(lbl)))
    return sorted(out, key=lambda p: p.key)


def _stratified_into_folds(items: list[_Pos], k: int, seed: int) -> list[list[str]]:
    """Deterministic stratified k-fold: bucket by disease, shuffle each
    bucket with a per-class seed, deal round-robin into k folds. Two images
    from the same class never end up in the same fold position by luck of
    sort order."""
    by_class: dict[int, list[_Pos]] = {}
    for it in items:
        by_class.setdefault(it.disease, []).append(it)
    folds: list[list[str]] = [[] for _ in range(k)]
    for c in sorted(by_class):
        bucket = sorted(by_class[c], key=lambda p: p.key)
        random.Random(seed * 31 + (c + 1) * 7).shuffle(bucket)
        for i, it in enumerate(bucket):
            folds[i % k].append(it.key)
    return folds


def build_or_load(path: Path, k: int = K) -> dict:
    """Idempotent: build splits.json once, re-load on every subsequent call.
    The on-disk file is the authority — if it already exists, ``k`` is taken
    from it and the argument is ignored (so different K splits live in
    different files, e.g. kfold10_splits.json vs kfold5_splits.json).
    """
    if path.exists():
        return json.loads(path.read_text())

    pos = _all_positives()
    pos_folds = _stratified_into_folds(pos, k, settings.SEED)

    neg_paths = [str(p.resolve()) for p in negatives.prepare()]
    random.Random(settings.SEED + 17).shuffle(neg_paths)
    neg_folds = [neg_paths[i::k] for i in range(k)]    # interleaved → balanced

    out = {
        "k": k,
        "seed": settings.SEED,
        "inner_val_frac": INNER_VAL_FRAC,
        "positives": {p.key: {
            "image": str(p.image.resolve()),
            "label": str(p.label.resolve()),
            "disease": p.disease,
        } for p in pos},
        "positives_by_fold": pos_folds,
        "negatives_by_fold": neg_folds,
        "negative_target_long_side": negatives.positive_target_size(),
        "n_positives": len(pos),
        "n_negatives": len(neg_paths),
    }
    # Atomic write so a second process reading concurrently can never see a
    # partially-written splits.json. If two processes race on the first build,
    # both produce byte-identical content (deterministic from seed) and the
    # last rename wins — no corruption either way.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    return out


def materialise_fold(split: dict, fold_idx: int) -> FoldSet:
    pos_meta = split["positives"]
    seed = split["seed"]
    inner_val_frac = split["inner_val_frac"]

    test_keys = list(split["positives_by_fold"][fold_idx])
    other_keys = [s for i, fold in enumerate(split["positives_by_fold"])
                  if i != fold_idx for s in fold]

    by_class: dict[int, list[str]] = {}
    for key in other_keys:
        by_class.setdefault(pos_meta[key]["disease"], []).append(key)

    rng = random.Random(seed + 1000 + fold_idx)
    train_keys: list[str] = []
    val_keys: list[str] = []
    for c in sorted(by_class):
        bucket = sorted(by_class[c])
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * inner_val_frac)) if len(bucket) > 1 else 0
        val_keys += bucket[:n_val]
        train_keys += bucket[n_val:]

    fold_neg = list(split["negatives_by_fold"][fold_idx])
    test_neg = fold_neg[:len(test_keys)]
    leftover_neg = fold_neg[len(test_keys):]
    other_neg = [p for i, fold in enumerate(split["negatives_by_fold"])
                 if i != fold_idx for p in fold] + leftover_neg

    rng = random.Random(seed + 2000 + fold_idx)
    rng.shuffle(other_neg)
    n_val_neg = round(len(other_neg) * inner_val_frac)
    val_neg = other_neg[:n_val_neg]
    train_neg = other_neg[n_val_neg:]

    def _attach(keys: list[str]) -> list[tuple[Path, Path, str]]:
        return [(Path(pos_meta[k]["image"]), Path(pos_meta[k]["label"]), k)
                for k in keys]

    return FoldSet(
        train_pos=_attach(train_keys),
        val_pos=_attach(val_keys),
        test_pos=_attach(test_keys),
        train_neg=[Path(p) for p in train_neg],
        val_neg=[Path(p) for p in val_neg],
        test_neg=[Path(p) for p in test_neg],
    )
