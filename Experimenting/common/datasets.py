"""
datasets.py — Build the generated YOLO trees for the six experiments.

Every experiment becomes a self-contained tree under Experimenting/_datasets/:

    _datasets/<name>/
        images/{train,val,test}/   (symlinks to read-only originals)
        labels/{train,val,test}/   (rewritten label .txt)
        data.yaml

Two builders:
  * build_original  — exp1 (5-class) / exp2 (binary). Train/val = pool split
    85/15 (seeded, stratified by class), test = data/test. Axis-aligned boxes.
  * build_expert    — exp3-6. Train/val = that disease's Roboflow images only,
    every label reformatted to single-class OBB. Test = ALL original images
    (pool + test + Normal); GT = that disease's original boxes, every other
    image (other disease or healthy) is a negative.

Originals are never modified — images are symlinked, labels are written fresh.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from common import negatives, settings
from common.obb_convert import format_obb_line, parse_lines, to_axis_box, to_obb


@dataclass
class DatasetSpec:
    name: str
    root: Path
    data_yaml: Path
    class_names: list[str]
    task: str                       # "detect" | "obb"
    test_images: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class RoboflowSource:
    dir_name: str                   # folder under data/additional
    kind: str                       # "box" | "obb" | "poly"
    keep_ids: frozenset[int]        # source class ids that are this disease
    prefix: str                     # filename namespace (collision-safe)


_SPLITS_RF = ("train", "valid", "test")    # all Roboflow splits feed train/val


# ── shared tree helpers ───────────────────────────────────────────────────────

def _fresh(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            (root / sub / split).mkdir(parents=True, exist_ok=True)


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def _image_for(label: Path, images_dir: Path) -> Path | None:
    for ext in settings.IMG_EXTS:
        cand = images_dir / f"{label.stem}{ext}"
        if cand.exists():
            return cand
    return None


def _write_yaml(spec_root: Path, names: list[str]) -> Path:
    y = spec_root / "data.yaml"
    lines = [
        f"path: {spec_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names: [" + ", ".join(names) + "]",
        "",
    ]
    y.write_text("\n".join(lines))
    return y


def _place(root: Path, split: str, stem: str, img: Path, label_text: str) -> None:
    _symlink(img, root / "images" / split / f"{stem}{img.suffix}")
    (root / "labels" / split / f"{stem}.txt").write_text(label_text)


# ── exp1 / exp2 : original data, axis-aligned detection ──────────────────────

def pool_train_val_split() -> tuple[list[Path], list[Path]]:
    """Deterministic pool train/val split, stratified by the image's first-box
    class. Shared by exp1/exp2 and exp7 so the pool half is byte-identical
    across them (the only thing exp7 adds is Roboflow images to TRAIN)."""
    pool = settings.list_images(settings.POOL_IMAGES)
    by_class: dict[int, list[Path]] = {}
    for img in pool:
        rows = _read_box_label(settings.POOL_LABELS / f"{img.stem}.txt")
        cid = rows[0][0] if rows else -1
        by_class.setdefault(cid, []).append(img)

    rng = random.Random(settings.SEED)
    train: list[Path] = []
    val: list[Path] = []
    for cid in sorted(by_class):
        imgs = sorted(by_class[cid], key=lambda p: p.name)
        rng.shuffle(imgs)
        k = max(1, round(len(imgs) * settings.VAL_FRACTION)) if len(imgs) > 1 else 0
        val += imgs[:k]
        train += imgs[k:]
    return train, val


def build_original(name: str, binary: bool) -> DatasetSpec:
    """5-class (binary=False) or single-class (binary=True) on original data."""
    root = settings.DATASETS_ROOT / name
    _fresh(root)

    train, val = pool_train_val_split()

    def emit(imgs: list[Path], split: str, labels_dir: Path) -> None:
        for img in imgs:
            rows = _read_box_label(labels_dir / f"{img.stem}.txt")
            out = [
                f"{0 if binary else c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for c, cx, cy, w, h in rows
            ]
            _place(root, split, img.stem, img, "\n".join(out))

    emit(train, "train", settings.POOL_LABELS)
    emit(val, "val", settings.POOL_LABELS)
    test_imgs = settings.list_images(settings.TEST_IMAGES)
    emit(test_imgs, "test", settings.TEST_LABELS)

    names = ["lesion"] if binary else settings.ORIG_CLASS_NAMES
    yaml = _write_yaml(root, names)
    test_paths = sorted((root / "images" / "test").iterdir(), key=lambda p: p.name)
    return DatasetSpec(name, root, yaml, names, "detect", test_paths)


def _read_box_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        try:
            rows.append((int(float(p[0])), *(float(v) for v in p[1:])))
        except ValueError:
            continue
    return rows


# ── exp8a / exp8b : original data + resolution-normalised FAIR negatives ─────

def fair_negative_split() -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """(train, val, test, reserve) of resolution-normalised negative paths,
    1:1 with the pool positive split at every split (the approved design):
    test = #locked-test positives (37), val = #pool-val positives,
    train = #pool-train positives, the rest reserved for a later ablation.

    Deterministic and shared by build_original_plus_negatives AND
    eval_fair_negatives, so the fair test slice is byte-identical across
    exp8a, exp8b and the no-retrain re-score (Number A)."""
    norm = negatives.prepare()                       # deterministic order
    order = list(norm)
    random.Random(settings.SEED).shuffle(order)

    train_pos, val_pos = pool_train_val_split()
    n_test = len(settings.list_images(settings.TEST_IMAGES))      # 37
    n_val = len(val_pos)
    n_train = len(train_pos)

    test = order[:n_test]
    val = order[n_test:n_test + n_val]
    train = order[n_test + n_val:n_test + n_val + n_train]
    reserve = order[n_test + n_val + n_train:]
    return train, val, test, reserve


def build_original_plus_negatives(name: str, binary: bool) -> DatasetSpec:
    """exp8a (binary=False) / exp8b (binary=True): exp1/exp2 EXACTLY on the
    positive side (same seeded pool split, same locked-37 test, same stems &
    labels — single-variable), with resolution-normalised negatives folded
    into train/val and 37 of them added to test (1:1). IoU≥0.5 match gate is
    unchanged (metrics.py), so this stays directly comparable to exp1/exp2."""
    root = settings.DATASETS_ROOT / name
    _fresh(root)

    train_pos, val_pos = pool_train_val_split()

    def emit_pos(imgs: list[Path], split: str, labels_dir: Path) -> None:
        for img in imgs:
            rows = _read_box_label(labels_dir / f"{img.stem}.txt")
            out = [
                f"{0 if binary else c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for c, cx, cy, w, h in rows
            ]
            _place(root, split, img.stem, img, "\n".join(out))

    emit_pos(train_pos, "train", settings.POOL_LABELS)
    emit_pos(val_pos, "val", settings.POOL_LABELS)
    test_pos = settings.list_images(settings.TEST_IMAGES)
    emit_pos(test_pos, "test", settings.TEST_LABELS)

    tr_neg, va_neg, te_neg, rs_neg = fair_negative_split()

    def emit_neg(paths: list[Path], split: str) -> None:
        for img in paths:                            # empty label = background
            _place(root, split, f"neg__{img.stem}", img, "")

    emit_neg(tr_neg, "train")
    emit_neg(va_neg, "val")
    emit_neg(te_neg, "test")

    names = ["lesion"] if binary else settings.ORIG_CLASS_NAMES
    yaml = _write_yaml(root, names)
    test_paths = sorted((root / "images" / "test").iterdir(),
                        key=lambda p: p.name)

    (root / "dataset_stats.json").write_text(json.dumps({
        "mode": "binary" if binary else "5class",
        "pool_train_pos": len(train_pos),
        "pool_val_pos": len(val_pos),
        "test_pos": len(test_pos),
        "neg_train": len(tr_neg),
        "neg_val": len(va_neg),
        "neg_test": len(te_neg),
        "neg_reserve": len(rs_neg),
        "negative_target_long_side": negatives.positive_target_size(),
        "note": "positives byte-identical to exp1/exp2; only negatives added",
    }, indent=2))
    return DatasetSpec(name, root, yaml, names, "detect", test_paths)


# ── exp3-6 : expert, Roboflow → single-class OBB ─────────────────────────────

def build_expert(
    name: str, disease: str, sources: list[RoboflowSource]
) -> DatasetSpec:
    """Train/val = this disease's Roboflow images (→ OBB, class 0).
    Test = every original image; GT = this disease's original boxes."""
    root = settings.DATASETS_ROOT / name
    _fresh(root)
    orig_id = settings.ORIG_CLASS_NAMES.index(disease)

    # 1. Collect + convert every Roboflow positive for this disease, then
    #    DEDUPE to one representative per unique source photo. Roboflow bakes
    #    flips/rotations/crops in as separate files sharing the stem before
    #    ".rf.<hash>"; augmented copies are NOT new data. We keep a single
    #    deterministic representative per base and let YOLO's own train-time
    #    augmentation do augmentation. Reported size = unique photos.
    by_base: dict[str, tuple[str, Path, str]] = {}
    raw_files = 0
    per_src_unique: dict[str, int] = {}
    for src in sources:
        sroot = settings.ADDITIONAL_DIR / src.dir_name
        seen_src: set[str] = set()
        for split in _SPLITS_RF:
            ldir, idir = sroot / split / "labels", sroot / split / "images"
            if not (ldir.is_dir() and idir.is_dir()):
                continue
            for lbl in sorted(ldir.glob("*.txt"), key=lambda p: p.stem):
                img = _image_for(lbl, idir)
                if img is None:
                    continue
                obb_lines = []
                for cid, coords in parse_lines(lbl.read_text()):
                    if cid not in src.keep_ids:
                        continue
                    corners = to_obb(coords, src.kind)
                    if corners is not None:
                        obb_lines.append(format_obb_line(corners, 0))
                if not obb_lines:                   # positives only
                    continue
                raw_files += 1
                # Base = source prefix + the Roboflow stem before ".rf.".
                base = f"{src.prefix}__{img.stem.split('.rf.')[0]}"
                if base in by_base:                 # an augmented sibling; drop
                    continue
                by_base[base] = (base, img, "\n".join(obb_lines))
                seen_src.add(base)
        per_src_unique[src.dir_name] = len(seen_src)

    # 2. Seeded 85/15 split over UNIQUE photos (split == by-base → no leakage).
    records = sorted(by_base.values(), key=lambda r: r[0])
    rng = random.Random(settings.SEED)
    rng.shuffle(records)
    k = max(1, round(len(records) * settings.VAL_FRACTION))
    for stem, img, text in records[:k]:
        _place(root, "val", stem, img, text)
    for stem, img, text in records[k:]:
        _place(root, "train", stem, img, text)

    (root / "dataset_stats.json").write_text(json.dumps({
        "disease": disease,
        "roboflow_label_files_positive": raw_files,
        "unique_source_photos": len(records),
        "augmented_copies_dropped": raw_files - len(records),
        "per_source_unique": per_src_unique,
        "train": len(records) - k, "val": k,
    }, indent=2))

    # 3. Test = ALL original images. GT = this disease's boxes (axis, class 0);
    #    other-disease and Normal images are negatives (empty label).
    test_paths: list[Path] = []
    for img, lbl_dir, prefix in _iter_originals():
        rows = _read_box_label(lbl_dir / f"{img.stem}.txt") if lbl_dir else []
        keep = [r for r in rows if r[0] == orig_id]
        text = "\n".join(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, cx, cy, w, h in keep
        )
        stem = f"{prefix}__{img.stem}"
        _place(root, "test", stem, img, text)
        test_paths.append(root / "images" / "test" / f"{stem}{img.suffix}")

    yaml = _write_yaml(root, [disease])
    return DatasetSpec(name, root, yaml, [disease], "obb", sorted(test_paths))


# ── exp7 : binary, original pool + UNIQUE Roboflow photos (no augmentation) ───

# Every lesion class across all four exports collapses to one "lesion" class.
# OSMF `non-osmf` (class 0) is not a lesion and is excluded.
_RF_BINARY_SOURCES = (
    RoboflowSource("Leukoplakia.v2i.yolov8", "box", frozenset({0}), "leukov2"),
    RoboflowSource("OPMD-SEG.v1i.yolov8", "poly", frozenset({0, 1, 2}), "opmdpoly"),
    RoboflowSource("OSMF DETECTION.v1i.yolov8", "poly", frozenset({1}), "osmf"),
)


def build_binary_plus_roboflow(name: str) -> DatasetSpec:
    """exp7 — exp2's binary setup with UNIQUE (de-augmented) Roboflow lesion
    photos appended to TRAIN only. Pool train/val + locked test are
    byte-identical to exp2; the single difference is the extra Roboflow images.
    """
    root = settings.DATASETS_ROOT / name
    _fresh(root)

    # Pool half — identical to exp2 (binary).
    train_pool, val_pool = pool_train_val_split()

    def emit_pool(imgs, split, labels_dir):
        for img in imgs:
            rows = _read_box_label(labels_dir / f"{img.stem}.txt")
            text = "\n".join(
                f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                for _, cx, cy, w, h in rows
            )
            _place(root, split, f"pool__{img.stem}", img, text)

    emit_pool(train_pool, "train", settings.POOL_LABELS)
    emit_pool(val_pool, "val", settings.POOL_LABELS)
    for img in settings.list_images(settings.TEST_IMAGES):
        rows = _read_box_label(settings.TEST_LABELS / f"{img.stem}.txt")
        text = "\n".join(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, cx, cy, w, h in rows
        )
        _place(root, "test", f"test__{img.stem}", img, text)
    test_paths = sorted((root / "images" / "test").iterdir(), key=lambda p: p.name)

    # Roboflow half — de-duped to unique source photos, axis boxes, → TRAIN.
    by_base: dict[str, tuple[Path, str]] = {}
    raw_files = 0
    per_src_unique: dict[str, int] = {}
    for src in _RF_BINARY_SOURCES:
        sroot = settings.ADDITIONAL_DIR / src.dir_name
        seen: set[str] = set()
        for split in _SPLITS_RF:
            ldir, idir = sroot / split / "labels", sroot / split / "images"
            if not (ldir.is_dir() and idir.is_dir()):
                continue
            for lbl in sorted(ldir.glob("*.txt"), key=lambda p: p.stem):
                img = _image_for(lbl, idir)
                if img is None:
                    continue
                lines = []
                for cid, coords in parse_lines(lbl.read_text()):
                    if cid not in src.keep_ids:
                        continue
                    box = to_axis_box(coords, src.kind)
                    if box is not None:
                        lines.append("0 " + " ".join(f"{v:.6f}" for v in box))
                if not lines:
                    continue
                raw_files += 1
                base = f"{src.prefix}__{img.stem.split('.rf.')[0]}"
                if base in by_base:
                    continue
                by_base[base] = (img, "\n".join(lines))
                seen.add(base)
        per_src_unique[src.dir_name] = len(seen)

    for base, (img, text) in sorted(by_base.items()):
        _place(root, "train", base, img, text)

    yaml = _write_yaml(root, ["lesion"])
    (root / "dataset_stats.json").write_text(json.dumps({
        "pool_train": len(train_pool),
        "pool_val": len(val_pool),
        "roboflow_label_files_positive": raw_files,
        "roboflow_unique_added_to_train": len(by_base),
        "augmented_copies_dropped": raw_files - len(by_base),
        "per_source_unique": per_src_unique,
        "total_train": len(train_pool) + len(by_base),
    }, indent=2))
    return DatasetSpec(name, root, yaml, ["lesion"], "detect", test_paths)


def _iter_originals():
    """Yield (image, labels_dir_or_None, namespace_prefix) for every original
    image used as an expert test image: pool, locked test, and Normal."""
    for img in settings.list_images(settings.POOL_IMAGES):
        yield img, settings.POOL_LABELS, "pool"
    for img in settings.list_images(settings.TEST_IMAGES):
        yield img, settings.TEST_LABELS, "test"
    for d in settings.NORMAL_DIRS:
        if d.is_dir():
            tag = "normal" if d.name == "Normal" else "normalnc"
            for img in settings.list_images(d):
                yield img, None, tag
