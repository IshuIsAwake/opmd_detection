"""
materialise.py — Build the on-disk per-fold trees the classifier reads from.

Three layouts, all under classifier_experiments/_datasets/:

  whole/fold_<k>/{train,val,test}/<Disease>/<image_key>.<ext>
       ← symlinks to the original full images (read-only). One file per image.

  gt_pad_<pad>/pool/<image_key>__b<i>.jpg
       ← actual cropped JPEGs (newly created, shared across folds). The pad
         pool is generated ONCE per pad fraction.
  gt_pad_<pad>/fold_<k>/{train,val,test}/<Disease>/<image_key>__b<i>.jpg
       ← per-fold ImageFolder layout. Each leaf is a symlink into pool/.

The whole-image arm symlinks originals; the GT-crop arms symlink into a
locally-generated pool of crops so the JPEG bytes exist once on disk even
though they appear in 5 folds' worth of split trees.

CPU-only. Idempotent: re-running rebuilds the trees from scratch each call
(matches datasets._fresh in spirit). The pool of cropped JPEGs is only
rewritten if the pad-pool dir is missing or --force.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2

# Allow direct invocation: `python common/materialise.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import settings
from common.crops import crop_from_gt_box
from common.folds import fold_entries, load_split


# ── filesystem helpers (mirror datasets.py shape) ────────────────────────────

def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def _fresh(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


# ── whole-image arm ──────────────────────────────────────────────────────────

def materialise_whole_image() -> Path:
    """5 folds × {train,val,test} of full-image symlinks. Returns root dir."""
    split = load_split()
    root = settings.DATASETS_ROOT / "whole"
    _fresh(root)
    for k in range(split["k"]):
        entries = fold_entries(k, mode="whole_image")
        for sp, items in (("train", entries.train),
                          ("val", entries.val),
                          ("test", entries.test)):
            for e in items:
                disease_name = settings.ORIG_ID_TO_CLASS[e.disease]
                dst = root / f"fold_{k}" / sp / disease_name / f"{e.image_key}{e.image.suffix}"
                _symlink(e.image, dst)
    print(f"  whole-image trees → {root}")
    return root


# ── GT-crop arms (3 pad fractions) ───────────────────────────────────────────

def _build_gt_pool(pad: float, force: bool) -> Path:
    """Generate one cropped JPEG per (image, box) into pool/. Idempotent."""
    pool_dir = settings.DATASETS_ROOT / f"gt_pad_{pad:.2f}" / "pool"
    if pool_dir.exists() and not force:
        n = len(list(pool_dir.glob("*.jpg")))
        print(f"  pad={pad:.2f}: pool exists ({n} crops) — reuse (use --force to rebuild)")
        return pool_dir
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)

    split = load_split()
    pos_meta = split["positives"]

    # Iterate every positive once (regardless of fold) and emit every box.
    n_boxes = 0
    n_degen = 0
    seen_imgs: dict[Path, "cv2.Mat | None"] = {}
    for ikey in sorted(pos_meta):
        # Recover the boxes via fold_entries-equivalent expansion. Simpler:
        # use read_yolo_boxes directly (fold-agnostic).
        from common.crops import read_yolo_boxes
        img_path = Path(pos_meta[ikey]["image"])
        lbl_path = Path(pos_meta[ikey]["label"])
        boxes = read_yolo_boxes(lbl_path)
        if not boxes:
            raise RuntimeError(f"no GT boxes for positive {ikey}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"cv2 failed to read {img_path}")
        for i, (_cid, cx, cy, w, h) in enumerate(boxes):
            crop = crop_from_gt_box(bgr, (cx, cy, w, h), pad)
            if crop is None:
                n_degen += 1
                continue
            out = pool_dir / f"{ikey}__b{i}.jpg"
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_boxes += 1
        seen_imgs.pop(img_path, None)  # explicit drop; helps the GC on big sets

    print(f"  pad={pad:.2f}: wrote {n_boxes} crops to {pool_dir}"
          + (f"  (degenerate-after-clip dropped: {n_degen})" if n_degen else ""))
    return pool_dir


def materialise_gt_pad(pad: float, force: bool) -> Path:
    """Per-pad ImageFolder trees, one per fold. Symlinks into pool/."""
    pool_dir = _build_gt_pool(pad, force=force)
    root = settings.DATASETS_ROOT / f"gt_pad_{pad:.2f}"

    # Clear out only the fold_* subtrees; keep pool/ as-is.
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("fold_"):
            shutil.rmtree(child)

    split = load_split()
    for k in range(split["k"]):
        entries = fold_entries(k, mode="box")
        for sp, items in (("train", entries.train),
                          ("val", entries.val),
                          ("test", entries.test)):
            for e in items:
                src = pool_dir / f"{e.key}.jpg"
                if not src.exists():
                    # Crop was degenerate at this pad — entry has no crop. Skip.
                    continue
                disease_name = settings.ORIG_ID_TO_CLASS[e.disease]
                dst = root / f"fold_{k}" / sp / disease_name / f"{e.key}.jpg"
                _symlink(src, dst)
    print(f"  gt_pad_{pad:.2f} per-fold trees → {root}")
    return root


# ── entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Materialise classifier trees.")
    ap.add_argument("--arms", nargs="+",
                    default=["whole", "0.00", "0.20", "0.40"],
                    help="subset of: whole, 0.00, 0.20, 0.40")
    ap.add_argument("--force", action="store_true",
                    help="rebuild GT crop pools even if they already exist")
    args = ap.parse_args()

    settings.DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"writes to: {settings.DATASETS_ROOT}\n")

    if "whole" in args.arms:
        print("→ whole-image arm")
        materialise_whole_image()

    for pad in settings.GT_PADS:
        token = f"{pad:.2f}"
        if token in args.arms:
            print(f"\n→ gt_pad_{token}")
            materialise_gt_pad(pad, force=args.force)


if __name__ == "__main__":
    main()
