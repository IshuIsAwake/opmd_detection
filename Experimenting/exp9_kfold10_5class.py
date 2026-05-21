"""
exp9_kfold10_5class.py — 10-fold CV of yolov8n 5-class + fair negatives.

The exp8a recipe, repeated 10 times on the SAME folds as
exp9_kfold10_binary.py (shared _datasets/kfold10_splits.json). Inner train/val
85/15 stratified by disease class; per-fold test slice is a blackbox (~36
pos + ~36 fair-1:1 neg).

No augmentation (single-variable lift vs exp8a; augmentation = next
experiment, layered on the same folds for a second controlled comparison).

Usage:
    python Experimenting/exp9_kfold10_5class.py                # all 10 folds
    python Experimenting/exp9_kfold10_5class.py --fold 0       # single fold
    ORAL_SMOKE=1 python Experimenting/exp9_kfold10_5class.py --fold 0

After (or during):
    python Experimenting/summarize_kfold.py kfold10_5class
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.datasets import build_kfold_fold
from common.kfold import K, build_or_load
from common.summarize import write_summary
from common.train_eval import run

EXPERIMENT = "kfold10_5class"
SPLITS_PATH = settings.DATASETS_ROOT / "kfold10_splits.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="run a single fold (0..K-1). Default: all folds.")
    ap.add_argument("--no-summary", action="store_true",
                    help="skip writing summary.{json,txt} after the runs.")
    args = ap.parse_args()

    build_or_load(SPLITS_PATH)
    folds = [args.fold] if args.fold is not None else list(range(K))

    for k in folds:
        if not (0 <= k < K):
            raise SystemExit(f"fold {k} out of range (0..{K - 1})")
        spec = build_kfold_fold(EXPERIMENT, k, binary=False,
                                splits_path=SPLITS_PATH)
        run(spec, "yolov8n.pt")

    if not args.no_summary:
        out_dir = settings.RESULTS_ROOT / EXPERIMENT
        write_summary(out_dir)
        print(f"\n→ {out_dir}/summary.txt\n")
        print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
