"""
exp9_kfold10_binary.py — 10-fold CV of yolov8n binary + fair negatives.

The exp8b recipe, repeated 10 times with deterministic stratified k-fold
splits over (pool + locked test, 362 positive imgs) and 10-way shuffled
splits over the ~570 resolution-normalised negatives. For each fold, the
test slice (~36 pos + ~36 fair-1:1 neg) is a TRUE blackbox — never seen in
train or val of that fold. Inner train/val on the remaining 9 folds is 85/15
stratified by disease class.

Shared splits.json under _datasets/kfold10_splits.json drives THIS script
AND exp9_kfold10_5class.py — both models train on byte-identical folds, so
binary vs 5-class is a controlled comparison on every fold.

No augmentation (deliberately — keep this a single-variable lift vs exp8b).
Same training recipe as exp8 otherwise: epochs/imgsz/batch/seed defaults from
common/settings.py, AMP off, all knobs env-overridable.

Usage:
    python Experimenting/exp9_kfold10_binary.py                # all 10 folds
    python Experimenting/exp9_kfold10_binary.py --fold 0       # single fold
    ORAL_SMOKE=1 python Experimenting/exp9_kfold10_binary.py --fold 0   # plumbing

After (or during — partial summaries are fine):
    python Experimenting/summarize_kfold.py kfold10_binary
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

EXPERIMENT = "kfold10_binary"
SPLITS_PATH = settings.DATASETS_ROOT / "kfold10_splits.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None,
                    help="run a single fold (0..K-1). Default: all folds.")
    ap.add_argument("--no-summary", action="store_true",
                    help="skip writing summary.{json,txt} after the runs.")
    args = ap.parse_args()

    build_or_load(SPLITS_PATH)                     # idempotent
    folds = [args.fold] if args.fold is not None else list(range(K))

    for k in folds:
        if not (0 <= k < K):
            raise SystemExit(f"fold {k} out of range (0..{K - 1})")
        spec = build_kfold_fold(EXPERIMENT, k, binary=True,
                                splits_path=SPLITS_PATH)
        run(spec, "yolov8n.pt")

    if not args.no_summary:
        out_dir = settings.RESULTS_ROOT / EXPERIMENT
        write_summary(out_dir)
        print(f"\n→ {out_dir}/summary.txt\n")
        print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
