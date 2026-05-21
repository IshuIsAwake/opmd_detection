"""
exp11_kfold5_aug_binary.py — 5-fold CV of yolov8n binary at a given aug level.

Faster variant of exp9 (10 → 5 folds) for the head-to-head between two
augmentation levels. Same K-fold logic, K=5 → each fold's blackbox test slice
is ~72 pos + ~72 neg (2× the kfold10 per-fold N → tighter per-fold numbers,
half the trainings).

Designed for two PARALLEL processes — both read the SAME
_datasets/kfold5_splits.json (shared, byte-identical 5 folds), each writes to
its own results/kfold5_<aug>_binary/ tree. The cross-aug comparison is
controlled by the shared split; the only variable is `--aug-level`.

Pair with `compare_aug_kfold.py` to produce a head-to-head table once both
processes finish.

Usage (parallel — two terminals):
    python Experimenting/exp11_kfold5_aug_binary.py --aug-level geom_no_color
    python Experimenting/exp11_kfold5_aug_binary.py --aug-level heavy_no_color

Per-fold:
    python Experimenting/exp11_kfold5_aug_binary.py --aug-level geom_no_color --fold 0

Smoke (plumbing check):
    ORAL_SMOKE=1 python Experimenting/exp11_kfold5_aug_binary.py --aug-level geom_no_color --fold 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.aug import AUG_LEVELS, describe
from common.datasets import build_kfold_fold
from common.kfold import build_or_load
from common.summarize import write_summary
from common.train_eval import run

K_FOLDS = 5
SPLITS_PATH = settings.DATASETS_ROOT / "kfold5_splits.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-level", required=True, choices=list(AUG_LEVELS),
                    help="augmentation config from common/aug.py")
    ap.add_argument("--fold", type=int, default=None,
                    help="run a single fold (0..4). Default: all 5 folds.")
    ap.add_argument("--no-summary", action="store_true",
                    help="skip writing summary.{json,txt} after the runs.")
    args = ap.parse_args()

    # Build (or load) the shared 5-fold split — idempotent, atomic-write safe
    # under concurrent reads from a parallel process.
    build_or_load(SPLITS_PATH, k=K_FOLDS)

    experiment = f"kfold5_{args.aug_level}_binary"
    folds = [args.fold] if args.fold is not None else list(range(K_FOLDS))

    print(f"\n══ exp11 5-fold CV [{args.aug_level}] ══")
    print(f"   {describe(args.aug_level)}")
    print(f"   folds to run: {folds}")
    print(f"   writes to:    Experimenting/results/{experiment}/\n")

    for k in folds:
        if not (0 <= k < K_FOLDS):
            raise SystemExit(f"fold {k} out of range (0..{K_FOLDS - 1})")
        spec = build_kfold_fold(experiment, k, binary=True,
                                splits_path=SPLITS_PATH)
        run(spec, "yolov8n.pt", train_kwargs=AUG_LEVELS[args.aug_level])

    if not args.no_summary:
        out_dir = settings.RESULTS_ROOT / experiment
        write_summary(out_dir)
        print(f"\n→ {out_dir}/summary.txt\n")
        print((out_dir / "summary.txt").read_text())


if __name__ == "__main__":
    main()
