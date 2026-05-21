"""
exp10_aug_sweep_binary.py — Augmentation sweep on exp8b's splits (binary).

Same _datasets/binary_negatives/ tree as exp8b (rebuilt deterministically; the
positive side is byte-identical to exp2/exp8b's seeded pool split + locked-37
test + 1:1 resolution-normalised neg slice). Single variable across runs =
the YOLO augmentation config. Seed 42 throughout; one training per level.

Five levels (see common/aug.py for exact configs):
  off            true no-aug except fliplr=0.5 (free symmetry of a mouth photo)
  light          mild geometric only, colour locked at 0
  default        YOLO defaults — reproduces exp8b
  geom_no_color  defaults but hsv_*=0 — isolates the "colour is diagnostic" rule
  heavy          defaults + rotate/scale/shear + mixup + copy_paste + stronger erasing

Each level lands in Experimenting/results/aug_<level>_binary/ with the usual
metrics.{json,txt} + match_rule_sweep + val_stock + run.json (run.json now
includes the train_kwargs_override for full provenance).

Usage:
    python Experimenting/exp10_aug_sweep_binary.py                  # all levels
    python Experimenting/exp10_aug_sweep_binary.py --level off      # single
    ORAL_SMOKE=1 python Experimenting/exp10_aug_sweep_binary.py --level off   # plumbing
After (or during):
    python Experimenting/summarize_aug.py                            # comparison table
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.aug import AUG_LEVELS, describe
from common.datasets import build_original_plus_negatives
from common.train_eval import run

LEVELS_ORDER = ("off", "light", "default", "geom_no_color", "heavy", "heavy_no_color")
DATASET_NAME = "binary_negatives"        # shared with exp8b on disk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=LEVELS_ORDER, default=None,
                    help="run a single level. Default: all 5 levels in order.")
    args = ap.parse_args()

    # Rebuild the binary_negatives tree once (idempotent + byte-identical to
    # exp8b; the build is fast, the train is the cost). All levels share it.
    spec = build_original_plus_negatives(DATASET_NAME, binary=True)

    levels = [args.level] if args.level else list(LEVELS_ORDER)
    for level in levels:
        print(f"\n══ exp10 aug-sweep [{level}] ══")
        print(f"   {describe(level)}\n")
        run(spec, "yolov8n.pt",
            run_name=f"aug_{level}_binary",
            train_kwargs=AUG_LEVELS[level])


if __name__ == "__main__":
    main()
