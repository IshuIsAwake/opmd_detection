"""
exp1_whole_image.py — Arm #1 (DINOv2): train on full positive images, no crop.

Sanity probe: does cropping matter at all? 5-class DINOv2-S head trained on
whole positive images, paired 5-fold CV on the detector's kfold5 splits.

Usage:
    cd /home/ishu/Projects/AI/Oral_cancer
    eval "$(conda shell.bash hook)" && conda activate ai_env

    # one-time: materialise the per-fold symlink trees (CPU)
    python Experimenting/classifier_experiments/common/materialise.py --arms whole

    # all 5 folds (GPU)
    python Experimenting/classifier_experiments/exp1_whole_image.py

    # one fold:
    python Experimenting/classifier_experiments/exp1_whole_image.py --fold 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model import DinoV2Classifier, _load_dinov2
from common.run_arm import run_arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    backbone = _load_dinov2()                       # shared across folds
    run_arm(arm_base="whole_image",
            backbone_tag="dinov2",
            model_factory=lambda: DinoV2Classifier(backbone=backbone),
            fold=args.fold,
            write_summary=not args.no_summary)


if __name__ == "__main__":
    main()
