"""
exp1_b0_whole_image.py — Round 2 / Arm #1: EfficientNet-B0 end-to-end on
whole positive images.

CNN counterpart of exp1_whole_image.py. Same data, same folds, same metrics
— only the backbone changes. Results land at results/whole_image_b0/ so the
DINOv2 results at results/whole_image/ are untouched.

Usage:
    cd /home/ishu/Projects/AI/Oral_cancer
    eval "$(conda shell.bash hook)" && conda activate ai_env
    python Experimenting/classifier_experiments/common/materialise.py --arms whole
    python Experimenting/classifier_experiments/exp1_b0_whole_image.py
    python Experimenting/classifier_experiments/exp1_b0_whole_image.py --fold 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model_b0 import EfficientNetB0Classifier
from common.run_arm import run_arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    run_arm(arm_base="whole_image",
            backbone_tag="b0",
            model_factory=lambda: EfficientNetB0Classifier(pretrained=True),
            fold=args.fold,
            write_summary=not args.no_summary)


if __name__ == "__main__":
    main()
