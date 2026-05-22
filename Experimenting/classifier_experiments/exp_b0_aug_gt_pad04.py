"""
exp_b0_aug_gt_pad04.py — Round 3: B0 at GT pad=0.4 with the strong-aug suite.

Single-variable test on top of the Round-2 best cell
(B0 train_pad=0.40 was the headline). Adds: RandomResizedCrop (0.7–1.0),
±20° rotation, ColorJitter (brightness/contrast/sat; no hue), RandomErasing
(p=0.25). See common.dataset.strong_train_transform.

Results land at results/gt_pad_0.40_b0_aug/ so the Round-2 numbers at
results/gt_pad_0.40_b0/ are untouched. Phase 2 for this arm via:
    python phase2_pipeline.py --backbone b0_aug --train-pads 0.4

Usage:
    python Experimenting/classifier_experiments/exp_b0_aug_gt_pad04.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.model_b0 import EfficientNetB0Classifier
from common.run_arm import run_arm
from common.train_eval import TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--no-summary", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(aug_strong=True)
    run_arm(arm_base="gt_pad", pad=0.4,
            backbone_tag="b0_aug",
            model_factory=lambda: EfficientNetB0Classifier(pretrained=True),
            fold=args.fold,
            write_summary=not args.no_summary,
            cfg=cfg)


if __name__ == "__main__":
    main()
