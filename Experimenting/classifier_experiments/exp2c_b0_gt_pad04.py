"""
exp2c_b0_gt_pad04.py — Round 2 / Arm #2c: EfficientNet-B0 on GT crops, pad = 0.4.
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

    run_arm(arm_base="gt_pad", pad=0.4,
            backbone_tag="b0",
            model_factory=lambda: EfficientNetB0Classifier(pretrained=True),
            fold=args.fold,
            write_summary=not args.no_summary)


if __name__ == "__main__":
    main()
