"""
exp2a_gt_pad00.py — Arm #2a (DINOv2): tight GT crop, pad = 0.0.
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

    backbone = _load_dinov2()
    run_arm(arm_base="gt_pad", pad=0.0,
            backbone_tag="dinov2",
            model_factory=lambda: DinoV2Classifier(backbone=backbone),
            fold=args.fold,
            write_summary=not args.no_summary)


if __name__ == "__main__":
    main()
