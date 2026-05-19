"""
CLI — Step 1: build the binary detector dataset for ONE experiment arm.

    python scripts/01_build_detector_dataset.py --arm original_only
    python scripts/01_build_detector_dataset.py --arm plus_roboflow

Conversion + dedup + web_holdout run in BOTH arms (fast, deterministic).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.common import console  # noqa: E402
from src.detector.build_dataset import build  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    args = ap.parse_args()

    console.start(
        "Step 1 · build detector dataset",
        arm=args.arm,
        smoke=config.SMOKE,
        new_data_root=config.NEW_DATA_ROOT,
    )
    stats = build(args.arm)
    console.end("Step 1 · build detector dataset", **stats)
