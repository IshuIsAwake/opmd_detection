"""
Experiment 1 — TRUE BASELINE.

Flat YOLOv8-nano, 5 classes, original data only (pool → train/val 85/15,
locked data/test as test). Nothing collapsed, nothing tuned. This is the
number every later idea has to beat.

    python Experimenting/exp1_5class_original.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import build_original
from common.train_eval import run


def main() -> None:
    spec = build_original("5class_original", binary=False)
    run(spec, "yolov8n.pt")


if __name__ == "__main__":
    main()
