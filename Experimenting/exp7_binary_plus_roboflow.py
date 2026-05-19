"""
Experiment 7 — BINARY + ROBOFLOW (original photos only, no augmentation).

exp2 exactly (binary yolov8n, pool train/val split, locked data/test as test)
with one change: every UNIQUE Roboflow lesion photo (all four exports, all
classes → "lesion", augmented copies dropped) is appended to TRAIN.

Controlled A/B vs exp2 — same pool split, same original-domain val, same
locked 37-image test. The only variable is the added real Roboflow data.
This is the clean original-domain analog of the main project's
`plus_roboflow` arm.

    python Experimenting/exp7_binary_plus_roboflow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import build_binary_plus_roboflow
from common.train_eval import run


def main() -> None:
    spec = build_binary_plus_roboflow("binary_plus_roboflow")
    run(spec, "yolov8n.pt")


if __name__ == "__main__":
    main()
