"""
Experiment 2 — BINARY BASELINE.

Same data and model as exp1, but every disease class collapsed to a single
"lesion" class. Tells us how much the detector struggles purely at "is there
a lesion", separate from telling the diseases apart.

    python Experimenting/exp2_binary_original.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import build_original
from common.train_eval import run


def main() -> None:
    spec = build_original("binary_original", binary=True)
    run(spec, "yolov8n.pt")


if __name__ == "__main__":
    main()
