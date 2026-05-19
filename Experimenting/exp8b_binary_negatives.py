"""
Experiment 8b — BINARY + FAIR NEGATIVES (the active next step).

exp2 EXACTLY on the positive side (binary yolov8n, same seeded pool split,
same locked-37 test), with one change: resolution-normalised Normal images
folded into train/val and 37 added to test (1:1, so 0.50 is the no-skill
line). The single variable vs exp2 is "negatives, fairly sized".

Read the screening result off det_rate_pos + false_alarm together, as the
raw counts now printed by metrics.py ("X of 37 lesion images flagged, Y of 74
total") — NOT box P/R. Box P/R + IoU/IoP/IoG are kept (IoU≥0.5 gate unchanged)
only for exp1/exp2 continuity.

Pair with: python Experimenting/eval_fair_negatives.py binary_original
(Number A — old exp2 weights on this same fair test, no retrain).

    python Experimenting/exp8b_binary_negatives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import build_original_plus_negatives
from common.train_eval import run


def main() -> None:
    spec = build_original_plus_negatives("binary_negatives", binary=True)
    run(spec, "yolov8n.pt")


if __name__ == "__main__":
    main()
