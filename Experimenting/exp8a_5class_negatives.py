"""
Experiment 8a — 5-CLASS + FAIR NEGATIVES.

exp1 EXACTLY on the positive side (5-class yolov8n, same seeded pool split,
same locked-37 test), with resolution-normalised Normal images folded into
train/val and 37 added to test (1:1). Single variable vs exp1 = "negatives,
fairly sized". Comparator = exp1's per-class numbers on the identical split.

Same reading rule as exp8b: screening = det_rate_pos + false_alarm as raw
counts, not box P/R. Pair with:
    python Experimenting/eval_fair_negatives.py 5class_original

    python Experimenting/exp8a_5class_negatives.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import build_original_plus_negatives
from common.train_eval import run


def main() -> None:
    spec = build_original_plus_negatives("5class_negatives", binary=False)
    run(spec, "yolov8n.pt")


if __name__ == "__main__":
    main()
