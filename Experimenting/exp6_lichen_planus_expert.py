"""
Experiment 6 — LICHEN PLANUS EXPERT (Roboflow-trained, OBB).

Trained ONLY on OPMD-SEG-obb class 1 ("Oral Lichen Planus"), single-class
OBB. Tested on EVERY original image; GT = original Lichen_Planus boxes.

NOTE: known weak split — Roboflow has very little Lichen Planus, so the
original test set is comparable in size to train+val. Accepted on purpose;
the number is reported as-is, not patched.

    python Experimenting/exp6_lichen_planus_expert.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import RoboflowSource, build_expert
from common.train_eval import run

SOURCES = [
    RoboflowSource("OPMD-SEG.v1i.yolov8-obb", "obb", frozenset({1}), "opmdobb"),
]


def main() -> None:
    spec = build_expert("lichen_planus_expert", "Lichen_Planus", SOURCES)
    run(spec, "yolov8n-obb.pt")


if __name__ == "__main__":
    main()
