"""
Experiment 4 — ERYTHROPLAKIA EXPERT (Roboflow-trained, OBB).

Trained ONLY on OPMD-SEG-obb class 2 (erythroplakia), single-class OBB.
Tested on EVERY original image; GT = original Erythroplakia boxes.

    python Experimenting/exp4_erythroplakia_expert.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import RoboflowSource, build_expert
from common.train_eval import run

SOURCES = [
    RoboflowSource("OPMD-SEG.v1i.yolov8-obb", "obb", frozenset({2}), "opmdobb"),
]


def main() -> None:
    spec = build_expert("erythroplakia_expert", "Erythroplakia", SOURCES)
    run(spec, "yolov8n-obb.pt")


if __name__ == "__main__":
    main()
