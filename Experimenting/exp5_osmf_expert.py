"""
Experiment 5 — OSMF EXPERT (Roboflow-trained, OBB).

OSMF DETECTION ships polygons (no OBB/box export). Each osmf polygon
(class 1) is reduced to its minimum-area oriented rectangle — a tight
oriented box derived from the annotation, not hand-drawn — then trained
single-class OBB. Tested on EVERY original image; GT = original OSMF boxes.

    python Experimenting/exp5_osmf_expert.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import RoboflowSource, build_expert
from common.train_eval import run

SOURCES = [
    RoboflowSource("OSMF DETECTION.v1i.yolov8", "poly", frozenset({1}), "osmf"),
]


def main() -> None:
    spec = build_expert("osmf_expert", "OSMF", SOURCES)
    run(spec, "yolov8n-obb.pt")


if __name__ == "__main__":
    main()
