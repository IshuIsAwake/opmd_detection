"""
Experiment 3 — LEUKOPLAKIA EXPERT (Roboflow-trained, OBB).

Trained ONLY on Roboflow Leukoplakia images, single-class OBB:
  * Leukoplakia.v2i.yolov8   (axis boxes, class 0 → reformatted to OBB)
  * OPMD-SEG.v1i.yolov8-obb  (OBB, class 0 → passthrough)
Tested on EVERY original image (pool + test + Normal); GT = original
Leukoplakia boxes, every other image is a negative.

    python Experimenting/exp3_leukoplakia_expert.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.datasets import RoboflowSource, build_expert
from common.train_eval import run

SOURCES = [
    RoboflowSource("Leukoplakia.v2i.yolov8", "box", frozenset({0}), "leukov2"),
    RoboflowSource("OPMD-SEG.v1i.yolov8-obb", "obb", frozenset({0}), "opmdobb"),
]


def main() -> None:
    spec = build_expert("leukoplakia_expert", "Leukoplakia", SOURCES)
    run(spec, "yolov8n-obb.pt")


if __name__ == "__main__":
    main()
