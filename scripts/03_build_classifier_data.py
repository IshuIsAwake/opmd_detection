"""
CLI — Step 3: emit classifier crops from the current run's detector.

Reads CURRENT_RUN; the arm is resolved from the run manifest (set by Step 2).
Only original pool/ images are cropped — NO Roboflow image ever reaches the
classifier in this change.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.classifier.build_data import build  # noqa: E402
from src.common import console, run_dir  # noqa: E402

if __name__ == "__main__":
    run = run_dir.current_run()
    arm = run_dir.run_arm(run)
    meta = run_dir.detector_meta(run)
    conf = (
        json.loads(meta.read_text())["conf"]
        if meta.exists()
        else config.DET_CONF
    )
    console.start(
        "Step 3 · build classifier data",
        run=run.name, arm=arm, detector_conf=conf,
    )
    stats = build(conf=conf)
    print(json.dumps(stats, indent=2))
    console.end("Step 3 · build classifier data", run=run.name, arm=arm)
