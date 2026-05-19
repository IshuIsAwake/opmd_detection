"""
CLI — Step 2: start a new run, train the detector for ONE arm, pick conf.

    python scripts/02_train_detector.py --arm original_only --out_dir yolov8n@640_original_only
    python scripts/02_train_detector.py --arm plus_roboflow --out_dir yolov8n@640_plus_roboflow

--out_dir (optional) names the run dir verbatim ('@' kept); omit it and the
run is named '<model>_<imgsz>_<timestamp>'. Steps 03/04/05 then read
CURRENT_RUN and resolve the arm from the run manifest — no need to re-pass it.

Model/imgsz/batch stay env-overridable (DET_MODEL / DET_IMGSZ / DET_BATCH).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.common import console, run_dir  # noqa: E402
from src.detector.evaluate import recommend_conf, sweep  # noqa: E402
from src.detector.train import train  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(config.ARMS))
    ap.add_argument("--out_dir", default=None,
                    help="run dir name (verbatim); default <tag>_<ts>")
    args = ap.parse_args()

    console.start(
        "Step 2 · train detector",
        model=config.DETECTOR_MODEL,
        imgsz=config.DET_IMGSZ,
        batch=config.DET_BATCH,
        arm=args.arm,
        out_dir=args.out_dir or f"{config.detector_tag()}_<ts>",
        smoke=config.SMOKE,
    )

    run, weights = train(args.arm, args.out_dir)
    print(f"\nRun: {run.name}\nDetector weights → {weights}")

    val_imgs = config.det_dataset_dir(args.arm) / "images" / "val"
    val_lbls = config.det_dataset_dir(args.arm) / "labels" / "val"
    rows = sweep(weights, val_imgs, val_lbls)

    console.phase("Val lesion recall sweep (primary = IoP containment)")
    print(f"  {'conf':>5} {'box_recall':>11} {'(iou_ref)':>10} {'image_recall':>13}")
    for r in rows:
        print(f"  {r['conf']:>5} {r['box_recall']:>11} "
              f"{r['box_recall_iou']:>10} {r['image_recall']:>13}")

    chosen = recommend_conf(rows)
    run_dir.detector_meta(run).write_text(
        json.dumps({"conf": chosen, "sweep": rows}, indent=2)
    )
    run_dir.update_manifest(run, "detector", {"chosen_conf": chosen, "sweep": rows})

    console.end(
        "Step 2 · train detector",
        run=run.name, arm=args.arm, chosen_conf=chosen,
    )
