"""
eval_with_negatives.py — Retroactive, un-gameable re-evaluation.

Takes any finished run's saved best.pt and scores it, WITHOUT retraining, on a
negative-bearing single-class test set:

    positives = locked data/test  (37 lesion images, GT = any box → "lesion")
    negatives = data/Normal       (570 healthy images, no GT)

Everything is collapsed to one class ("is there a lesion at all"), so this
works for the 5-class (exp1), binary (exp2/exp7) or OBB runs alike, and is
directly comparable across them. Because negatives are present, a model that
just fires everywhere can no longer hide behind recall — false_alarm_neg and
screening_acc expose it exactly as they did for the experts.

    python Experimenting/eval_with_negatives.py binary_original
    python Experimenting/eval_with_negatives.py binary_plus_roboflow
    python Experimenting/eval_with_negatives.py 5class_original

Writes results/<run>/metrics_with_negatives.{json,txt}. Pure inference,
~seconds; the original metrics.* files are left untouched.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import metrics as M
from common import settings


def _resolve_weights(run: str) -> Path:
    p = Path(run)
    if p.suffix == ".pt" and p.exists():
        return p
    w = settings.RESULTS_ROOT / run / "train" / "weights" / "best.pt"
    if not w.exists():
        raise FileNotFoundError(
            f"no weights for run {run!r} (looked at {w}). "
            f"Pass a run name under results/ or a path to a .pt"
        )
    return w


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python Experimenting/eval_with_negatives.py <run_name|path-to-best.pt>")
    run = sys.argv[1]
    weights = _resolve_weights(run)

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # Build the single-class, negative-bearing test set.
    test_set: list[tuple[Path, Path | None]] = []
    for img in settings.list_images(settings.TEST_IMAGES):              # positives
        test_set.append((img, settings.TEST_LABELS / f"{img.stem}.txt"))
    for d in settings.NORMAL_DIRS:                                      # negatives
        if d.is_dir():
            for img in settings.list_images(d):
                test_set.append((img, None))

    per_image = []
    for img, lbl in test_set:
        res = model.predict(str(img), device=settings.DEVICE,
                             verbose=False, conf=min(M.CONFS),
                             imgsz=settings.IMGSZ)[0]
        w, h = res.orig_shape[1], res.orig_shape[0]
        gts = M._gt_for(lbl, w, h) if lbl is not None else []
        gts = [(0, b) for _, b in gts]                       # collapse → lesion
        preds = [(0, cf, bx) for _, cf, bx in M._preds_from_result(res)]
        per_image.append((gts, preds))

    n_pos = sum(1 for g, _ in per_image if g)
    report = {
        "experiment": f"{run}  [re-eval with negatives]",
        "task": "single-class lesion / no-lesion",
        "class_names": ["lesion(any)"],
        "n_test_images": len(per_image),
        "n_positive_images": n_pos,
        "n_negative_images": len(per_image) - n_pos,
        "weights": str(weights),
        "iou_match_threshold": M.IOU_MATCH,
        "metrics": {f"conf_{c}": M._score(per_image, 1, c) for c in M.CONFS},
    }

    out = settings.RESULTS_ROOT / run if (settings.RESULTS_ROOT / run).is_dir() \
        else weights.parent
    (out / "metrics_with_negatives.json").write_text(json.dumps(report, indent=2))
    (out / "metrics_with_negatives.txt").write_text(M._pretty(report))
    print(f"[{run}] {n_pos} pos / {len(per_image)-n_pos} neg "
          f"→ {out}/metrics_with_negatives.txt\n")
    print((out / "metrics_with_negatives.txt").read_text())


if __name__ == "__main__":
    main()
