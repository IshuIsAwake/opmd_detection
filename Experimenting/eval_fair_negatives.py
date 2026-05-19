"""
eval_fair_negatives.py — "Number A": the un-retrained fair re-score.

Takes a finished run's saved best.pt and scores it, WITHOUT retraining, on
exactly the fair test exp8a/exp8b are evaluated on:

    positives = locked data/test            (37 lesion images)
    negatives = 37 RESOLUTION-NORMALISED Normal images
                (the deterministic test slice from fair_negative_split — the
                 SAME 37 files exp8a/exp8b see, so this is directly comparable)

Everything is collapsed to one class ("is there a lesion at all"), so this
works for 5-class (exp1) or binary (exp2) weights alike.

This separates the two effects the audit conflated:
  * old-37-only  → THIS (A) : the *measurement* effect (a fair ruler)
  * (A)          → exp8x     : the *negative-training* effect

    python Experimenting/eval_fair_negatives.py binary_original
    python Experimenting/eval_fair_negatives.py 5class_original

Writes results/<run>/metrics_fair_negatives.{json,txt}. Pure inference,
~seconds; the original metrics.* files are left untouched.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import metrics as M
from common import settings
from common.datasets import fair_negative_split
from common.negatives import positive_target_size


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
        sys.exit("usage: python Experimenting/eval_fair_negatives.py "
                 "<run_name|path-to-best.pt>")
    run = sys.argv[1]
    weights = _resolve_weights(run)

    from ultralytics import YOLO
    model = YOLO(str(weights))

    # Fair test = locked 37 positives + the SAME 37 normalised negatives
    # exp8a/exp8b are scored on (deterministic shared split).
    _, _, te_neg, _ = fair_negative_split()
    test_set: list[tuple[Path, Path | None]] = []
    for img in settings.list_images(settings.TEST_IMAGES):              # pos
        test_set.append((img, settings.TEST_LABELS / f"{img.stem}.txt"))
    for img in te_neg:                                                  # neg
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
        "experiment": f"{run}  [Number A — fair (normalised 1:1) negatives]",
        "task": "single-class lesion / no-lesion",
        "class_names": ["lesion(any)"],
        "n_test_images": len(per_image),
        "n_positive_images": n_pos,
        "n_negative_images": len(per_image) - n_pos,
        "negative_target_long_side": positive_target_size(),
        "weights": str(weights),
        "iou_match_threshold": M.IOU_MATCH,
        "metrics": {f"conf_{c}": M._score(per_image, 1, c) for c in M.CONFS},
    }

    out = settings.RESULTS_ROOT / run if (settings.RESULTS_ROOT / run).is_dir() \
        else weights.parent
    (out / "metrics_fair_negatives.json").write_text(
        json.dumps(report, indent=2))
    (out / "metrics_fair_negatives.txt").write_text(M._pretty(report))
    print(f"[{run}] {n_pos} pos / {len(per_image) - n_pos} fair-neg "
          f"→ {out}/metrics_fair_negatives.txt\n")
    print((out / "metrics_fair_negatives.txt").read_text())


if __name__ == "__main__":
    main()
