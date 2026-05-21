"""
eval_match_rules.py — How much box recall is the IoU>=0.5 gate hiding?

Pure re-scoring of a FINISHED run's saved best.pt on its OWN test set, under
several box-match rules. No retraining, no GPU, seconds.

Only box P/R/F1 + loc-on-hits change between rules. det_rate_pos /
false_alarm_neg / screening_acc are geometry-free and identical under every
rule — printed ONCE per conf as a reminder that the *screening* recall does
not move; this only re-reads localisation / crop quality.

    python Experimenting/eval_match_rules.py binary_negatives
    python Experimenting/eval_match_rules.py 5class_negatives
    python Experimenting/eval_match_rules.py binary_original     # any run

Weights : results/<run>/train/weights/best.pt   (or a path to a .pt)
Test set: _datasets/<run>/{images,labels}/test  (the tree that run was built
          on — deterministic, already on disk)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import metrics as M
from common import settings


def _resolve(run: str) -> tuple[Path, Path]:
    p = Path(run)
    if p.suffix == ".pt" and p.exists():
        # weights path given; assume dataset name == its run dir name
        name = p.parent.parent.name
        return p, settings.DATASETS_ROOT / name
    w = settings.RESULTS_ROOT / run / "train" / "weights" / "best.pt"
    if not w.exists():
        raise FileNotFoundError(f"no weights for run {run!r} (looked at {w})")
    ds = settings.DATASETS_ROOT / run
    if not (ds / "images" / "test").is_dir():
        raise FileNotFoundError(f"no test tree at {ds}/images/test")
    return w, ds


def _class_names(ds: Path) -> list[str]:
    for line in (ds / "data.yaml").read_text().splitlines():
        if line.startswith("names:"):
            inside = line.split("[", 1)[1].rsplit("]", 1)[0]
            return [s.strip() for s in inside.split(",") if s.strip()]
    return ["lesion"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python Experimenting/eval_match_rules.py "
                  "<run_name|path-to-best.pt>")
    run = sys.argv[1]
    weights, ds = _resolve(run)
    names = _class_names(ds)
    n_classes = len(names)

    from ultralytics import YOLO
    model = YOLO(str(weights))

    test_imgs = sorted((ds / "images" / "test").iterdir(),
                       key=lambda p: p.name)
    per_image = []
    for img in test_imgs:
        res = model.predict(str(img), device=settings.DEVICE, verbose=False,
                            conf=min(M.CONFS), imgsz=settings.IMGSZ)[0]
        w, h = res.orig_shape[1], res.orig_shape[0]
        gts = M._gt_for(ds / "labels" / "test" / f"{img.stem}.txt", w, h)
        per_image.append((gts, M._preds_from_result(res)))

    header = (f"\n# {run}  —  match-rule sweep "
              f"({len(test_imgs)} test imgs, {n_classes}-class)\n"
              f"# weights: {weights}\n"
              f"# det_rate / false_alarm / screening_acc are geometry-free → "
              f"IDENTICAL under every rule (shown once per conf).")
    print(header)

    report: dict = {
        "run": run,
        "weights": str(weights),
        "n_test_images": len(test_imgs),
        "n_classes": n_classes,
        "class_names": names,
        "sweep_by_conf": {},
    }
    txt_lines: list[str] = [header]

    for conf in M.CONFS:
        il = M._score(per_image, n_classes, conf)["image_level"]
        head = (f"\n== conf {conf} =="
                f"  [screening invariant: "
                f"det_rate_pos={il['detection_rate_on_positives']:.3f} "
                f"({il['positives_flagged']}/{il['positives']})  "
                f"false_alarm_neg={il['false_alarm_rate_on_negatives']:.3f} "
                f"({il['negatives_flagged']}/{il['negatives']})  "
                f"screening_acc={il['screening_accuracy']:.3f}]")
        cols = (f"{'rule':<26} {'TP':>3} {'FP':>4} {'FN':>3}  "
                f"{'P':>5} {'R':>5} {'F1':>5}   "
                f"{'IoU':>5} {'IoP':>5} {'IoG':>5}  n")
        print(head)
        print(cols)
        txt_lines += [head, cols]

        rule_rows: list[dict] = []
        for rule in M.MATCH_RULES:
            b = M._score(per_image, n_classes, conf, rule=rule)
            o, loc = b["overall"], b["localisation_on_hits"]
            line = (f"{rule.name:<26} {o['tp']:>3} {o['fp']:>4} {o['fn']:>3}  "
                    f"{o['precision']:>5.3f} {o['recall']:>5.3f} "
                    f"{o['f1']:>5.3f}   "
                    f"{loc['mean_iou']:>5.3f} {loc['mean_iop']:>5.3f} "
                    f"{loc['mean_iog']:>5.3f}  {loc['n']}")
            print(line)
            txt_lines.append(line)
            rule_rows.append({
                "rule": rule.name,
                "overall": o,
                "localisation_on_hits": loc,
            })

        report["sweep_by_conf"][f"conf_{conf}"] = {
            "image_level_invariant": il,
            "rules": rule_rows,
        }

    out_dir = settings.RESULTS_ROOT / run if (settings.RESULTS_ROOT / run).is_dir() \
        else weights.parent
    (out_dir / "metrics_match_rules.json").write_text(
        json.dumps(report, indent=2))
    (out_dir / "metrics_match_rules.txt").write_text("\n".join(txt_lines) + "\n")
    print(f"\n→ wrote {out_dir}/metrics_match_rules.{{json,txt}}")


if __name__ == "__main__":
    main()
