"""
sweep_conf_threshold.py — Post-hoc confidence threshold sweep.

Inference-only re-analysis of a finished kfold OR single-run experiment. For
every fold's saved best.pt (or, for single-run experiments, the run's best.pt),
re-runs prediction at conf=0.001 so every detection above the model's noise
floor is collected, then for a sweep of conf thresholds recomputes the
image-level screening triple (screening_acc, det_rate_pos, false_alarm_neg).

This is the cheapest possible "calibration": we don't recalibrate the
probabilities themselves, we just find a better operating threshold on the
model we already have. For a screener that only cares about "fire / don't
fire" decisions, this captures most of what proper temperature scaling would
give. No retraining, no GPU training — only inference at conf=0.001.

Works on both shapes:
  * kfold experiments  → results/<exp>/fold_*/train/weights/best.pt
  * single-run aug exp → results/<exp>/train/weights/best.pt

Usage:
    python Experimenting/sweep_conf_threshold.py kfold5_geom_no_color_binary
    python Experimenting/sweep_conf_threshold.py aug_geom_no_color_binary
    python Experimenting/sweep_conf_threshold.py kfold5_geom_no_color_binary \\
        --thresholds 0.20 0.25 0.30 0.35 0.40 0.45

Writes results/<exp>/threshold_sweep.{json,txt}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80)


def _has_gt(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text().strip().splitlines():
        if len(line.split()) == 5:
            return True
    return False


def _resolve_dataset_root(run_dir: Path, fallback_name: str) -> Path:
    """run.json stores dataset_spec_name (e.g. 'kfold5_<aug>_binary/fold_0' or
    'binary_negatives'). Use it to find the YOLO tree this run was built on."""
    rj = run_dir / "run.json"
    if rj.exists():
        d = json.loads(rj.read_text())
        ds_name = d.get("dataset_spec_name") or d.get("experiment") or fallback_name
    else:
        ds_name = fallback_name
    return settings.DATASETS_ROOT / ds_name


def _sweep_one(weights: Path, ds_root: Path, thresholds: tuple) -> dict:
    """Run inference at conf=0.001 over the test split, then per threshold
    compute image-level screening metrics. Returns rows + per-image diagnostics."""
    from ultralytics import YOLO
    model = YOLO(str(weights))

    img_dir = ds_root / "images" / "test"
    lbl_dir = ds_root / "labels" / "test"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"no test images at {img_dir}")

    per_image: list[dict] = []
    for img in sorted(img_dir.iterdir(), key=lambda p: p.name):
        if not img.is_file():
            continue
        res = model.predict(str(img), device=settings.DEVICE, verbose=False,
                            conf=0.001, imgsz=settings.IMGSZ)[0]
        boxes = res.boxes
        max_conf = 0.0
        if boxes is not None and len(boxes):
            max_conf = float(boxes.conf.max().cpu())
        per_image.append({
            "img": img.name,
            "max_conf": max_conf,
            "has_gt": _has_gt(lbl_dir / f"{img.stem}.txt"),
        })

    rows = []
    for t in thresholds:
        tp = fn = fp = tn = 0
        for r in per_image:
            fired = r["max_conf"] >= t
            if r["has_gt"]:
                if fired: tp += 1
                else: fn += 1
            else:
                if fired: fp += 1
                else: tn += 1
        pos, neg = tp + fn, fp + tn
        rows.append({
            "threshold": t,
            "screening_acc": (tp + tn) / (pos + neg) if (pos + neg) else 0.0,
            "det_rate_pos": tp / pos if pos else 0.0,
            "false_alarm_neg": fp / neg if neg else 0.0,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "positives": pos, "negatives": neg,
        })
    return {"rows": rows, "per_image_n": len(per_image)}


def _agg(values: list) -> tuple | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0: return None
    if n == 1: return (vals[0], 0.0)
    return (mean(vals), stdev(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment dir name under results/")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=list(DEFAULT_THRESHOLDS),
                    help="conf thresholds to sweep (defaults to a wide grid)")
    args = ap.parse_args()

    results_root = settings.RESULTS_ROOT / args.experiment
    if not results_root.is_dir():
        raise SystemExit(f"no such results dir: {results_root}")

    # Discover folds: either fold_* subdirs OR the results_root itself if it
    # contains train/weights/best.pt (single-run experiments like aug_*_binary).
    fold_dirs = sorted(
        [p for p in results_root.iterdir()
         if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    if not fold_dirs:
        if (results_root / "train" / "weights" / "best.pt").exists():
            fold_dirs = [results_root]
        else:
            raise SystemExit(f"no fold_* subdirs and no best.pt under {results_root}")

    thresholds = tuple(sorted(set(round(t, 4) for t in args.thresholds)))
    print(f"\n# Conf-threshold sweep: {args.experiment}")
    print(f"# Thresholds: {list(thresholds)}")
    print(f"# Folds to sweep: {len(fold_dirs)}\n")

    per_fold = {}
    for fd in fold_dirs:
        weights = fd / "train" / "weights" / "best.pt"
        if not weights.exists():
            print(f"  [{fd.name}] missing best.pt — skipping")
            continue
        ds_root = _resolve_dataset_root(fd, args.experiment)
        result = _sweep_one(weights, ds_root, thresholds)
        per_fold[fd.name] = result
        print(f"  [{fd.name}] swept {result['per_image_n']} test imgs  "
              f"(data: {ds_root.name})")

    if not per_fold:
        raise SystemExit("no folds produced predictions; nothing to aggregate")

    # Per-threshold aggregation
    agg_rows = []
    for t in thresholds:
        screens, dets, falses = [], [], []
        for data in per_fold.values():
            for r in data["rows"]:
                if abs(r["threshold"] - t) < 1e-9:
                    screens.append(r["screening_acc"])
                    dets.append(r["det_rate_pos"])
                    falses.append(r["false_alarm_neg"])
                    break
        a_s = _agg(screens) or (None, None)
        a_d = _agg(dets)    or (None, None)
        a_f = _agg(falses)  or (None, None)
        agg_rows.append({
            "threshold": t, "n_folds": len(screens),
            "screen_mean": a_s[0], "screen_std": a_s[1],
            "det_mean": a_d[0],    "det_std": a_d[1],
            "false_mean": a_f[0],  "false_std": a_f[1],
        })

    # Baseline reference at conf=0.25
    ref_row = next((r for r in agg_rows
                    if abs(r["threshold"] - 0.25) < 1e-9), None)
    best = max((r for r in agg_rows if r["screen_mean"] is not None),
               key=lambda r: r["screen_mean"])

    # Print + save
    L = [
        f"# Conf-threshold sweep: {args.experiment}",
        f"# Folds swept: {', '.join(per_fold.keys())}  ({len(per_fold)} folds)",
        "",
        "# Aggregate (mean ± std across folds, image-level metrics)",
        f"  {'thresh':>7}  {'screen_acc':>16}  {'det_rate':>16}  {'false_alm':>16}",
    ]
    for r in agg_rows:
        if r["screen_mean"] is None:
            continue
        marker = ""
        if ref_row is not None and r["threshold"] == ref_row["threshold"]:
            marker += "  <- current default"
        if r["threshold"] == best["threshold"]:
            marker += "  ⭐ BEST by screening"
        L.append(
            f"  {r['threshold']:>7.3f}  "
            f"{r['screen_mean']:>7.3f} ± {r['screen_std']:>5.3f}  "
            f"{r['det_mean']:>7.3f} ± {r['det_std']:>5.3f}  "
            f"{r['false_mean']:>7.3f} ± {r['false_std']:>5.3f}"
            f"{marker}"
        )
    L.append("")

    L.append("# Headline")
    if ref_row is not None and ref_row["screen_mean"] is not None:
        L.append(
            f"  conf=0.250 (current): screening {ref_row['screen_mean']:.3f} "
            f"± {ref_row['screen_std']:.3f}  det {ref_row['det_mean']:.3f}  "
            f"false {ref_row['false_mean']:.3f}"
        )
    L.append(
        f"  conf={best['threshold']:.3f} (best): "
        f"screening {best['screen_mean']:.3f} ± {best['screen_std']:.3f}  "
        f"det {best['det_mean']:.3f}  false {best['false_mean']:.3f}"
    )
    if ref_row is not None and ref_row["screen_mean"] is not None:
        d = best["screen_mean"] - ref_row["screen_mean"]
        L.append(f"  Δ vs current 0.25: {d:+.3f} screening_acc")
    L.append("")

    text = "\n".join(L)
    print()
    print(text)

    (results_root / "threshold_sweep.json").write_text(json.dumps({
        "experiment": args.experiment,
        "thresholds": list(thresholds),
        "aggregate": agg_rows,
        "per_fold": per_fold,
        "best_by_screening_acc": best,
        "reference_at_0.25": ref_row,
    }, indent=2, default=float))
    (results_root / "threshold_sweep.txt").write_text(text + "\n")
    print(f"→ {results_root}/threshold_sweep.txt")


if __name__ == "__main__":
    main()
