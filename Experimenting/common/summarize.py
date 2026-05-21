"""
summarize.py — Aggregate fold_*/metrics.json into summary.{json,txt}.

For every numeric leaf in a fold's metrics (overall box P/R/F1, image-level
screening, loc-on-hits, per-class for 5-class, match_rule_sweep, and the
val_stock.json mAP cross-reference) we report across folds:

    mean ± std, [min, max], CI95 half-width (= 1.96 * std / sqrt(n))

n is the number of folds present on disk — partial runs aggregate fine.
Idempotent: re-running overwrites summary.{json,txt} from whatever's there.

Headline metrics are surfaced at the top of summary.txt; everything else
follows alphabetically so a diff between two summaries is comprehensible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev


def _ci(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"n": 0}
    if n == 1:
        return {"n": 1, "mean": round(values[0], 4), "std": 0.0,
                "min": round(values[0], 4), "max": round(values[0], 4),
                "ci95_half": 0.0}
    m = mean(values)
    s = stdev(values)
    return {
        "n": n,
        "mean": round(m, 4),
        "std": round(s, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci95_half": round(1.96 * s / math.sqrt(n), 4),
    }


def _walk(d: dict, prefix: str = "") -> dict:
    """Flatten nested numeric leaves into 'a.b.c' = value. Non-numeric leaves
    are skipped (lists, strings, bools — none of which we aggregate)."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_walk(v, key))
        elif isinstance(v, bool):
            continue
        elif isinstance(v, (int, float)):
            out[key] = float(v)
    return out


def _collect(fold_dir: Path) -> dict | None:
    mfile = fold_dir / "metrics.json"
    if not mfile.exists():
        return None
    m = json.loads(mfile.read_text())
    flat = _walk(m.get("metrics", {}), prefix="metrics")

    sweep = m.get("match_rule_sweep") or {}
    for conf_key, rows in sweep.items():
        for row in rows:
            rule = row["rule"].replace(" ", "_")
            flat.update(_walk(row.get("overall", {}),
                              prefix=f"sweep.{conf_key}.{rule}.overall"))
            flat.update(_walk(row.get("localisation_on_hits", {}),
                              prefix=f"sweep.{conf_key}.{rule}.loc"))

    vfile = fold_dir / "val_stock.json"
    if vfile.exists():
        v = json.loads(vfile.read_text())
        if isinstance(v, dict):
            flat.update(_walk({k: val for k, val in v.items()
                                if isinstance(val, (int, float))},
                              prefix="val_stock"))
    return flat


_HEADLINE_KEYS = (
    "metrics.conf_0.25.image_level.screening_accuracy",
    "metrics.conf_0.25.image_level.detection_rate_on_positives",
    "metrics.conf_0.25.image_level.false_alarm_rate_on_negatives",
    "metrics.conf_0.25.overall.precision",
    "metrics.conf_0.25.overall.recall",
    "metrics.conf_0.25.overall.f1",
    "sweep.conf_0.25.iog>=0.5.overall.precision",
    "sweep.conf_0.25.iog>=0.5.overall.recall",
    "sweep.conf_0.25.iog>=0.5.overall.f1",
    "metrics.conf_0.25.localisation_on_hits.mean_iou",
    "metrics.conf_0.25.localisation_on_hits.mean_iog",
    "metrics.conf_0.001.image_level.false_alarm_rate_on_negatives",
    "val_stock.map50",
)


def write_summary(experiment_dir: Path) -> dict:
    """Read every fold_*/metrics.json under experiment_dir, write summary.*."""
    fold_dirs = sorted(
        [p for p in experiment_dir.iterdir()
         if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    rows: list[dict] = []
    for fd in fold_dirs:
        flat = _collect(fd)
        if flat is None:
            continue
        rows.append({"fold": fd.name, **flat})

    if not rows:
        out = {"experiment": experiment_dir.name, "n_folds": 0,
               "folds_included": [], "aggregates": {}, "per_fold": []}
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "summary.json").write_text(json.dumps(out, indent=2))
        (experiment_dir / "summary.txt").write_text(
            f"# {experiment_dir.name}  (0 folds — nothing to summarise yet)\n")
        return out

    keys = sorted({k for r in rows for k in r if k != "fold"})
    agg = {k: _ci([r[k] for r in rows if k in r]) for k in keys}

    out = {
        "experiment": experiment_dir.name,
        "n_folds": len(rows),
        "folds_included": [r["fold"] for r in rows],
        "headline_keys": list(_HEADLINE_KEYS),
        "aggregates": agg,
        "per_fold": rows,
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "summary.json").write_text(json.dumps(out, indent=2))
    (experiment_dir / "summary.txt").write_text(_pretty(out))
    return out


# Compact column spec for the per-fold table. (display_label, flat_key).
# Only the columns the user has asked to see — keep this list short on purpose.
_PERFOLD_COLS = (
    ("screen_acc", "metrics.conf_0.25.image_level.screening_accuracy"),
    ("det_rate",   "metrics.conf_0.25.image_level.detection_rate_on_positives"),
    ("false_alm",  "metrics.conf_0.25.image_level.false_alarm_rate_on_negatives"),
    ("P_iou",      "metrics.conf_0.25.overall.precision"),
    ("R_iou",      "metrics.conf_0.25.overall.recall"),
    ("F1_iou",     "metrics.conf_0.25.overall.f1"),
    ("P_iog",      "sweep.conf_0.25.iog>=0.5.overall.precision"),
    ("R_iog",      "sweep.conf_0.25.iog>=0.5.overall.recall"),
    ("F1_iog",     "sweep.conf_0.25.iog>=0.5.overall.f1"),
    ("mAP50",      "val_stock.map50"),
    ("FA@.001",    "metrics.conf_0.001.image_level.false_alarm_rate_on_negatives"),
)

_HEADLINE_LABELS = {
    "metrics.conf_0.25.image_level.screening_accuracy":              "screening_acc",
    "metrics.conf_0.25.image_level.detection_rate_on_positives":     "det_rate_pos",
    "metrics.conf_0.25.image_level.false_alarm_rate_on_negatives":   "false_alarm_neg",
    "metrics.conf_0.25.overall.precision":                            "box P (iou>=0.5)",
    "metrics.conf_0.25.overall.recall":                               "box R (iou>=0.5)",
    "metrics.conf_0.25.overall.f1":                                   "box F1 (iou>=0.5)",
    "sweep.conf_0.25.iog>=0.5.overall.precision":                     "box P (iog>=0.5)",
    "sweep.conf_0.25.iog>=0.5.overall.recall":                        "box R (iog>=0.5)",
    "sweep.conf_0.25.iog>=0.5.overall.f1":                            "box F1 (iog>=0.5)",
    "metrics.conf_0.25.localisation_on_hits.mean_iou":                "loc IoU (hits)",
    "metrics.conf_0.25.localisation_on_hits.mean_iog":                "loc IoG (hits)",
    "metrics.conf_0.001.image_level.false_alarm_rate_on_negatives":   "FA neg @ conf 0.001",
    "val_stock.map50":                                                "val_stock mAP50",
}

_BEST_BY = "metrics.conf_0.25.image_level.screening_accuracy"


def _averages_table(agg: dict) -> list[str]:
    """Section 1: averages across folds (the headline aggregate)."""
    L = ["# 1. AVERAGES ACROSS FOLDS  (mean ± std,  [min, max],  CI95 half)",
         f"  {'metric':<22} {'n':>2}  {'mean':>7}  {'± std':>7}  "
         f"{'min':>7}  {'max':>7}  {'CI95±':>7}"]
    for k in _HEADLINE_KEYS:
        if k not in agg or agg[k].get("n", 0) == 0:
            continue
        a = agg[k]
        label = _HEADLINE_LABELS.get(k, k.split(".")[-1])
        L.append(
            f"  {label:<22} {a['n']:>2}  "
            f"{a['mean']:>7.3f}  {a['std']:>7.3f}  "
            f"{a['min']:>7.3f}  {a['max']:>7.3f}  {a['ci95_half']:>7.3f}"
        )
    return L


def _perfold_table(rows: list[dict]) -> list[str]:
    """Section 2: per-fold results (only what matters), one row per fold."""
    L = ["# 2. PER FOLD"]
    header = f"  {'fold':>5}  " + "  ".join(f"{lbl:>9}" for lbl, _ in _PERFOLD_COLS)
    L.append(header)
    for r in rows:
        cells = [f"{r['fold'].split('_')[-1]:>5}"]
        for _, key in _PERFOLD_COLS:
            v = r.get(key)
            cells.append(f"{v:>9.3f}" if isinstance(v, (int, float)) else f"{'-':>9}")
        L.append("  " + "  ".join(cells))
    return L


def _best_fold(rows: list[dict]) -> list[str]:
    """Section 3: best fold by screening_acc @ conf 0.25, with its full row."""
    scored = [r for r in rows if isinstance(r.get(_BEST_BY), (int, float))]
    if not scored:
        return ["# 3. BEST FOLD  (no folds with screening_acc to rank)"]
    best = max(scored, key=lambda r: r[_BEST_BY])
    L = [f"# 3. BEST FOLD  (ranked by {_HEADLINE_LABELS[_BEST_BY]} @ conf 0.25)",
         f"  {best['fold']}  →  "
         f"{_HEADLINE_LABELS[_BEST_BY]} = {best[_BEST_BY]:.4f}"]
    for k in _HEADLINE_KEYS:
        v = best.get(k)
        if not isinstance(v, (int, float)):
            continue
        label = _HEADLINE_LABELS.get(k, k.split(".")[-1])
        L.append(f"    {label:<22} {v:.4f}")
    return L


def _pretty(out: dict) -> str:
    L = [f"# {out['experiment']}  ({out['n_folds']} folds: "
         f"{', '.join(out['folds_included'])})", ""]
    agg = out["aggregates"]
    rows = out["per_fold"]
    L += _averages_table(agg)
    L.append("")
    L += _perfold_table(rows)
    L.append("")
    L += _best_fold(rows)
    L.append("")
    L += ["# 4. ALL AGGREGATES  (full flat key list, alphabetical)"]
    for k in sorted(agg):
        if k in _HEADLINE_KEYS:
            continue
        a = agg[k]
        if a.get("n", 0) == 0:
            continue
        L.append(f"  {k:<62} {a['mean']:.4f} ± {a['std']:.4f}")
    L.append("")
    return "\n".join(L)
