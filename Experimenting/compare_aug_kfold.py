"""
compare_aug_kfold.py — Head-to-head between two kfold experiments.

Reads each experiment's per-fold metrics.json + val_stock.json and prints:

  1. AGGREGATE     — mean ± std per metric, side by side, with Δ
  2. PER-FOLD      — fold k of A vs fold k of B (same split — that's the point), Δ
  3. WIN COUNTS    — how many folds A beats B on each metric

A's screening / det / box / mAP metrics with + Δ = A better. `false_alm` and
`FA@.001` are sign-flipped on the Δ column so positive is always "A better".

Use after both 5-fold runs finish (in either order):
    python Experimenting/compare_aug_kfold.py \\
        kfold5_geom_no_color_binary kfold5_heavy_no_color_binary

Writes Experimenting/results/compare__<A>__vs__<B>/comparison.{json,txt}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import settings


# (display_name, keys_into_metrics_json, invert_for_winner)
_METRICS: tuple[tuple[str, tuple, bool], ...] = (
    ("screen_acc", ("metrics", "conf_0.25", "image_level", "screening_accuracy"), False),
    ("det_rate",   ("metrics", "conf_0.25", "image_level", "detection_rate_on_positives"), False),
    ("false_alm",  ("metrics", "conf_0.25", "image_level", "false_alarm_rate_on_negatives"), True),
    ("P_iou",      ("metrics", "conf_0.25", "overall", "precision"), False),
    ("R_iou",      ("metrics", "conf_0.25", "overall", "recall"), False),
    ("F1_iou",     ("metrics", "conf_0.25", "overall", "f1"), False),
    # P/R/F1 under iog>=0.5 are pulled from match_rule_sweep, not metrics
    ("P_iog",      None, False),
    ("R_iog",      None, False),
    ("F1_iog",     None, False),
    ("mAP50",      ("__val_stock__", "map50"), False),
    ("FA@.001",    ("metrics", "conf_0.001", "image_level", "false_alarm_rate_on_negatives"), True),
)


def _get(d: dict, keys: tuple):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _iog_overall(metrics_json: dict) -> tuple:
    sweep = metrics_json.get("match_rule_sweep", {}).get("conf_0.25", [])
    for row in sweep:
        if row.get("rule") == "iog>=0.5":
            o = row.get("overall", {})
            return o.get("precision"), o.get("recall"), o.get("f1")
    return None, None, None


def _per_fold(experiment_dir: Path) -> list[dict]:
    fold_dirs = sorted(
        [p for p in experiment_dir.iterdir()
         if p.is_dir() and p.name.startswith("fold_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    rows: list[dict] = []
    for fd in fold_dirs:
        mfile = fd / "metrics.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text())
        vfile = fd / "val_stock.json"
        v = json.loads(vfile.read_text()) if vfile.exists() else {}
        row: dict = {"fold": fd.name}
        for name, keys, _ in _METRICS:
            if keys is None:
                continue
            if keys[0] == "__val_stock__":
                row[name] = v.get(keys[1])
            else:
                row[name] = _get(m, keys)
        p_iog, r_iog, f1_iog = _iog_overall(m)
        row["P_iog"], row["R_iog"], row["F1_iog"] = p_iog, r_iog, f1_iog
        rows.append(row)
    return rows


def _agg(values: list) -> tuple | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    n = len(vals)
    if n == 0:
        return None
    if n == 1:
        return (vals[0], 0.0)
    return (mean(vals), stdev(vals))


def _aggregate_section(rA: list[dict], rB: list[dict], lA: str, lB: str) -> list[str]:
    L = ["# 1. AGGREGATE  (mean ± std,  Δ = A − B,  + on Δ = A better)",
         f"  {'metric':<12}  {lA + ' mean':>14} {lA + ' std':>10}    "
         f"{lB + ' mean':>14} {lB + ' std':>10}    {'Δ':>8}"]
    for name, _, invert in _METRICS:
        a = _agg([r.get(name) for r in rA])
        b = _agg([r.get(name) for r in rB])
        if a is None or b is None:
            continue
        d = (a[0] - b[0]) * (-1 if invert else 1)
        L.append(f"  {name:<12}  {a[0]:>14.3f} {a[1]:>10.3f}    "
                 f"{b[0]:>14.3f} {b[1]:>10.3f}    {d:>+8.3f}")
    return L


def _perfold_section(rA: list[dict], rB: list[dict]) -> list[str]:
    L = ["# 2. PER-FOLD HEAD-TO-HEAD  (same fold = same split; Δ in 'A better' sign)"]
    cols = (("screen_acc", False), ("det_rate", False),
            ("false_alm", True),   ("F1_iog", False),
            ("mAP50", False))
    hdr = f"  {'fold':>5}  " + "  ".join(
        f"{'A_' + c:>10} {'B_' + c:>10} {'Δ':>7}" for c, _ in cols
    )
    L.append(hdr)
    n = min(len(rA), len(rB))
    for i in range(n):
        cells = [f"{rA[i]['fold']:>5}"]
        for c, invert in cols:
            va, vb = rA[i].get(c), rB[i].get(c)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = (va - vb) * (-1 if invert else 1)
                cells.append(f"{va:>10.3f} {vb:>10.3f} {d:>+7.3f}")
            else:
                cells.append(f"{'-':>10} {'-':>10} {'-':>7}")
        L.append("  " + "  ".join(cells))
    return L


def _wins_section(rA: list[dict], rB: list[dict], lA: str, lB: str) -> list[str]:
    L = ["# 3. WIN COUNTS  (per metric; ties go to B)"]
    n = min(len(rA), len(rB))
    for name, _, invert in _METRICS:
        wins_a = valid = 0
        for i in range(n):
            va, vb = rA[i].get(name), rB[i].get(name)
            if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                continue
            valid += 1
            better_a = (va > vb) if not invert else (va < vb)
            if better_a:
                wins_a += 1
        if valid:
            L.append(f"  {name:<12}  {lA} wins {wins_a}/{valid}   "
                     f"{lB} wins {valid - wins_a}/{valid}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("A", help="experiment dir name A (under results/)")
    ap.add_argument("B", help="experiment dir name B (under results/)")
    ap.add_argument("--label-a", default=None,
                    help="short label for A in tables (default = A)")
    ap.add_argument("--label-b", default=None,
                    help="short label for B in tables (default = B)")
    args = ap.parse_args()

    pA = settings.RESULTS_ROOT / args.A
    pB = settings.RESULTS_ROOT / args.B
    if not pA.is_dir(): raise SystemExit(f"no such results dir: {pA}")
    if not pB.is_dir(): raise SystemExit(f"no such results dir: {pB}")

    rA, rB = _per_fold(pA), _per_fold(pB)
    if not rA or not rB:
        raise SystemExit(f"empty fold sets — A:{len(rA)}  B:{len(rB)}")

    lA = args.label_a or args.A.split("kfold5_")[-1].replace("_binary", "") or "A"
    lB = args.label_b or args.B.split("kfold5_")[-1].replace("_binary", "") or "B"
    lA, lB = lA[:8], lB[:8]                          # keep table tidy

    L = [f"# head-to-head: {args.A}  vs  {args.B}",
         f"# A = {lA}   B = {lB}   |  folds: A={len(rA)} B={len(rB)}", ""]
    L += _aggregate_section(rA, rB, lA, lB);  L.append("")
    L += _perfold_section(rA, rB);            L.append("")
    L += _wins_section(rA, rB, lA, lB);       L.append("")
    text = "\n".join(L)

    safe = lambda s: s.replace("/", "_")
    out_dir = settings.RESULTS_ROOT / f"compare__{safe(args.A)}__vs__{safe(args.B)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.txt").write_text(text)
    (out_dir / "comparison.json").write_text(json.dumps({
        "A": {"experiment": args.A, "label": lA, "per_fold": rA},
        "B": {"experiment": args.B, "label": lB, "per_fold": rB},
    }, indent=2))
    print(text)
    print(f"→ {out_dir}/comparison.txt")


if __name__ == "__main__":
    main()
