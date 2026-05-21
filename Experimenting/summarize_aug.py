"""
summarize_aug.py — One-shot comparison table across exp10 aug-sweep runs.

Reads Experimenting/results/aug_<level>_binary/metrics.json for every level
present (partial sweeps work fine) and prints:

  1. AUG SWEEP TABLE — one row per level, same columns as the kfold per-fold
     table (screen_acc, det_rate, false_alarm, P/R/F1 under both IoU and IoG,
     mAP50, FA@conf 0.001). Compare against the exp8b baseline = the `default`
     row, or the kfold-10 mean reference line at the bottom.
  2. BEST LEVEL — ranked by screening_acc @ conf 0.25.
  3. Δ vs DEFAULT — every level's headline deltas vs the `default` row, so
     "did this level beat exp8b's recipe?" is the first thing you read.

Writes Experimenting/results/aug_sweep_summary.{json,txt}. Idempotent.

    python Experimenting/summarize_aug.py
    python Experimenting/summarize_aug.py --suffix binary       # default
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings

LEVELS_ORDER = ("off", "light", "default", "geom_no_color", "heavy", "heavy_no_color")

_COLS = (
    ("screen_acc", "metrics", "conf_0.25", "image_level", "screening_accuracy"),
    ("det_rate",   "metrics", "conf_0.25", "image_level", "detection_rate_on_positives"),
    ("false_alm",  "metrics", "conf_0.25", "image_level", "false_alarm_rate_on_negatives"),
    ("P_iou",      "metrics", "conf_0.25", "overall", "precision"),
    ("R_iou",      "metrics", "conf_0.25", "overall", "recall"),
    ("F1_iou",     "metrics", "conf_0.25", "overall", "f1"),
    ("FA@.001",    "metrics", "conf_0.001", "image_level", "false_alarm_rate_on_negatives"),
)


def _get(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _iog_row(metrics: dict) -> tuple[float | None, float | None, float | None]:
    """Pull P/R/F1 for iog>=0.5 from match_rule_sweep at conf 0.25."""
    sweep = metrics.get("match_rule_sweep", {}).get("conf_0.25", [])
    for row in sweep:
        if row.get("rule") == "iog>=0.5":
            o = row.get("overall", {})
            return o.get("precision"), o.get("recall"), o.get("f1")
    return None, None, None


def _read_level(level: str, suffix: str) -> dict | None:
    rdir = settings.RESULTS_ROOT / f"aug_{level}_{suffix}"
    mfile = rdir / "metrics.json"
    if not mfile.exists():
        return None
    m = json.loads(mfile.read_text())
    out = {"level": level}
    for col, *keys in _COLS:
        v = _get(m, *keys)
        if isinstance(v, (int, float)):
            out[col] = float(v)
    p_iog, r_iog, f1_iog = _iog_row(m)
    if p_iog is not None:
        out["P_iog"] = p_iog
        out["R_iog"] = r_iog
        out["F1_iog"] = f1_iog
    vfile = rdir / "val_stock.json"
    if vfile.exists():
        v = json.loads(vfile.read_text())
        if isinstance(v.get("map50"), (int, float)):
            out["mAP50"] = float(v["map50"])
    # carry the train_kwargs override so we can echo it in the table
    runj = rdir / "run.json"
    if runj.exists():
        out["_aug_kwargs"] = json.loads(runj.read_text()).get(
            "train_kwargs_override", {})
    return out


_DISPLAY_COLS = ("screen_acc", "det_rate", "false_alm",
                 "P_iou", "R_iou", "F1_iou",
                 "P_iog", "R_iog", "F1_iog",
                 "mAP50", "FA@.001")


def _fmt(v):
    return f"{v:>9.3f}" if isinstance(v, (int, float)) else f"{'-':>9}"


def _table(rows: list[dict]) -> list[str]:
    L = ["# 1. AUG SWEEP TABLE  (binary, exp8b splits)"]
    hdr = f"  {'level':<14}  " + "  ".join(f"{c:>9}" for c in _DISPLAY_COLS)
    L.append(hdr)
    for r in rows:
        cells = [f"{r['level']:<14}"]
        cells += [_fmt(r.get(c)) for c in _DISPLAY_COLS]
        L.append("  " + "  ".join(cells))
    return L


def _best(rows: list[dict]) -> list[str]:
    scored = [r for r in rows if isinstance(r.get("screen_acc"), (int, float))]
    if not scored:
        return ["# 2. BEST LEVEL  (no levels with screening_acc to rank)"]
    best = max(scored, key=lambda r: r["screen_acc"])
    L = [f"# 2. BEST LEVEL  (ranked by screening_acc @ conf 0.25)",
         f"  {best['level']}  →  screening_acc = {best['screen_acc']:.4f}",
         f"  det_rate={best.get('det_rate', float('nan')):.4f}  "
         f"false_alarm={best.get('false_alm', float('nan')):.4f}  "
         f"F1_iog={best.get('F1_iog', float('nan')):.4f}  "
         f"mAP50={best.get('mAP50', float('nan')):.4f}  "
         f"FA@.001={best.get('FA@.001', float('nan')):.4f}"]
    return L


def _delta(rows: list[dict]) -> list[str]:
    by_level = {r["level"]: r for r in rows}
    base = by_level.get("default")
    if base is None:
        return ["# 3. Δ vs DEFAULT  (default level not present; skipping)"]
    L = ["# 3. Δ vs DEFAULT  (positive = better than exp8b's recipe; "
         "false_alm and FA@.001 are inverted: − = better)"]
    hdr = f"  {'level':<14}  " + "  ".join(f"{c:>9}" for c in _DISPLAY_COLS)
    L.append(hdr)
    for r in rows:
        if r["level"] == "default":
            continue
        cells = [f"{r['level']:<14}"]
        for c in _DISPLAY_COLS:
            v, b = r.get(c), base.get(c)
            if isinstance(v, (int, float)) and isinstance(b, (int, float)):
                d = v - b
                if c in ("false_alm", "FA@.001"):
                    d = -d                                  # better = more negative→print as positive
                cells.append(f"{d:>+9.3f}")
            else:
                cells.append(f"{'-':>9}")
        L.append("  " + "  ".join(cells))
    return L


def _aug_kwargs_block(rows: list[dict]) -> list[str]:
    L = ["# 4. AUG CONFIGS (the train_kwargs_override per level, from run.json)"]
    for r in rows:
        cfg = r.get("_aug_kwargs") or {}
        if not cfg:
            L.append(f"  {r['level']:<14}  YOLO defaults (no overrides)")
            continue
        cfg_s = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        L.append(f"  {r['level']:<14}  {cfg_s}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="binary",
                    help="results dir suffix (default 'binary' → aug_<level>_binary/)")
    args = ap.parse_args()

    rows: list[dict] = []
    for level in LEVELS_ORDER:
        r = _read_level(level, args.suffix)
        if r is not None:
            rows.append(r)

    out_path = settings.RESULTS_ROOT / f"aug_sweep_summary_{args.suffix}"
    out_path.mkdir(parents=True, exist_ok=True)

    if not rows:
        msg = (f"# aug sweep summary ({args.suffix}) — no level results found "
               f"under {settings.RESULTS_ROOT}/aug_*_{args.suffix}/\n")
        (out_path / "summary.txt").write_text(msg)
        print(msg)
        return

    L = [f"# aug sweep summary  ({args.suffix})  "
         f"— levels present: {', '.join(r['level'] for r in rows)}",
         f"# reference: exp8b single fold screening_acc 0.865 / "
         f"kfold10 mean 0.840 ± 0.063 (CI95 ±0.039)", ""]
    L += _table(rows); L.append("")
    L += _best(rows);  L.append("")
    L += _delta(rows); L.append("")
    L += _aug_kwargs_block(rows); L.append("")
    text = "\n".join(L)

    (out_path / "summary.txt").write_text(text)
    (out_path / "summary.json").write_text(json.dumps({
        "suffix": args.suffix,
        "levels_present": [r["level"] for r in rows],
        "rows": rows,
    }, indent=2))
    print(text)
    print(f"→ {out_path}/summary.txt")


if __name__ == "__main__":
    main()
