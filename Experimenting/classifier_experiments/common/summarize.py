"""
summarize.py — Roll fold-level Phase 1 metrics into an arm-level summary.

Mirrors Experimenting/common/summarize.py in shape: write a summary.{json,txt}
under the arm's results dir with mean ± std across folds for the headline
numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

from common.metrics import mean_std


def write_phase1_summary(arm_results_root: Path, fold_indices: list[int]) -> Path:
    rows = []
    for k in fold_indices:
        fp = arm_results_root / f"fold_{k}" / "phase1_metrics.json"
        if not fp.exists():
            continue
        rep = json.loads(fp.read_text())
        rows.append({
            "fold": k,
            "n": rep["n"],
            "micro_accuracy": rep["micro_accuracy"],
            "macro_accuracy": rep["macro_accuracy"],
            "per_class_recall": rep["per_class_recall"],
        })

    micro_mean, micro_std = mean_std([r["micro_accuracy"] for r in rows])
    macro_mean, macro_std = mean_std([r["macro_accuracy"] for r in rows])

    classes = rows[0]["per_class_recall"].keys() if rows else []
    per_class_agg = {}
    for c in classes:
        vals = [r["per_class_recall"][c] for r in rows]
        m, s = mean_std(vals)
        per_class_agg[c] = {"mean": m, "std": s, "per_fold": vals}

    summary = {
        "arm_root": str(arm_results_root),
        "folds": [r["fold"] for r in rows],
        "micro_accuracy_mean": micro_mean,
        "micro_accuracy_std": micro_std,
        "macro_accuracy_mean": macro_mean,
        "macro_accuracy_std": macro_std,
        "per_fold": rows,
        "per_class_recall_summary": per_class_agg,
    }

    arm_results_root.mkdir(parents=True, exist_ok=True)
    (arm_results_root / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"folds              : {summary['folds']}",
        f"micro accuracy     : {micro_mean:.4f} ± {micro_std:.4f}",
        f"macro accuracy     : {macro_mean:.4f} ± {macro_std:.4f}",
        "",
        "per-fold:",
        f"  {'fold':>4s}  {'n':>4s}  {'micro':>7s}  {'macro':>7s}",
    ]
    for r in rows:
        lines.append(f"  {r['fold']:>4d}  {r['n']:>4d}  "
                     f"{r['micro_accuracy']:>7.4f}  {r['macro_accuracy']:>7.4f}")
    lines.append("")
    lines.append("per-class recall (mean ± std):")
    for c, agg in per_class_agg.items():
        lines.append(f"  {c:14s} {agg['mean']:.4f} ± {agg['std']:.4f}")
    (arm_results_root / "summary.txt").write_text("\n".join(lines) + "\n")
    return arm_results_root / "summary.txt"
