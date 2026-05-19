"""
CLI — control vs treatment, side by side.

    python scripts/06_compare_runs.py \
        artifacts/runs/yolov8n@640_original_only \
        artifacts/runs/yolov8n@640_plus_roboflow

Reads each run's run.json (eval + detector sections). The valid comparison is
control (original_only) vs treatment (plus_roboflow) — the old 54.1% predates
this code and is NOT the baseline.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.common import console  # noqa: E402


def _load(run_dir: Path) -> dict:
    rj = run_dir / "run.json"
    if not rj.exists():
        raise FileNotFoundError(f"{rj} missing — has this run been evaluated?")
    return json.loads(rj.read_text())


def _row(label: str, a, b) -> str:
    def f(v):
        return "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    return f"  {label:<22} {f(a):>20} {f(b):>20}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    args = ap.parse_args()

    A, B = _load(Path(args.run_a)), _load(Path(args.run_b))
    ea, eb = A.get("eval", {}), B.get("eval", {})
    da, db = A.get("detector", {}), B.get("detector", {})
    ca, cb = A.get("config", {}), B.get("config", {})

    console.phase("CONTROL vs TREATMENT")
    print(_row("", Path(args.run_a).name, Path(args.run_b).name))
    print(_row("arm", ca.get("arm"), cb.get("arm")))
    print(_row("locked37_accuracy",
               ea.get("end_to_end_accuracy"), eb.get("end_to_end_accuracy")))
    print(_row("fallback_used", ea.get("fallback_used_count"),
               eb.get("fallback_used_count")))
    print(_row("recommend_conf", da.get("chosen_conf"), db.get("chosen_conf")))

    console.phase("Per-class recall (locked 37)")
    pa = ea.get("per_class_recall", {}) or {}
    pb = eb.get("per_class_recall", {}) or {}
    for cls in config.CLASS_NAMES:
        print(_row(cls, pa.get(cls), pb.get(cls)))

    console.phase("web_holdout detector (secondary, higher-N)")
    wa = ea.get("web_holdout_detector") or {}
    wb = eb.get("web_holdout_detector") or {}
    print(_row("map50", wa.get("map50"), wb.get("map50")))
    print(_row("recall", wa.get("recall"), wb.get("recall")))
    print(_row("images", wa.get("images"), wb.get("images")))
    print()
