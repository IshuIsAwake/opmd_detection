"""CLI — Final: run the locked test/ set through the full pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import console, run_dir  # noqa: E402
from src.evaluate_pipeline import evaluate  # noqa: E402

if __name__ == "__main__":
    run = run_dir.current_run()
    arm = run_dir.run_arm(run)
    console.start("Step 5 · evaluate pipeline", run=run.name, arm=arm)

    report = evaluate()

    console.phase("CONSOLIDATED REPORT — locked 37 + web_holdout")
    console.kv({
        "run": report["run"],
        "arm": report["arm"],
        "locked37_accuracy": report["end_to_end_accuracy"],
        "test_images_evaluated": report["test_images_evaluated"],
        "fallback_used_count": report["fallback_used_count"],
        "detector_conf": report["detector_conf"],
        "match_rule": report["match_rule"],
    })
    console.phase("Per-class recall (locked 37)")
    console.kv(report["per_class_recall"])

    wh = report.get("web_holdout_detector")
    console.phase("web_holdout detector (secondary, higher-N)")
    console.kv(wh if wh else {"status": "absent — run 01 to build web_holdout"})

    print("\n" + json.dumps(report, indent=2))
    console.end("Step 5 · evaluate pipeline", run=run.name, arm=arm)
