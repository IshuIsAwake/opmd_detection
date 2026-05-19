"""CLI — Step 4: train EfficientNet-B2 on detector-emitted crops."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier.train import train  # noqa: E402
from src.common import console, run_dir  # noqa: E402

if __name__ == "__main__":
    run = run_dir.current_run()
    arm = run_dir.run_arm(run)
    console.start("Step 4 · train classifier", run=run.name, arm=arm)
    out = train()
    console.end(
        "Step 4 · train classifier",
        run=run.name, arm=arm,
        best_val_acc=round(out["best_val_acc"], 4),
        weights=out["weights"],
    )
