"""
run_dir.py — One self-contained, human-readable folder per experiment pass.

Layout:
    artifacts/runs/<YYYYmmdd_HHMMSS>_<tag>/
        run.json            manifest: config snapshot + final metrics
        detector/  best.pt  detector.json   (weights + chosen conf + sweep)
        classifier/ best.pt classifier.json (weights + train history)
        eval/      pipeline_report.json  confusion_matrix.png  samples/

    artifacts/CURRENT_RUN   text pointer — the step chain (02→05) writes here
    artifacts/latest -> runs/<...>   symlink — what pipeline.py / app.py serve

Step 02 (detector) starts a new run. Steps 03/04/05 append into that same run
(read via CURRENT_RUN). Training three detector variants therefore yields three
independent, directly comparable run folders. No MLflow, no random hashes.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import config

_SUBDIRS = ("detector", "classifier", "eval")


def _sanitize(name: str) -> str:
    """Filesystem-safe run name. Keep '@' (e.g. yolov8n@640_plus_roboflow);
    strip surrounding whitespace; replace path separators / whitespace runs."""
    name = name.strip().replace("/", "_")
    return "_".join(name.split())


def new_run(tag: str, name: str | None = None) -> Path:
    """
    Create a fresh run dir, point CURRENT_RUN and `latest` at it.

    name given  → used verbatim (only sanitized: whitespace/'/', '@' kept).
    name None   → f"{tag}_{ts}"  (tag is already model-first, e.g.
                  'yolov8n_640'; the timestamp keeps repeat runs distinct).
    """
    if name:
        name = _sanitize(name)
    else:
        name = f"{tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run = config.RUNS_ROOT / name
    for sub in _SUBDIRS:
        (run / sub).mkdir(parents=True, exist_ok=True)
    (run / "eval" / "samples").mkdir(parents=True, exist_ok=True)
    config.CURRENT_RUN_FILE.write_text(name)
    set_latest(run)
    return run


def current_run() -> Path:
    """The run the step chain is operating on (set by step 02)."""
    if not config.CURRENT_RUN_FILE.exists():
        raise FileNotFoundError(
            "No CURRENT_RUN — run scripts/02_train_detector.py first."
        )
    run = config.RUNS_ROOT / config.CURRENT_RUN_FILE.read_text().strip()
    if not run.exists():
        raise FileNotFoundError(f"CURRENT_RUN points at a missing dir: {run}")
    return run


def latest_run() -> Path:
    """The run served by the demo (the `latest` symlink)."""
    if not config.LATEST_LINK.exists():
        raise FileNotFoundError(
            f"{config.LATEST_LINK} missing — train a model first."
        )
    return config.LATEST_LINK.resolve()


def set_latest(run: Path) -> None:
    """Repoint artifacts/latest → run (atomic-ish replace)."""
    link = config.LATEST_LINK
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(run.resolve(), target_is_directory=True)


# ── Well-known paths inside a run (one place that knows the layout) ────────────

def detector_weights(run: Path) -> Path:
    return run / "detector" / "best.pt"


def detector_meta(run: Path) -> Path:
    return run / "detector" / "detector.json"


def classifier_data_dir(run: Path) -> Path:
    """Detector-emitted crops for this run (depend on this run's detector)."""
    return run / "classifier" / "data"


def classifier_weights(run: Path) -> Path:
    return run / "classifier" / "best.pt"


def classifier_meta(run: Path) -> Path:
    return run / "classifier" / "classifier.json"


def eval_dir(run: Path) -> Path:
    return run / "eval"


def manifest_path(run: Path) -> Path:
    return run / "run.json"


def update_manifest(run: Path, section: str, payload: dict) -> None:
    """Merge a section into run.json (config/detector/classifier/eval/...)."""
    path = manifest_path(run)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = payload
    path.write_text(json.dumps(data, indent=2))


def run_arm(run: Path) -> str:
    """
    Which experiment arm this run trained on (written by Step 2 into
    run.json::config.arm). Steps 03/05 read it back to resolve arm-aware
    paths without re-passing --arm. Defaults to original_only if absent.
    """
    path = manifest_path(run)
    if path.exists():
        cfg = json.loads(path.read_text()).get("config", {})
        arm = cfg.get("arm")
        if arm in config.ARMS:
            return arm
    return "original_only"
