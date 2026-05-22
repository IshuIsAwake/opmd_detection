"""
run_arm.py — Shared body of the per-arm entry scripts (whole-image + the
three GT-crop pads, × both backbones).

Builds the (data_root, results_root) pair from arm + backbone, instantiates a
fresh model per fold via the supplied factory, and drives the per-fold loop.
Results land at ``results/<arm>/<backbone-suffix>/fold_k/``; the suffix is
empty for DINOv2 (Round 1's existing layout) and ``_b0`` for B0 (Round 2),
so the two backbones never overwrite each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch.nn as nn

from common import settings
from common.summarize import write_phase1_summary
from common.train_eval import TrainConfig, train_one_fold


def _arm_paths(arm_base: str, pad: float | None, backbone_tag: str
               ) -> tuple[Path, Path]:
    """Return (data_root, results_root) for a given arm + backbone."""
    if arm_base == "whole_image":
        data_root = settings.DATASETS_ROOT / "whole"
        arm_results = "whole_image"
    elif arm_base == "gt_pad":
        if pad is None:
            raise ValueError("gt_pad arms require a pad value")
        data_root = settings.DATASETS_ROOT / f"gt_pad_{pad:.2f}"
        arm_results = f"gt_pad_{pad:.2f}"
    else:
        raise ValueError(f"unknown arm_base: {arm_base}")

    suffix = "" if backbone_tag == "dinov2" else f"_{backbone_tag}"
    results_root = settings.RESULTS_ROOT / f"{arm_results}{suffix}"
    return data_root, results_root


def run_arm(arm_base: str,
            backbone_tag: str,
            model_factory: Callable[[], nn.Module],
            pad: float | None = None,
            fold: int | None = None,
            write_summary: bool = True,
            cfg: TrainConfig | None = None) -> None:
    data_root, results_root = _arm_paths(arm_base, pad, backbone_tag)

    if not data_root.exists():
        materialise_arms = "whole" if arm_base == "whole_image" else f"{pad:.2f}"
        raise SystemExit(
            f"{data_root} not found. Run "
            f"`python Experimenting/classifier_experiments/common/materialise.py "
            f"--arms {materialise_arms}` first.")

    folds = [fold] if fold is not None else list(settings.ALL_FOLDS)
    arm_label = results_root.name
    pad_str = f"pad={pad:.2f}" if pad is not None else "whole image"
    print(f"\n══ {arm_label} — {pad_str}, backbone={backbone_tag} ══")
    print(f"   data:    {data_root}")
    print(f"   results: {results_root}")
    print(f"   folds:   {folds}\n")

    cfg = cfg or TrainConfig()
    for k in folds:
        train_one_fold(
            arm=arm_label, fold_idx=k,
            fold_root=data_root / f"fold_{k}",
            results_root=results_root / f"fold_{k}",
            model_factory=model_factory,
            cfg=cfg,
        )

    if write_summary and fold is None:
        out = write_phase1_summary(results_root, list(settings.ALL_FOLDS))
        print(f"\n→ {out}\n")
        print(out.read_text())
