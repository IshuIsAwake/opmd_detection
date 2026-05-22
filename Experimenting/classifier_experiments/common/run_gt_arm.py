"""
run_gt_arm.py — Shared body of exp2{a,b,c}_gt_pad*.py. The three pad-fraction
arms differ only by ``pad`` and the arm name; everything else is identical.
"""

from __future__ import annotations

from common import settings
from common.model import _load_dinov2
from common.summarize import write_phase1_summary
from common.train_eval import TrainConfig, train_one_fold


def run_gt_arm(pad: float, arm_name: str,
               fold: int | None = None, write_summary: bool = True) -> None:
    data_root = settings.DATASETS_ROOT / f"gt_pad_{pad:.2f}"
    results_root = settings.RESULTS_ROOT / arm_name

    if not data_root.exists():
        raise SystemExit(
            f"{data_root} not found. Run "
            f"`python Experimenting/classifier_experiments/common/materialise.py "
            f"--arms {pad:.2f}` first.")

    folds = [fold] if fold is not None else list(settings.ALL_FOLDS)
    print(f"\n══ Round 1 / {arm_name} — GT crop, pad={pad:.2f} ══")
    print(f"   data:    {data_root}")
    print(f"   results: {results_root}")
    print(f"   folds:   {folds}\n")

    backbone = _load_dinov2()
    cfg = TrainConfig()
    for k in folds:
        train_one_fold(
            arm=arm_name, fold_idx=k,
            fold_root=data_root / f"fold_{k}",
            results_root=results_root / f"fold_{k}",
            cfg=cfg, backbone=backbone,
        )

    if write_summary and fold is None:
        out = write_phase1_summary(results_root, list(settings.ALL_FOLDS))
        print(f"\n→ {out}\n")
        print(out.read_text())
