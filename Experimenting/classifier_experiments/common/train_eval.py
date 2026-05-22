"""
train_eval.py — One call per (arm, fold): train the model, write best.pt,
score Phase 1 on the matched test split, dump metrics.{json,txt}.

AdamW + cosine LR schedule + CE loss with inverse-frequency class weights +
early stopping on val macro accuracy.

Backbone-agnostic: the loop talks to the model only through
``trainable_param_groups(base_lr)``, ``trainable_state_dict()``, and
``load_trainable_state(state)``. Both DinoV2Classifier (frozen head) and
EfficientNetB0Classifier (full fine-tune) implement that interface, so
the same loop drives either via the ``model_factory`` argument.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import settings
from common.dataset import (CropFolder, class_weights_from_counts,
                            eval_transform, strong_train_transform,
                            train_transform)
from common.metrics import format_phase1, phase1_report, write_json
from common.model import trainable_params


@dataclass
class TrainConfig:
    epochs: int = settings.EPOCHS
    batch: int = settings.BATCH
    lr: float = settings.LR
    weight_decay: float = settings.WEIGHT_DECAY
    patience: int = settings.EARLY_STOP_PATIENCE
    device: str = settings.DEVICE
    num_workers: int = settings.NUM_WORKERS
    seed: int = settings.SEED
    cosine_schedule: bool = True
    aug_strong: bool = False         # Round-3 augmentation suite


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device
              ) -> tuple[list[int], list[int], list[str]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    stems: list[str] = []
    with torch.no_grad():
        for x, y, stem in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            y_true += list(y.numpy().tolist())
            y_pred += pred
            stems += list(stem)
    return y_true, y_pred, stems


def train_one_fold(arm: str,
                   fold_idx: int,
                   fold_root: Path,
                   results_root: Path,
                   model_factory: Callable[[], nn.Module],
                   cfg: TrainConfig | None = None) -> dict:
    """Train head on fold_root/train, early-stop on fold_root/val, score on
    fold_root/test. Writes best.pt + metrics under results_root/.

    ``model_factory`` is called once per fold to produce a fresh model.
    The model must expose ``trainable_param_groups``,
    ``trainable_state_dict``, ``load_trainable_state`` (see model.py /
    model_b0.py)."""
    cfg = cfg or TrainConfig()
    _seed_everything(cfg.seed + fold_idx)

    tr_tfm = strong_train_transform() if cfg.aug_strong else train_transform()
    train_ds = CropFolder(fold_root / "train", tr_tfm)
    val_ds = CropFolder(fold_root / "val", eval_transform())
    test_ds = CropFolder(fold_root / "test", eval_transform())

    counts = train_ds.class_counts()
    weights = class_weights_from_counts(counts)
    print(f"  fold {fold_idx}: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"             class counts={counts}")
    print(f"             class weights={[round(float(w), 3) for w in weights]}")

    device = torch.device(cfg.device if torch.cuda.is_available()
                          or cfg.device == "cpu" else "cpu")
    pin = device.type == "cuda"

    train_loader = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=pin)

    model = model_factory().to(device)
    n_train = trainable_params(model)
    arch = getattr(model, "arch", model.__class__.__name__)
    print(f"             backbone: {arch}    trainable params: {n_train:,}")

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    param_groups = model.trainable_param_groups(cfg.lr)
    optim = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    sched = None
    if cfg.cosine_schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val_macro = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for x, y, _stem in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running_loss += float(loss.detach()) * x.size(0)
            running_n += x.size(0)
            running_correct += int((logits.argmax(dim=1) == y).sum())

        if sched is not None:
            sched.step()

        train_loss = running_loss / max(running_n, 1)
        train_acc = running_correct / max(running_n, 1)

        y_true, y_pred, _ = _evaluate(model, val_loader, device)
        val_rep = phase1_report(y_true, y_pred)
        val_macro = val_rep["macro_accuracy"]
        val_micro = val_rep["micro_accuracy"]
        dt = time.time() - t0

        lrs = [g["lr"] for g in optim.param_groups]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_macro": val_macro, "val_micro": val_micro,
            "lrs": lrs, "secs": dt,
        })
        print(f"  ep {epoch:3d}/{cfg.epochs}  "
              f"loss {train_loss:.4f}  tr_acc {train_acc:.3f}  "
              f"val_macro {val_macro:.3f}  val_micro {val_micro:.3f}  "
              f"({dt:.1f}s)")

        if val_macro > best_val_macro:
            best_val_macro = val_macro
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.trainable_state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= cfg.patience:
                print(f"             early stop at epoch {epoch} "
                      f"(no val_macro improvement for {cfg.patience} epochs)")
                break

    assert best_state is not None, "training produced no checkpoint"
    model.load_trainable_state(best_state)

    # ── Phase 1 test evaluation on this fold's matched test split ────────────
    y_true, y_pred, stems = _evaluate(model, test_loader, device)
    test_rep = phase1_report(y_true, y_pred)

    results_root.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, results_root / "best.pt")
    write_json(results_root / "phase1_metrics.json", test_rep)
    (results_root / "phase1_metrics.txt").write_text(format_phase1(test_rep) + "\n")
    write_json(results_root / "history.json", {
        "history": history, "best_epoch": best_epoch,
        "best_val_macro": best_val_macro,
    })
    write_json(results_root / "train_config.json", {
        "arm": arm, "fold": fold_idx, "arch": arch,
        "fold_root": str(fold_root), "results_root": str(results_root),
        **asdict(cfg),
        "trainable_params": n_train,
        "class_counts_train": counts,
        "class_weights": [float(w) for w in weights],
    })
    with open(results_root / "phase1_predictions.jsonl", "w") as f:
        for stem, t, p in zip(stems, y_true, y_pred):
            f.write(json.dumps({"stem": stem, "gt": int(t), "pred": int(p)}) + "\n")

    print(f"\n  fold {fold_idx} test (matched, Phase 1):")
    print(format_phase1(test_rep))
    print(f"\n  → wrote {results_root}/")
    return test_rep
