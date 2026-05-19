"""
Step 4 — Train EfficientNet-B2 (5-class) on detector-emitted crops.

2-phase fine-tune:
  phase 1: backbone frozen, lr 1e-3, ~5 epochs
  phase 2: full network,    lr 2e-4, ~25 epochs
AdamW, CrossEntropy with inverse-frequency class weights (data is imbalanced;
Erythroplakia / OSMF are the smallest). Best val-accuracy checkpoint is saved.

The headline number is NOT classifier-val — that comes from the locked test/
set through the full pipeline (src/evaluate_pipeline.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import json

import config
from src.classifier.dataset import CropDataset
from src.classifier.model import build_model, set_backbone_frozen
from src.common import run_dir


def _inverse_freq_weights(counts: list[int]) -> torch.Tensor:
    total = sum(counts)
    n = len(counts)
    w = torch.zeros(n)
    for i, c in enumerate(counts):
        w[i] = total / (n * c) if c > 0 else 0.0
    return w


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        if train:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train() -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run = run_dir.current_run()
    train_ds = CropDataset("train", train_aug=True, run=run)
    val_ds = CropDataset("val", train_aug=False, run=run)
    train_ld = DataLoader(
        train_ds, batch_size=config.CLF_BATCH, shuffle=True,
        num_workers=config.CLF_NUM_WORKERS, pin_memory=(device == "cuda"),
    )
    val_ld = DataLoader(
        val_ds, batch_size=config.CLF_BATCH, shuffle=False,
        num_workers=config.CLF_NUM_WORKERS, pin_memory=(device == "cuda"),
    )

    model = build_model(pretrained=True).to(device)
    weights = _inverse_freq_weights(train_ds.class_counts()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    weights_path = run_dir.classifier_weights(run)
    best_val_acc = -1.0
    history = []

    def checkpoint(epoch, val_acc):
        torch.save(
            {
                "state_dict": model.state_dict(),
                "class_names": config.CLASS_NAMES,
                "img_size": config.CLF_IMG_SIZE,
                "epoch": epoch,
                "val_acc": val_acc,
            },
            weights_path,
        )

    # ── Phase 1: frozen backbone ─────────────────────────────────────────────
    set_backbone_frozen(model, frozen=True)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CLF_FREEZE_LR, weight_decay=config.CLF_WEIGHT_DECAY,
    )
    for e in range(config.CLF_FREEZE_EPOCHS):
        tl, ta = _run_epoch(model, train_ld, criterion, opt, device, True)
        vl, va = _run_epoch(model, val_ld, criterion, opt, device, False)
        history.append({"phase": 1, "epoch": e, "train_loss": tl,
                         "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"[P1 {e+1}/{config.CLF_FREEZE_EPOCHS}] "
              f"train_loss={tl:.4f} train_acc={ta:.3f} "
              f"val_loss={vl:.4f} val_acc={va:.3f}", flush=True)
        if va > best_val_acc:
            best_val_acc = va
            checkpoint(e, va)

    # ── Phase 2: full fine-tune ──────────────────────────────────────────────
    set_backbone_frozen(model, frozen=False)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.CLF_FINETUNE_LR, weight_decay=config.CLF_WEIGHT_DECAY,
    )
    for e in range(config.CLF_FINETUNE_EPOCHS):
        tl, ta = _run_epoch(model, train_ld, criterion, opt, device, True)
        vl, va = _run_epoch(model, val_ld, criterion, opt, device, False)
        history.append({"phase": 2, "epoch": e, "train_loss": tl,
                         "train_acc": ta, "val_loss": vl, "val_acc": va})
        print(f"[P2 {e+1}/{config.CLF_FINETUNE_EPOCHS}] "
              f"train_loss={tl:.4f} train_acc={ta:.3f} "
              f"val_loss={vl:.4f} val_acc={va:.3f}", flush=True)
        if va > best_val_acc:
            best_val_acc = va
            checkpoint(e, va)

    meta = {
        "best_val_acc": best_val_acc,
        "train_class_counts": dict(zip(config.CLASS_NAMES, train_ds.class_counts())),
        "val_class_counts": dict(zip(config.CLASS_NAMES, val_ds.class_counts())),
        "history": history,
    }
    run_dir.classifier_meta(run).write_text(json.dumps(meta, indent=2))
    run_dir.update_manifest(run, "classifier", {"best_val_acc": best_val_acc})
    return {"best_val_acc": best_val_acc, "weights": str(weights_path),
            "history": history}
