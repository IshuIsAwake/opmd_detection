"""
model_b0.py — EfficientNet-B0 (ImageNet-pretrained) fine-tuned end-to-end.

The CNN counterpart to model.DinoV2Classifier. Round-1 conclusion: a frozen
ViT-S at 224 caps around 0.60 Phase 1 micro, suggesting the bottleneck is
backbone capacity + locality bias, not crop geometry. B0 brings CNN inductive
biases (locality, translation invariance) and ~5 M trainable params — small
enough to fine-tune on a 6 GB GPU with batch 32.

Head replaces torchvision's default classifier: features (1280-d after global
avg-pool) → Linear(256) → GELU → Dropout → Linear(5). Same shape as
DinoV2Classifier so downstream metrics / state-dict round-trip the same way.

Optimizer-friendly: ``trainable_param_groups(base_lr)`` returns two groups —
head at base_lr, backbone at 0.1 × base_lr. Standard discriminative-LR
pattern: the pretrained backbone needs gentler updates than a freshly
initialised head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from common import settings

B0_FEATURE_DIM = 1280
BACKBONE_LR_MULT = 0.1


class EfficientNetB0Classifier(nn.Module):
    arch = "efficientnet_b0"

    def __init__(self,
                 n_classes: int = 5,
                 hidden: int = settings.HEAD_HIDDEN,
                 dropout: float = settings.DROPOUT,
                 pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
        # torchvision's B0 ends with: features → avgpool → classifier(Dropout, Linear).
        # Replace classifier; keep everything else trainable.
        net.classifier = nn.Identity()
        self.backbone = net   # outputs [B, 1280] after avgpool+flatten

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(B0_FEATURE_DIM, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                   # [B, 1280]
        return self.head(feats)

    # ── uniform interface for train_eval / phase2 ────────────────────────────

    def trainable_param_groups(self, base_lr: float) -> list[dict]:
        return [
            {"params": list(self.head.parameters()),
             "lr": base_lr, "name": "head"},
            {"params": list(self.backbone.parameters()),
             "lr": base_lr * BACKBONE_LR_MULT, "name": "backbone"},
        ]

    def trainable_state_dict(self) -> dict:
        # Whole model; the backbone is fine-tuned, can't be reloaded from
        # ImageNet weights at inference time.
        return self.state_dict()

    def load_trainable_state(self, state: dict) -> None:
        self.load_state_dict(state)
