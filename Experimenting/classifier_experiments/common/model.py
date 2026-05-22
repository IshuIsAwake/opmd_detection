"""
model.py — DINOv2-S (frozen) + small MLP classification head.

Backbone: facebookresearch/dinov2 via torch.hub, weights cached locally on
first call. CLS token (384-d) → Linear(256) → GELU → Dropout → Linear(5).

Only the head trains; the backbone is set to eval() and requires_grad=False
so BatchNorm/LayerNorm running stats and gradients are both frozen.

Two methods give train_eval.py a uniform interface across backbones (B0 in
model_b0.py provides the same shape):
  * ``trainable_param_groups(base_lr)`` — optimizer param-group list.
  * ``trainable_state_dict()`` / ``load_trainable_state(state)`` — what to
    persist as best.pt. DINOv2 saves only the head (frozen backbone is
    re-downloaded fresh on load); B0 saves the whole model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import settings


class DinoV2Classifier(nn.Module):
    arch = "dinov2_s"

    def __init__(self,
                 n_classes: int = 5,
                 hidden: int = settings.HEAD_HIDDEN,
                 dropout: float = settings.DROPOUT,
                 backbone: nn.Module | None = None):
        super().__init__()
        self.backbone = backbone if backbone is not None else _load_dinov2()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.head = nn.Sequential(
            nn.Linear(settings.DINOV2_EMBED, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)               # [B, 384] CLS token
        return self.head(feats)

    def train(self, mode: bool = True):
        # Keep backbone in eval() even when the module is set to train(),
        # so dropout/LN behave deterministically.
        super().train(mode)
        self.backbone.eval()
        return self

    # ── uniform interface for train_eval / phase2 ────────────────────────────

    def trainable_param_groups(self, base_lr: float) -> list[dict]:
        return [{"params": list(self.head.parameters()), "lr": base_lr,
                 "name": "head"}]

    def trainable_state_dict(self) -> dict:
        return self.head.state_dict()

    def load_trainable_state(self, state: dict) -> None:
        self.head.load_state_dict(state)


def _load_dinov2() -> nn.Module:
    """torch.hub load. First call downloads weights to ~/.cache/torch/hub/."""
    model = torch.hub.load(settings.DINOV2_REPO, settings.DINOV2_NAME,
                           trust_repo=True, verbose=False)
    return model


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
