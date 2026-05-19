"""
Classifier model — EfficientNet-B2 (timm), 5-class.

Also exposes the freeze/unfreeze switch for 2-phase fine-tuning and the
Grad-CAM target layer, so every consumer agrees on the same layer.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

import config


def build_model(pretrained: bool = True) -> nn.Module:
    return timm.create_model(
        config.CLF_MODEL,
        pretrained=pretrained,
        num_classes=config.NUM_CLASSES,
    )


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Phase 1 freezes everything but the classifier head; phase 2 unfreezes."""
    for p in model.parameters():
        p.requires_grad = not frozen
    # The head always trains.
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def gradcam_target_layers(model: nn.Module) -> list[nn.Module]:
    """Last conv stage of EfficientNet — sharp, class-discriminative maps."""
    # timm efficientnet exposes conv_head before global pooling.
    if hasattr(model, "conv_head"):
        return [model.conv_head]
    return [model.blocks[-1]]


def load_for_inference(weights_path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild the architecture and load trained weights for serving/eval."""
    ckpt = torch.load(weights_path, map_location=device)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, ckpt
