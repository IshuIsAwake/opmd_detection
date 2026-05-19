"""
Grad-CAM explainer.

One entry point: feed a raw RGB crop, get back the predicted class, softmax
confidence, full probability vector, and a Grad-CAM overlay. Preprocessing
lives here (matching the val transform) so the pipeline never re-implements it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from src.classifier.dataset import build_transforms
from src.classifier.model import gradcam_target_layers


class CropExplainer:
    """Predict + explain a single detector crop."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = device
        self.transform = build_transforms(train=False)
        self.target_layers = gradcam_target_layers(self.model)

    @torch.no_grad()
    def _probs(self, tensor: torch.Tensor) -> np.ndarray:
        logits = self.model(tensor)
        return F.softmax(logits, dim=1)[0].cpu().numpy()

    def explain(self, rgb_crop: np.ndarray) -> dict:
        """
        Args:
            rgb_crop: HxWx3 uint8 RGB image (the detector crop).
        Returns dict: pred_idx, pred_class, confidence, probs, overlay (RGB uint8).
        """
        from PIL import Image

        pil = Image.fromarray(rgb_crop)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)

        probs = self._probs(tensor)
        pred_idx = int(np.argmax(probs))

        # Normalised float image for the CAM overlay, same resolution as input.
        resized = np.array(
            pil.resize((config.CLF_IMG_SIZE, config.CLF_IMG_SIZE))
        ).astype(np.float32) / 255.0

        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale = cam(
                input_tensor=tensor,
                targets=[ClassifierOutputTarget(pred_idx)],
            )[0]
        overlay = show_cam_on_image(resized, grayscale, use_rgb=True)

        return {
            "pred_idx": pred_idx,
            "pred_class": config.CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probs": probs.tolist(),
            "overlay": overlay,
        }
