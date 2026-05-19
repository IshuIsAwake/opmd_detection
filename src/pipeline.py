"""
Step 5 — End-to-end pipeline. Full image in → result out.

    detect (low conf) → box? → SHARED crop → EfficientNet → Grad-CAM
                       → no box → NO_DETECTION_FALLBACK:
                            "healthy"     → "looks fine, no visit" (default)
                            "center_crop" → classify central region, flagged
                                            low-confidence (recall-max, opt-in)

The crop here is the exact same function used to build the classifier data
(src.common.crop) — the structural guarantee against mistake #2.

By default serves the run pointed to by artifacts/latest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from src.classifier.gradcam import CropExplainer
from src.classifier.model import load_for_inference
from src.common import run_dir
from src.common.crop import center_box, crop_with_padding   # the ONE shared fn


@dataclass
class PipelineResult:
    lesion_found: bool
    recommendation: str
    disease: str | None = None
    confidence: float | None = None
    probs: list[float] = field(default_factory=list)
    boxed_image: np.ndarray | None = None
    gradcam_overlay: np.ndarray | None = None
    detector_conf: float | None = None
    used_fallback: bool = False              # center-crop fallback was used


class OralLesionPipeline:
    """Loads both models once; reuse across many images."""

    def __init__(self, run: Path | None = None, device: str | None = None):
        self.run = run or run_dir.latest_run()
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.conf = self._resolve_conf()
        self.detector = YOLO(str(run_dir.detector_weights(self.run)))
        model, _ = load_for_inference(run_dir.classifier_weights(self.run), self.device)
        self.explainer = CropExplainer(model, self.device)

    def _resolve_conf(self) -> float:
        meta = run_dir.detector_meta(self.run)
        if meta.exists():
            try:
                return float(json.loads(meta.read_text())["conf"])
            except (KeyError, ValueError):
                pass
        return config.DET_CONF

    def _classify(self, rgb: np.ndarray, box, draw_color) -> dict:
        crop_rgb = crop_with_padding(rgb, box)          # SHARED crop fn
        ex = self.explainer.explain(crop_rgb)
        boxed = rgb.copy()
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(boxed, (x1, y1), (x2, y2), draw_color, 3)
        ex["boxed"] = boxed
        return ex

    def analyze(self, image: str | Path | np.ndarray) -> PipelineResult:
        bgr = cv2.imread(str(image)) if not isinstance(image, np.ndarray) else image
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        result = self.detector.predict(bgr, conf=self.conf, verbose=False)[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            box = tuple(xyxy[int(np.argmax(confs))])
            ex = self._classify(rgb, box, (255, 0, 0))
            return PipelineResult(
                lesion_found=True,
                recommendation=f"Visit a dentist — possible {ex['pred_class']}.",
                disease=ex["pred_class"],
                confidence=ex["confidence"],
                probs=ex["probs"],
                boxed_image=ex["boxed"],
                gradcam_overlay=ex["overlay"],
                detector_conf=self.conf,
            )

        # ── No detection ─────────────────────────────────────────────────────
        if config.NO_DETECTION_FALLBACK != "center_crop":
            return PipelineResult(
                lesion_found=False,
                recommendation="Looks fine — no dentist visit needed.",
                boxed_image=rgb,
                detector_conf=self.conf,
            )

        box = center_box(rgb, config.CENTER_CROP_FRAC)
        ex = self._classify(rgb, box, (255, 165, 0))
        return PipelineResult(
            lesion_found=False,
            used_fallback=True,
            recommendation=(
                f"Could not localize a lesion — central-region check suggests "
                f"possible {ex['pred_class']} (LOW confidence). Consider a dental "
                f"check to be safe."
            ),
            disease=ex["pred_class"],
            confidence=ex["confidence"],
            probs=ex["probs"],
            boxed_image=ex["boxed"],
            gradcam_overlay=ex["overlay"],
            detector_conf=self.conf,
        )


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()
