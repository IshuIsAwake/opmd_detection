"""
dataset.py — torch Dataset reading from the materialised per-fold trees.

The on-disk layout is ImageFolder-shaped, but our class index is FIXED by the
detector experiment's ORIG_ID_TO_CLASS (0..4 ordering), not by directory sort
order. Custom Dataset enforces that mapping so a future renamed folder cannot
silently relabel.

Two transforms exposed:
  * train_transform — horizontal flip + mild rotation, ImageNet norm (HANDOFF
    "minimal augmentation").
  * eval_transform  — resize + ImageNet norm only.

Resize policy: resize the shorter side to INPUT_SIZE, then centre-crop to
INPUT_SIZE×INPUT_SIZE. Padding to square would introduce black bars the
backbone has never seen.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common import settings

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(settings.INPUT_SIZE),
        transforms.CenterCrop(settings.INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def strong_train_transform() -> transforms.Compose:
    """Round-3 augmentation suite. Adds: wider rotation (±20°),
    RandomResizedCrop (0.7–1.0 area), ColorJitter (brightness/contrast/sat
    only — hue stays 0 because colour is diagnostic on this task, see
    `geom_no_color` in the detector chapter), RandomErasing (p=0.25).

    Intentionally heavier than `train_transform()` — this is the lever we pull
    once the model has saturated the basic-aug regime. Eval-time transform
    stays the same (no TTA here, that's a separate eval-time tool)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(settings.INPUT_SIZE,
                                     scale=(0.7, 1.0),
                                     ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.1, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(settings.INPUT_SIZE),
        transforms.CenterCrop(settings.INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class CropFolder(Dataset):
    """ImageFolder-style reader with FIXED class index (settings.ORIG_ID_TO_CLASS).

    Expects layout: <root>/<DiseaseName>/<file>.<ext>
    Files of any type in settings.IMG_EXTS are picked up.
    """

    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.transform = transform
        self.class_to_idx = {name: cid for cid, name
                             in settings.ORIG_ID_TO_CLASS.items()}

        self.samples: list[tuple[Path, int]] = []
        for disease_name, cid in self.class_to_idx.items():
            d = self.root / disease_name
            if not d.is_dir():
                continue
            for ext in settings.IMG_EXTS:
                for p in sorted(d.glob(f"*{ext}")):
                    self.samples.append((p, cid))
        # Stable order for reproducibility.
        self.samples.sort(key=lambda t: t[0].name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, label, path.stem

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {cid: 0 for cid in self.class_to_idx.values()}
        for _, c in self.samples:
            counts[c] += 1
        return counts


def class_weights_from_counts(counts: dict[int, int]) -> np.ndarray:
    """Inverse-frequency class weights (HANDOFF: do not resample). Normalised
    so the mean weight is 1.0 — keeps the effective loss magnitude comparable
    across folds with different counts."""
    n_classes = len(settings.ORIG_ID_TO_CLASS)
    w = np.zeros(n_classes, dtype=np.float32)
    total = sum(counts.values())
    for cid in range(n_classes):
        c = counts.get(cid, 0)
        w[cid] = (total / (n_classes * c)) if c > 0 else 0.0
    nonzero = w[w > 0]
    if len(nonzero):
        w = w * (len(nonzero) / nonzero.sum())
    return w
