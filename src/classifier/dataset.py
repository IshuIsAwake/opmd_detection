"""
Classifier dataset + transforms.

Raw RGB + ImageNet normalisation ONLY. No LAB/CLAHE (mistake #3). Minimal aug:
horizontal flip + mild rotation. The label index comes from config.CLASS_TO_IDX
so ordering is identical everywhere (dataset, model head, Grad-CAM, pipeline).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from src.common import run_dir
from src.common.io import list_images


def build_transforms(train: bool) -> transforms.Compose:
    base = [
        transforms.Resize((config.CLF_IMG_SIZE, config.CLF_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    if not train:
        return transforms.Compose(base)
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            *base,
        ]
    )


class CropDataset(Dataset):
    """Reads artifacts/classifier_data/<split>/<Class>/*.jpg."""

    def __init__(self, split: str, train_aug: bool, run: "Path | None" = None):
        run = run or run_dir.current_run()
        root = run_dir.classifier_data_dir(run) / split
        if not root.exists():
            raise FileNotFoundError(f"{root} missing — run Step 3 first.")

        self.samples: list[tuple[Path, int]] = []
        for cls_name, idx in config.CLASS_TO_IDX.items():
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for img in list_images(cls_dir):
                self.samples.append((img, idx))

        if not self.samples:
            raise RuntimeError(f"No crops found under {root}.")

        self.transform = build_transforms(train=train_aug)

    def class_counts(self) -> list[int]:
        counts = [0] * config.NUM_CLASSES
        for _, idx in self.samples:
            counts[idx] += 1
        return counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
