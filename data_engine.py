"""
data_engine.py — Crop extraction, dataset class, and transforms.

YOLO class ID → disease name mapping (from pool labels):
    0: Leukoplakia
    1: Erythroplakia
    2: OSMF
    3: Lichen_Planus
    4: NH_Ulcers

Crop strategy:
    - Each bbox annotation → one crop (Option A)
    - Multiple bboxes per image → multiple crops, all in same fold (image-level split)
    - LAB+CLAHE applied on-the-fly in DataLoader to each crop

Normal class (class 5) added later via Oral_Cancer_Self_Data.
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split


# ── Constants ─────────────────────────────────────────────────────────────────

YOLO_ID_TO_CLASS = {
    0: "Leukoplakia",
    1: "Erythroplakia",
    2: "OSMF",
    3: "Lichen_Planus",
    4: "NH_Ulcers",
}

CLASS_TO_IDX = {
    "Leukoplakia":   0,
    "Erythroplakia": 1,
    "OSMF":          2,
    "Lichen_Planus": 3,
    "NH_Ulcers":     4,
    "Normal":        5,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMG_SIZE = 260  # EfficientNet-B2 native resolution


# ── Filename → class normaliser ───────────────────────────────────────────────

def _infer_class_from_filename(stem: str) -> str | None:
    """
    Handles the naming inconsistencies in the raw pool:
    Erythoplakia / Erythyplakia → Erythroplakia
    OSFM → OSMF
    NH_Ulcer / non_healing_ulcer → NH_Ulcers
    """
    s = stem.lower()
    if s.startswith("eryth"):
        return "Erythroplakia"
    if s.startswith("leuko"):
        return "Leukoplakia"
    if s.startswith("lichen"):
        return "Lichen_Planus"
    if s.startswith("nh_ulcer") or s.startswith("non_healing"):
        return "NH_Ulcers"
    if s.startswith("osfm") or s.startswith("osmf"):
        return "OSMF"
    return None


# ── LAB + CLAHE transform ─────────────────────────────────────────────────────

def apply_lab_clahe(bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel of the LAB colour space."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class LabClaheTransform:
    """Callable transform: applies LAB+CLAHE then converts to PIL Image."""

    def __call__(self, img: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        bgr = apply_lab_clahe(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


# ── Crop extraction ───────────────────────────────────────────────────────────

def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalised bbox to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    return x1, y1, x2, y2


def extract_crops(
    img_dir: str,
    lbl_dir: str,
    out_dir: str,
    min_side: int = 32,
) -> dict:
    """
    Extract bbox crops from raw pool images and save to out_dir/<ClassName>/.

    Args:
        img_dir:  Directory containing pool images.
        lbl_dir:  Directory containing matching YOLO .txt label files.
        out_dir:  Root output directory.  Sub-dirs created per class.
        min_side: Discard crops smaller than this in either dimension.

    Returns:
        stats dict: {class_name: crop_count}
    """
    img_dir = Path(img_dir)
    lbl_dir = Path(lbl_dir)
    out_dir = Path(out_dir)

    # Create class output dirs
    for cls in YOLO_ID_TO_CLASS.values():
        (out_dir / cls).mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_no_label = 0
    skipped_bad_crop = 0
    skipped_unknown_class = 0

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.png"))

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped_no_label += 1
            continue

        # Infer class from filename for sanity check
        expected_class = _infer_class_from_filename(img_path.stem)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        img_h, img_w = img_bgr.shape[:2]

        with open(lbl_path) as f:
            annotations = f.read().strip().splitlines()

        for i, line in enumerate(annotations):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            if class_id not in YOLO_ID_TO_CLASS:
                skipped_unknown_class += 1
                continue

            class_name = YOLO_ID_TO_CLASS[class_id]
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, bw, bh, img_w, img_h)

            if (x2 - x1) < min_side or (y2 - y1) < min_side:
                skipped_bad_crop += 1
                continue

            crop = img_bgr[y1:y2, x1:x2]

            # Unique filename: stem + bbox index
            crop_filename = f"{img_path.stem}_crop{i}.jpg"
            out_path = out_dir / class_name / crop_filename
            cv2.imwrite(str(out_path), crop)
            stats[class_name] += 1

    print("\n=== Crop Extraction Complete ===")
    for cls in sorted(stats):
        print(f"  {cls:20s}: {stats[cls]} crops")
    print(f"  {'TOTAL':20s}: {sum(stats.values())}")
    print(f"\nSkipped — no label file : {skipped_no_label}")
    print(f"Skipped — crop too small: {skipped_bad_crop}")
    print(f"Skipped — unknown class : {skipped_unknown_class}")

    return dict(stats)


# ── Dataset class ─────────────────────────────────────────────────────────────

class OralCancerDataset(Dataset):
    """
    PyTorch Dataset for the processed crop directory structure:

        data_dir/
            Leukoplakia/
            Erythroplakia/
            OSMF/
            Lichen_Planus/
            NH_Ulcers/
            Normal/       (optional)

    LAB+CLAHE is applied on-the-fly here, before any torchvision transforms.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],   # [(path, label_idx), ...]
        augment: bool = False,
        use_clahe: bool = True,
        custom_transform: transforms.Compose | None = None,
    ):
        self.samples = samples
        self.augment = augment
        self.use_clahe = use_clahe

        self.lab_clahe = LabClaheTransform()

        if custom_transform is not None:
            self.transform = custom_transform
        else:
            base_transforms = [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]

            if augment:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                    transforms.RandomGrayscale(p=0.05),
                    *base_transforms,
                    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                ])
            else:
                self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.use_clahe:
            img = self.lab_clahe(img)
        img = self.transform(img)
        return img, label


# ── Split helpers ─────────────────────────────────────────────────────────────

def _collect_images(data_dir: str) -> tuple[list[str], list[int], list[str]]:
    """
    Walk data_dir/<ClassName>/ and return (paths, labels, image_stems).
    image_stems used for image-level grouping (crops from same source image).
    """
    paths, labels, stems = [], [], []
    data_dir = Path(data_dir)

    for cls_name, cls_idx in CLASS_TO_IDX.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png")):
            paths.append(str(img_path))
            labels.append(cls_idx)
            # Strip _cropN suffix to get the source image stem
            stem = re.sub(r"_crop\d+$", "", img_path.stem)
            stems.append(stem)

    return paths, labels, stems


def _image_level_label(stems: list[str], labels: list[int]) -> tuple[list[str], list[int]]:
    """
    Return unique (stem, label) pairs for stratified splitting at image level.
    Uses the first label seen for each stem (all crops from one image share a label).
    """
    seen = {}
    unique_stems, unique_labels = [], []
    for stem, lbl in zip(stems, labels):
        if stem not in seen:
            seen[stem] = lbl
            unique_stems.append(stem)
            unique_labels.append(lbl)
    return unique_stems, unique_labels


def make_kfold_splits(
    data_dir: str,
    n_splits: int = 5,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[list[dict], list[tuple[str, int]], list[tuple[str, int]]]:
    """
    Build k-fold cross-validation splits + a held-out test set.

    Split order:
        1. Hold out test_size fraction at IMAGE level (stratified).
        2. Run StratifiedKFold on remaining images.
        3. Map image-level splits back to individual crops.

    Returns:
        folds      : list of {'train': [(path, label)], 'val': [(path, label)]}
        test_samples: [(path, label)]  — never seen during training
        all_samples : all (path, label) pairs (useful for diagnostics)
    """
    paths, labels, stems = _collect_images(data_dir)
    assert len(paths) > 0, f"No images found under {data_dir}"

    unique_stems, unique_labels = _image_level_label(stems, labels)

    # Hold-out test split
    train_val_stems, test_stems, _, _ = train_test_split(
        unique_stems, unique_labels,
        test_size=test_size,
        stratify=unique_labels,
        random_state=random_state,
    )
    test_set = set(test_stems)

    # Filter to train+val stems only
    tv_stems = [s for s in unique_stems if s not in test_set]
    tv_labels = [unique_labels[unique_stems.index(s)] for s in tv_stems]

    # Build stem → crop mapping
    stem_to_crops: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path, label, stem in zip(paths, labels, stems):
        stem_to_crops[stem].append((path, label))

    test_samples = [
        item for stem in test_set for item in stem_to_crops[stem]
    ]

    # K-fold on train+val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(tv_stems, tv_labels):
        train_stems = {tv_stems[i] for i in train_idx}
        val_stems   = {tv_stems[i] for i in val_idx}

        train_samples = [item for s in train_stems for item in stem_to_crops[s]]
        val_samples   = [item for s in val_stems   for item in stem_to_crops[s]]

        folds.append({"train": train_samples, "val": val_samples})

    all_samples = [(p, l) for p, l in zip(paths, labels)]

    print(f"\n=== Split Summary ===")
    print(f"  Total crops  : {len(paths)}")
    print(f"  Test crops   : {len(test_samples)}")
    print(f"  Train+val    : {len(paths) - len(test_samples)}")
    print(f"  Folds        : {n_splits}")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: train={len(fold['train'])}, val={len(fold['val'])}")

    return folds, test_samples, all_samples


# ── Class weights for imbalanced training ─────────────────────────────────────

def compute_class_weights(samples: list[tuple[str, int]], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = torch.zeros(n_classes)
    for _, label in samples:
        counts[label] += 1
    # Zero counts get weight 0 (class not present)
    weights = torch.zeros(n_classes)
    present = counts > 0
    weights[present] = counts[present].sum() / (n_classes * counts[present])
    return weights


# ── CLI: run crop extraction directly ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract crops from raw pool")
    parser.add_argument("--img_dir", default="../datasets/PreProcessed_Data/pool/images")
    parser.add_argument("--lbl_dir", default="../datasets/PreProcessed_Data/pool/labels")
    parser.add_argument("--out_dir", default="../Oral_Cancer_Data")
    parser.add_argument("--min_side", type=int, default=32)
    args = parser.parse_args()

    extract_crops(args.img_dir, args.lbl_dir, args.out_dir, args.min_side)
