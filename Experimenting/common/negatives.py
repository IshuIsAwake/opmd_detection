"""
negatives.py — Resolution-normalise the Normal/ images to the lesion domain.

The audit's "below trivial" verdict was measured against the 570 raw
data/Normal images: 960–4000 px phone photos versus ~250 px lesion thumbnails.
A detector can false-alarm on those for being an unseen *resolution/framing
domain*, not for failing to tell healthy from diseased. exp8a/exp8b remove
that confound by resizing every Normal image so its long side matches the
positives' median long side, then letting YOLO's imgsz upscale BOTH the same
way. The transform is applied identically in train and test (train/serve
consistency).

Pure-ish and deterministic: the normalised files are written once under
_datasets/_normalized_negatives/ (idempotent — skipped if already present),
and the source ordering is a stable function of the read-only originals, so
exp8a, exp8b and eval_fair_negatives all see byte-identical negatives.

Originals are never modified — we only ever read them and write resized copies
under Experimenting/_datasets/.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from common import settings

try:                                            # Pillow ≥ 9.1
    from PIL.Image import Resampling
    _LANCZOS = Resampling.LANCZOS
except ImportError:                             # older Pillow
    _LANCZOS = Image.LANCZOS                     # type: ignore[attr-defined]

NORM_ROOT = settings.DATASETS_ROOT / "_normalized_negatives"


def _positive_images() -> list[Path]:
    """Every lesion-domain image (pool + locked test) — defines the target
    resolution we normalise the negatives down to."""
    return (settings.list_images(settings.POOL_IMAGES)
            + settings.list_images(settings.TEST_IMAGES))


def positive_target_size() -> int:
    """Median long side (max(w, h)) over all positive images. Computed from
    PIL's lazy header read — no full decode. Deterministic."""
    longs: list[int] = []
    for p in _positive_images():
        with Image.open(p) as im:
            longs.append(max(im.size))
    longs.sort()
    n = len(longs)
    if n == 0:
        raise RuntimeError("no positive images found to size negatives against")
    mid = n // 2
    med = longs[mid] if n % 2 else (longs[mid - 1] + longs[mid]) / 2.0
    return int(round(med))


def _raw_negatives() -> list[Path]:
    """All Normal/ images from BOTH dirs (list_images is non-recursive — the
    documented landmine — so each dir is enumerated explicitly), in a stable
    global order: (parent dir name, filename)."""
    out: list[Path] = []
    for d in settings.NORMAL_DIRS:
        if d.is_dir():
            out.extend(settings.list_images(d))
    return sorted(out, key=lambda p: (p.parent.name, p.name))


def prepare() -> list[Path]:
    """Resolution-normalise every Normal image to the positive median long
    side. Idempotent (an existing output is left as-is). Returns the
    normalised paths in the same deterministic order as ``_raw_negatives()``.
    """
    target = positive_target_size()
    NORM_ROOT.mkdir(parents=True, exist_ok=True)

    norm_paths: list[Path] = []
    for src in _raw_negatives():
        # Namespace by parent so Normal/ vs Normal/NON CANCER/ stems can never
        # collide (then again under neg__ in the dataset builder).
        tag = "n" if src.parent.name == "Normal" else "nc"
        dst = NORM_ROOT / f"{tag}__{src.stem}.jpg"
        if not dst.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                s = target / float(max(w, h))
                new = (max(1, round(w * s)), max(1, round(h * s)))
                im.resize(new, _LANCZOS).save(dst, "JPEG", quality=95)
        norm_paths.append(dst)

    (NORM_ROOT / "_negatives.json").write_text(json.dumps({
        "positive_target_long_side": target,
        "n_normalized": len(norm_paths),
        "source_dirs": [str(d) for d in settings.NORMAL_DIRS],
        "seed": settings.SEED,
        "resample": "LANCZOS",
    }, indent=2))
    return norm_paths
