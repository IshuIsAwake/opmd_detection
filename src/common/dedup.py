"""
dedup.py — perceptual-hash leakage check before any number is trusted.

The locked 37 test/ images are the sole headline metric. If a Roboflow image
is a near-duplicate of one of them, the comparison is contaminated. We pHash
every Roboflow image and:

  * EXCLUDE it from both arms if it is within PHASH_HAMMING_THRESH of any
    locked test/ image (hard leak into the headline metric).
  * REPORT (do not exclude) Roboflow-vs-pool/ near-dupes — pool is train/val,
    a near-dupe there is mild redundancy, not headline contamination.

Everything found is written to dedup_report.json. Uses imagehash (installed
into ai_env for this experiment).
"""

from __future__ import annotations

import json
from pathlib import Path

import imagehash
from PIL import Image

import config


def phash(path: Path):
    """Perceptual hash of one image; None if unreadable."""
    try:
        with Image.open(path) as im:
            return imagehash.phash(im.convert("RGB"))
    except Exception:
        return None


def _hash_many(paths: list[Path]) -> dict[Path, object]:
    out = {}
    for p in paths:
        h = phash(p)
        if h is not None:
            out[p] = h
    return out


def _nearest(h, ref: dict[Path, object]) -> tuple[Path | None, int]:
    best_p, best_d = None, 1 << 30
    for rp, rh in ref.items():
        d = h - rh
        if d < best_d:
            best_p, best_d = rp, d
    return best_p, best_d


def build_report(
    roboflow_images: list[Path],
    test_images: list[Path],
    pool_images: list[Path],
    thresh: int = None,
) -> tuple[set[str], dict]:
    """
    Returns (excluded, report).

    excluded — resolved-path strings of Roboflow images to drop (test leak).
    report   — written verbatim to dedup_report.json.
    """
    thresh = config.PHASH_HAMMING_THRESH if thresh is None else thresh

    test_h = _hash_many(test_images)
    pool_h = _hash_many(pool_images)

    excluded: set[str] = set()
    test_collisions: list[dict] = []
    pool_near_dupes: list[dict] = []
    n_hashed = 0

    for rp in roboflow_images:
        h = phash(rp)
        if h is None:
            continue
        n_hashed += 1

        tp, td = _nearest(h, test_h)
        if tp is not None and td <= thresh:
            excluded.add(str(rp.resolve()))
            test_collisions.append(
                {"roboflow": rp.name, "test": tp.name, "hamming": int(td)}
            )

        pp, pd = _nearest(h, pool_h)
        if pp is not None and pd <= thresh:
            pool_near_dupes.append(
                {"roboflow": rp.name, "pool": pp.name, "hamming": int(pd)}
            )

    report = {
        "phash": "imagehash.phash (8x8 DCT, 64-bit)",
        "hamming_threshold": thresh,
        "roboflow_images_hashed": n_hashed,
        "test_images_hashed": len(test_h),
        "pool_images_hashed": len(pool_h),
        "policy": {
            "vs_test": "EXCLUDE from both arms (headline leak)",
            "vs_pool": "REPORT only (train/val redundancy, not excluded)",
        },
        "excluded_vs_test_count": len(test_collisions),
        "excluded_vs_test": sorted(
            test_collisions, key=lambda r: r["hamming"]
        ),
        "pool_near_dupes_count": len(pool_near_dupes),
        "pool_near_dupes": sorted(
            pool_near_dupes, key=lambda r: r["hamming"]
        ),
    }
    return excluded, report


def write_report(report: dict, path: Path = None) -> None:
    path = config.DEDUP_REPORT if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
