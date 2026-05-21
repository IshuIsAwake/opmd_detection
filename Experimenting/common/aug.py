"""
aug.py — Named augmentation levels for the exp10 sweep.

Each level is a dict of Ultralytics YOLO train kwargs. Anything missing falls
back to Ultralytics defaults — which matter, because exp1–8 implicitly used
those defaults (mosaic=1, fliplr=0.5, scale=0.5, translate=0.1, hsv_h=0.015,
hsv_s=0.7, hsv_v=0.4, degrees=0, mixup=0, copy_paste=0, erasing=0.4). The
project notes "exp1–8 = YOLO built-in only" never meant "no aug" — it meant
"moderate default aug including HSV colour distortion".

The sweep:
  off            true no-aug control (fliplr only — free symmetry of a mouth)
  light          mild geometric only, COLOUR LOCKED (respects instructions.md
                 "colour is diagnostic" rule)
  default        empty override → exp8b recipe = YOLO defaults
  geom_no_color  defaults but hsv_h=hsv_s=hsv_v=0 — isolates "does HSV hurt?"
                 against `default`, a clean controlled test of the colour rule
  heavy          defaults + bigger rotate/scale/shear + mixup + copy_paste +
                 stronger erasing — re-tests the "mixup/heavy aug premature"
                 line from instructions.md, now that exp8 confirmed the
                 detector works.
"""

from __future__ import annotations


AUG_LEVELS: dict[str, dict] = {
    "off": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "light": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 5.0, "translate": 0.05, "scale": 0.3,
        "shear": 0.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0,
        "erasing": 0.0,
    },
    "default": {},                                   # = YOLO defaults = exp8b
    "geom_no_color": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,    # colour locked, geometry untouched
    },
    "heavy": {
        # colour: keep YOLO defaults (compare against geom_no_color to isolate)
        "degrees": 10.0, "translate": 0.15, "scale": 0.7,
        "shear": 2.0,
        "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.3,
        "erasing": 0.6,
    },
    "heavy_no_color": {
        # heavy's exact geometry + composition + erasing, colour locked at 0.
        # Pairs with `heavy` the same way `geom_no_color` pairs with `default`:
        # one-variable test of "does the colour fix rescue the kitchen sink,
        # or are mixup/copy_paste/big-rotate independently too aggressive at
        # ~277 imgs?"
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 10.0, "translate": 0.15, "scale": 0.7,
        "shear": 2.0,
        "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.3,
        "erasing": 0.6,
    },
}


def describe(level: str) -> str:
    """One-line config dump for the run.json / banner."""
    cfg = AUG_LEVELS[level]
    if not cfg:
        return f"{level}: YOLO defaults (no overrides)"
    parts = [f"{k}={v}" for k, v in sorted(cfg.items())]
    return f"{level}: " + ", ".join(parts)
