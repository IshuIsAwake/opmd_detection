"""
predict_internet.py — qualitative sanity check on out-of-domain images.

Runs fold-0 of kfold5_geom_no_color_binary against Experimenting/internet_images/
at multiple operating points (conf=0.05, 0.10, 0.25) and saves annotated
images + a console summary of fired/didn't-fire per image.

No GT, no metric — purely "does it look sensible". Useful before the demo;
NOT a number to put in a doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ultralytics import YOLO

WEIGHTS = HERE / "results" / "kfold5_geom_no_color_binary" / "fold_0" / "train" / "weights" / "best.pt"
IMAGES_DIR = HERE / "internet_images"
OUT_ROOT = HERE / "results" / "internet_sanity_fold0"
CONFS = [0.05, 0.10, 0.25]
IMGSZ = 640


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}")
    images = sorted(p for p in IMAGES_DIR.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise SystemExit(f"no images in {IMAGES_DIR}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    print(f"\nweights : {WEIGHTS.relative_to(HERE.parent)}")
    print(f"images  : {len(images)} in {IMAGES_DIR.relative_to(HERE.parent)}")
    print(f"imgsz   : {IMGSZ}\n")

    for conf in CONFS:
        out_dir = OUT_ROOT / f"conf_{conf:.2f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"── conf={conf:.2f} ──")
        for img in images:
            r = model.predict(source=str(img), conf=conf, imgsz=IMGSZ,
                              device=0, verbose=False, save=False)[0]
            n = len(r.boxes) if r.boxes is not None else 0
            confs = ([float(c) for c in r.boxes.conf.cpu().tolist()]
                     if n else [])
            annotated = r.plot()
            from PIL import Image
            Image.fromarray(annotated[..., ::-1]).save(out_dir / img.name)
            tag = "FIRED" if n else "silent"
            conf_str = ", ".join(f"{c:.2f}" for c in confs) if confs else "—"
            print(f"  {img.name:25s}  {tag:6s}  boxes={n}  confs=[{conf_str}]")
        print()

    print(f"annotated images → {OUT_ROOT.relative_to(HERE.parent)}/conf_*/\n")


if __name__ == "__main__":
    main()
