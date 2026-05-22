"""
inspect_pipeline.py — Visual diagnostic for the full image → YOLO → crops →
B0 classifier path.

Motivated by the suspicion that the classifier ceiling has more to do with
WHAT we feed B0 than with B0's capacity: YOLO sometimes emits multiple boxes
per lesion, sometimes lands on a lesion edge instead of centring on the
whole thing, and our training data (GT crops with uniform pad=0.4) doesn't
look like that at all.

For each image in Experimenting/internet_images/:
  1. Run the production detector (kfold5_geom_no_color_binary fold 2 — middle
     of pack per §10) at conf=0.10.
  2. Draw every box on the original with a coloured rectangle + index +
     confidence label.
  3. Emit each box's crop at pad ∈ {0.0, 0.2, 0.4} as a separate JPEG.
  4. Run the MVP classifier (B0 aug, fold 2) on each crop and record the
     softmax / argmax in info.json. Skipped silently if weights missing.

Outputs land in Experimenting/full_pipeline_test/<image_stem>/, one folder
per image, mirroring the user's request.

Usage:
    eval "$(conda shell.bash hook)" && conda activate ai_env
    python Experimenting/classifier_experiments/inspect_pipeline.py
    # custom detector fold:
    python Experimenting/classifier_experiments/inspect_pipeline.py --fold 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import settings
from common.crops import pad_and_crop
from common.dataset import eval_transform
from common.model_b0 import EfficientNetB0Classifier

PROJECT_ROOT = settings.PROJECT_ROOT
INPUT_DIR = settings.EXP_ROOT / "internet_images"
OUTPUT_ROOT = settings.EXP_ROOT / "full_pipeline_test"

PAD_FRACS = (0.0, 0.2, 0.4)
DEFAULT_FOLD = 2
DEFAULT_CONF = settings.DETECTOR_CONF
CLASSIFIER_ARM = "gt_pad_0.40_b0_aug"

# BGR palette for box overlays — cycled per box index. Bright/distinct.
BOX_COLOURS = [
    (0, 255, 0),       # green
    (0, 165, 255),     # orange
    (255, 0, 255),     # magenta
    (0, 255, 255),     # yellow
    (255, 255, 0),     # cyan
    (0, 0, 255),       # red
]


def detector_weights(fold_idx: int) -> Path:
    return (settings.DET_RESULTS_ROOT / f"fold_{fold_idx}"
            / "train" / "weights" / "best.pt")


def classifier_weights(fold_idx: int) -> Path | None:
    """Best.pt for the MVP arm at this fold. None if not trained yet."""
    p = (settings.RESULTS_ROOT / CLASSIFIER_ARM
         / f"fold_{fold_idx}" / "best.pt")
    return p if p.exists() else None


def load_detector(fold_idx: int):
    from ultralytics import YOLO
    p = detector_weights(fold_idx)
    if not p.exists():
        raise FileNotFoundError(f"detector weights missing: {p}")
    return YOLO(str(p))


def load_classifier(fold_idx: int, device: torch.device
                    ) -> EfficientNetB0Classifier | None:
    p = classifier_weights(fold_idx)
    if p is None:
        return None
    model = EfficientNetB0Classifier(pretrained=False).to(device)
    state = torch.load(str(p), map_location="cpu", weights_only=True)
    model.load_trainable_state(state)
    model.eval()
    return model


def detect_boxes(detector, image_path: Path, conf: float
                 ) -> list[tuple[float, float, float, float, float]]:
    """Returns [(x1, y1, x2, y2, conf), ...] in pixel coords, conf-sorted desc."""
    res = detector.predict(source=str(image_path), conf=conf, verbose=False)
    out: list[tuple] = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    out.sort(key=lambda b: b[4], reverse=True)
    return out


def annotate_image(bgr: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    """Draw numbered boxes with confidence labels on a copy of the image."""
    canvas = bgr.copy()
    H, W = canvas.shape[:2]
    thick = max(2, min(H, W) // 400)
    font_scale = max(0.5, min(H, W) / 1200)

    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        colour = BOX_COLOURS[i % len(BOX_COLOURS)]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(canvas, p1, p2, colour, thick)

        label = f"#{i} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, thick)
        # Anchor label above-left of the box; if off-image, slip below.
        ly = p1[1] - 6 if p1[1] - 6 - th > 0 else p1[1] + th + 6
        lx = p1[0]
        # Filled background for legibility.
        cv2.rectangle(canvas,
                      (lx, ly - th - 4), (lx + tw + 4, ly + 4),
                      colour, -1)
        cv2.putText(canvas, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), thick)
    return canvas


_TFM = eval_transform()


def classify_crop(model: EfficientNetB0Classifier,
                  crop_bgr: np.ndarray,
                  device: torch.device) -> dict:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = _TFM(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return {
        "pred_id": pred,
        "pred_name": settings.ORIG_ID_TO_CLASS[pred],
        "softmax": {settings.ORIG_ID_TO_CLASS[c]: float(probs[c])
                    for c in range(len(settings.ORIG_ID_TO_CLASS))},
    }


def process_image(image_path: Path, detector, classifier, device, conf: float
                  ) -> dict:
    stem = image_path.stem
    out_dir = OUTPUT_ROOT / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        # Fall back via PIL for .webp etc.; cv2 build may lack the codec.
        pil = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]

    boxes = detect_boxes(detector, image_path, conf=conf)
    annotated = annotate_image(bgr, boxes)
    cv2.imwrite(str(out_dir / "00_original_with_boxes.jpg"), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    # Also stash a clean copy of the original (re-encoded; preserves
    # codec-agnostic JPEG that webp inputs can't keep).
    cv2.imwrite(str(out_dir / "00_original.jpg"), bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    info = {
        "image": str(image_path),
        "image_size_wh": [W, H],
        "detector_conf": conf,
        "n_boxes": len(boxes),
        "boxes": [],
    }

    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        bw = x2 - x1
        bh = y2 - y1
        box_info = {
            "index": i,
            "xyxy": [x1, y1, x2, y2],
            "wh": [bw, bh],
            "confidence": c,
            "area_frac": (bw * bh) / (W * H),
            "crops": {},
        }
        for pad in PAD_FRACS:
            crop = pad_and_crop(bgr, (x1, y1, x2, y2), pad)
            if crop is None:
                box_info["crops"][f"{pad:.2f}"] = {"error": "degenerate_crop"}
                continue
            ch, cw = crop.shape[:2]
            out_path = out_dir / f"box_{i}_pad_{pad:.2f}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            crop_meta: dict = {
                "path": out_path.name,
                "wh": [cw, ch],
            }
            if classifier is not None:
                crop_meta["classifier"] = classify_crop(classifier, crop,
                                                       device)
            box_info["crops"][f"{pad:.2f}"] = crop_meta
        info["boxes"].append(box_info)

    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
    return info


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'image':<28s}  {'boxes':>5s}  {'top_conf':>8s}  "
          f"{'largest_area%':>13s}  pred@pad0.4")
    print("=" * 78)
    for r in results:
        stem = Path(r["image"]).stem
        n = r["n_boxes"]
        if n == 0:
            print(f"{stem:<28s}  {0:>5d}  {'-':>8s}  {'-':>13s}  -")
            continue
        top = r["boxes"][0]
        top_conf = top["confidence"]
        max_area = max(b["area_frac"] for b in r["boxes"]) * 100
        # Top-conf box's pad=0.4 prediction (if classifier ran).
        pred = "(no classifier)"
        clf = top["crops"].get("0.40", {}).get("classifier")
        if clf:
            pred = clf["pred_name"]
        print(f"{stem:<28s}  {n:>5d}  {top_conf:>8.3f}  "
              f"{max_area:>13.1f}  {pred}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD,
                    help="kfold5 fold to use for detector + classifier weights "
                         f"(default {DEFAULT_FOLD}, middle-of-pack)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--no-classifier", action="store_true",
                    help="skip classifier inference; emit crops + boxes only")
    args = ap.parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit(f"input dir not found: {INPUT_DIR}")
    images = sorted(p for p in INPUT_DIR.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not images:
        raise SystemExit(f"no images in {INPUT_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:           {device}")
    print(f"detector fold:    {args.fold}")
    print(f"detector conf:    {args.conf}")
    print(f"classifier arm:   {CLASSIFIER_ARM} (fold {args.fold})")
    print(f"images:           {len(images)} in {INPUT_DIR}")
    print(f"output:           {OUTPUT_ROOT}\n")

    detector = load_detector(args.fold)
    classifier = None if args.no_classifier else load_classifier(args.fold,
                                                                 device)
    if classifier is None and not args.no_classifier:
        print(f"NOTE: classifier weights for {CLASSIFIER_ARM}/fold_{args.fold} "
              "not found — emitting crops + boxes only.\n")

    results = []
    for i, img in enumerate(images, 1):
        print(f"[{i:2d}/{len(images)}] {img.name}")
        info = process_image(img, detector, classifier, device, args.conf)
        results.append(info)
        n = info["n_boxes"]
        print(f"           {n} box(es); → {OUTPUT_ROOT / img.stem}/")

    print_summary(results)
    print(f"\n→ all outputs under {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
