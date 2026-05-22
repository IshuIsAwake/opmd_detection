"""
predict_with_old_classifier.py — Borrow-the-old-classifier experiment.

Question being answered:
  Does the EfficientNet-B2 classifier from the discarded Dentilligence/OCD
  project work acceptably when fed crops produced by our YOLO binary detector
  (kfold5_geom_no_color_binary, fold 3)? The OCD classifier was trained on
  tight human-bbox crops of the same pool/ images we use here — it failed on
  full-image input. Hypothesis: it might work on detector-emitted crops since
  those are also lesion-zoomed.

Pipeline per image:
  full image → YOLO detector (conf=0.10, fold 3 blackbox weights)
              ├─ no boxes  → "Looks fine" (no classifier call)
              └─ ≥1 box    → for each box: pad → crop → CLAHE → resize 260 →
                              classify (softmax over 6 classes incl. Normal)
                            mean softmax across boxes → argmax → predicted class

Positive test set:  fold 3's test slice from kfold5_splits.json (72 lesion
                    images, blackbox to the fold-3 detector weights).
Negative test set:  OCD's 120 Normal images (or 12 OCD-test-blackbox subset).

Padding policy:
  For each pad_frac in {0.0, 0.2, 0.4}, expand each YOLO bbox by
  pad_frac × w on each horizontal side and pad_frac × h on each vertical side
  (so total width grows by 2 × pad_frac). Clipped to image bounds.

Caveat preserved in writeup: the OCD classifier was trained on crops from
SOURCE IMAGES that overlap fold 3's test slice (~90% of OCD train shares the
pool). When fed YOLO crops from those same source images, it's a DIFFERENT
crop of a SEEN source image — not full leakage, not full blackbox.
Conditional disease accuracy may be slightly optimistic on that account.

Borrowed assets (read-only):
  - OCD EfficientNet weights: Dentilligence/OCD/Oral_Cancer/runs/.../fold_5/best.pt
  - OCD model definition + CLAHE transform: imported via sys.path

Usage:
    cd /home/ishu/Projects/AI/Oral_cancer
    eval "$(conda shell.bash hook)" && conda activate ai_env
    python Experimenting/predict_with_old_classifier.py
    # optional flags:
    #   --conf 0.10
    #   --pads 0.0 0.2 0.4
    #   --negatives_mode all          # or 'blackbox' for 12 OCD-test Normals
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────

ORAL_CANCER_ROOT = Path("/home/ishu/Projects/AI/Oral_cancer")
DETECTOR_WEIGHTS = ORAL_CANCER_ROOT / "Experimenting/results/kfold5_geom_no_color_binary/fold_3/train/weights/best.pt"
KFOLD_SPLITS = ORAL_CANCER_ROOT / "Experimenting/_datasets/kfold5_splits.json"
FOLD_INDEX = 3  # fold 3 test slice = 72 lesion images, blackbox to fold-3 weights

OCD_ROOT = Path("/home/ishu/Projects/AI/Dentilligence/OCD/Oral_Cancer")
CLASSIFIER_WEIGHTS = OCD_ROOT / "runs/run_1773770685/fold_5/best.pt"
OCD_NORMAL_DIR = OCD_ROOT.parent / "Oral_Cancer_Data/Normal"
OCD_TEST_MANIFEST = OCD_ROOT / "runs/run_1773770685/test_manifest.json"

# OCD src import (model definition + CLAHE) — read-only borrow
sys.path.insert(0, str(OCD_ROOT))
from src.data_engine import apply_lab_clahe, IMG_SIZE  # noqa: E402
from src.model import build_model  # noqa: E402

OUTPUT_ROOT = Path(__file__).resolve().parent / "results/yolo_to_old_classifier"

DISEASE_NAMES = {0: "Leukoplakia", 1: "Erythroplakia", 2: "OSMF",
                 3: "Lichen_Planus", 4: "NH_Ulcers", 5: "Normal"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_positives() -> list[dict]:
    """Fold-3 test slice: 72 lesion images with disease label (0-4)."""
    with open(KFOLD_SPLITS) as f:
        split = json.load(f)
    fold_keys = split["positives_by_fold"][FOLD_INDEX]
    positives = []
    for key in fold_keys:
        entry = split["positives"][key]
        positives.append({
            "stem": key,
            "image": entry["image"],
            "gt_class": int(entry["disease"]),
            "gt_class_name": DISEASE_NAMES[int(entry["disease"])],
        })
    return positives


def load_negatives(mode: str) -> list[dict]:
    """OCD Normal images. mode='blackbox' = 12 OCD-test Normals; 'all' = 120."""
    if mode == "blackbox":
        with open(OCD_TEST_MANIFEST) as f:
            manifest = json.load(f)
        normals = [e for e in manifest if e["label"] == 5]
        out = []
        for e in normals:
            p = (OCD_ROOT / e["path"]).resolve()
            out.append({"stem": p.stem, "image": str(p), "gt_class": 5,
                        "gt_class_name": "Normal"})
        return out
    elif mode == "all":
        return [{"stem": p.stem, "image": str(p), "gt_class": 5,
                 "gt_class_name": "Normal"}
                for p in sorted(OCD_NORMAL_DIR.glob("*.jpeg"))]
    else:
        raise ValueError(f"unknown negatives_mode: {mode}")


# ── Crop / preprocess ─────────────────────────────────────────────────────────

def pad_and_crop(bgr: np.ndarray, xyxy: tuple[float, float, float, float],
                 pad_frac: float):
    """Expand bbox by pad_frac * (w, h) on each side, clip to image, return BGR crop."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    px = pad_frac * w
    py = pad_frac * h
    nx1 = int(max(0, round(x1 - px)))
    ny1 = int(max(0, round(y1 - py)))
    nx2 = int(min(W, round(x2 + px)))
    ny2 = int(min(H, round(y2 + py)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return bgr[ny1:ny2, nx1:nx2].copy()


_imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
_to_tensor = transforms.ToTensor()


def crop_to_tensor(crop_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """BGR crop → CLAHE → RGB → 260x260 → ImageNet-norm tensor [1,3,260,260]."""
    bgr = apply_lab_clahe(crop_bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE))
    t = _imagenet_norm(_to_tensor(pil)).unsqueeze(0).to(device)
    return t


# ── Model loading ─────────────────────────────────────────────────────────────

def load_detector(device: torch.device):
    print(f"  detector: {DETECTOR_WEIGHTS}")
    return YOLO(str(DETECTOR_WEIGHTS))


def load_classifier(device: torch.device):
    print(f"  classifier: {CLASSIFIER_WEIGHTS}")
    model = build_model().to(device)
    model.load_state_dict(torch.load(str(CLASSIFIER_WEIGHTS),
                                     map_location=device, weights_only=True))
    model.eval()
    return model


# ── Per-image inference ───────────────────────────────────────────────────────

def detect_boxes(detector: YOLO, image_path: str, conf: float) -> list[tuple]:
    """Return list of (x1, y1, x2, y2, conf) in pixel coords."""
    results = detector.predict(source=image_path, conf=conf, verbose=False)
    boxes = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    return boxes


def classify_crops(classifier, crops_bgr: list[np.ndarray],
                   device: torch.device) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run classifier on each crop, return (mean softmax [6], per_box softmax list)."""
    per_box = []
    with torch.no_grad():
        for crop in crops_bgr:
            t = crop_to_tensor(crop, device)
            logits = classifier(t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            per_box.append(probs)
    mean = np.mean(per_box, axis=0) if per_box else np.zeros(6)
    return mean, per_box


# ── Single padding-value run ──────────────────────────────────────────────────

def run_one_pad(positives, negatives, detector, classifier,
                device, conf: float, pad: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== pad={pad} ===")
    per_image_rows = []

    pos_caught = 0
    pos_missed = 0
    pos_correct = 0
    pos_wrong = 0
    conf_mat = np.zeros((5, 6), dtype=int)  # rows GT 0-4, cols pred 0-5

    # ── Positives ────────────────────────────────────────────────────────────
    for i, pos in enumerate(positives, 1):
        boxes = detect_boxes(detector, pos["image"], conf=conf)
        if not boxes:
            pos_missed += 1
            per_image_rows.append({
                "side": "positive", "stem": pos["stem"],
                "gt_class": pos["gt_class"], "gt_name": pos["gt_class_name"],
                "n_boxes": 0, "pred_class": None, "pred_name": "MISSED",
                "mean_probs": None, "outcome": "missed",
            })
            print(f"  [{i:3d}/{len(positives)}] {pos['stem']:40s} GT={pos['gt_class_name']:14s} MISSED")
            continue

        bgr = cv2.imread(pos["image"])
        crops = [pad_and_crop(bgr, b[:4], pad) for b in boxes]
        crops = [c for c in crops if c is not None]
        if not crops:
            pos_missed += 1
            per_image_rows.append({
                "side": "positive", "stem": pos["stem"],
                "gt_class": pos["gt_class"], "gt_name": pos["gt_class_name"],
                "n_boxes": len(boxes), "pred_class": None,
                "pred_name": "DEGEN_CROP",
                "mean_probs": None, "outcome": "missed",
            })
            continue
        pos_caught += 1
        mean, per_box = classify_crops(classifier, crops, device)
        pred = int(np.argmax(mean))
        is_correct = (pred == pos["gt_class"])
        if is_correct:
            pos_correct += 1
        else:
            pos_wrong += 1
        conf_mat[pos["gt_class"], pred] += 1
        per_image_rows.append({
            "side": "positive", "stem": pos["stem"],
            "gt_class": pos["gt_class"], "gt_name": pos["gt_class_name"],
            "n_boxes": len(crops), "pred_class": pred,
            "pred_name": DISEASE_NAMES[pred],
            "mean_probs": {DISEASE_NAMES[k]: float(mean[k]) for k in range(6)},
            "outcome": "correct" if is_correct else "wrong",
        })
        tag = "OK " if is_correct else "ERR"
        print(f"  [{i:3d}/{len(positives)}] {pos['stem']:40s} GT={pos['gt_class_name']:14s} pred={DISEASE_NAMES[pred]:14s} boxes={len(crops)} {tag}")

    # ── Negatives ────────────────────────────────────────────────────────────
    neg_silent = 0
    neg_fa = 0
    fa_pred_counts = np.zeros(6, dtype=int)

    for i, neg in enumerate(negatives, 1):
        boxes = detect_boxes(detector, neg["image"], conf=conf)
        if not boxes:
            neg_silent += 1
            per_image_rows.append({
                "side": "negative", "stem": neg["stem"],
                "gt_class": 5, "gt_name": "Normal",
                "n_boxes": 0, "pred_class": None, "pred_name": "SILENT",
                "mean_probs": None, "outcome": "silent_ok",
            })
            continue
        neg_fa += 1
        bgr = cv2.imread(neg["image"])
        crops = [pad_and_crop(bgr, b[:4], pad) for b in boxes]
        crops = [c for c in crops if c is not None]
        if not crops:
            per_image_rows.append({
                "side": "negative", "stem": neg["stem"],
                "gt_class": 5, "gt_name": "Normal",
                "n_boxes": len(boxes), "pred_class": None,
                "pred_name": "DEGEN_CROP",
                "mean_probs": None, "outcome": "fa_degen",
            })
            continue
        mean, _ = classify_crops(classifier, crops, device)
        pred = int(np.argmax(mean))
        fa_pred_counts[pred] += 1
        per_image_rows.append({
            "side": "negative", "stem": neg["stem"],
            "gt_class": 5, "gt_name": "Normal",
            "n_boxes": len(crops), "pred_class": pred,
            "pred_name": DISEASE_NAMES[pred],
            "mean_probs": {DISEASE_NAMES[k]: float(mean[k]) for k in range(6)},
            "outcome": "fa",
        })

    # ── Headline ─────────────────────────────────────────────────────────────
    n_pos = len(positives)
    n_neg = len(negatives)
    cond_acc = pos_correct / pos_caught if pos_caught else 0.0

    summary = {
        "config": {
            "pad_frac": pad, "conf": conf,
            "detector": str(DETECTOR_WEIGHTS),
            "classifier": str(CLASSIFIER_WEIGHTS),
            "n_positives": n_pos, "n_negatives": n_neg,
        },
        "positives": {
            "caught": pos_caught, "missed": pos_missed,
            "caught_correct": pos_correct, "caught_wrong": pos_wrong,
            "system_accuracy": pos_correct / n_pos if n_pos else 0.0,
            "conditional_disease_accuracy": cond_acc,
            "catch_rate": pos_caught / n_pos if n_pos else 0.0,
        },
        "negatives": {
            "silent_ok": neg_silent, "false_alarm": neg_fa,
            "false_alarm_rate": neg_fa / n_neg if n_neg else 0.0,
            "fa_classifier_pred_counts": {
                DISEASE_NAMES[k]: int(fa_pred_counts[k]) for k in range(6)
            },
        },
        "confusion_matrix_caught": {
            "rows_gt_0_to_4": DISEASE_NAMES,
            "cols_pred_0_to_5": DISEASE_NAMES,
            "matrix": conf_mat.tolist(),
        },
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "per_image.jsonl", "w") as f:
        for row in per_image_rows:
            f.write(json.dumps(row) + "\n")

    lines = []
    lines.append(f"=== pad={pad}, conf={conf} ===")
    lines.append(f"Positives (n={n_pos}, fold {FOLD_INDEX} test slice):")
    lines.append(f"  Caught                       : {pos_caught} / {n_pos}  ({pos_caught/n_pos:.1%})")
    lines.append(f"    Caught + correct disease   : {pos_correct}")
    lines.append(f"    Caught + wrong disease     : {pos_wrong}")
    lines.append(f"  Missed (YOLO silent)         : {pos_missed}")
    lines.append(f"  Conditional disease accuracy : {cond_acc:.1%}  ← headline")
    lines.append(f"  System-level accuracy        : {pos_correct/n_pos:.1%}")
    lines.append("")
    lines.append(f"Negatives (n={n_neg}, OCD Normals):")
    lines.append(f"  Correctly silent             : {neg_silent} / {n_neg}  ({neg_silent/n_neg:.1%})")
    lines.append(f"  False alarmed                : {neg_fa} / {n_neg}  ({neg_fa/n_neg:.1%})")
    lines.append(f"  FA classifier predictions    : "
                 + ", ".join(f"{DISEASE_NAMES[k]}={fa_pred_counts[k]}" for k in range(6)))
    lines.append("")
    lines.append("Confusion matrix (caught positives only, rows=GT, cols=pred):")
    lines.append("             " + "  ".join(f"{DISEASE_NAMES[c][:6]:>6s}" for c in range(6)))
    for r in range(5):
        row = f"  {DISEASE_NAMES[r][:10]:10s} " + "  ".join(f"{conf_mat[r,c]:>6d}" for c in range(6))
        lines.append(row)
    txt = "\n".join(lines)
    with open(out_dir / "summary.txt", "w") as f:
        f.write(txt + "\n")
    print()
    print(txt)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--pads", type=float, nargs="+", default=[0.0, 0.2, 0.4])
    ap.add_argument("--negatives_mode", choices=["blackbox", "all"], default="all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"conf: {args.conf}, pads: {args.pads}, negatives_mode: {args.negatives_mode}")

    print("\nLoading test sets...")
    positives = load_positives()
    negatives = load_negatives(args.negatives_mode)
    print(f"  positives: {len(positives)} (fold {FOLD_INDEX} test slice)")
    print(f"  negatives: {len(negatives)} (OCD Normals, mode={args.negatives_mode})")

    print("\nLoading models...")
    detector = load_detector(device)
    classifier = load_classifier(device)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = {}
    for pad in args.pads:
        sub = OUTPUT_ROOT / f"pad_{pad:.2f}_neg_{args.negatives_mode}"
        s = run_one_pad(positives, negatives, detector, classifier,
                        device, args.conf, pad, sub)
        all_summaries[pad] = s

    print("\n\n" + "=" * 78)
    print("HEADLINE COMPARISON ACROSS PADDING VALUES")
    print("=" * 78)
    print(f"{'pad':>6s}  {'caught':>8s}  {'correct':>8s}  {'cond_acc':>9s}  {'sys_acc':>8s}  {'neg_FA':>8s}")
    rows = []
    for pad in args.pads:
        s = all_summaries[pad]
        p = s["positives"]; n = s["negatives"]
        rows.append(f"{pad:>6.2f}  {p['caught']:>4d}/{s['config']['n_positives']:<3d}  "
                    f"{p['caught_correct']:>4d}/{p['caught']:<3d}  "
                    f"{p['conditional_disease_accuracy']:>8.1%}  "
                    f"{p['system_accuracy']:>7.1%}  "
                    f"{n['false_alarm_rate']:>7.1%}")
    for r in rows:
        print(r)
    with open(OUTPUT_ROOT / f"comparison_neg_{args.negatives_mode}.txt", "w") as f:
        f.write(f"conf={args.conf}, negatives_mode={args.negatives_mode}\n")
        f.write(f"{'pad':>6s}  {'caught':>8s}  {'correct':>8s}  {'cond_acc':>9s}  {'sys_acc':>8s}  {'neg_FA':>8s}\n")
        f.write("\n".join(rows) + "\n")

    print(f"\n→ outputs in {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()
