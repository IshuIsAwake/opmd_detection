"""
metrics.py — One honest evaluation, no metric tuned.

For every test image we run the model, reduce each prediction to its
axis-aligned box (OBB → enclosing rectangle of its 4 corners) and compare to
the axis-aligned ground truth. Per class, predictions are greedily matched to
GT by IoU (highest-confidence first); a match counts at IoU >= 0.5.

Reported (per class + micro overall), at the Ultralytics default conf=0.25 and
again at conf=0.001 as an untuned recall ceiling:

  precision, recall, f1                 box level
  mean IoU / IoP / IoG over matches     localisation quality on hits
  mean best-IoU per GT (any conf)       shows the loose-box penalty directly
  image-level: detection rate on positives, false-alarm rate on negatives,
               and overall screening accuracy (catches "fires on everything")

OBB models are NOT scored with Ultralytics' own val() (its test labels here
are axis boxes, not OBB) — only this custom pass. Detect models additionally
get a stock model.val() mAP for cross-reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONFS = (0.25, 0.001)        # headline (stock default) + untuned recall ceiling
IOU_MATCH = 0.5


# ── geometry ──────────────────────────────────────────────────────────────────

def _inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _area(x) -> float:
    return max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])


def iou(a, b) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def iop(pred, gt) -> float:           # intersection / prediction area
    a = _area(pred)
    return _inter(pred, gt) / a if a > 0 else 0.0


def iog(pred, gt) -> float:           # intersection / GT area
    a = _area(gt)
    return _inter(pred, gt) / a if a > 0 else 0.0


# ── match rules ───────────────────────────────────────────────────────────────
# A rule picks, among unused GTs, the best one by ``key(pred, gt)`` and accepts
# the pair iff ``ok(pred, gt)``. The default is IoU≥0.5 — byte-identical to the
# original behaviour, so every exp1–8 / Number-A number stays comparable. The
# alternatives are screening-leaning: IoG (did we cover the lesion) primary,
# with an IoP floor so a box can't game IoG by ballooning over the image.

class MatchRule:
    __slots__ = ("name", "key", "ok")

    def __init__(self, name, key, ok):
        self.name = name
        self.key = key            # (pred, gt) -> float : how good is this pair
        self.ok = ok              # (pred, gt) -> bool  : does it count as TP


IOU_RULE = MatchRule("iou>=0.5", iou, lambda p, g: iou(p, g) >= IOU_MATCH)

# Decided 2026-05-19 from the exp8 match-rule sweep: the IoP floor was inert
# (the model does not balloon boxes — coverage-only ≈ floored, ≤1 box across
# every run/conf), so it is dropped. IoG≥0.5 ("the detection covers ≥half the
# annotated lesion") is the HEADLINE localisation metric — defensible, not
# gameable by a tiny speck, and what the downstream padded crop needs.
# IoU≥0.5 is kept for exp1–8 continuity; IoG≥0.3 is a looser secondary view.
IOG_RULE = MatchRule("iog>=0.5", iog, lambda p, g: iog(p, g) >= 0.5)

MATCH_RULES = (
    IOU_RULE,                                       # exp1–8 continuity
    IOG_RULE,                                        # HEADLINE localisation
    MatchRule("iog>=0.3 & iop>=0.10", iog,           # screening-loose secondary
              lambda p, g: iog(p, g) >= 0.3 and iop(p, g) >= 0.10),
)


# ── label / prediction extraction ─────────────────────────────────────────────

def _gt_for(label_path: Path, w: int, h: int) -> list[tuple[int, tuple]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        try:
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(v) for v in p[1:])
        except ValueError:
            continue
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        out.append((c, (x1, y1, x2, y2)))
    return out


def _preds_from_result(res) -> list[tuple[int, float, tuple]]:
    """(class, conf, axis-aligned xyxy) for detect OR obb results."""
    out = []
    if getattr(res, "obb", None) is not None and res.obb is not None and len(res.obb):
        corners = res.obb.xyxyxyxy.cpu().numpy()      # (N, 4, 2)
        confs = res.obb.conf.cpu().numpy()
        clss = res.obb.cls.cpu().numpy()
        for pts, cf, cl in zip(corners, confs, clss):
            xs, ys = pts[:, 0], pts[:, 1]
            out.append((int(cl), float(cf),
                        (float(xs.min()), float(ys.min()),
                         float(xs.max()), float(ys.max()))))
    elif res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy()
        for b, cf, cl in zip(xyxy, confs, clss):
            out.append((int(cl), float(cf),
                        (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
    return out


# ── core scoring ──────────────────────────────────────────────────────────────

def _score(per_image, n_classes, conf, rule=None):
    """per_image: list of (gts, preds). Returns metrics dict at this conf.
    ``rule`` defaults to IoU≥0.5 (original behaviour); pass a MatchRule to
    re-score box P/R/F1 under a different criterion. Image-level metrics
    (det_rate / false_alarm / screening_acc) are geometry-free and identical
    under every rule."""
    rule = rule or IOU_RULE
    tp = [0] * n_classes
    fp = [0] * n_classes
    fn = [0] * n_classes
    ious, iops, iogs = [], [], []
    best_iou_per_gt = []
    img_total = img_correct = pos_total = pos_hit = neg_total = neg_fp = 0

    for gts, preds in per_image:
        preds = [p for p in preds if p[1] >= conf]
        gt_any = len(gts) > 0
        pred_any = len(preds) > 0

        img_total += 1
        img_correct += int(gt_any == pred_any)
        if gt_any:
            pos_total += 1
            pos_hit += int(pred_any)
        else:
            neg_total += 1
            neg_fp += int(pred_any)

        for c in range(n_classes):
            gt_c = [g[1] for g in gts if g[0] == c]
            pr_c = sorted([p for p in preds if p[0] == c],
                          key=lambda x: -x[1])
            used = [False] * len(gt_c)
            for _, _, pbox in pr_c:
                best_j, best_v = -1, -1.0
                for j, gbox in enumerate(gt_c):
                    if used[j]:
                        continue
                    v = rule.key(pbox, gbox)
                    if v > best_v:
                        best_v, best_j = v, j
                if best_j >= 0 and rule.ok(pbox, gt_c[best_j]):
                    used[best_j] = True
                    tp[c] += 1
                    g = gt_c[best_j]
                    ious.append(iou(pbox, g))
                    iops.append(iop(pbox, g))
                    iogs.append(iog(pbox, g))
                else:
                    fp[c] += 1
            fn[c] += used.count(False)

            for gbox in gt_c:
                bv = max((iou(p[2], gbox) for p in pr_c), default=0.0)
                best_iou_per_gt.append(bv)

    def pr(t, f):
        return t / (t + f) if (t + f) else 0.0

    per_class = {}
    for c in range(n_classes):
        p = pr(tp[c], fp[c])
        r = pr(tp[c], fn[c])
        per_class[c] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c],
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if (p + r) else 0.0,
        }
    TP, FP, FN = sum(tp), sum(fp), sum(fn)
    P, R = pr(TP, FP), pr(TP, FN)
    return {
        "conf": conf,
        "overall": {
            "tp": TP, "fp": FP, "fn": FN,
            "precision": round(P, 4), "recall": round(R, 4),
            "f1": round(2 * P * R / (P + R), 4) if (P + R) else 0.0,
        },
        "per_class": per_class,
        "localisation_on_hits": {
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "mean_iop": round(float(np.mean(iops)), 4) if iops else 0.0,
            "mean_iog": round(float(np.mean(iogs)), 4) if iogs else 0.0,
            "n": len(ious),
        },
        "mean_best_iou_per_gt": (
            round(float(np.mean(best_iou_per_gt)), 4) if best_iou_per_gt else 0.0
        ),
        "image_level": {
            "screening_accuracy": round(img_correct / img_total, 4) if img_total else 0.0,
            "detection_rate_on_positives": round(pos_hit / pos_total, 4) if pos_total else 0.0,
            "false_alarm_rate_on_negatives": round(neg_fp / neg_total, 4) if neg_total else 0.0,
            "positives": pos_total, "negatives": neg_total,
            # Raw counts so results read in plain English ("27 of 37 lesion
            # images flagged, 30 of 74 total"). Image-level: a lesion *image*
            # is "flagged" if the model emitted any box, regardless of where.
            "positives_flagged": pos_hit,
            "negatives_flagged": neg_fp,
            "total_flagged": pos_hit + neg_fp,
            "total_images": img_total,
        },
    }


def evaluate(model, spec, results_dir: Path) -> dict:
    """Run the model over the test split and write metrics.{json,txt}."""
    n_classes = len(spec.class_names)
    per_image = []
    for img in spec.test_images:
        res = model.predict(str(img), device=_dev(), verbose=False,
                             conf=min(CONFS), imgsz=_imgsz())[0]
        w, h = res.orig_shape[1], res.orig_shape[0]
        gts = _gt_for(spec.root / "labels" / "test" / f"{Path(img).stem}.txt", w, h)
        per_image.append((gts, _preds_from_result(res)))

    blocks = {f"conf_{c}": _score(per_image, n_classes, c) for c in CONFS}
    # Box P/R/F1 + loc under every reported match rule (image-level metrics are
    # geometry-free → identical across rules, so only overall+loc are stored).
    sweep = {
        f"conf_{c}": [
            {"rule": r.name,
             "overall": _score(per_image, n_classes, c, rule=r)["overall"],
             "localisation_on_hits":
                 _score(per_image, n_classes, c, rule=r)["localisation_on_hits"]}
            for r in MATCH_RULES
        ]
        for c in CONFS
    }
    report = {
        "experiment": spec.name,
        "task": spec.task,
        "class_names": spec.class_names,
        "n_test_images": len(spec.test_images),
        "iou_match_threshold": IOU_MATCH,
        "headline_localisation_rule": IOG_RULE.name,
        "metrics": blocks,
        "match_rule_sweep": sweep,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (results_dir / "metrics.txt").write_text(_pretty(report))
    return report


# ── helpers shared with train_eval ────────────────────────────────────────────

def _dev():
    from common import settings
    return settings.DEVICE


def _imgsz():
    from common import settings
    return settings.IMGSZ


def _pretty(r: dict) -> str:
    L = [f"# {r['experiment']}  ({r['task']}, {r['n_test_images']} test imgs)",
         f"# classes: {', '.join(r['class_names'])}",
         f"# IoU match threshold: {r['iou_match_threshold']}", ""]
    for key, b in r["metrics"].items():
        o = b["overall"]
        il = b["image_level"]
        loc = b["localisation_on_hits"]
        L += [
            f"== {key} ==",
            f"  box     P={o['precision']:.3f} R={o['recall']:.3f} "
            f"F1={o['f1']:.3f}  (TP={o['tp']} FP={o['fp']} FN={o['fn']})",
            f"  loc/hit IoU={loc['mean_iou']:.3f} IoP={loc['mean_iop']:.3f} "
            f"IoG={loc['mean_iog']:.3f}  (n={loc['n']})",
            f"  mean best IoU per GT (any conf) = {b['mean_best_iou_per_gt']:.3f}",
            f"  image   screening_acc={il['screening_accuracy']:.3f}  "
            f"det_rate_pos={il['detection_rate_on_positives']:.3f}  "
            f"false_alarm_neg={il['false_alarm_rate_on_negatives']:.3f}  "
            f"(pos={il['positives']} neg={il['negatives']})",
            f"  counts  {il['positives_flagged']}/{il['positives']} lesion-imgs "
            f"flagged · {il['negatives_flagged']}/{il['negatives']} neg "
            f"false-alarm · {il['total_flagged']}/{il['total_images']} "
            f"total flagged",
        ]
        if len(r["class_names"]) > 1:
            for c, pc in b["per_class"].items():
                L.append(f"    [{r['class_names'][c]:<14}] "
                         f"P={pc['precision']:.3f} R={pc['recall']:.3f} "
                         f"F1={pc['f1']:.3f}")
        L.append("")

    sweep = r.get("match_rule_sweep")
    if sweep:
        head = r.get("headline_localisation_rule", "")
        L += ["== match-rule sweep (box P/R/F1 only; image-level is "
              "geometry-free → see above) =="]
        for key, rows in sweep.items():
            L.append(f"  {key}")
            for row in rows:
                o, loc = row["overall"], row["localisation_on_hits"]
                tag = "  <- headline" if row["rule"] == head else ""
                L.append(f"    {row['rule']:<22} "
                         f"P={o['precision']:.3f} R={o['recall']:.3f} "
                         f"F1={o['f1']:.3f}  (TP={o['tp']} FP={o['fp']} "
                         f"FN={o['fn']})  IoG={loc['mean_iog']:.3f}{tag}")
        L.append("")
    return "\n".join(L)
