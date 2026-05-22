"""
predict_bulk_sanity.py — fire-rate + det_rate + IoG + screening_acc on
Roboflow + Normal images. Sanity check; not a CV claim.

Runs (one image at a time → OOM-proof on 6 GB):
  fold 0  →  Leukoplakia.v2  +  OPMD-SEG  +  Normal
  fold 4  →  OSMF DETECTION  +  Normal

Per Roboflow folder reports fire_rate, det_rate (label-positives only),
mean IoG on matched hits. Per Normal folder reports false_alarm.
Combines (positives + Normal) on the same fold into screening_acc.

Label parsing handles BOTH 5-token YOLO boxes (Leukoplakia.v2) AND polygons
(OPMD-SEG, OSMF DETECTION) — poly→bbox by min/max of x/y. OSMF class 0 =
non-osmf (treated as non-lesion); only class 1 counts as a real GT lesion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from ultralytics import YOLO

K5_ROOT = HERE / "results" / "kfold5_geom_no_color_binary"
FOLD_WEIGHTS = {k: K5_ROOT / f"fold_{k}" / "train" / "weights" / "best.pt"
                for k in range(5)}
CONFS = [0.05, 0.075, 0.10, 0.25]
IMGSZ = 640
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IOG_MATCH = 0.5            # threshold for counting a pred as a "hit" on a GT


# ── label loading (box + polygon) ────────────────────────────────────────────

def load_gt(label_path: Path, w: int, h: int,
            lesion_classes: set[int] | None) -> list[tuple[float, float, float, float]]:
    """Return GT boxes (xyxy pixel coords). Polygons → min/max bbox.

    ``lesion_classes`` filters which class IDs count as a real lesion. None ⇒
    every line is a lesion (Leukoplakia.v2, OPMD-SEG). For OSMF pass {1} so
    class-0 non-osmf lines are ignored.
    """
    if not label_path.exists():
        return []
    out: list[tuple[float, float, float, float]] = []
    for raw in label_path.read_text().strip().splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            nums = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if lesion_classes is not None and cls not in lesion_classes:
            continue
        if len(nums) == 4:                              # standard YOLO box
            cx, cy, bw, bh = nums
            x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
            x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        elif len(nums) >= 6 and len(nums) % 2 == 0:     # polygon
            xs = [nums[i] * w for i in range(0, len(nums), 2)]
            ys = [nums[i] * h for i in range(1, len(nums), 2)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            continue
        if x2 - x1 > 1 and y2 - y1 > 1:
            out.append((x1, y1, x2, y2))
    return out


def list_images_with_labels(root: Path) -> list[tuple[Path, Path | None]]:
    """For Roboflow trees, label is `<split>/labels/<stem>.txt` (sibling of
    `images/`). Return (image_path, label_path_or_None).

    Dedupes Roboflow augmented copies: filenames share the stem before
    `.rf.<hash>`. Same convention `common/datasets.py` uses — keep one
    deterministic representative (first alphabetically) per base stem.
    """
    seen: set[str] = set()
    pairs: list[tuple[Path, Path | None]] = []
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS):
        base = img.stem.split(".rf.")[0]
        if base in seen:
            continue
        seen.add(base)
        label = None
        for anc in img.parents:
            if anc.name == "images":
                lbl = anc.parent / "labels" / (img.stem + ".txt")
                if lbl.exists():
                    label = lbl
                break
        pairs.append((img, label))
    return pairs


def list_normal(root: Path) -> list[tuple[Path, Path | None]]:
    return [(p, None) for p in sorted(root.rglob("*"))
            if p.suffix.lower() in EXTS]


# ── geometry helpers ────────────────────────────────────────────────────────

def inter(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def area(x) -> float:
    return max(0.0, x[2] - x[0]) * max(0.0, x[3] - x[1])


def iog_val(pred, gt) -> float:
    a = area(gt)
    return inter(pred, gt) / a if a > 0 else 0.0


# ── one folder ──────────────────────────────────────────────────────────────

def run_folder(model: YOLO, label: str, root: Path,
               lesion_classes: set[int] | None, is_normal: bool) -> dict:
    if is_normal:
        items = list_normal(root)
    else:
        items = list_images_with_labels(root)
    n = len(items)
    print(f"── {label}")
    print(f"   {n} images under {root.relative_to(PROJECT_ROOT)}")

    fires = {c: 0 for c in CONFS}                       # image-level fire counts
    pos_fires = {c: 0 for c in CONFS}                   # only on label-positives
    n_pos = 0
    iog_hits_sum = 0.0
    iog_hits_n = 0

    min_conf = min(CONFS)
    for img_path, lbl_path in items:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            continue
        gts = (load_gt(lbl_path, w, h, lesion_classes)
               if lbl_path is not None else [])
        has_gt = bool(gts)
        if has_gt:
            n_pos += 1

        res = model.predict(source=str(img_path), conf=min_conf, imgsz=IMGSZ,
                            device=0, verbose=False, save=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().tolist()
        confs = res.boxes.conf.cpu().tolist()
        max_c = max(confs)
        for c in CONFS:
            if max_c >= c:
                fires[c] += 1
                if has_gt:
                    pos_fires[c] += 1

        # IoG on hits: for each GT, take the best-IoG pred (any conf above min)
        # — counts as a "hit" if IoG >= IOG_MATCH. Averaged at folder level.
        if has_gt:
            for gt in gts:
                best = max((iog_val(tuple(p), gt) for p in xyxy), default=0.0)
                if best >= IOG_MATCH:
                    iog_hits_sum += best
                    iog_hits_n += 1

    out = {
        "label": label, "n": n, "n_pos": n_pos,
        "fires": fires, "pos_fires": pos_fires,
        "iog_mean_on_hits": (iog_hits_sum / iog_hits_n) if iog_hits_n else None,
        "iog_n_hits": iog_hits_n,
        "is_normal": is_normal,
    }
    return out


def fmt_pct(num: int, den: int) -> str:
    return f"{num}/{den} ({100.0*num/den:5.1f}%)" if den else "—"


def print_folder(out: dict) -> None:
    """One folder block. Positives show fire_rate + det_rate; Normal shows
    fire_rate only (= false_alarm). No cross-distribution mixing."""
    n, n_pos = out["n"], out["n_pos"]
    for c in CONFS:
        f = out["fires"][c]
        line = f"     conf={c:.3f}: fired {f:5d} / {n:5d} ({100.0*f/n:5.1f}%)"
        if not out["is_normal"] and n_pos:
            pf = out['pos_fires'][c]
            line += f"   det_rate={pf}/{n_pos} ({100.0*pf/n_pos:5.1f}%)"
        print(line)
    if out["iog_mean_on_hits"] is not None:
        print(f"     IoG mean on hits (≥{IOG_MATCH}): {out['iog_mean_on_hits']:.3f}  "
              f"(n_hits={out['iog_n_hits']})")
    print()


def screening(pos: dict, neg: dict, conf: float) -> dict:
    """screening_acc = (pos correctly fired + neg correctly silent) / total.
    Uses image-level fire (≥1 box) — geometry-free, matches `metrics.py`."""
    n_p = pos["n_pos"] if not pos["is_normal"] else 0   # use label-positives only
    tp = pos["pos_fires"][conf]
    n_n = neg["n"]
    fp = neg["fires"][conf]
    tn = n_n - fp
    fn = n_p - tp
    total = n_p + n_n
    return {
        "screening_acc": (tp + tn) / total if total else 0.0,
        "det_rate_pos":  tp / n_p if n_p else 0.0,
        "false_alarm":   fp / n_n if n_n else 0.0,
        "n_pos": n_p, "n_neg": n_n,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # folder configs (lesion_classes filter): None = any class is lesion
    cfgs = {
        "Leukoplakia.v2 (Roboflow)":
            (PROJECT_ROOT / "data" / "additional" / "Leukoplakia.v2i.yolov8",
             None, False),
        "OPMD-SEG (Roboflow)":
            (PROJECT_ROOT / "data" / "additional" / "OPMD-SEG.v1i.yolov8",
             None, False),
        "OSMF DETECTION (Roboflow, cls1=osmf only)":
            (PROJECT_ROOT / "data" / "additional" / "OSMF DETECTION.v1i.yolov8",
             {1}, False),
        "Normal (healthy mouths)":
            (PROJECT_ROOT / "data" / "Normal", None, True),
    }
    # All 4 folders × all 5 folds. Normal first per fold so positives can
    # print inline screening_acc.
    NORMAL = "Normal (healthy mouths)"
    pos_names = [n for n in cfgs if n != NORMAL]
    folds = sorted(FOLD_WEIGHTS)

    # results[fold][folder_name] = run_folder dict
    results: dict[int, dict[str, dict]] = {}

    for fold in folds:
        w = FOLD_WEIGHTS[fold]
        if not w.exists():
            raise SystemExit(f"fold {fold} weights missing: {w}")
        print(f"\n══════ FOLD {fold}  ({w.relative_to(PROJECT_ROOT)}) ══════")
        model = YOLO(str(w))
        results[fold] = {}

        root, lesion_classes, is_normal = cfgs[NORMAL]
        neg = run_folder(model, NORMAL, root, lesion_classes, is_normal)
        print_folder(neg)
        results[fold][NORMAL] = neg

        for name in pos_names:
            root, lesion_classes, is_normal = cfgs[name]
            if not root.exists():
                print(f"!! missing: {root}"); continue
            out = run_folder(model, name, root, lesion_classes, is_normal)
            print_folder(out)
            results[fold][name] = out

    # ── summary: mean ± std across folds, per folder per conf ────────────────
    import statistics as st

    def mstd(xs: list[float]) -> str:
        if not xs:
            return "—"
        if len(xs) == 1:
            return f"{xs[0]:.3f}"
        return f"{st.mean(xs):.3f}±{st.stdev(xs):.3f}"

    print("\n══════ SUMMARY (mean ± std across 5 folds) ══════")
    print(f"\n── {NORMAL}  —  false_alarm only (no positives in this set)")
    print(f"   {'conf':>6}  {'FA':>14}")
    for c in CONFS:
        fa = [results[f][NORMAL]["fires"][c] / results[f][NORMAL]["n"]
              for f in folds if NORMAL in results[f]]
        print(f"   {c:>6.3f}  {mstd(fa):>14}")

    for name in pos_names:
        runs = [results[f][name] for f in folds if name in results[f]]
        if not runs:
            continue
        print(f"\n── {name}  —  positives only (no negatives in this set)")
        print(f"   {'conf':>6}  {'fire_rate':>14}  {'det_rate':>14}")
        for c in CONFS:
            fire = [r["fires"][c] / r["n"] for r in runs]
            det = [r["pos_fires"][c] / r["n_pos"]
                   for r in runs if r["n_pos"]]
            print(f"   {c:>6.3f}  {mstd(fire):>14}  {mstd(det):>14}")
        iogs = [r["iog_mean_on_hits"] for r in runs
                if r["iog_mean_on_hits"] is not None]
        if iogs:
            print(f"   IoG mean on hits: {mstd(iogs)}  (over {len(iogs)} folds)")


if __name__ == "__main__":
    main()
