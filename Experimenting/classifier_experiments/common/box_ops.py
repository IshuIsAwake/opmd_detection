"""
box_ops.py — Box-geometry helpers for post-detection processing.

Currently: IoU + greedy union-merge for the classifier-feeding stage.

Motivation. The detector tends to emit several overlapping boxes per lesion
(see Experimenting/full_pipeline_test/erythro2: 7 stacked boxes on one
inflamed region). Plain mean-softmax aggregation treats each of those views
as an independent vote, which both adds variance and pushes the input
distribution away from training (training crops cover a whole lesion +
padding; sub-box crops cover a sub-region). Merging overlapping boxes into a
single union rectangle restores the "one lesion → one classification" shape
the classifier was trained on.

Single-link clustering would risk bridging two distinct lesions via a third
overlapping box. We use anchor-based greedy clustering (top-conf box seeds a
cluster; remaining boxes join whichever existing cluster they overlap most
with, ≥ iou_thresh). Two genuine lesions that don't overlap each other stay
as separate clusters (e.g. lichen1, with one lesion on tongue + one on cheek).
"""

from __future__ import annotations

from dataclasses import dataclass

XYXY = tuple[float, float, float, float]


@dataclass
class MergedBox:
    xyxy: XYXY
    conf: float                # max conf across merged boxes
    n_merged: int              # how many raw boxes formed this cluster
    member_idx: list[int]      # original indices (debugging / diagnostics)


def iou(a: XYXY, b: XYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _union_xyxy(boxes: list[XYXY]) -> XYXY:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def merge_boxes(boxes: list[tuple[float, float, float, float, float]],
                iou_thresh: float = 0.3) -> list[MergedBox]:
    """Greedy anchor-based clustering.

    ``boxes`` is the detector's emit list: (x1, y1, x2, y2, conf), pixel
    coords. Returns a list of merged clusters, each with the union rectangle,
    the max original conf, and how many raw boxes formed it.

    Algorithm:
      1. Sort boxes by conf descending.
      2. Walk the sorted list. For each unassigned box:
           - For each existing cluster, compute IoU between the candidate
             and the cluster's *current union rectangle*. Assign to the
             cluster with the highest IoU, provided that IoU >= iou_thresh.
           - Otherwise the candidate seeds a new cluster.
      3. After all boxes are assigned, recompute final union per cluster.

    Using the cluster's running union (not just its anchor) lets a long
    chain of overlapping boxes merge cleanly — which is the erythro2 case.
    But we still require the candidate to overlap with the *union so far*,
    which prevents bridging two non-overlapping clusters via a third box
    that happens to touch both (the multi-lesion case).
    """
    if not boxes:
        return []

    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][4], reverse=True)

    clusters: list[dict] = []     # {"boxes": [xyxy...], "confs": [...], "idx": [...]}
    for orig_idx, b in indexed:
        candidate_xyxy: XYXY = b[:4]
        candidate_conf = b[4]

        best_ci = -1
        best_iou = 0.0
        for ci, cl in enumerate(clusters):
            cu = _union_xyxy(cl["boxes"])
            v = iou(candidate_xyxy, cu)
            if v >= iou_thresh and v > best_iou:
                best_iou = v
                best_ci = ci

        if best_ci == -1:
            clusters.append({"boxes": [candidate_xyxy],
                             "confs": [candidate_conf],
                             "idx": [orig_idx]})
        else:
            clusters[best_ci]["boxes"].append(candidate_xyxy)
            clusters[best_ci]["confs"].append(candidate_conf)
            clusters[best_ci]["idx"].append(orig_idx)

    out: list[MergedBox] = []
    for cl in clusters:
        out.append(MergedBox(
            xyxy=_union_xyxy(cl["boxes"]),
            conf=max(cl["confs"]),
            n_merged=len(cl["boxes"]),
            member_idx=sorted(cl["idx"]),
        ))
    # Stable order by conf desc, matches detect_image's emit order.
    out.sort(key=lambda m: m.conf, reverse=True)
    return out
