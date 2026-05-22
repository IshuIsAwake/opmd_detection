# Experiment Results — chronological record

Every detector and classifier experiment we have run on this project,
most recent first. Each entry says **what** was done, **why**, the
**numbers**, and the **verdict**.

> Authority. Where this file and code disagree, this file describes
> intent and measured outcomes. Investigate the gap — do not silently
> rewrite one to match the other.

---

## Timeline (most recent → least recent)

| # | date | name | one-line result |
|---|---|---|---|
| 17 | 2026-05-23 | **TTA-only adopted as MVP** | B0 aug + 4-view TTA at serve_pad=0.20: **cond_acc 0.660 ± 0.041, sys_acc 0.597** on folds 1-4. Clears the §12 bar by +3.3 pp / +1.4 pp in leak-free eval. Production config locked. |
| 16 | 2026-05-23 | **Union-box merging — null on headline** | Best merged 0.616 ± 0.064 vs un-merged 0.630 ± 0.042 — within noise. Best serve_pad shifts 0.20 → 0.00 (mechanism confirmed; ceiling unchanged). Kept behind a flag, default off. |
| 15 | 2026-05-22 | **Round 3 — B0 strong augmentation** | ColorJitter + RandomResizedCrop + RandomErasing + wider rotation. Best cell 0.630 ± 0.042 — **no measurable lift** over basic-aug B0. Data scale, not input diversity, is the bottleneck. |
| 14 | 2026-05-22 | **Round 2 — EfficientNet-B0 end-to-end** | Full fine-tune (~5 M params, discriminative LR). Best cell **0.626 ± 0.025** (+0.05 over Round 1, std halved). Fold-3 outlier disappears. Locality bias confirmed: best serve_pad 0.00 → 0.20. |
| 13 | 2026-05-22 | **Round 1 — DINOv2-S frozen head** | 4-arm Phase 1 + 3×3 Phase 2 matrix. Best cell **0.576 ± 0.050**. Right scaffold, wrong backbone — frozen ViT-S/14 at 224 px under-fits this domain. |
| 12 | 2026-05-22 | **YOLO → old EffNet-B2 classifier (MVP measurement)** | fold-3 detector + borrowed OCD classifier: caught 67/72 (93.1%), conditional disease accuracy **62.7%** at pad = 0.0, system-level accuracy **58.3%**. Padding hurts. |
| 11 | 2026-05-21 | **§10 OOD sanity check** | Normal FA jumps 0.047 → **0.147 ± 0.049** on raw phone photos; Roboflow det_rate 0.77–0.86. Demo-readiness signal, not a CV claim. Fold 0 is globally less aggressive — do **not** ship fold 0. |
| 10 | 2026-05-21 | **§9 confidence-threshold sweep** | Same kfold5 weights, inference only. Screening lifts from 0.842 (conf=0.25) → **0.917 (conf=0.10)** — +0.076 free. Adopted operating point: conf = 0.10. |
|  9 | 2026-05-21 | **exp11 — paired 5-fold CV, `geom_no_color` vs default** | HSV-off validated: +0.101 paired screening, 3.6× tighter std, wins 4/5 folds. This is the model whose weights the §9 sweep used. |
|  8 | 2026-05-20 | **exp10 — augmentation sweep (6 levels)** | `geom_no_color` (= YOLO defaults but HSV = 0) wins on a single 85/15 split: 0.919. Caveat: single split. exp11 paired-tested it. |
|  7 | 2026-05-20 | **exp9 — 10-fold CV of exp8b** | Reproduced exp8b's 0.865 within noise: mean **0.840 ± 0.063**. The FA@conf-0.001 ≈ 0.95 framing was a tail draw — typical is 0.62. |
|  6 | 2026-05-19 | **exp8a / exp8b — the "detection is dead" retraction** | exp8b binary, resolution-normalized 1:1 negatives in train: **28/37 caught, 1/37 false alarm, screening 0.865**. Detection is a credible screener. |
|  5 | 2026-05-19 | **eval helpers — Number A & match-rule sweep** | Re-scored exp1/exp2 on the fair test, no retraining. Confirmed the 570-Normal "below trivial" verdict was a base-rate + missing-negatives artifact, not a model bug. IoG ≥ 0.5 adopted as headline localisation rule. |
|  4 | 2026-05-19 | **exp7 — binary + 955 Roboflow images** | F1 / mAP50 flat, recall ↓, only specificity ↑. **Roboflow scale-up does not transfer** to the original domain. |
|  3 | 2026-05-19 | **exp3 / exp5 — Roboflow per-disease experts** | mAP50 0.57–0.62 on Roboflow's own val, **precision 0.015–0.031 on original**. Structural domain shift, not augmentation or label noise. |
|  2 | 2026-05-19 | **exp1 / exp2 — 5-class vs binary, original data** | Binary YOLO ≈ **3× the image-level recall** of 5-class on identical data. The two-stage structure is empirically justified. |
|  1 | 2026-05-18 | **`src/` pipeline — `original_only` vs `plus_roboflow`** | Originally concluded "Roboflow clearly helps"; later overturned by exp7. Pipeline is runnable history; numbers are stale. |
|  0 | — | **Phase 0 — three abandoned approaches** | YOLO → YOLO cascade (~25%), EfficientNet on human-bbox crops (75% eval / 0% prod), LAB/CLAHE preprocessing. All three retired. |

---

# 17. TTA-only adopted as MVP — paired evaluation (2026-05-23)

**Script:** `Experimenting/classifier_experiments/phase2_pipeline.py --tta`
**Outputs:** `Experimenting/classifier_experiments/results/phase2_b0_aug_tta/`,
`phase2_b0_aug_merged_tta/`

### Purpose

Last lever before locking the MVP: does 4-view test-time augmentation
move the B0 fine-tune past the §12 baseline in a leak-free way? The
alternative (5-fold ensemble of B0 classifier weights) carries
4-of-5 train/test leakage on the Phase 2 evaluation slices — every
test image was in 4 folds' training sets — so ensembling cannot be
honestly evaluated on the existing held-out data. TTA is the same
model on the same fold's held-out slice, augmented only at inference.

### Setup

Same B0 strong-aug weights (§15), same Phase 2 folds {1, 2, 3, 4}
(fold 0 excluded per §10), same detector
kfold5_geom_no_color_binary at conf = 0.10, same train_pad = 0.4
GT-crop training. Single variable change: at inference each crop
expands into 4 views (identity, hflip, rotate +10°, rotate −10°);
softmaxes averaged across views per crop, then across boxes per image.
Post-softmax averaging (not logits) — respects the simplex.

4-view recipe stays in the geometric range the strong-aug training
already saw (±20° rotation + hflip). Avoided vflip (changes oral
anatomy semantics — upper vs lower mucosa) and crop-jitter (unstable
on small lesion crops).

### Headline (best cell, train_pad = 0.4)

| config | serve | cond_acc | sys_acc | Δ vs baseline |
|---|---|---|---|---|
| baseline (§14 R3 aug) | 0.20 | 0.630 ± 0.042 | 0.569 | — |
| **TTA only** | **0.20** | **0.660 ± 0.041** | **0.597** | **+3.0 pp** |
| merged only (§16) | 0.20 | 0.616 ± 0.064 | 0.555 | −1.4 pp |
| merged + TTA | 0.20 | 0.655 ± 0.047 | 0.594 | +2.5 pp |

### Findings

1. **TTA alone +3.0 pp cond_acc, +2.8 pp sys_acc**, std unchanged
   (0.041 vs 0.042). Clean lift, no robustness penalty.
2. **Merging on top of TTA is redundant.** Merged + TTA ≈ TTA alone
   (within noise). The geometric variance merging tries to reduce is
   the same variance TTA averages over — they fight for the same lift.
3. **Mechanism**: TTA does not add model capacity, only reduces
   prediction variance across geometric views. That's why it works
   where the architecture-scale and aug-suite levers (§14, §15) had
   already plateaued — different statistical pool.

### vs §12 borrowed-OCD baseline

| | cond_acc | sys_acc |
|---|---|---|
| §12 (fold 3, ~90% leakage) | 0.627 | 0.583 |
| **MVP (folds 1-4, no leakage)** | **0.660** | **0.597** |

+3.3 pp / +1.4 pp on a stricter eval. The §12 bar is decisively cleared.

### Verdict

**Adopted production config:**

| component | choice |
|---|---|
| detector | `Experimenting/results/kfold5_geom_no_color_binary/fold_2/train/weights/best.pt` |
| detector conf | 0.10 |
| classifier | `Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/fold_2/best.pt` |
| serve_pad | 0.20 |
| TTA | 4 views (identity, hflip, rot ±10°) |
| merge | off |
| aggregation | mean softmax across TTA views → across boxes → argmax |

This is the MVP. Per-fold worst case at this cell: fold 4 cond_acc
0.625 (sys_acc 0.563). Mean across folds 0.660. Headline number
to report: **0.660 ± 0.041**.

---

# 16. Union-box merging — null result on the headline (2026-05-23)

**Script:** `Experimenting/classifier_experiments/phase2_pipeline.py --merge-boxes`
**Outputs:** `Experimenting/classifier_experiments/results/phase2_b0_aug_merged/`,
`phase2_b0_aug_merged_tta/`

### Purpose

Visual inspection (`inspect_pipeline.py` on `Experimenting/internet_images/`)
showed the detector frequently emits N overlapping boxes on one lesion
(erythro2: 7 boxes on one inflamed region; lichen2: 2 nested boxes;
several others 3-5). Each box passes through the classifier as an
independent vote, then mean-softmax aggregates. This is a train/serve
crop-shape mismatch: training crops cover a whole lesion + 40 % pad,
but the pipeline feeds the classifier multiple sub-views.

Hypothesis: cluster boxes with IoU ≥ 0.3, replace each cluster with
its bounding-rect union, then crop + classify per cluster. One
classification per lesion region, matching the training distribution.

### Setup

Anchor-based greedy clustering at `iou_thresh = 0.3`
(`common/box_ops.py`). The candidate joins whichever existing cluster
has the highest IoU with its current running union, provided that
IoU ≥ threshold; otherwise it seeds a new cluster. Single-link
clustering avoided to prevent bridging genuine multi-lesion images
(e.g. lichen1: two lesions, tongue + cheek, IoU 0.0 between them).

Validated against 4 actual cases from `inspect_pipeline` before the
Phase 2 run:

| case | raw → merged | check |
|---|---|---|
| erythro2 (7 stacked) | 7 → 1 | ✓ |
| lichen1 (2 distinct lesions) | 2 → 2 | ✓ |
| eythro1 (lesion + finger FP) | 3 → 2 | ✓ (FP stays separate) |
| leuko3 (mixed) | 4 → 3 | ✓ |

### Headline (best cell, train_pad = 0.4, no TTA)

| serve | un-merged (§14 R3) | merged | Δ |
|---|---|---|---|
| 0.00 | 0.588 ± 0.058 | 0.610 ± 0.036 | +0.022 |
| **0.20** | **0.630 ± 0.042** | 0.616 ± 0.064 | −0.014 |
| 0.40 | 0.600 ± 0.061 | 0.581 ± 0.063 | −0.019 |

### Findings

1. **Headline unmoved.** Best merged 0.616 vs un-merged 0.630 — within
   noise both ways.
2. **Structural shift in best serve_pad.** Un-merged wants
   serve_pad = 0.20 (extra padding fills the lesion); merged wants
   serve_pad = 0.00 (the union rectangle already widened the crop).
   Same total cropped area, same accuracy. Mechanism confirmed;
   ceiling unmoved.
3. **Std improves at serve = 0.00.** Tightest cell across both
   variants (0.036 vs un-merged-at-0.00 of 0.058). Useful for
   per-fold robustness even though mean didn't move.

### Verdict

Crop aggregation is not the bottleneck. Not the right lever. The
flag stays in `phase2_pipeline.py --merge-boxes` for future use
(multi-lesion visualisation, sanity checks), default off. §17's TTA
work was done on the un-merged path.

---

# 15. Round 3 — B0 with strong augmentation (2026-05-22)

**Script:** `Experimenting/classifier_experiments/exp_b0_aug_gt_pad04.py`
**Outputs:** `Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/`,
`phase2_b0_aug/`

### Purpose

Round 2's B0 fine-tune (§14) showed clear overfitting (train_acc → 1.0
by epoch 10 while val plateaus at ~0.65). Test whether stronger
augmentation closes the train/val gap and lifts Phase 2.

### Setup

Single arm, B0 at train_pad = 0.4 (Round 2's headline) with the strong
aug suite: `RandomResizedCrop (0.7–1.0)`, ±20° rotation,
`ColorJitter (b = 0.2, c = 0.2, sat = 0.1, hue = 0)`,
`RandomErasing (p = 0.25)`. **Hue stays at 0** — colour is diagnostic
on this task, same rule as the detector's `geom_no_color` (see §9).
Everything else (B0 weights init, LR, epochs, fold split, head)
identical to Round 2.

### Headline (Phase 2, train_pad = 0.4, no TTA, no merge)

| serve | Round 2 (basic aug) | **Round 3 (strong aug)** | Δ |
|---|---|---|---|
| 0.00 | 0.589 ± 0.049 | 0.588 ± 0.058 | −0.001 |
| **0.20** | **0.626 ± 0.025** | **0.630 ± 0.042** | +0.004 |
| 0.40 | 0.599 ± 0.045 | 0.600 ± 0.061 | +0.001 |

### Findings

1. **No measurable lift.** Means unchanged within noise; std actually
   increased slightly on the best cell.
2. **Mechanism.** ImageNet pretraining already absorbed substantial
   geometric/colour variation across 1.3 M photos. Strong aug perturbs
   inputs the backbone is already invariant to; the head's overfitting
   is to feature-combinations (~400 examples × 5 classes), not raw
   pixels.
3. **Right negative result.** Rules out augmentation as the lever and
   shifts the diagnosis to either data scale (§14 verdict) or
   post-detection logic (§16, §17). §17 succeeds; §16 doesn't.

### Verdict

Augmentation is not the bottleneck on this data scale. Round 3 weights
are the ones §16 and §17 build on — kept on disk as the official
classifier weights:
`Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/`.

---

# 14. Round 2 — EfficientNet-B0 end-to-end fine-tune (2026-05-22)

**Scripts:** `Experimenting/classifier_experiments/exp1_b0_whole_image.py`,
`exp2{a,b,c}_b0_gt_pad*.py`,
`phase2_pipeline.py --backbone b0`

### Purpose

Round 1's frozen DINOv2-S (§13) maxed at 0.576 Phase 2 cond_acc.
Architectural hypothesis: ViT-S frozen at 224 px lacks the locality
bias appropriate for textural medical imaging on this data scale.
CNNs with full fine-tune are the natural counterpoint — ~5 M params
fit comfortably on a 6 GB GPU at batch 32, full backbone adapts to
mucosal texture, ImageNet pretraining provides domain-agnostic
low-level filters.

### Setup

EfficientNet-B0 (`torchvision.models.efficientnet_b0`,
`IMAGENET1K_V1`), full end-to-end fine-tune. Discriminative LR: head
1e-3, backbone 1e-4 (10× lower). Cosine schedule. Same head shape as
Round 1 (`Linear(1280→256) → GELU → Dropout → Linear(256→5)`). Same
5-fold split, same 4-arm experiment (whole image + GT crop at
pad ∈ {0.0, 0.2, 0.4}), same Phase 2 protocol on folds {1-4} at
detector conf = 0.10. Single seed per fold.

### Phase 1 (matched-distribution, mean ± std across 5 folds)

| arm | DINOv2-S (R1) | **B0 (R2)** | Δ |
|---|---|---|---|
| whole_image | 0.530 ± 0.059 | 0.588 ± 0.074 | +0.058 |
| gt_pad_0.00 | 0.603 ± 0.068 | 0.600 ± 0.067 | −0.003 |
| gt_pad_0.20 | 0.530 ± 0.079 | **0.634 ± 0.035** | **+0.104** |
| gt_pad_0.40 | 0.569 ± 0.048 | 0.620 ± ~0.05 | +0.05 |

### Phase 2 (best cell, folds 1-4)

train_pad = 0.40, serve_pad = 0.20 → **cond_acc 0.626 ± 0.025**,
sys_acc 0.566. **+0.050 cond_acc over Round 1**, **std halves** from
0.050 → 0.025.

### Findings

1. **CNN locality bias confirmed.** B0 prefers serve_pad = 0.20
   (uses surrounding tissue meaningfully); DINOv2 preferred
   serve_pad = 0.00 (tight crops only). Direct evidence of
   inductive-bias difference between ViT-S and CNN at this scale.
2. **Fold-3 outlier disappeared.** In DINOv2 fold 3 tanked across
   every arm; in B0 it sits middle-of-pack. The frozen-feature blind
   spot was specific to the DINOv2 backbone, not the data.
3. **Std halved on the best cell.** Real production robustness gain
   — single-fold worst case becomes much more predictable.
4. **Still under the §12 bar** (0.626 vs 0.627) — but in a stricter
   eval (no source-image leakage). Morally at parity; not yet
   over the line. Round 3 (aug) and post-processing (§16 merge, §17
   TTA) explored next.

### Verdict

B0 fine-tune is the right architecture for this scale. ~+5 pp lift
from the Round 1 baseline at half the per-fold variance. Locked in
as the backbone for subsequent rounds.

---

# 13. Round 1 — DINOv2-S frozen head + MLP (2026-05-22)

**Scripts:** `Experimenting/classifier_experiments/exp1_whole_image.py`,
`exp2{a,b,c}_gt_pad*.py`,
`phase2_pipeline.py --backbone dinov2`

### Purpose

Build the simplest possible 5-class disease classifier on the detector's
own data — frozen DINOv2-S (ViT-S/14) + small MLP head (384 → 256 → 5,
~100 k trainable params). Phase 1 measures whether GT-crop training
reaches usable accuracy at all; Phase 2 measures whether GT-trained
classifiers transfer to YOLO detector crops well enough to ship.

### Setup

DINOv2-S backbone (frozen via torch.hub, ~21 M params, eval-locked
so BatchNorm/LayerNorm running stats stay frozen). Head ~100 k
trainable. AdamW lr 1e-3, cosine schedule, 50 epochs with early
stopping patience 10 on val macro-acc. Same 5-fold split as the
detector (`kfold5_splits.json`). Raw RGB + ImageNet norm only.
Minimal aug (horizontal flip + ±10° rotation). All 579 GT boxes from
`data/pool/` + `data/test/` (multi-box images contribute every box).

4 arms: whole image, GT crop at pad ∈ {0.0, 0.2, 0.4}.
Phase 2: 3 × 3 train_pad × serve_pad matrix on folds {1, 2, 3, 4}
(fold 0 excluded per §10 landmine), each fold uses its own detector
at conf = 0.10.

### Phase 1 headline (mean ± std across 5 folds)

| arm | micro | macro |
|---|---|---|
| whole_image | 0.530 ± 0.059 | 0.546 ± 0.052 |
| gt_pad_0.00 | 0.603 ± 0.068 | 0.607 ± 0.052 |
| gt_pad_0.20 | 0.530 ± 0.079 | 0.537 ± 0.076 |
| gt_pad_0.40 | 0.569 ± 0.048 | 0.567 ± 0.040 |

### Phase 2 best cell

train_pad = 0.40, serve_pad = 0.00 → **cond_acc 0.576 ± 0.050**,
sys_acc 0.521, catch 0.907, neg_FA 0.059.

### Findings

1. **Cropping matters** (whole image 0.530 → tight crops 0.603 →
   wider crops 0.569 in Phase 1). The trade is between resolution
   and context.
2. **Padding is non-monotonic.** pad = 0.20 is worse than both 0.0
   and 0.4 on Phase 1 — uncanny-valley behaviour for the frozen ViT
   (lesion no longer fills the frame but the crop isn't scene-like
   enough either).
3. **Fold 3 outlier** — micro 0.47 in pad = 0.0 vs other folds
   0.61-0.65. Same fold's detector pattern (§10) gave a different
   reason; this one looks model-specific (resolved in R2).
4. **§12 borrowed-OCD baseline not cleared** (0.576 vs 0.627). The
   frozen-ViT-S features at 224 px are not discriminative enough on
   this task — confirmed by Round 2's CNN-fine-tune lift.

### Verdict

Right scaffold (4 arms, 5-fold paired CV, Phase 1/2 protocol), wrong
backbone for low-data medical imaging. Round 2 swaps in B0 fine-tune.
Weights kept at
`Experimenting/classifier_experiments/results/{whole_image,gt_pad_0.0X}/`.

---

# 12. YOLO → old EffNet-B2 classifier — MVP measurement (2026-05-22)

**Script:** `Experimenting/predict_with_old_classifier.py`
**Outputs:** `Experimenting/results/yolo_to_old_classifier/`

### Purpose

Before building a new classifier from scratch we asked: *does the
EfficientNet-B2 classifier from the previous discarded Dentilligence/OCD
project work acceptably when fed crops from our YOLO detector?*

The OCD classifier was trained on **tight human-bbox crops** of the same
`pool/` images we use here, and it failed in production because it was
**served full images** in deployment — the same train/serve
crop-distribution mismatch our two-stage architecture is designed to
avoid. The hypothesis was: it might work on **detector-emitted crops**
because those are also lesion-zoomed, just produced differently.

### Setup

- **Detector:** fold 3 of `kfold5_geom_no_color_binary` (a
  middle-of-pack fold, not the conservative fold 0).
- **Classifier:** OCD EffNet-B2, weights from
  `Dentilligence/OCD/Oral_Cancer/runs/run_1773770685/fold_5/best.pt`.
  6-class head (5 diseases + Normal).
- **Positives:** 72 images from fold 3's blackbox test slice.
- **Negatives:** 120 OCD Normal images (`all`) or 12-image blackbox
  subset (`blackbox`).
- **Detector conf:** 0.10 (adopted operating point).
- **Padding sweep:** {0.0, 0.2, 0.4} of bbox dimension on each side.

Per image: full image → YOLO → for each box, pad → crop → CLAHE →
resize 260 px → classify (softmax over 6 classes). Per-image prediction
= argmax of mean softmax across boxes.

### Honest caveat

~90% of the OCD classifier's training set shares pool/ source images
with fold 3's test slice. The crops fed in here are a **different crop
of an image the classifier may have seen** — not full leakage, but not
full blackbox either. Conditional disease accuracy is therefore likely
mildly optimistic.

### Headline (negatives_mode = all, 120 OCD Normals)

| pad | caught (det) | + correct disease | conditional disease acc | system-level acc | neg FA |
|---|---|---|---|---|---|
| **0.0** | 67/72 (93.1%) | **42 / 67** | **62.7%** | **58.3%** | **15.0%** |
| 0.2 | 67/72 (93.1%) | 38 / 67 | 56.7% | 52.8% | 15.0% |
| 0.4 | 67/72 (93.1%) | 35 / 67 | 52.2% | 48.6% | 15.0% |

### Confusion matrix at pad = 0.0 (caught positives only, n = 67)

| GT \ pred | Leukop | Erythr | OSMF | Lichen | NH_Ulc | Normal | per-class |
|---|---|---|---|---|---|---|---|
| Leukoplakia | **7** | 0 | 1 | 0 | 0 | 0 | 7 / 8 = 87.5% |
| Erythroplakia | 0 | **11** | 1 | 1 | 2 | 0 | 11 / 15 = 73.3% |
| OSMF | 0 | 2 | **6** | 1 | 0 | 0 | 6 / 9 = 66.7% |
| Lichen_Planus | 0 | 2 | 2 | **9** | 2 | 1 | 9 / 16 = 56.3% |
| NH_Ulcers | 2 | 3 | 3 | 2 | **9** | 0 | 9 / 19 = 47.4% |

### Findings

1. **Padding hurts.** pad = 0.0 (tight YOLO crop) → 62.7%; pad = 0.4 →
   52.2%. The OCD classifier was trained on tight human-bbox crops;
   loosening the crop pushes the input distribution away from what it
   has seen. The YOLO box at IoG ≈ 0.833 is closer to a "tight crop"
   than a padded crop is.
2. **Detection recall (fold 3) is 93.1%** on this slice (67 / 72) —
   slightly above the kfold5 mean of 88.2% at conf = 0.10. Consistent
   with fold 3 sitting middle-of-pack in §10.
3. **NH_Ulcers is the weakest disease class** in the OCD classifier
   (47.4%) — it inherits the same imbalance / noise the OCD project
   reported.
4. **Padding does not hurt negative FA** (steady 15.0% across pads) —
   FA is driven by detector firings, the classifier just relabels them.

### Verdict

The borrowed classifier is **demoable but not the final answer**. ~63%
conditional disease accuracy on detector-emitted crops at pad = 0.0 is
above the investor bar (60%) but below where the new classifier
training run should land — because (a) this measurement leaks ~90% of
training source images, and (b) the OCD classifier still has the
"trained on tight human bboxes" bias the new classifier will avoid by
being trained directly on detector-emitted crops.

Useful baselines that come out of this experiment:

- **System-level accuracy 58.3%** (correct disease over **all 72**
  positives, including detector misses) — a floor for what the new
  classifier needs to beat at minimum.
- **Conditional disease accuracy 62.7%** (correct disease over **caught**
  positives) — what the new classifier directly competes with.
- **Pad = 0.0 is the right starting point** when feeding the
  classifier — match the crop distribution it will be served in
  production.

---

# 11. §10 OOD sanity check — pre-classifier demo readiness (2026-05-21)

**Script:** `Experimenting/predict_bulk_sanity.py`
**Outputs:** `Experimenting/results/internet_sanity_fold0/`

### Purpose

The §9 numbers are measured on the project's own fair test distribution
(1:1, resolution-normalized). Before the investor demo we needed to
check what the model does on data it has *never* seen and *will never*
see in the fair-test setup — raw high-resolution phone photos, and
out-of-domain Roboflow lesion sets.

### Setup

Each of the 5 fold weights × 4 image sets, evaluated one image at a
time (OOM-safe on 6 GB), at confs {0.05, 0.075, 0.10, 0.25}.

| set | source | size |
|---|---|---|
| Normal | raw `data/Normal/` + `NON CANCER/`, **not** resolution-normalized | 570 phone photos, 960–4000 px |
| Leukoplakia.v2 | Roboflow | 74 |
| OPMD-SEG | Roboflow, deduped on `.rf.<hash>` | 576 unique |
| OSMF DETECTION | Roboflow, class-1 only | 305 positive |

### Headline (mean ± std across 5 folds, conf = 0.10)

| set | metric | conf 0.05 | conf 0.10 | conf 0.25 |
|---|---|---|---|---|
| Normal (raw, 570) | FA | 0.239 ± 0.056 | **0.147 ± 0.049** | 0.051 ± 0.025 |
| Leukoplakia.v2 (74) | det_rate | 0.903 ± 0.051 | **0.854 ± 0.069** | 0.619 ± 0.135 |
| OPMD-SEG (576) | det_rate | 0.874 ± 0.058 | **0.771 ± 0.086** | 0.476 ± 0.164 |
| OSMF DETECTION (305) | det_rate | 0.932 ± 0.040 | **0.856 ± 0.061** | 0.588 ± 0.063 |

IoG mean on hits across folds: Leuk 0.908 ± 0.021, OPMD 0.957 ± 0.004,
OSMF 0.866 ± 0.014. **But IoG on hits is survivorship-biased** — it
averages only over GTs already covered ≥ 0.5, and is gameable by big
boxes. Read as "when the model commits to a region, the crop is usable
for the classifier" — *not* as a clean localisation quality number.

### Per-fold variance — fold 0 is globally less aggressive

| fold | Normal FA | Leuk det | OPMD det | OSMF det | RF det avg |
|---|---|---|---|---|---|
| **0** | **6.5%** | 75.7% | 69.1% | 77.0% | **73.9%** |
| 1 | 18.9% | 81.1% | 67.4% | 90.5% | 79.7% |
| 2 | 14.9% | 90.5% | 80.7% | 89.5% | 86.9% |
| 3 | 15.1% | 91.9% | 80.9% | 81.3% | 84.7% |
| 4 | 17.9% | 87.8% | 87.5% | 89.8% | 88.4% |

Fold 0 fires less on lesions **and** less on negatives — sits ~2σ below
the other 4 folds on both axes. **Production weight pick: do NOT ship
fold 0.** Fold 2 or fold 3 are middle-of-pack; an ensemble of all 5 is
~250 ms on a 3050 if you want to flatten the variance.

### What §10 does and does not change

- **§9's 0.917 ± 0.031 remains the headline.** It is measured on the
  project's own fair test (1:1, resolution-normalized). §10 is a
  separate, looser test on OOD data.
- §10 quantifies what the §5-flagged resolution confound costs: Normal
  FA balloons from 0.047 (CV-fair) to 0.147 (raw phone photos) at
  conf = 0.10. Roughly **1 in 7 healthy selfies will be flagged**.
- Roboflow det_rate transfers within ~5–10 pp of the CV number. Domain
  transfer is real and costs about a tenth of recall.

### Verdict

The detector is **good enough to ship into the MVP**. The remaining
weaknesses (high OOD FA on raw phone photos, fold-0 conservatism) are
v2 work or operating-point choices, not blockers.

---

# 10. §9 confidence-threshold sweep — the operating-point fix (2026-05-21)

**Script:** `Experimenting/sweep_conf_threshold.py`

### Purpose

The exp11 headline at conf = 0.25 was 0.842 ± 0.041. The
`recommend_conf` helper in `src/` had been pinning near 0.001 for
months, and the CLAUDE.md note "the detector is tuned for recall at low
conf, not mAP" suggested the YOLO-default threshold of 0.25 was leaving
signal on the table. We measured whether that was true.

### Setup

For each of the 5 kfold5_geom_no_color_binary fold weights, re-ran
prediction at conf = 0.001 (to collect every box the model would ever
consider), then for a grid of decision thresholds recomputed the
image-level screening triple. No retraining, no GPU training — ~3 min
of inference total.

### The sweep

| conf | screening_acc | det_rate | false_alarm |
|---|---|---|---|
| 0.050 | **0.917 ± 0.019** | **0.932** | 0.097 |
| **0.100** | **0.917 ± 0.031** | 0.882 | 0.047 |
| 0.150 | 0.899 ± 0.021 | 0.838 | 0.039 |
| 0.200 | 0.869 ± 0.028 | 0.769 | 0.031 |
| **0.250 (prev default)** | 0.842 ± 0.041 | 0.703 | 0.020 |
| 0.300 | 0.808 ± 0.056 | 0.631 | 0.014 |
| 0.350 | 0.782 ± 0.077 | 0.573 | 0.008 |
| 0.400 | 0.740 ± 0.103 | 0.485 | 0.006 |
| 0.500 | 0.666 ± 0.094 | 0.333 | 0.000 |
| 0.800 | 0.517 ± 0.015 | 0.033 | 0.000 |

**Δ vs the previous default**: +0.076 screening_acc, +0.179 det_rate
(≈ 13 more lesions caught per 72-image fold), at the cost of +0.027
false_alarm (≈ 2 more false alarms per fold).

### Findings

1. **The model was always capable of 0.917-class screening.** Conf =
   0.25 was throwing away signal — the model was firing on real lesions
   with conf in [0.05, 0.25] and those firings were being suppressed.
2. **Screening_acc is flat from conf = 0.05 to 0.10** (both at 0.917
   mean). The detector emits a clear bimodal confidence distribution:
   real lesions above ~0.05, noise above ~0.001 but below 0.05. The
   right operator threshold sits in the gap.
3. **The "calibration is the residual" framing is partially resolved.**
   The FA @ conf = 0.001 ≈ 0.70 figure from exp8 / exp9 / exp11 was an
   evaluation-knob artifact — no operator runs at conf = 0.001. At conf
   = 0.10, FA = 0.047. The logits are still technically miscalibrated,
   but it doesn't matter operationally.

### Honest caveat

One-parameter test-set leakage. The threshold was selected by maximising
mean screening_acc over the same 5 test slices we report on. This is a
mild form of test-set adaptation for a single scalar. A fully-held-out
estimate (choose conf on inner val per fold, apply to outer test) would
likely land **0.90–0.91**, not 0.917. The direction and the +0.076 Δ
are real; the absolute number is slightly optimistic.

### Verdict

**Adopted operating point: conf = 0.10.** Other points (0.05
recall-first, 0.15 specificity-leaning) are knobs on the same model.

---

# 9. exp11 — paired 5-fold CV, `geom_no_color` vs default (2026-05-21)

**Script:** `Experimenting/exp11_kfold5_aug_binary.py`
**Outputs:** `Experimenting/results/kfold5_geom_no_color_binary/`,
`kfold5_default_binary/`

### Purpose

exp10 found `geom_no_color` (= YOLO defaults but `hsv_h = hsv_s = hsv_v
= 0`) at 0.919 on a single 85/15 split. That single split was lucky,
unlucky, or representative — exp10 alone could not tell. We needed a
**paired CV** where both recipes train on identical folds, so the only
difference is the augmentation config.

### Setup

- Fresh stratified 5-fold split (`_datasets/kfold5_splits.json`), K = 5.
- **Per fold's blackbox test slice = ~72 positives + ~72 fair
  (resolution-normalized) negatives** (1:1, no-skill 0.50).
- **Per fold's train + val pool = ~290 positives + ~498 negatives**
  (the other 4 folds' positives, plus the inner train/val negative
  pool). Inner 85/15 stratified split inside each fold for early
  stopping.
- Same 5 folds for both runs; only variable = the aug config.
- Seed 42, 100 epochs, imgsz 640, batch 8, AMP off.

> **Note on training-set size.** Each exp11 fold trained on ~290
> positives, vs ~325 in exp1 / exp2 / exp8 / exp10 (which used a single
> 85/15 split of pool/ with a separate 37-image locked test). This is an
> ~11% smaller training set per fold. The exp11 headline is therefore
> if anything a *mildly conservative* estimate of what the same recipe
> would deliver if trained on all 362 positives. Direction and paired
> deltas are unaffected — the comparison within exp11 (default vs
> `geom_no_color`) is internally clean.

### Paired headline (conf = 0.25)

| metric | default (HSV on) | **`geom_no_color`** | Δ |
|---|---|---|---|
| **screening_acc** | 0.741 ± 0.147 | **0.842 ± 0.041** | **+0.101** (3.6× tighter) |
| **det_rate_pos** | 0.484 ± 0.297 | **0.703 ± 0.099** | **+0.219** (3.0× tighter) |
| false_alarm_neg | 0.003 ± 0.006 | 0.020 ± 0.027 | −0.017 |
| box F1 (iou ≥ 0.5) | 0.254 ± 0.145 | 0.379 ± 0.041 | +0.125 |
| box F1 (iog ≥ 0.5) | 0.362 ± 0.193 | 0.525 ± 0.036 | +0.163 |
| loc IoG (hits) | 0.826 ± 0.042 | 0.833 ± 0.017 | +0.007 |
| val_stock mAP50 | 0.252 ± 0.085 | 0.328 ± 0.067 | +0.075 |

### Per-fold screening_acc

| fold | default | `geom_no_color` | Δ |
|---|---|---|---|
| 0 | **0.534** | 0.824 | **+0.290** |
| 1 | 0.678 | 0.781 | +0.103 |
| 2 | 0.868 | 0.875 | +0.007 |
| 3 | 0.729 | 0.847 | +0.118 |
| 4 | **0.894** | 0.880 | −0.014 |

`geom_no_color` wins 4 / 5 folds. The fold-0 datum is the striking one:
same data, same fold, same seed, only HSV differs — default caught
**5 / 72 lesions** (training near-collapse), `geom_no_color` caught
**47 / 72**.

### Findings

1. **The `instructions.md` "colour is diagnostic, no HSV" rule is
   empirically validated.** Was a design assertion through exp1–exp8 —
   YOLO's defaults include `hsv_h = 0.015, hsv_s = 0.7, hsv_v = 0.4`,
   and every experiment up to here silently applied them.
2. **Likely mechanism (hypothesis, not proof):** HSV jitter
   destabilises training on small medical datasets, not just corrupts
   features. fold-0 default's near-total collapse is consistent with
   that. We have one seed per fold, so this is a hypothesis — multi-seed
   would be needed to harden it.
3. **`geom_no_color` is the adopted default detector recipe.** All new
   detector experiments inherit it.

### Verdict

`geom_no_color` is the binary-detector baseline. The exp11 weights are
what the §9 threshold sweep operated on — together, exp11 + §9 produce
the current headline 0.917 ± 0.031 at conf = 0.10.

---

# 8. exp10 — augmentation sweep, 6 levels (2026-05-20)

**Script:** `Experimenting/exp10_aug_sweep_binary.py`

### Purpose

Sweep 6 named YOLO augmentation configs on a single 85/15 split (= the
exp8b split) to find the most promising recipe. Single seed, single
split — meant as a *signal-finder*, not a CV verdict.

### Results (one 85/15 split, seed 42, conf = 0.25)

| level | screen_acc | det_rate | false_alm | F1 (iog) | mAP50 | notes |
|---|---|---|---|---|---|---|
| off | 0.595 | 0.243 | 0.054 | 0.203 | 0.160 | fliplr only — under-trains |
| light | 0.676 | 0.378 | 0.027 | 0.354 | 0.224 | mild geom, no mosaic |
| default | 0.865 | 0.757 | 0.027 | 0.500 | 0.280 | YOLO defaults — reproduces exp8b |
| **`geom_no_color`** | **0.919** | **0.838** | **0.000** | **0.545** | **0.385** | defaults + hsv = 0 |
| heavy | 0.757 | 0.513 | 0.000 | 0.405 | 0.277 | + mixup / copy_paste / big geom |
| heavy_no_color | 0.784 | 0.568 | 0.000 | 0.447 | 0.330 | heavy + hsv = 0 |

### Findings

1. **HSV-off helps both recipes.** default → `geom_no_color` = +0.054
   screening; heavy → heavy_no_color = +0.027. Same sign.
2. **Mosaic + scale + translate are load-bearing.** off → default =
   +0.270 screening. Without mosaic (`light`), the model never reaches
   default's performance.
3. **Heavy aug is independently bad.** `geom_no_color` →
   heavy_no_color = −0.135 screening. mixup + copy_paste + big rotate /
   scale / shear overwhelm a ~277-image positive pool.
4. **Aug is not the lever for calibration.** FA @ conf = 0.001 stayed
   in [0.81, 0.95] across all six levels.

### Caveat (resolved by exp11)

0.919 is a single 85/15 split. exp9 kfold10 default mean was
0.840 ± 0.063, putting `geom_no_color`'s 0.919 at ~+1.25σ above
default's CV mean. exp11 paired-tested it cleanly.

### Verdict

Real signal at the wrong measurement granularity. The direction
(HSV-off helps, mosaic is essential, heavy hurts) was reproduced under
paired CV in exp11.

---

# 7. exp9 — 10-fold CV of the exp8b recipe (2026-05-20)

**Scripts:** `Experimenting/exp9_kfold10_binary.py`,
`exp9_kfold10_5class.py`

### Purpose

exp8b's 0.865 was a single fold. Was it a typical draw, a tail event, or
between? A CV would tell us, and the per-fold variance would tell us how
reliable any future single-fold experiment is.

### Setup

Stratified 10-fold on (pool + locked-test = 362 positives) + 10-way
shuffled ~570 resolution-normalised negatives. Fair 1:1 test slice per
fold (~37 + 37, no-skill 0.50). Inner 85/15 stratified train/val on the
other 9 folds. Same recipe as exp8b (YOLO defaults = HSV on).

### Results — binary, 10/10 folds beat no-skill (0.50)

| metric | exp8b (single fold) | **kfold10 mean ± std** | range |
|---|---|---|---|
| screening_acc | 0.865 | **0.840 ± 0.063** | [0.757, 0.932] |
| det_rate_pos | 0.757 | 0.694 ± 0.126 | [0.514, 0.892] |
| false_alarm_neg | 0.027 | 0.013 ± 0.034 | [0.000, 0.108] |
| FA @ conf 0.001 | 0.946 | **0.618 ± 0.140** | [0.400, 0.829] |
| box R (iog ≥ 0.5) | 0.439 | 0.429 ± 0.097 | [0.270, 0.633] |
| loc IoG (hits) | 0.811 | 0.811 ± 0.026 | [0.752, 0.843] |

### Findings

1. exp8b's 0.865 sits slightly above the CV mean, well inside CI — not
   an outlier.
2. **exp8b's "FA @ conf = 0.001 = 0.946 fires on everything" was a tail
   draw.** Typical FA at that threshold is 0.618; 4 of 10 folds are
   already below 0.50 at conf = 0.001. Calibration is broken but not
   catastrophically.
3. **Localisation IoG on hits = 0.811 ± 0.026** — when the detector
   hits, the box covers ≥ 80% of the annotation in every single fold,
   std 2.6 pp. Crop quality is rock-solid.
4. **Det_rate is the bottleneck**, not specificity — recall variance is
   what aug or capacity needs to flatten.

### Verdict

exp8b reproduces under CV; exp9's mean is the honest detector number
*for the default-aug recipe*. exp10 then asked whether a better aug
recipe lifts that mean — yes (exp11 confirmed).

---

# 6. exp8a / exp8b — the "detection is dead" retraction (2026-05-19)

**Scripts:** `Experimenting/exp8a_5class_negatives.py`,
`exp8b_binary_negatives.py`

### Purpose

Phase 2's exp1–exp7 audit concluded the binary detector was "below a
trivial baseline / detection is dead". A discussion pass found two
problems with that verdict:

1. The "trivial baseline" of 0.939 was computed on 570 Normals vs 37
   positives — a base-rate artifact.
2. **exp1 and exp2 trained on zero negatives** (`build_original` adds
   none). The detector was never taught to stay silent.

exp8 tests both: re-train with resolution-normalized negatives in train,
evaluate on a **fair 1:1 test** (no-skill 0.50).

### Setup

Identical to exp1 / exp2 on the positive side (verified byte-identical
pool split + locked-37 stems / labels). Only variable = negatives.
`data/Normal/` resized so its long side = positives' median long side
(measured **276 px**) — independently confirms the ~275 px figure in
the data audit. 37-image 1:1 slice in test.

### Number A — fair ruler alone, no retraining

`eval_fair_negatives.py` re-scored exp1 / exp2's existing `best.pt` on
the same fair 37 + 37 test. Isolates the measurement effect from the
training effect.

| | det_rate | false_alarm | screening_acc |
|---|---|---|---|
| exp2 raw 570-neg | 0.838 | 0.621 | 0.407 |
| **Number A** (exp2 wts, fair 37-neg) | 0.838 | 0.568 | 0.635 |

Fair ruler alone moves false-alarm only 0.621 → 0.568 — the resolution
confound is real but only ~5 pp. The rest was the missing-negatives bug.

### exp8 — with negatives in training

| | det_rate | false_alarm | screening_acc |
|---|---|---|---|
| **exp8b** (binary, neg in train) | 0.757 (28/37) | **0.027 (1/37)** | **0.865** |
| exp8a (5-class, neg in train) | 0.622 (23/37) | **0.054 (2/37)** | **0.784** |
| Number A (exp1 wts, fair) | 0.595 | 0.351 | 0.622 |

**False-alarm: 21 / 37 → 1 / 37 (binary)**, det_rate ≈ held. Binary
beats 5-class with a usable number (0.865 vs 0.784). exp8a also revived
exp1's two dead disease classes (Leuk / OSMF no longer 0 / 0 / 0,
directionally).

### Calibration still broken (confound-free)

At conf = 0.001 on the fair test: exp8b FA = 0.946; exp8a FA = 0.892.
Negative training barely dents the conf = 0.001 collapse. **This left
calibration as the only confound-free survivor of the audit** — until
the §9 sweep showed that conf = 0.001 is a regime no operator uses.

### Verdict

"Detection is dead" retracted. Binary YOLO with negatives in training =
**screening 0.865, det 0.757, FA 0.027** — a credible screener on the
project's own fair distribution. The exp9 → exp10 → exp11 chain then
cross-validated and improved this further.

### Match-rule sweep (companion finding)

`eval_match_rules.py` re-scored exp8b's saved predictions under three
overlap rules. Det_rate / false_alarm / screening are geometry-free and
**do not move**; only box P / R / F1 change.

| rule | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| iou ≥ 0.5 | 21 | 22 | 36 | 0.488 | 0.368 | 0.420 |
| **iog ≥ 0.5 (headline)** | 25 | 18 | 32 | 0.581 | **0.439** | 0.500 |
| iog ≥ 0.3 & iop ≥ 0.10 | 28 | 15 | 29 | 0.651 | 0.491 | 0.560 |

TP + FP constant — the IoU gate was relabelling well-placed offset
boxes as FP. Under IoG, precision and recall rise together (not a
trade). **Headline rule adopted: `iog ≥ 0.5`** ("≥ half the annotated
lesion covered"). `iou ≥ 0.5` kept for continuity.

---

# 5. exp7 — binary + 955 unique Roboflow images (2026-05-19)

**Script:** `Experimenting/exp7_binary_plus_roboflow.py`

### Purpose

The earlier `src/` pipeline experiment concluded "Roboflow data clearly
helps". That conclusion was measured on `web_holdout`, which is itself
Roboflow-derived (in-domain). Single-variable controlled test: does
+955 unique Roboflow photos help when measured on the **original
domain**?

### Setup

Identical to exp2 in every way except training data: exp2's pool split +
955 unique de-augmented Roboflow images appended to train. Same locked
37-image test. No negatives in train (matches exp2's bug).

### Results (locked-37 test)

| metric @ conf 0.25 | exp2 binary | exp7 binary + Roboflow |
|---|---|---|
| box P / R / F1 | .382 / .368 / .375 | .460 / .298 / .362 |
| stock val mAP50 | 0.326 | 0.316 |
| det_rate_pos | **0.838** | 0.703 |
| false_alarm_neg (raw 570-neg) | 0.621 | 0.312 |
| screening_acc (raw 570-neg) | 0.407 | 0.689 |

### Findings

- F1 / mAP50 **flat**, recall **down**.
- Only specificity goes up — the Roboflow data makes the model **more
  conservative**, not better at detecting.
- "Roboflow helps" in `src/` was measured on Roboflow-derived
  `web_holdout`. On the original domain, controlled, the gain is ≈ 0.

### Verdict

**Roboflow scale-up does not transfer.** Specificity gains are not what
a recall-first screener wants. The earlier `src/` "Roboflow clearly
helps" headline is an in-domain measurement artifact.

---

# 4. exp3 / exp5 — Roboflow per-disease experts (2026-05-19)

**Scripts:** `Experimenting/exp3_leukoplakia_expert.py`,
`exp5_osmf_expert.py`

### Purpose

If Roboflow as bulk data doesn't transfer (exp7), maybe it works for
**per-disease experts** — large, single-disease training pools from
clean Roboflow sources.

### Setup

`yolov8n-obb`, 100 epochs, imgsz 640. Two cleanest available sources:

- **exp3 Leukoplakia.v2** (74 imgs, no aug) + OPMD-SEG cls0 (435 imgs).
  509 unique total. Messy multi-source.
- **exp5 OSMF DETECTION** (305 unique, **0 augmentation**). The
  cleanest possible Roboflow expert test.

### Results — 932-img test (46–52 positives, ~880 negatives)

| @ conf 0.25 | Leukoplakia expert | OSMF expert |
|---|---|---|
| Roboflow-own val mAP50 / P | **0.568 / 0.676** | **0.619 / 0.803** |
| original-domain box P | 0.031 | 0.015 |
| det_rate_pos | 0.957 | 0.712 |
| false_alarm_neg | 0.525 | 0.533 |
| screening_acc | 0.499 | 0.481 |

### Findings

- mAP50 ≈ 0.6 on Roboflow's own val; precision **0.015 – 0.031** on
  the original domain.
- The clean OSMF expert (305 unaugmented photos, single source, no
  multi-label) fails **identically** to the messy Leukoplakia expert.
- → Not augmentation, not label noise — **structural domain shift**
  (camera / lighting + a 250 px vs 432/640 px resolution gap).

### Verdict

Roboflow → original domain shift is severe and structural. exp4 / exp6
(Erythroplakia / Lichen Planus experts) were designed but
**deliberately not run** — exp5 was the decisive test, and Lichen is
data-dead anyway (48 train).

---

# 3. exp1 / exp2 — 5-class vs binary on original data (2026-05-19)

**Scripts:** `Experimenting/exp1_5class_original.py`,
`exp2_binary_original.py`

### Purpose

Clean restart. Independent of `config.py` / `artifacts/` / `src/`. No
classifier head, nothing tuned, no metric optimised. The first question:
**should YOLO classify, or just detect?**

### Setup

`yolov8n`, pool 85 / 15 stratified split, 100 epochs, imgsz 640. Same
data, only difference: exp1 keeps 5 disease classes, exp2 collapses to
1 (`lesion`). Locked 37-image test. **No negatives in training** (the
bug exp8 later fixed).

### Results (locked-37 test, conf = 0.25)

| metric | exp1 5-class | exp2 binary |
|---|---|---|
| box P / R / F1 | .280 / .123 / .171 | .382 / .368 / **.375** |
| TP / FP / FN | 7 / 18 / 50 | 21 / 34 / 36 |
| stock val mAP50 (test) | 0.168 | **0.326** |
| det_rate_pos | 0.595 | **0.838** |
| loc-on-hits IoU / IoP / IoG | .66 / .70 / .92 | .65 / .81 / .80 |

exp1 dead classes @ conf 0.25: **Leukoplakia 0 / 0 / 0, OSMF 0 / 0 / 0**.

### Findings

1. **Binary YOLO ≈ 3 × the recall, 2 × the F1, 2 × the mAP50 of
   5-class** on identical data — the price of asking YOLO to classify,
   measured. This is empirical justification for the two-stage
   architecture.
2. **Localisation is not the problem.** loc-on-hits IoG 0.80–0.92 every
   run — when the model hits, the box covers most of the lesion. The
   "loose boxes" premise from `instructions.md` (predictions are tight
   boxes inside huge loose GTs, IoU tiny) was empirically false.

### Verdict

- Two-stage **structure** is sound: binary YOLO + separate classifier.
- "Loose boxes" rationale is wrong; the IoP-matching machinery in `src/`
  solves a non-problem (it works, just not for the reason given).
- Right structure, wrong reason.

---

# 2. `src/` pipeline — `original_only` vs `plus_roboflow` (2026-05-18)

**Scripts:** `scripts/01..05_*.py`

### Purpose

The original two-stage rebuild (`src/`, per `instructions.md`) was
augmented with a data experiment: control (original pool) vs treatment
(original pool + ~2.3 k Roboflow images appended to detector training
only). Identical code; only detector training data differs.

### Results (control vs treatment)

| Metric | `original_only` | `plus_roboflow` |
|---|---|---|
| web_holdout mAP50 (higher-N) | 0.0185 | **0.263** (~14×) |
| web_holdout recall | 0.054 | **0.298** (~5.5×) |
| locked-37 accuracy | 0.5405 | **0.5676** (+2.7 pp) |
| Leukoplakia recall (locked-37) | 0.00 | **0.50** (dead class revived) |

### Verdict (now overturned)

Originally concluded "Roboflow data clearly helps". The audit
(exp7, above) overturned this: the decisive `web_holdout` ~14× lift was
in-domain (web_holdout is Roboflow-derived), and on the original domain
the gain is ≈ 0. The locked-37 +2.7 pp is within 37-image noise.

The `src/` pipeline still runs end-to-end as documented. Treat the
numbers above as **stale**; the design rationale (IoP matching,
proportional padding, per-run storage, shared crop function) is sound
and binding.

---

# 1. Phase 0 — three abandoned approaches

These are documented so they are **not repeated**. Source:
`instructions.md` §1.

1. **YOLO → YOLO cascade** (binary YOLO → 5-class YOLO). ~20 – 30 %
   accuracy. YOLO optimises mAP, not classification; the cascade also
   tanked recall.
   → Use a real *classifier* for the disease stage.

2. **EfficientNet on tight human-annotation crops.** ~75 % in eval,
   **~0 % in production.** Trained on tight bbox crops, served full
   images — total train / serve distribution mismatch.
   → Train the classifier on the **detector's own crops**, never on
   raw annotations. One shared crop function imported by both the data
   builder and the live pipeline.

3. **LAB / CLAHE preprocessing.** Hurt accuracy and was applied
   inconsistently between train and serve.
   → Raw RGB + ImageNet normalisation only.

Also retired in Phase 0: k-fold (later reintroduced at exp9), Mixup /
CutMix (re-tested in exp10 `heavy`, still bad), heavy augmentation (same),
Gradio (Streamlit kept).

These three lessons remain binding. Every later experiment was designed
to *not* re-derive them.
