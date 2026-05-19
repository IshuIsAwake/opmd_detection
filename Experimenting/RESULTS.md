# Experimenting/ — RESULTS & VERDICT (2026-05-19)

Source of truth for the clean-baseline investigation. `README.md` (this dir)
describes *how* to run; this file is *what we found and what it means*.
Everything below is from runs the user executed on the RTX 3050; no number is
tuned or cherry-picked.

> One-line verdict (revised after exp8, §7): the "loose boxes" rationale is
> empirically false and the Roboflow scale-up does not transfer — both still
> hold. **But the "detector is below a trivial baseline / detection is dead"
> conclusion is RETRACTED.** It was a base-rate artifact (570 negatives) on
> top of a real bug: exp1/exp2 trained on **zero** negatives. Give the binary
> detector the negative signal it always lacked and measure it fairly (exp8b:
> 1:1 resolution-normalized negatives): **28/37 lesion images caught, 1/37
> false alarm, screening_acc 0.865** (no-skill = 0.50). Detection is a
> credible screener at conf 0.25. The one confound-free survivor is
> **confidence calibration** (conf-0.001 still fires on everything). Active
> next step: **augmentation + k-fold CV** (§6) for a trustworthy ceiling.

---

## 1. Why this tree exists

The main pipeline's headline ("Roboflow data clearly helps", 54%→57%) felt
off. `Experimenting/` rebuilds clean, dependency-light, no-classifier YOLO
baselines to isolate, with controlled single-variable experiments, *whether
the Roboflow data actually helps detection on the domain that matters* (the
original images). Independent of `config.py` / `artifacts/` / `src/`.

## 2. Data reality established (write this down)

| | what we found |
|---|---|
| original `pool/` | 325 imgs, **one disease per image**, 522 boxes. Per-class imgs: Leuk 42, Eryth 70, OSMF 47, Lichen 76, NH 90 |
| locked `test/` | 37 imgs, 57 boxes (Leuk 4 / Eryth 9 / OSMF 5 / Lichen 9 / NH 10) — **noise-dominated** |
| OPMD-SEG (obb & poly) | 1770 label files → **only 576 unique photos** (~3.1× Roboflow augmentation). Unique per class: Leuk 435, Lichen(OLP) 48, Eryth 223. 141/576 are multi-disease |
| Leukoplakia.v2 | 74 imgs, no augmentation |
| OSMF DETECTION | polygons, **305 unique, 0 augmentation** (cleanest Roboflow source) |
| expert unique training pools | Leuk **509**, Eryth **223**, OSMF **305**, Lichen **48** (not the "1000+" the file counts implied) |
| **image dimensions** | original lesions ≈ **250 px thumbnails** (~275×183); Roboflow pre-stretched squares (OPMD 432², OSMF/Leuko 640²); Normal = 960×1280–4000 px phone photos. `imgsz=640` upscales originals ~2.3× (blur+pad). A **third domain gap**: resolution + aspect, on top of camera/lighting |
| trivial baseline | on the 607-img neg-bearing set, "always say no lesion" = 570/607 = **0.939** |

A train/val **augmentation-leakage bug** in the expert builder was found and
fixed (split was over augmented files, not source photos; variants of one
photo leaked across train/val). Experts now train on unique photos only;
`_datasets/<name>/dataset_stats.json` records the dedup per run.

## 3. Runs the user executed

Model = yolov8n (exp1/2/7, detect) or yolov8n-obb (experts). 100 epochs,
imgsz 640, batch 8, AMP off. exp4 (erythroplakia) and exp6 (lichen) were
**designed but deliberately not run** — exp5 (OSMF, the clean source) was the
decisive expert test and exp4/6 could only echo it.

### 3a. Original-data baselines — locked-37 test (no negatives)

| metric @conf 0.25 | exp1 5-class | exp2 binary | exp7 binary +955 Roboflow |
|---|---|---|---|
| box P / R / F1 | .280 / .123 / .171 | .382 / .368 / **.375** | .460 / .298 / .362 |
| TP / FP / FN | 7 / 18 / 50 | 21 / 34 / 36 | 17 / 20 / 40 |
| stock val mAP50 (test) | 0.168 | **0.326** | 0.316 |
| loc-on-hits IoU/IoP/IoG | .66/.70/.92 | .65/.81/.80 | .69/.83/.84 |
| mean best IoU / GT | 0.140 | 0.333 | 0.267 |
| det_rate_pos | 0.595 | **0.838** | 0.703 |

exp1 dead classes @conf0.25: **Leukoplakia 0/0/0, OSMF 0/0/0**.

### 3b. Roboflow experts — 932-img test (46–52 pos, ~880 neg)

| @conf 0.25 | Leukoplakia (509, messy) | OSMF (305, clean, 0-aug) |
|---|---|---|
| Roboflow-own val mAP50 / P | **0.568 / 0.676** | **0.619 / 0.803** |
| original-domain box P | 0.031 | 0.015 |
| false_alarm_neg | 0.525 | 0.533 |
| screening_acc | 0.499 | **0.481** |
| det_rate_pos | 0.957 | 0.712 |

### 3c. Un-gameable re-eval — `eval_with_negatives.py`, 37 pos / 570 neg, single-class

| @conf 0.25 | exp1 5-class | exp2 binary | exp7 +Roboflow |
|---|---|---|---|
| det_rate_pos (recall) | 0.595 | 0.838 | 0.703 |
| **false_alarm_neg** | 0.356 | **0.621** | **0.312** |
| **screening_acc** | 0.641 | **0.407** | **0.689** |
| box F1 | 0.050 | 0.061 | 0.108 |

**Every detect run at conf 0.001 collapses to the identical
`screening_acc 0.061 / false_alarm_neg 1.000`** — fires on everything.

## 4. Conclusions (the actual findings)

1. **The "loose boxes" premise is false.** The main project's whole rationale
   (annotator GT huge/loose → tight correct pred sits inside → IoU tiny, so
   use IoP and train the classifier on detector crops) predicts IoP≈1.0,
   IoG low. Every run shows the **opposite**: loc-on-hits IoU 0.63–0.69,
   IoG 0.80–0.92. When the detector hits, boxes are fine. Box geometry was
   never the problem. *Wrong reason.*
2. **The two-stage *structure* is sound.** exp1→exp2 (identical data, only the
   5→1 class collapse) ≈ **3× recall / 2× F1 / 2× mAP50**. That gap is the
   price of asking YOLO to classify — empirical justification for "binary
   detector + separate classifier". *Right structure, wrong reason.*
3. **Roboflow→original domain shift is severe and structural.** Experts score
   mAP50 0.57–0.62 on Roboflow's own val and **P 0.015–0.031** on original
   images. The *clean* OSMF expert (305 unaugmented photos, single source, ~6×
   the original data, no multi-label) fails **identically** to the messy
   Leukoplakia one → not augmentation, not label noise — structural domain
   shift (camera/lighting **+** the resolution/aspect gap in §2).
4. **More Roboflow data ≠ better detection (controlled).** exp2 vs exp7 differ
   by exactly one variable (+955 unique Roboflow photos, same pool split, same
   original-domain val, same locked test). F1/mAP50 **flat**; recall **down**
   (det_rate_pos 0.838→0.703); specificity **up** (false_alarm_neg
   0.621→0.312, screening_acc 0.407→0.689). Roboflow makes it *more
   conservative* — real specificity, but the wrong direction for a
   recall-first screener. It does not buy detection ability.
5. **Confidence calibration is totally broken — and it is now the ONLY
   confound-free survivor (reinforced by exp8, §7).** Identical conf-0.001
   collapse across 5-class / binary / +Roboflow, and exp8 shows negative
   training barely dents it (false_alarm_neg 1.000→0.946). Not data volume,
   not resolution, not base rate, not missing negatives — calibration. This is
   THE open problem and the next conversation's real target (after aug/k-fold
   establish the ceiling).
6. **~~The detector approach is below a trivial baseline.~~ RETRACTED by exp8
   (§7).** This was true *as measured* (607-img set, "always healthy" = 0.939,
   exp2 = 0.407) but the measurement was the artifact: 570 negatives make the
   trivial baseline 0.939 by base rate alone, and exp1/exp2 trained on **zero**
   negatives so they were never taught to stay silent. At a fair 1:1
   resolution-normalized test, with negatives in training, exp8b = **0.865**
   screening_acc (no-skill 0.50). See §7. The resolution confound was real but
   minor (~5pp of false-alarm); the base rate + missing-negatives bug were the
   real story.
7. **The main pipeline's "data helps" is a measurement artifact.** It judged
   the Roboflow benefit on `web_holdout` (Roboflow-derived = in-domain), where
   it of course looks ~14×. On the original domain, controlled, the gain is
   ≈0. The locked-37 54%→57% is within 37-image noise.
8. **~~The root cause is data, not anything tunable.~~ PARTLY CORRECTED by
   exp8.** The data is still tiny (277 ~250 px thumbnails, 37-img test) and
   that caps the ceiling — but one *non-tunable-knob* change (adding the
   negative training signal exp1/exp2 never had) collapsed false-alarm
   21→1 /37 with ~no recall cost. The substrate limits the *ceiling*; it was
   not the reason the detector "failed". Whether augmentation + k-fold lift
   the ceiling further is the open question (§6).

## 5. Caveats (intellectual honesty)

- 37-img test / tens-per-class is noisy in absolute terms — but exp1/2/7 share
  the identical test, so the *deltas* are controlled signal, not noise.
- OBB predictions are reduced to axis-aligned enclosing boxes before overlap
  math → a small bias *against* the experts (slightly inflates their FP).
  Doesn't explain a 0.80→0.015 precision collapse.
- `det_rate_pos` / `false_alarm_neg` measure "did it predict *any* box on the
  image", not whether the box is correct — read alongside box P/R.
- exp4 (Eryth) / exp6 (Lichen) not run; Lichen is data-dead anyway (48 train).
- **The negative set is resolution-confounded (added 2026-05-19, post-discussion).**
  §3c / conclusion 6 use the 570 `data/Normal` images as negatives — these are
  960–4000 px phone photos vs ~250 px lesion thumbnails. A model can false-alarm
  on them for being an unseen *resolution/framing domain*, not for failing to
  tell healthy from diseased. So `false_alarm_neg`, `screening_acc`, the
  "always healthy = 0.939" baseline, and the "below trivial" framing inherit a
  confound — the *direction* (over-fires; calibration broken at conf 0.001 on
  the positives' own domain too) holds, the *magnitude* is not cleanly
  measured. Also: exp1/exp2 trained on **zero** negatives (`build_original`),
  and the locked-37 test has none, so its `det_rate_pos` is uninterpretable in
  isolation. **exp8a/exp8b (§7) re-measured this with resolution-normalized
  1:1 negatives in both train and test — confound resolved; see §7.**

## 6. Next direction — augmentation + k-fold CV (active, fresh conversation)

exp8 (§7) settled that detection is *not* dead and that negative training is
the single biggest lever. Two things are now untested and both are expected to
help and/or harden the numbers:

- **Augmentation** — none has ever been applied (YOLO's built-in only). exp1–8
  are all un-augmented. With ~277 thumbnails this is the most likely real lift.
- **k-fold cross-validation** — every number so far is a single 85/15 split on
  a 37-img test (±2–3 imgs = noise). At ~277 imgs, k-fold buys a *trustworthy*
  number, not a bigger one. exp8's deltas are big enough to trust; the absolute
  values are not, until k-fold.

Run both as new `Experimenting/` experiments in the exp8 pattern (binary first
— it is the better front-end, §7). Keep the fair 1:1 resolution-normalized
negatives and the `iog>=0.5` headline localisation rule. After that, the real
target is **confidence calibration** (conclusion 5) — the only confound-free
failure left.

**Deferred, not dead — the whole-image classification pivot** (EfficientNet-B2
/ DINOv2 frozen, the Normal-class resolution trap). exp8 revived detection, so
this is now a *later comparison arm*, not the escape hatch. Same seeded pool
split when it happens, so it is a controlled counterpart to exp1/exp8a.

## 7. exp8 — the correction (2026-05-19, runs the user executed)

exp8a/exp8b = exp1/exp2 **byte-identical on the positive side** (same seeded
pool split, same locked-37, same stems/labels — verified) with the only
variable = **resolution-normalized negatives** (Normal resized so its long
side = the positives' median long side, measured **276 px** — independently
confirming the ~275 px figure in §2) folded into train **and** a 37-img 1:1
slice into test. So no-skill screening_acc = **0.50** at every split.
**Number A** = the existing exp1/exp2 `best.pt` re-scored on the *same* fair
test with no retraining (`eval_fair_negatives.py`) — isolates the ruler effect
from the negative-training effect.

### 7a. Screening (conf 0.25, ratio-invariant metrics; geometry-free)

| | det_rate_pos | false_alarm_neg | screening_acc |
|---|---|---|---|
| exp2 raw 570-neg (old §3c) | 0.838 | 0.621 | 0.407 |
| **Number A** (exp2 wts, fair 37-neg) | 0.838 (31/37) | 0.568 (21/37) | 0.635 |
| **exp8b** (binary, neg in train) | 0.757 (28/37) | **0.027 (1/37)** | **0.865** |
| Number A (exp1 wts, fair) | 0.595 (22/37) | 0.351 (13/37) | 0.622 |
| **exp8a** (5-class, neg in train) | 0.622 (23/37) | **0.054 (2/37)** | **0.784** |

Reading: the fair ruler alone (raw→Number A) moved false-alarm only 0.621→0.568
— **resolution confound real but ~5pp**. The *negative training* did the work:
false-alarm **21→1 /37 (binary), 13→2 (5-class)**, det_rate ≈ held (31→28 is
within 37-img noise; 5-class even 22→23). Binary > 5-class confirmed with a
*usable* number (0.865 vs 0.784). exp8a also revived exp1's two dead classes
(Leuk/OSMF no longer 0/0/0 — directionally; per-class on 37 imgs is noisy).

### 7b. Calibration still broken (confound-free)

conf 0.001, fair test: exp8b false_alarm_neg **0.946 (35/37)**, exp8a 0.892 —
negative training barely dents the collapse. screening_acc → 0.50 (no-skill).
This is the one thing every correction leaves standing → §6's real target.

### 7c. Match-rule decision (the IoU≥0.5 gate was hiding localisations)

A no-retrain sweep (`eval_match_rules.py`) over the saved predictions. Box
P/R/F1 only — det_rate/false_alarm/screening are geometry-free and **do not
move** (so "the recall we care about" stays det_rate ≈ 0.75, rule-invariant).
exp8b, conf 0.25:

| rule | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| iou>=0.5 (continuity) | 21 | 22 | 36 | 0.488 | 0.368 | 0.420 |
| **iog>=0.5 (HEADLINE)** | 25 | 18 | 32 | 0.581 | **0.439** | 0.500 |
| iog>=0.3 & iop>=0.10 | 28 | 15 | 29 | 0.651 | 0.491 | 0.560 |

TP+FP constant (43) — the IoU gate was relabelling well-placed offset boxes as
FP; under IoG, **precision AND recall rise together** (not a trade). The IoP
floor was **inert** (coverage-only ≈ floored, ≤1 box every run/conf → the
model does not balloon boxes), so it is dropped. **Adopted:** `iog>=0.5`
("detection covers ≥half the annotated lesion") is the reported headline
localisation metric; `iou>=0.5` kept for exp1–8 continuity; `iog>=0.3` a
looser secondary. Wired into `metrics.py` (`MATCH_RULES`, `match_rule_sweep`
in every run's `metrics.{json,txt}`). Ceiling note: at conf 0.25 the model
emits only 43 boxes for 57 GT — box recall caps ~0.75 there regardless of
rule; raising it needs lower conf (calibration) or aug/k-fold, not a looser
rule.
