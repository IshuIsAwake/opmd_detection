# Experimenting/ — RESULTS & VERDICT (2026-05-19)

Source of truth for the clean-baseline investigation. `README.md` (this dir)
describes *how* to run; this file is *what we found and what it means*.
Everything below is from runs the user executed on the RTX 3050; no number is
tuned or cherry-picked.

> One-line verdict (after exp8 §7 and the §8 chain — current authority): the
> "loose boxes" rationale is empirically false and the Roboflow scale-up does
> not transfer — both hold. The "detection is dead" verdict is **retracted**
> (was a base-rate artifact + a zero-negatives bug; §7). The cross-validated
> binary detector with `geom_no_color` aug (YOLO defaults but `hsv_*=0`) is:
> **screening_acc 0.842 ± 0.041, det_rate 0.703 ± 0.099, false_alarm
> 0.020 ± 0.027, FA@conf-0.001 0.702 ± 0.068, loc IoG hits 0.833 ± 0.017**
> (paired 5-fold CV, §8c). The `instructions.md` "colour is diagnostic" rule
> is empirically **validated** (was design assertion until §8c — HSV-off
> beats defaults +0.101 paired screening, 3.6× tighter std, 4/5 folds).
> exp10's single-fold 0.919 was a lucky split, not the headline. Active
> residual: **confidence calibration** (FA@conf-0.001 ≈ 0.70, the one thing
> aug + neg-training + CV didn't move). Active next: yolov8 transfer-learning
> sweep on the same kfold5 splits + geom_no_color aug — see `HANDOFF.md`
> §NEXT.

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

## 6. Next direction — DONE (k-fold + aug both executed; see §8)

Historical: this section called for augmentation + k-fold CV next. Both ran
(§8). Current active brief = transfer-learning sweep, then confidence
calibration. See **`HANDOFF.md` §NEXT**.

The original deferred whole-image classification pivot (EfficientNet-B2 /
DINOv2 frozen) remains deferred — exp8 revived detection (§7) and §8 confirmed
the recipe is stable; the pivot is now a *later comparison arm*, not the path.

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

## 8. exp9 → exp10 → exp11 — the k-fold + augmentation chain (2026-05-20/21)

The §6 plan ran in three controlled stages: cross-validate the exp8b headline
(exp9), sweep augmentation levels (exp10), then paired-CV the most promising
recipe (exp11). The chain replaced the single-fold exp8b 0.865 with a
**cross-validated, paired headline** and validated the original
`instructions.md` "colour is diagnostic" rule.

### 8a. exp9 — 10-fold CV of exp8b (verify the single-fold number)

Stratified 10-fold on (pool + locked-test = 362 positives) + 10-way shuffled
~570 resolution-normalised negatives, fair 1:1 test slice per fold (no-skill
0.50), inner 85/15 stratified train/val on the other 9 folds, blackbox test
in every fold. Same recipe as exp8b otherwise (YOLO defaults — note this
includes HSV jitter; see §8b).

| metric | exp8b (single fold) | **kfold10 (mean ± std)** | range |
|---|---|---|---|
| screening_acc | 0.865 | **0.840 ± 0.063** (CI95 ±0.039) | [0.757, 0.932] |
| det_rate_pos | 0.757 | 0.694 ± 0.126 | [0.514, 0.892] |
| false_alarm_neg | 0.027 | **0.013 ± 0.034** | [0.000, 0.108] |
| box R (iog≥0.5) | 0.439 | 0.429 ± 0.097 | [0.270, 0.633] |
| loc IoG (hits) | 0.811 | **0.811 ± 0.026** | [0.752, 0.843] |
| FA neg @ conf 0.001 | 0.946 | **0.618 ± 0.140** | [0.400, 0.829] |
| val_stock mAP50 | 0.280 | 0.326 ± 0.095 | [0.221, 0.546] |

**10/10 folds beat no-skill (0.50).** exp8b's 0.865 reproduces inside CI95
(slightly above mean, not an outlier). **The FA@conf-0.001 "0.946 fires on
everything" framing from §7b was a tail draw** — typical is 0.618 ± 0.140;
4/10 folds are below 0.50 at conf 0.001 already. Calibration is still broken,
just not catastrophically.

Localisation IoG on hits = **0.811 ± 0.026** across folds — when the detector
hits, the box covers >80% of the annotation in every fold, std 2.6 pp.
Crop-quality is rock-solid.

### 8b. exp10 — augmentation sweep on the exp8b split (6 levels, 1 seed)

A controlled sweep across six YOLO augmentation configs on exp8b's static
split — same data, same seed (42), only variable = the train kwargs. The
seventh "level" missing from §7 was that **"exp1–8 = YOLO built-in only"
quietly meant "YOLO defaults including `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`"**.
The `instructions.md` §3 step 4 rule "No HSV — colour is diagnostic" was thus
silently violated for every run on record.

| level | screen_acc | det_rate | false_alm | F1_iog | mAP50 | FA@.001 | notes |
|---|---|---|---|---|---|---|---|
| off | 0.595 | 0.243 | 0.054 | 0.203 | 0.160 | 0.811 | fliplr only — model under-trains |
| light | 0.676 | 0.378 | 0.027 | 0.354 | 0.224 | 0.946 | mild geom, no mosaic, no colour |
| default (exp8b) | 0.865 | 0.757 | 0.027 | 0.500 | 0.280 | 0.946 | YOLO defaults — reproduces exp8b |
| **geom_no_color** | **0.919** | **0.838** | **0.000** | **0.545** | **0.385** | **0.892** | defaults but hsv_*=0 |
| heavy | 0.757 | 0.513 | 0.000 | 0.405 | 0.277 | 0.946 | defaults + mixup/copy_paste/big geom |
| heavy_no_color | 0.784 | 0.568 | 0.000 | 0.447 | 0.330 | 0.892 | heavy + hsv_*=0 |

Findings on the single split:

1. **HSV-off lifts both recipes.** default → geom_no_color = +0.054 screening;
   heavy → heavy_no_color = +0.027. Same sign, same direction.
2. **Mosaic + scale + translate (in YOLO defaults) are essential** at ~277
   thumbnails. off → default = +0.270 screening; without mosaic (light) the
   model never reaches default's performance.
3. **The kitchen sink (heavy) is independently bad,** beyond colour:
   geom_no_color → heavy_no_color = −0.135 screening. mixup + copy_paste +
   big rotate/scale/shear overwhelm the small dataset.
4. **Aug is NOT the lever for calibration.** FA@.001 stayed in [0.81, 0.95]
   across all six levels.

**Caveat: 0.919 is one 85/15 split.** kfold10 default mean is 0.840 ± 0.063
(§8a). geom_no_color's 0.919 sits +1.25σ above that — plausibly real, plausibly
lucky. exp11 paired-tests it.

### 8c. exp11 — paired 5-fold CV (the definitive HSV-off test)

A fresh stratified 5-fold split (`_datasets/kfold5_splits.json`, K=5,
~72 test pos + ~72 fair neg per fold). **Same five folds for both runs**;
only variable = the aug config. This is the clean paired comparison §8b
flagged was missing.

| paired metric | default (HSV on) | **geom_no_color (HSV off)** | Δ (geom − default) |
|---|---|---|---|
| **screening_acc** | 0.741 ± **0.147** | **0.842 ± 0.041** | **+0.101**  (geom 3.6× tighter) |
| **det_rate_pos** | **0.484** ± **0.297** | **0.703** ± 0.099 | **+0.219**  (geom 3.0× tighter) |
| false_alarm_neg | 0.003 ± 0.006 | 0.020 ± 0.027 | −0.017 |
| box F1 (iou≥0.5) | 0.254 ± 0.145 | 0.379 ± 0.041 | +0.125 |
| box F1 (iog≥0.5) | 0.362 ± 0.193 | 0.525 ± 0.036 | +0.163 |
| loc IoG (hits) | 0.826 ± 0.042 | 0.833 ± 0.017 | +0.007 |
| FA neg @ conf 0.001 | 0.767 ± 0.128 | 0.702 ± 0.068 | +0.065 |
| val_stock mAP50 | 0.252 ± 0.085 | 0.328 ± 0.067 | +0.075 |

**Per-fold screening_acc (same fold = same split):**

| fold | default | geom_no_color | Δ |
|---|---|---|---|
| 0 | **0.534** | 0.824 | **+0.290** |
| 1 | 0.678 | 0.781 | +0.103 |
| 2 | 0.868 | 0.875 | +0.007 |
| 3 | 0.729 | 0.847 | +0.118 |
| 4 | **0.894** | 0.880 | −0.014 |

**geom_no_color wins 4/5 folds.** Win counts across every screening / box /
mAP metric: **geom 4/5 vs default 1/5**. The only metric default "wins" is
`false_alarm` (4/5) — but that's because default barely detects, not because
it discriminates better; det_rate collapsed to 0.484 ± 0.297.

**The fold-0 datum is striking.** Same data, same fold, same seed, only HSV
differs:

- default fold 0: caught **5 / 72 lesions** (det_rate 0.068) — training
  essentially failed to converge into a useful detector on this fold.
- geom_no_color fold 0: caught **47 / 72 lesions** (det_rate 0.649) — normal
  performance.

This is consistent with a **training-stability hypothesis** for HSV-off on
small medical data, not just "colour is a feature, don't corrupt it." On
~250-photo positive pools, HSV jitter appears to derail some training
trajectories outright. **Hypothesis, not proof — we have one seed per fold,
not multiple retrains.** Worth confirming with a multi-seed ablation if the
finding ever matters more than the absolute number does.

### 8d. Adopted verdict

- **`geom_no_color` is the binary-detector baseline** (YOLO defaults +
  `hsv_h=hsv_s=hsv_v=0`). Validated by paired 5-fold CV: +0.101 mean,
  3.6× tighter std, wins 4/5 folds, better calibration, better mAP50,
  comparable localisation.
- **The `instructions.md` §3 step 4 "colour is diagnostic" rule is
  empirically validated** (was "design assertion" until §8c).
- **Honest cross-validated numbers** for the binary detector:
  - screening_acc **0.842 ± 0.041**  (CI95 ±0.036, n=5 folds, paired-CV)
  - det_rate_pos **0.703 ± 0.099**
  - false_alarm_neg **0.020 ± 0.027**
  - FA@conf-0.001 **0.702 ± 0.068**  (calibration: still the residual)
  - loc IoG on hits **0.833 ± 0.017**
- exp10's single-fold **0.919 was a lucky split**, not the headline. The
  *direction* was real (HSV-off helps); the magnitude wasn't.
- Cumulative across the 5 folds (all 362 positives + 362 fair negatives each
  appeared in test once): **~254/362 lesion images caught, ~108 missed, ~7/362
  false alarms**.

Active brief now lives in **`HANDOFF.md` §NEXT**: yolov8 transfer-learning
sweep on the same kfold5 splits (with `geom_no_color` aug locked in), then
confidence calibration as the remaining residual. Whole-image classification
pivot stays deferred — a later comparison arm, not the path.
