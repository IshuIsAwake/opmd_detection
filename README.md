# Oral Lesion Screening

Two-stage screening of oral photographs: a **YOLOv8-nano binary lesion
detector** → one shared crop → an **EfficientNet-B2 5-class disease
classifier** → Grad-CAM. If the detector finds nothing, the answer is
"healthy" — there is no `Normal` classifier class.

> **Authority for what is actually true: `Experimenting/RESULTS.md`** (§7
> results, §6 plan). `instructions.md` = original design + the three abandoned
> approaches. `HANDOFF.md` §NEXT = active brief. This README summarises; those
> govern. (Not a git-history project — these docs are the record.)

---

## 1. Current Status and Results

**State (2026-05-21): cross-validated binary detector is the headline.** The
exp8 retraction held; the exp9 → exp10 → exp11 chain produced a paired,
cross-validated baseline and validated the `instructions.md` "colour is
diagnostic" rule. Active residual = **confidence calibration**.

The clean-baseline investigation under `Experimenting/` (no classifier head,
nothing tuned, all single-variable controlled comparisons) settled:

**Headline binary detector** — `yolov8n`, 100 epochs, `imgsz=640`, `batch=8`,
seed 42, fair 1:1 resolution-normalized negatives, **`geom_no_color`
augmentation** (= YOLO defaults but `hsv_h=hsv_s=hsv_v=0`); paired 5-fold CV
on (pool + locked test = 362 positives) + 5-way negative slices:

| metric (paired 5-fold, n=5) | mean ± std | range |
|---|---|---|
| **screening_acc** | **0.842 ± 0.041** | [0.781, 0.880] |
| det_rate_pos | 0.703 ± 0.099 | [0.562, 0.817] |
| false_alarm_neg | 0.020 ± 0.027 | [0.000, 0.056] |
| box F1 (iog≥0.5) | 0.525 ± 0.036 | — |
| loc IoG on hits | 0.833 ± 0.017 | — |
| val_stock mAP50 | 0.328 ± 0.067 | — |
| FA neg @ conf 0.001 | 0.702 ± 0.068 | — |

Cumulative across all 5 folds (every positive + every fair negative tested
once): **~254 of 362 lesion images caught, ~108 missed, ~7 of 362 false
alarms**. No-skill screening = 0.50.

**Settled / measured (this is what's true now):**
- **Two-stage structure is sound.** Binary YOLO ≈ **3× the recall** of 5-class
  YOLO on identical data (exp1→exp2). Don't make YOLO classify.
- **The "loose boxes" rationale is empirically false.** Localisation-on-hits
  IoU 0.63–0.69 every run — when the detector hits, boxes are fine.
- **Roboflow scale-up does not transfer** (controlled exp2 vs exp7; experts
  P 0.015–0.031 on the original domain).
- **Detection is NOT dead.** "Below trivial baseline" was a 570-negative
  base-rate artifact on top of a real bug (exp1/exp2 trained on zero negatives).
  Fair 1:1 resolution-normalized negatives + negatives in train → screening
  works (exp8b 0.865 single fold; 0.842 ± 0.041 across the paired 5-fold CV).
- **Colour is diagnostic — VALIDATED.** YOLO's default `hsv_*` jitter was
  silently applied through exp1–8. Paired 5-fold CV (exp11): **geom_no_color
  beats default by +0.101 screening, with 3.6× tighter std, winning 4/5 folds**
  (RESULTS.md §8c). On fold 0 the default recipe caught 5/72 lesions
  (training near-collapse) where geom_no_color caught 47/72. The
  `instructions.md` §3 step 4 rule is now empirically validated, not an
  assertion.
- **exp10's single-fold 0.919 was a lucky split.** The *direction* (HSV-off
  helps) was real; the magnitude (+0.054 vs default on that split) was not.
  CV mean is 0.842, not 0.919.
- **Calibration is the one residual.** FA@conf-0.001 = 0.702 ± 0.068 across
  the paired CV — better than the single-fold 0.946 exp8b figure suggested,
  but still well above no-skill. Aug + neg-training + CV none of them moved
  this; it's the next real target.

**Localisation metric.** The `IoU≥0.5` match gate understated localisation
(it relabelled well-placed offset boxes as false positives). Headline is
**`IoG≥0.5`** ("the detection covers ≥half the annotated lesion"); `IoU≥0.5`
kept for continuity.

**Active next:** **yolov8 transfer-learning sweep** on the same kfold5
splits, with `geom_no_color` aug locked in (single-variable comparison
against the kfold5_geom_no_color_binary headline). Then confidence
calibration. The whole-image classification pivot (EfficientNet-B2 / DINOv2
frozen) stays deferred — a later comparison arm, not the path.

---

## 2. Problem Statement

Oral potentially-malignant lesions are under-screened; a phone-photo triage
("see a dentist / looks fine") has real value. The data available makes this
hard:

- **Tiny, low-resolution.** `pool/` = 325 annotated images, **one disease per
  image**, 522 boxes; ~277 are ≈250 px thumbnails. Locked `test/` = 37 images
  (57 boxes). Per-class images: Leukoplakia 42, Erythroplakia 70, OSMF 47,
  Lichen Planus 76, NH_Ulcers 90.
- **Noisy boxes.** Annotation `class_id` is reliable; box coordinates are
  rough ("roughly where the lesion is").
- **Domain gaps.** `data/Normal/` healthy mouths are 960–4000 px phone photos
  vs the ≈250 px lesion thumbnails; Roboflow third-party data is a different
  imaging domain again (and does not transfer).
- **No `Normal` annotations** — healthy images are unlabelled negatives.

Five disease classes: `Leukoplakia, Erythroplakia, OSMF, Lichen_Planus,
NH_Ulcers`.

---

## 3. Objective

A reliable, demoable screener that, from one raw photo, answers **"is there a
lesion (→ see a dentist) or not (→ looks fine)"** and, if yes, names the
likely disease with a Grad-CAM. Concretely:

- **Sensitivity + specificity together** on a fair, resolution-matched test —
  the screening decision is `det_rate_pos` paired with `false_alarm_neg`, not
  box geometry and not box recall.
- The locked 37-image `test/` is the sole headline; it is never trained or
  tuned on.
- Honest measurement over leaderboard numbers — every experiment is a
  controlled single-variable comparison, nothing tuned for a metric.

---

## 4. Every Approach (the whole story)

The project did not start at exp1 — that is the *clean restart*. The full arc:

### Phase 0 — Early approaches, all abandoned (`instructions.md` §1)

These are documented so they are **not repeated**:

1. **YOLO → YOLO cascade** (binary YOLO → 5-class YOLO): ~20–30% accuracy.
   YOLO optimises mAP, not classification; the cascade also tanked recall.
   → use a real *classifier* for the disease stage.
2. **EfficientNet on "smart crops":** ~75% in eval, **~0% in production.**
   Trained on tight human-annotation crops, served full images — total
   train/serve distribution mismatch. → train the classifier on the
   *detector's own crops*, never on annotations.
3. **LAB/CLAHE preprocessing:** hurt accuracy and was applied inconsistently
   between train and serve. → raw RGB + ImageNet norm only.

Also retired here: k-fold, Mixup/CutMix, heavy augmentation, Gradio.

### Phase 1 — The two-stage rebuild (`src/`, `instructions.md`)

The clean design: **YOLOv8n binary detector → one shared crop fn →
EfficientNet-B2 5-class → Grad-CAM**, classifier trained on the detector's own
crops (fixes mistake #2), IoP-containment matching, per-run storage. A
controlled data experiment was bolted on — `original_only` vs `plus_roboflow`
— which concluded "Roboflow data clearly helps." **That conclusion is now
stale** (Phase 2 showed it was an in-domain measurement artifact).

### Phase 2 — The clean-baseline audit (`Experimenting/`, exp1–7)

Dependency-light, no classifier head, nothing tuned. "fair test" later = 37
locked lesion + 37 resolution-normalized negatives (1:1, no-skill 0.50).

| # | what | key finding |
|---|---|---|
| **exp1** | 5-class YOLOv8n, pool 85/15, locked-37 | weak; Leuk & OSMF dead (0/0/0 @0.25) |
| **exp2** | binary YOLOv8n, same data | ≈3× exp1 recall → **structure sound** |
| **exp3** | Leukoplakia expert (Roboflow→OBB) | mAP50 0.57 Roboflow val, **P 0.031** original |
| **exp5** | OSMF expert (clean Roboflow source) | fails identically to exp3 → structural domain shift |
| **exp7** | exp2 + 955 unique Roboflow imgs | F1/mAP50 flat, recall ↓ → **Roboflow doesn't transfer** |
| (exp4/exp6) | Eryth / Lichen experts | designed, deliberately not run (exp5 decisive) |

Audit verdicts: loose-box rationale **false**, Roboflow **doesn't transfer**,
and (then) "detector below a trivial baseline" → pivot to whole-image
classification recommended.

### Phase 3 — The exp8 correction (the audit was too gloomy)

A discussion pass found the "below trivial" verdict was a 570-negative
base-rate artifact on top of a real bug — exp1/exp2 trained on **zero
negatives**. Tested controlled:

| # | what | key finding |
|---|---|---|
| **exp8a** | exp1 + resolution-normalized negatives | false-alarm 13→2 /37, screening **0.784**, dead classes revived |
| **exp8b** | exp2 + resolution-normalized negatives | **false-alarm 21→1 /37, 28/37 caught, screening 0.865** |
| **Number A** | re-score exp1/exp2 weights on the fair test, no retrain | fair ruler alone: 0.621→0.568 → resolution confound only ~5 pp |
| **match-rule sweep** | re-score exp8 under IoU vs IoG, no retrain | IoU gate hid localisations → adopt **`IoG≥0.5`** |

→ **detection revived**; "detection is dead" retracted. Helpers (no retrain):
`eval_with_negatives.py` (raw 570-neg, superseded), `eval_fair_negatives.py`
(Number A), `eval_match_rules.py`.

### Phase 4 — k-fold + augmentation chain (2026-05-20/21, exp9 → exp10 → exp11)

| # | what | key finding |
|---|---|---|
| **exp9** | 10-fold CV of exp8b recipe | screening 0.840 ± 0.063 — verified exp8b within noise; exp8b's FA@.001=0.946 was a tail draw (typical 0.618 ± 0.140) |
| **exp10** | 6-level aug sweep on exp8b split | `geom_no_color` 0.919 single-fold — best of sweep; `heavy` and `light` both worse than `default`; `instructions.md` HSV rule directionally supported |
| **exp11** | **Paired 5-fold CV: geom_no_color vs default** | **HSV-off validated.** Geom beats default +0.101 screening / +0.219 det_rate, 3.6× tighter std, wins 4/5 folds. exp10's 0.919 was lucky; CV mean is 0.842 ± 0.041 |

The paired exp11 mechanism is striking: fold-0 of default caught 5/72 lesions
(training near-collapse); same fold + HSV off caught 47/72. Consistent with
HSV jitter destabilising training on ~250-photo medical data, not just
"corrupting a useful feature" — though that mechanism remains a hypothesis
(one seed per fold, not multi-seed retrains). See `Experimenting/RESULTS.md`
§8 for the full chain.

### Phase 5 — Active next

**yolov8 transfer-learning sweep** on the same `kfold5_splits.json` with
`geom_no_color` aug locked in — single-variable comparison against the
kfold5 binary headline. Then **confidence calibration** (the one residual
neg-training + aug + CV did not move). Whole-image classification
(EfficientNet-B2 / DINOv2 frozen) stays a **deferred comparison arm**.

---

## 5. Repository Structure

```
Experimenting/             ← the truth: clean baselines + exp8 (CURRENT WORK)
├── common/                  settings · datasets · negatives · metrics · obb_convert
├── exp1..exp8b_*.py         one script per experiment (run by hand)
├── eval_fair_negatives.py   Number A (no retrain)
├── eval_match_rules.py      IoU vs IoG sweep (no retrain)
├── results/<run>/           metrics.{json,txt} = the findings (weights gitignored)
└── RESULTS.md / README.md   verdict (§7) / how-to-run

src/                        ← the original two-stage pipeline (runnable history)
├── common/                  geometry · crop (THE shared fn) · io · run_dir · dedup
├── detector/ classifier/    build · train · evaluate · gradcam
├── pipeline.py              full image → result (+ fallback)
scripts/01..06_*.py          thin CLI entrypoints for the src/ pipeline
app.py                       Streamlit demo (serves artifacts/latest)
config.py                    static roots + knobs (arm-aware helpers)

data/                       originals (READ-ONLY): pool/ test/ Normal/ additional/
                            data/new_data/ + artifacts/ = generated (gitignored)
CLAUDE.md HANDOFF.md instructions.md   design, brief, history
```

Generated (`data/new_data/`, `artifacts/`, `Experimenting/_datasets/`,
`*/train/`, `*.pt`) is regenerable and **gitignored** — see `.gitignore`.

---

## 6. Setup

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env
pip install -r requirements.txt        # includes imagehash (dedup)
```

GPU: RTX 3050 6 GB (batch sizes tuned for it). Ultralytics' AMP self-check is
disabled (`DET_AMP=False`) — its helper-model download 404s here.
**All GPU training is run by the user**; eval/re-score scripts are seconds.

---

## 7. How to Reproduce the Results

### The current work — `Experimenting/` (exp8, the real state)

```bash
# binary first — it is the better front-end
python Experimenting/exp8b_binary_negatives.py        # GPU
python Experimenting/exp8a_5class_negatives.py        # GPU
python Experimenting/eval_fair_negatives.py binary_original   # Number A, seconds
python Experimenting/eval_fair_negatives.py 5class_original   # Number A, seconds
python Experimenting/eval_match_rules.py   binary_negatives   # sweep, seconds
python Experimenting/eval_match_rules.py   5class_negatives   # sweep, seconds
# earlier baselines: exp1_5class_original.py, exp2_binary_original.py, exp7_…
```
Each writes `Experimenting/results/<run>/metrics.{json,txt}` (now incl. the
`match_rule_sweep` block). Knobs are env-overridable: `EXP_EPOCHS` (100),
`EXP_IMGSZ` (640), `EXP_BATCH` (8), `EXP_DEVICE` (`0`).

### The original two-stage pipeline — `src/` (runnable history)

An arm-aware experiment (`original_only` control vs `plus_roboflow`
treatment); its "Roboflow helps" conclusion is **stale** (overturned — §1).
Retained because it runs end-to-end:

```bash
python scripts/01_build_detector_dataset.py --arm original_only
python scripts/02_train_detector.py --arm original_only --out_dir yolov8n@640_original_only   # GPU
python scripts/03_build_classifier_data.py
python scripts/04_train_classifier.py                                                          # GPU
python scripts/05_evaluate_pipeline.py
python scripts/06_compare_runs.py artifacts/runs/<A> artifacts/runs/<B>
streamlit run app.py                                   # demo
```
`03/04/05` read the arm from `artifacts/CURRENT_RUN`. `ORAL_SMOKE=1` slashes
epochs for a fast end-to-end plumbing check (numbers are throwaway). Don't
interleave the two arms.

---

## 8. Notes & Invariants

- **Locked test.** `data/test/` (37) is the sole headline, touched only at
  final eval — never trained or tuned on.
- **Screening = det_rate + false_alarm together**, geometry-free. Box recall
  and IoU/IoP/IoG are localisation / crop-quality, not the screening verdict.
  Headline localisation rule: `IoG≥0.5` (`IoU≥0.5` kept for continuity).
- **No `Normal` classifier class** — "healthy" = the detector finds nothing.
- **Negatives must be resolution-normalized** (to the positives' ~276 px
  median long side) in train *and* test, or the specificity number is a
  resolution artifact.
- **Originals are read-only**: `data/pool/`, `data/test/`, `data/Normal/`,
  `data/additional/`. All generated output goes under `data/new_data/` /
  `artifacts/` / `Experimenting/_datasets/` (gitignored, rebuildable).
- Deliberately deferred until the baseline is trustworthy: heavy augmentation
  zoo, LAB/CLAHE (raw RGB + ImageNet norm only). Light augmentation + k-fold
  are the *active next step*, not part of the deferral.
- Roboflow polygon labels need the dedicated parser (`read_yolo_label()`
  silently drops them); use the polygon yolov8 exports, never `-obb`.
