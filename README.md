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

**State (2026-05-19): detection is a credible screener; the earlier "detection
is dead" verdict was retracted by a controlled experiment (exp8).**

The clean-baseline investigation under `Experimenting/` (no classifier head,
nothing tuned) settled the following. Deltas are controlled single-variable
comparisons; absolute numbers are on a small test (see caveats).

**Settled / measured:**
- **Two-stage structure is sound.** Binary YOLO ≈ **3× the recall** of 5-class
  YOLO on identical data (exp1→exp2). Don't make YOLO classify.
- **The "loose boxes" rationale is empirically false.** Localisation-on-hits
  IoU 0.63–0.69 every run — when the detector hits, boxes are fine.
- **Roboflow scale-up does not transfer** to the original imaging domain
  (controlled exp2 vs exp7; experts P 0.015–0.031 on the original domain).
- **"Detector is below a trivial baseline" — RETRACTED.** It was a
  570-negative base-rate artifact on top of a real bug: exp1/exp2 trained on
  **zero** negatives. Give the binary detector the negative signal and measure
  it fairly (resolution-normalized 1:1 negatives):

  | conf 0.25, fair test | lesions caught | false alarm | screening_acc |
  |---|---|---|---|
  | exp2 weights, no neg training (*Number A*) | 31/37 | 21/37 | 0.635 |
  | **exp8b — binary, neg in train** | **28/37** | **1/37** | **0.865** |
  | **exp8a — 5-class, neg in train** | 23/37 | 2/37 | 0.784 |

  No-skill line = 0.50 (1:1). The fair ruler alone moved false-alarm only
  0.621→0.568 (resolution confound real but ~5 pp); **negative training did
  the work** (false-alarm 21→1 /37, recall ≈ held). Binary > 5-class with a
  usable number.

**The one open, confound-free problem: confidence calibration.** At conf
0.001 every detector — even after negative training — fires on everything
(exp8b false-alarm 0.946). Not data, not resolution, not base rate.

**Localisation metric.** The `IoU≥0.5` match gate understated localisation
(it relabelled well-placed offset boxes as false positives). Headline is now
**`IoG≥0.5`** ("the detection covers ≥half the annotated lesion"); `IoU≥0.5`
kept for continuity. This does **not** move the screening recall (det_rate is
geometry-free, ≈0.75) — it is the honest crop-quality number.

**Active next (separate work):** **augmentation + k-fold CV** (neither ever
tried; expected to lift/harden the numbers), then confidence calibration. The
whole-image classification pivot (EfficientNet-B2 / DINOv2 frozen) is
**deferred — a later comparison arm, not the path.**

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

### Phase 4 — Active next (not yet done)

**Augmentation + k-fold CV** (never tried; k-fold was retired in Phase 0 as
premature, now appropriate), then **confidence calibration** (the only
confound-free failure left). Whole-image classification (EfficientNet-B2 /
DINOv2 frozen) is a **deferred comparison arm, not the path**.

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
