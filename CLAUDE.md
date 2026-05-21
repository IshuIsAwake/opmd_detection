# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two-stage oral-lesion screening: **YOLOv8-nano binary lesion detector → one
shared crop → EfficientNet-B2 5-class disease classifier → Grad-CAM**. If the
detector finds nothing, the answer is "healthy" — there is no `Normal`
classifier class.

## ⚠ Read this before trusting the design (audit + paired CV, 2026-05-21)

A clean-baseline investigation under `Experimenting/` (controlled, no-classifier
YOLO baselines) overturned two of the main pipeline's premises, retracted a
third gloomy verdict via exp8, and validated the colour rule via exp11.
**These were measured, not argued — do not re-derive or re-defend:**

- **"Loose boxes" rationale is empirically false.** Every run shows
  localisation-on-hits IoU 0.63–0.69 / IoG 0.80–0.92 — predictions are not
  "tight boxes lost inside huge loose GT". The IoP-matching + train-
  classifier-on-detector-crops machinery solves a problem that isn't there.
- **Roboflow scale-up does not transfer.** "Data clearly helps" was measured
  on `web_holdout` (Roboflow-derived = in-domain). Controlled, on the original
  domain, the gain is ≈0; the Roboflow→original domain shift is severe and
  structural (250 px vs 640 px resolution gap, camera/lighting).
- **RETRACTED by exp8: "the detector is below a trivial baseline / detection
  is dead".** That was a 570-negative base-rate artifact on top of a real bug
  — exp1/exp2 trained on **zero negatives**. exp8b single-fold = 28/37 lesion
  images caught, 1/37 false alarm, screening_acc 0.865. Detection is a
  credible screener. Do **not** repeat "detection is hopeless".
- **VALIDATED by exp11 paired 5-fold CV (2026-05-21): the
  `instructions.md` "colour is diagnostic / no HSV" rule.** YOLO's defaults
  silently apply `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` — exp1–10 all ran with
  HSV jitter on. Paired 5-fold CV (`geom_no_color` = defaults but `hsv_*=0`)
  beats default by **+0.101 paired screening, 3.6× tighter std, winning
  4/5 folds**; on fold 0 default caught 5/72 lesions where geom_no_color
  caught 47/72, suggesting HSV jitter destabilises training on small medical
  data (one seed per fold — hypothesis, not proof). `geom_no_color` is now
  the adopted binary-detector aug recipe.
- **Cross-validated headline (`yolov8n` binary, `geom_no_color` aug, paired
  5-fold CV) — these are the current trustworthy numbers:**
  - screening_acc **0.842 ± 0.041** (CI95 ±0.036), range [0.781, 0.880]
  - det_rate_pos **0.703 ± 0.099**
  - false_alarm_neg **0.020 ± 0.027**
  - FA neg @ conf 0.001 **0.702 ± 0.068** ← residual
  - loc IoG on hits **0.833 ± 0.017**
  - val_stock mAP50 **0.328 ± 0.067**
  - Cumulative across the 5 folds: ~254/362 lesions caught, ~108 missed,
    ~7/362 false alarms.
  - exp10's single-fold 0.919 was a lucky split, **not** the headline.
- **The two-stage *structure* is sound** (binary YOLO ≈3× the recall of
  5-class on identical data; exp1→exp2). Localisation headline rule:
  **`iog>=0.5`** ("≥half the lesion covered"), `iou>=0.5` kept for continuity
  — wired into `metrics.py`. Headline is screening (det_rate +
  false_alarm together), geometry-free.
- **The one confound-free residual: confidence calibration.** FA@conf-0.001
  ≈ 0.70 across the paired CV — better than exp8b's single-fold 0.946 tail
  suggested, still bad. Aug + neg-training + CV none of them moved this.

Full evidence in **`Experimenting/RESULTS.md` §8** (the exp9 → exp10 → exp11
chain). The **active next step is the yolov8 transfer-learning sweep** on
the same kfold5 splits + `geom_no_color` aug, then confidence calibration —
`HANDOFF.md` §NEXT. The whole-image classification pivot
(EfficientNet-B2 / DINOv2 frozen) is **deferred, a later comparison arm —
not the path**. The `src/` pipeline still runs as documented; treat its
design rationale as historical record. **Use `geom_no_color` (= YOLO
defaults but `hsv_h=hsv_s=hsv_v=0`) as the default detector training recipe
in any new experiment.**

## Source-of-truth docs (read these; do not re-derive history from git — not a git repo)

- **`Experimenting/RESULTS.md`** — the audit verdict + all controlled results;
  the current authority on what is actually true. Read first.
- **`Experimenting/README.md`** — the clean-baseline harness (how to run it).
- **`instructions.md`** — the original two-stage design, rationale, and the
  **three abandoned approaches** (binding history; note the loose-box rationale
  is now contradicted — see the audit note above).
- **`HANDOFF.md`** — concluded `plus_roboflow` experiment + the active next
  plan (the model pivot).
- **`README.md`** — current pipeline, commands, tunable knobs (results stale).
- **`config.py`** — single source of truth for all paths and hyperparameters;
  every knob is env-overridable. No logic lives here.

When these docs and the code disagree, the docs describe intent — investigate
the gap, don't silently "fix" code to match or vice versa.

## Environment & the GPU-training rule

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env
```

GPU is an RTX 3050 **6 GB** (batch sizes are tuned for it). **The user runs all
GPU training (Steps 02 and 04).** You write code and hand back exact commands;
you do **not** launch training yourself. Ultralytics' AMP self-check is disabled
(`DET_AMP = False`) because its helper-model download 404s in this environment —
do not re-enable it.

## Running the pipeline

The pipeline is a 6-step script chain. It is an **arm-aware experiment**: run
each arm's full chain end-to-end, never interleave the two. `01`/`02` take
`--arm {original_only,plus_roboflow}`; `03/04/05` take no `--arm` — they read it
from `artifacts/CURRENT_RUN`'s `run.json`.

```bash
# one arm, in order:
python scripts/01_build_detector_dataset.py --arm <arm>
python scripts/02_train_detector.py --arm <arm> --out_dir <name>   # GPU (user runs)
python scripts/03_build_classifier_data.py
python scripts/04_train_classifier.py                              # GPU (user runs)
python scripts/05_evaluate_pipeline.py
python scripts/06_compare_runs.py <runA> <runB>                    # side-by-side

streamlit run app.py                                               # serves artifacts/latest
```

**Smoke test** (proves plumbing end-to-end in minutes; throw the numbers away):
prefix any command with `ORAL_SMOKE=1`. It only shrinks epoch counts —
data-build/dedup/web_holdout are not shrunk. Must pass for **both** arms.

There is no test framework — the smoke chain is the integration check.

## Architecture (the parts that span files)

- **One shared crop function** (`src/common/crop.py`) is imported by *both*
  Step 3 (building classifier training data) and `src/pipeline.py` (serving).
  The classifier is trained on the detector's *own crops*, never on human
  annotations. This identity is the structural fix for the train/serve mismatch
  that killed an earlier attempt — the single highest-risk regression spot.
- **Matching is IoP containment, not IoU** (`MATCH_IOP_THRESH=0.70`). Annotator
  GT boxes are huge/loose; a correct tight prediction sits *inside* the GT, so
  IoU is tiny but containment ≈ 1. IoU is computed/reported only, never gates.
- **Detector is tuned for recall at low conf**, not mAP. mAP is irrelevant to
  the product; `recommend_conf` currently pins near 0.001 (calibration is the
  known live bottleneck).
- **Run storage** (`src/common/run_dir.py`): one self-contained folder per pass
  under `artifacts/runs/<name>/` — no MLflow, no hash dirs. Step 02 starts a
  run; 03/04/05 append into it via `CURRENT_RUN`; `latest` symlink is what the
  demo/pipeline serve. `--out_dir` names the run verbatim (`@` is preserved).
- **The experiment**: control (`original_only`) vs treatment (`plus_roboflow`,
  adds ~2.3k tight-box Roboflow images to the detector's *train only*). Identical
  code; only detector training data differs.

## Invariants you must not violate

- **`data/test/` (37 images) is the locked, sole headline metric** — only
  touched in Step 05, never trained or tuned on. `data/new_data/web_holdout/`
  is a *secondary, eval-only* higher-N detector signal — never train/tune on it.
- **Originals are read-only**: `data/pool/`, `data/test/`, `data/Normal/`,
  `data/Normal/NON CANCER/`, `data/additional/`. Everything generated goes under
  `data/new_data/` (and `artifacts/`), which is safe to delete and rebuild.
- The `pool/` train/val split must be **byte-identical across both arms**
  (computed once → `pool_split.json`, reused by both arms and Step 3). The only
  permitted cross-arm difference is the Roboflow images appended to treatment's
  train split.
- **No Roboflow image ever reaches the classifier.** Detector labels are
  single-class `0`; disease classes live only in `disease_sidecar.json`
  (provenance for a *future* classifier phase — do not wire it in now).
- Raw RGB + ImageNet normalization only. No LAB/CLAHE, no augmentation zoo, no
  k-fold — these are deliberately deferred (see `instructions.md` §1).

## Landmines (cost a debugging session each)

- `src/common/io.py::list_images` uses non-recursive `glob`, so it sees
  `data/Normal/` (120) but **not** `data/Normal/NON CANCER/` (450). Negatives
  (~570) must be sourced from both dirs explicitly; do not switch it to `rglob`
  (other call sites depend on the current behavior).
- Roboflow polygon labels: `read_yolo_label()` hard-drops any line where
  `len(parts) != 5`, silently eating every polygon. `convert_roboflow.py` has a
  dedicated polygon parser — use it; never route polygons through
  `read_yolo_label()`.
- Use the Roboflow **polygon** yolov8 exports, never the `-obb` variant.
- Roboflow stems collide with each other and with `pool/`; generated files are
  namespaced (`opmdseg__`, `osmf__`, `leukov2__`). No silent `copy2` overwrites.
- Eval is small (37 test / ~49 val): ±2% accuracy is noise — that's why
  `web_holdout` (~142 imgs) exists as the higher-N detector check.

## Code style

Many small, single-responsibility files, each editable in isolation
(`src/common/`, `src/detector/`, `src/classifier/`, thin CLI wrappers in
`scripts/`). Match this when adding code. Conversion/dedup/geometry helpers are
pure and unit-testable — keep training deps out of them.
