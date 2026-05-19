# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two-stage oral-lesion screening: **YOLOv8-nano binary lesion detector → one
shared crop → EfficientNet-B2 5-class disease classifier → Grad-CAM**. If the
detector finds nothing, the answer is "healthy" — there is no `Normal`
classifier class.

## ⚠ Read this before trusting the design (audit, 2026-05-19)

A clean-baseline investigation under `Experimenting/` (controlled, no-classifier
YOLO baselines) overturned two of the main pipeline's premises. **Do not
re-derive or re-defend these — they were measured:**

- The **"loose boxes" rationale is empirically false.** Every run shows
  localisation-on-hits IoU 0.63–0.69 / IoG 0.80–0.92 — predictions are not
  "tight boxes lost inside huge loose GT". The IoP-matching + train-classifier-
  on-detector-crops machinery solves a problem that isn't there.
- The **Roboflow scale-up does not transfer.** The main project's
  "data clearly helps" was measured on `web_holdout` (Roboflow-derived =
  in-domain). Controlled, on the original domain, the gain is ≈0; the
  Roboflow→original domain shift is severe and structural (incl. a 250 px-vs-
  640 px resolution gap).
- **RETRACTED by exp8 (2026-05-19): "the detector is below a trivial baseline /
  detection is dead".** That was a 570-negative base-rate artifact on top of a
  real bug — exp1/exp2 trained on **zero negatives**. With negatives in train
  and a fair 1:1 resolution-normalized test, **exp8b binary = 28/37 lesion
  images caught, 1/37 false alarm, screening_acc 0.865** (no-skill 0.50);
  exp8a 5-class 0.784. Detection is a credible screener; binary > 5-class with
  a usable number. Do **not** repeat "detection is hopeless".
- What *is* sound: the binary-detector-then-classifier **structure** (binary
  ≈3× the recall of 5-class). The **only confound-free failure left is
  confidence calibration** (conf-0.001 still fires on everything even after
  negative training). Localisation headline metric is now **`iog>=0.5`** (the
  IoU≥0.5 gate understated it; `iou>=0.5` kept for continuity) — wired into
  `metrics.py`; this does not change det_rate (geometry-free, screening ≈0.75).

Full evidence is in **`Experimenting/RESULTS.md` §7** (results) **/ §6**
(plan). The **active next step is augmentation + k-fold CV** (then
calibration) — `HANDOFF.md` §NEXT. The whole-image classification pivot
(EfficientNet/DINOv2) is **deferred, a later comparison arm — not the path**. The `src/` pipeline still runs as documented; treat its rationale as
historical, not validated.

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
