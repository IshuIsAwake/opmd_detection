# CLAUDE.md

Project-level guidance for Claude Code (claude.ai/code) and other AI
assistants working in this repo.

## What this is

Two-stage oral-lesion screening: **YOLOv8n binary lesion detector →
shared crop → EfficientNet-B0 5-class disease classifier (TTA at
inference)**. If the detector finds nothing, the answer is "healthy" —
there is no `Normal` class in the classifier.

The MVP is **complete**. See `HANDOFF.md` for the locked production
config (detector + classifier weight paths, conf, serve_pad, TTA
recipe) and `Experiment_Results.md` §17 for the measurement.

Grad-CAM was dropped from the MVP scope (Round 1 brief) and the
originally-named EfficientNet-B2 was downsized to B0 (Round 2).
Re-introduce only if a specific demo requirement comes back for them.

## Authoritative docs (read these, in order)

1. **`README.md`** — current headline numbers, problem statement,
   architecture, metrics, directory layout.
2. **`Experiment_Results.md`** — chronological record of every
   experiment we ran. Where this file and code disagree, this is the
   authority on what was actually measured.
3. **`HANDOFF.md`** — the current active brief (what the next chat
   should pick up).
4. **`instructions.md`** — original architecture intent + the three
   abandoned approaches (binding lessons; do not re-derive).
5. **`Experimenting/exp_readme.md`** — how to run the experiment
   harness.

## Environment

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env
```

GPU: RTX 3050 **6 GB** — batch sizes are tuned for it. Ultralytics' AMP
self-check is disabled (`DET_AMP = False`) because its helper-model
download 404s here. Do not re-enable.

## The GPU-training rule

**The user runs all GPU training (Steps 02 and 04, and any
`Experimenting/exp*` training script).** You write code and hand back
exact copy-pasteable commands; you do **not** launch training yourself.

## Invariants you must not violate

- **Originals are read-only:** `data/pool/`, `data/test/`,
  `data/Normal/`, `data/Normal/NON CANCER/`, `data/additional/`.
  Everything generated goes under `data/new_data/` and `artifacts/`
  (and `Experimenting/_datasets/` / `Experimenting/results/`). All
  generated paths are safe to delete and rebuild.
- **No Roboflow image ever reaches the classifier.** Detector labels
  are single-class `0`; disease provenance lives only in
  `disease_sidecar.json` (do not wire it in).
- **Raw RGB + ImageNet normalization only.** No LAB / CLAHE.
- **The classifier is trained on the detector's own crops**, never on
  raw human-annotation crops. One shared crop function imported by
  both the data builder and the live pipeline. *(Caveat: the shipped
  Round-3 classifier was trained on GT crops at pad=0.4 — not detector
  crops — because Round 1 / Round 2 measurements showed train/serve
  crop-distribution mismatch is no longer the bottleneck on B0
  fine-tune. See §16 and §17. This is a knowing departure from the
  original invariant, justified by measurement.)*
- **No `Normal` class in the classifier.** "Healthy" is decided
  upstream by "detector found nothing".
- **Classifier serve-pad = 0.20** at inference, with **4-view TTA**
  (identity, hflip, rot ±10°). Mean post-softmax across views, then
  across boxes. See `Experiment_Results.md` §17. Other config (merge,
  serve_pad = 0.00 etc.) is supported via flags but worse on the
  Phase 2 headline.

## Landmines (each one costs a debugging session)

- **`src/common/io.py::list_images` is non-recursive.** It sees
  `data/Normal/` (120 imgs) but **not** `data/Normal/NON CANCER/`
  (450 imgs). Negatives (~570 total) must be sourced from both dirs
  explicitly. Do not switch to `rglob` — other call sites depend on
  the current behavior.
- **Roboflow polygon labels need the dedicated parser**
  (`src/detector/convert_roboflow.py`). `read_yolo_label()` hard-drops
  any line where `len(parts) != 5`, silently eating every polygon.
  Use the polygon YOLOv8 exports, never the `-obb` variant.
- **Roboflow stems collide** with each other and with `pool/`.
  Generated files are namespaced (`opmdseg__`, `osmf__`, `leukov2__`).
  No silent `shutil.copy2` overwrites.
- **Adopted detector aug recipe = `geom_no_color`** (YOLO defaults but
  `hsv_h = hsv_s = hsv_v = 0`). YOLO's defaults silently apply HSV
  jitter — do not re-introduce it. See `Experiment_Results.md` §9 / §10
  for why.
- **Adopted operating point = conf = 0.10** for the binary detector.
  Do not benchmark at conf = 0.25 alone (that was the YOLO default and
  it hid signal).
- **Headline localisation rule = `IoG ≥ 0.5`** (`IoU ≥ 0.5` kept for
  continuity). Wired into `Experimenting/common/metrics.py`.
- **Do not ship fold 0** of `kfold5_geom_no_color_binary` as the
  production weight — it is globally less aggressive than folds 1–4
  (see `Experiment_Results.md` §11). Fold 2 or 3 are middle-of-pack.
  Adopted production weight: **fold 2**.
- **5-fold ensemble on the existing Phase 2 test set is not honest.**
  Each test image was in 4/5 folds' training sets. The Phase 2 numbers
  reported in §13-§17 use each fold's own held-out test slice; any
  ensemble evaluation would be ~80 % leaked. Shipping single-fold
  weights (fold 2) for the MVP. See §17.
- **`src/` and `scripts/` are archived.** They implement the abandoned
  EfficientNet-B2-on-tight-human-crops design (Phase 0 mistake #2).
  Numbers in §1 were retracted by §7. Do not port from there — the
  current MVP is in `Experimenting/classifier_experiments/`.

## Code style

Many small, single-responsibility files, each editable in isolation.
Originally `src/common/`, `src/detector/`, `src/classifier/` with thin
CLI wrappers in `scripts/`; the working version is now
`Experimenting/classifier_experiments/common/` (`crops.py`, `folds.py`,
`dataset.py`, `model.py`, `model_b0.py`, `train_eval.py`, `metrics.py`,
`summarize.py`, `box_ops.py`, `run_arm.py`, `materialise.py`) with thin
entry scripts at the level above. Match this when adding code.
Conversion / dedup / geometry helpers are pure and unit-testable —
keep training deps out of them.

## When docs and code disagree

The docs describe intent. Investigate the gap, do not silently "fix"
code to match docs or docs to match code. If the gap is real, surface
it.
