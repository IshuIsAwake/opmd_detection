# Experimenting/ — the true baseline

> **Results & verdict: `RESULTS.md`** (read that for what we found and what it
> means). This file is just *how to run it*.

Independent YOLO models, **no classifier head, nothing tuned, no metric
optimised**. Each is one script you run by hand. The point is honest baseline
numbers that answer one question: *does the Roboflow data actually help?*
Conclusion (see `RESULTS.md` §7): Roboflow does **not** help on the original
domain. The "detector is below trivial / dead" verdict was **retracted by
exp8** — it was a base-rate artifact + a zero-negatives bug; with negatives in
train and a fair 1:1 test, **exp8b binary = 28/37 caught, 1/37 false alarm,
screening_acc 0.865**. Detection is a credible screener. Active next:
**augmentation + k-fold CV** (`RESULTS.md` §6, `HANDOFF.md` §NEXT); the
whole-image classification pivot is deferred, a later comparison arm.

This tree is self-contained. It reads the original data and the Roboflow
exports **read-only** and writes only under `Experimenting/`. It does not touch
the main pipeline (`config.py`, `artifacts/`, `data/new_data/`).

## The experiments

| # | script | model | train / val | test |
|---|---|---|---|---|
| 1 | `exp1_5class_original.py` | `yolov8n` | pool 85/15, 5 classes | `data/test` (37) |
| 2 | `exp2_binary_original.py` | `yolov8n` | pool 85/15, classes→`lesion` | `data/test` (37) |
| 3 | `exp3_leukoplakia_expert.py` | `yolov8n-obb` | Leukoplakia.v2 (box→obb) + OPMD-obb cls0 | all originals |
| 4 | `exp4_erythroplakia_expert.py` | `yolov8n-obb` | OPMD-obb cls2 | all originals |
| 5 | `exp5_osmf_expert.py` | `yolov8n-obb` | OSMF polygons → min-area obb | all originals |
| 6 | `exp6_lichen_planus_expert.py` | `yolov8n-obb` | OPMD-obb cls1 | all originals |
| 7 | `exp7_binary_plus_roboflow.py` | `yolov8n` | exp2 pool split **+ 955 unique Roboflow** | `data/test` (37) |
| 8a | `exp8a_5class_negatives.py` | `yolov8n` | exp1 + resolution-normalized negatives | 37 lesion + **37 fair neg** |
| 8b | `exp8b_binary_negatives.py` | `yolov8n` | exp2 + resolution-normalized negatives | 37 lesion + **37 fair neg** |

Eval helpers (no retraining):
- `eval_with_negatives.py <run>` — re-scores `best.pt` on the **raw** 607-img
  set (37 lesion + 570 Normal). The 570 are resolution-confounded — superseded
  by the fair version below for any specificity claim.
- `eval_fair_negatives.py <run>` — **Number A**: re-scores `best.pt` on the
  fair test (37 lesion + the *same* 37 resolution-normalized negatives exp8
  uses). Isolates the measurement effect from negative-training.
- `eval_match_rules.py <run>` — re-scores a finished run's predictions under
  `iou>=0.5` / `iog>=0.5` (headline) / `iog>=0.3` to show what the IoU gate hid.

**Runs executed: exp1, exp2, exp3, exp5, exp7, exp8a, exp8b** (+ Number A on
exp1/exp2 and the match-rule sweep on exp8a/exp8b). exp4/exp6 deliberately not
run (exp5/OSMF was the decisive expert; exp6/Lichen is data-dead, 48 train).
**Verdict & numbers: `RESULTS.md` §7. Active next: aug + k-fold, `RESULTS.md` §6.**

"all originals" = every `data/pool` + `data/test` + `data/Normal` image. For an
expert, only its own disease's original boxes are ground truth; **every other
image (other disease or healthy) is a negative** — that is what produces a real
precision / false-alarm number and catches a model that just fires everywhere.
No NH_Ulcers expert: no Roboflow source exists for it.

## Run

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env
python Experimenting/exp1_5class_original.py
python Experimenting/exp2_binary_original.py
python Experimenting/exp3_leukoplakia_expert.py
python Experimenting/exp4_erythroplakia_expert.py
python Experimenting/exp5_osmf_expert.py
python Experimenting/exp6_lichen_planus_expert.py

# exp8 — the correction (binary first; it is the better front-end)
python Experimenting/exp8b_binary_negatives.py
python Experimenting/exp8a_5class_negatives.py
python Experimenting/eval_fair_negatives.py binary_original   # Number A
python Experimenting/eval_fair_negatives.py 5class_original   # Number A
python Experimenting/eval_match_rules.py   binary_negatives
python Experimenting/eval_match_rules.py   5class_negatives
```

Run from the project root (so the bundled `yolov8n.pt` resolves).
`yolov8n-obb.pt` is auto-downloaded by Ultralytics on first use (needs network
once; if offline, drop the file in the project root beforehand).

Plain defaults, env-overridable, **not tuning**: `EXP_EPOCHS` (100),
`EXP_IMGSZ` (640), `EXP_BATCH` (8, safe for a 6 GB GPU), `EXP_DEVICE` (`0`;
`cpu` to force CPU). AMP is off (Ultralytics' AMP self-check 404s in this env).

## Outputs — `Experimenting/results/<name>/`

```
run.json            exact config snapshot for the run
train/              Ultralytics training dir (weights/best.pt, curves)
val/                stock Ultralytics val() — detect experiments only
val_stock.json      mAP50 / mAP50-95 / P / R cross-reference (detect only)
metrics.json        full honest evaluation (machine-readable)
metrics.txt         the same, human-readable
```

## What is measured (per class + micro overall)

At Ultralytics' default `conf=0.25` **and** at `conf=0.001` (an *untuned*
recall ceiling, not optimisation):

- **precision, recall, F1** at IoU≥0.5 box matching
- **mean IoU / IoP / IoG over the hits** — localisation quality on what it did
  catch
- **mean best IoU per GT at any confidence** — shows the loose-box penalty
  directly; if original GT boxes are loose this stays low even when the model
  is "right"
- **image level**: detection rate on positives, false-alarm rate on negatives,
  overall screening accuracy

## Caveats baked in (read before trusting a number)

- **OBB scored against axis-aligned GT.** Original labels are axis boxes; OBB
  predictions are reduced to their enclosing rectangle before overlap math.
  IoU is therefore a slight under-estimate for tilted lesions — expected, and
  exactly why IoP/IoG are reported alongside.
- **`poly→minAreaRect` runs in normalised space** (image dims aren't read at
  build time). The 4 corners still contain every polygon vertex; rectangle
  aspect is mildly distorted vs pixel space. Baseline-acceptable.
- **Experts train on positives only** — a Roboflow image with no box of the
  expert's disease is skipped, not added as a negative. Simple and on purpose.
- **Augmented copies are dropped.** Roboflow bakes flips/rotations/crops in as
  separate files sharing the stem before `.rf.<hash>`; these are not new data.
  One deterministic representative per source photo is kept (`split` is
  by-base, so no photo's variants leak across train/val) and YOLO's own
  train-time augmentation does augmentation. Real sizes after dedup:
  Leukoplakia 509, Erythroplakia 223, OSMF 305 (0 augmented), Lichen 48.
  Each run writes `_datasets/<name>/dataset_stats.json` with the counts.
- **Lichen Planus is hopeless by data** — 48 unique photos (41 train / 7 val)
  against 85 original test positives. Run as-is; the low number is the result.
- Eval sets are small (37; per-disease originals are tens of images). Treat
  small per-class numbers as noisy.
