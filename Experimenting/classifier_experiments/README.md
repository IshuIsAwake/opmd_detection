# classifier_experiments — Round 1 (DINOv2-S) + Round 2 (EfficientNet-B0)

Two parallel investigations of the MVP classifier on the detector's
`kfold5_splits.json`. Same data, same folds, same metrics — only the
backbone changes.

| | Round 1 | Round 2 |
|---|---|---|
| backbone | DINOv2-S (ViT-S/14) | EfficientNet-B0 |
| pretraining | DINOv2 SSL on 142 M images | ImageNet supervised |
| trainable | head only (~100 k params) | end-to-end (~5 M params) |
| optimizer | AdamW, lr 1e-3 (head) | AdamW, lr 1e-3 (head) / 1e-4 (backbone) |
| schedule | cosine | cosine |
| input | 224 × 224 | 224 × 224 |
| entry scripts | `exp1_whole_image.py`, `exp2{a,b,c}_gt_pad*.py` | `exp1_b0_whole_image.py`, `exp2{a,b,c}_b0_gt_pad*.py` |
| results dir | `results/<arm>/` | `results/<arm>_b0/` |
| Phase 2 dir | `results/phase2/` | `results/phase2_b0/` |

The two write to disjoint paths — running Round 2 does **not** disturb the
saved Round 1 results.

## What's here

```
classifier_experiments/
  common/
    settings.py        knobs + paths (env-overridable, see below)
    crops.py           pad_and_crop, byte-equivalent to predict_with_old_classifier
    folds.py           splits.json → per-fold {train,val,test} BoxEntries
    materialise.py     build _datasets/ trees (CPU, one-time)
    dataset.py         torch Dataset + train/eval transforms
    model.py           DinoV2Classifier (frozen head)
    model_b0.py        EfficientNetB0Classifier (full fine-tune)
    train_eval.py      one-call-per-fold train + Phase-1 eval (backbone-agnostic)
    metrics.py         Phase 1 + Phase 2 reports
    summarize.py       per-arm fold roll-up
    run_arm.py         shared body of the per-arm entry scripts
  exp1_whole_image.py        | exp1_b0_whole_image.py
  exp2a_gt_pad00.py          | exp2a_b0_gt_pad00.py
  exp2b_gt_pad02.py          | exp2b_b0_gt_pad02.py
  exp2c_gt_pad04.py          | exp2c_b0_gt_pad04.py
  phase2_pipeline.py   --backbone {dinov2,b0}; 3×3 train_pad × serve_pad
```

## Run (Round 2 — B0)

```bash
cd /home/ishu/Projects/AI/Oral_cancer
eval "$(conda shell.bash hook)" && conda activate ai_env

# 1. Materialise — already done if Round 1 ran. Idempotent.
python Experimenting/classifier_experiments/common/materialise.py

# 2. Four B0 training arms (GPU). Fine-tuning B0 ≈ 4× slower than the frozen
#    DINOv2 head, so plan ~10–15 min per fold = ~60 min per arm.
python Experimenting/classifier_experiments/exp1_b0_whole_image.py
python Experimenting/classifier_experiments/exp2a_b0_gt_pad00.py
python Experimenting/classifier_experiments/exp2b_b0_gt_pad02.py
python Experimenting/classifier_experiments/exp2c_b0_gt_pad04.py

# 3. Phase 2 pipeline eval for B0.
python Experimenting/classifier_experiments/phase2_pipeline.py --backbone b0
```

Single-fold reruns: `--fold <k>`.

## Run (Round 1 — DINOv2, already done)

```bash
python Experimenting/classifier_experiments/common/materialise.py
python Experimenting/classifier_experiments/exp1_whole_image.py
python Experimenting/classifier_experiments/exp2a_gt_pad00.py
python Experimenting/classifier_experiments/exp2b_gt_pad02.py
python Experimenting/classifier_experiments/exp2c_gt_pad04.py
python Experimenting/classifier_experiments/phase2_pipeline.py --backbone dinov2
```

## Knobs (env-overridable; defaults are not tuned)

| var | default | meaning |
|---|---|---|
| `CLF_EPOCHS` | 50 | max epochs per fold |
| `CLF_BATCH` | 32 | batch size |
| `CLF_LR` | 1e-3 | AdamW base learning rate (head); backbone gets 0.1× this on B0 |
| `CLF_WD` | 1e-4 | AdamW weight decay |
| `CLF_PATIENCE` | 10 | early-stopping patience on val macro-acc |
| `CLF_DROPOUT` | 0.1 | head dropout |
| `CLF_HIDDEN` | 256 | head hidden dim |
| `CLF_DEVICE` | `cuda` | `cpu` to force CPU |
| `CLF_WORKERS` | 2 | DataLoader workers |

## Invariants honoured (see ../../CLAUDE.md)

- Raw RGB + ImageNet norm only. No LAB / CLAHE.
- No `Normal` class in the classifier.
- Trains on every GT box across `pool/` + `test/` (579 total; multi-box images
  contribute every box).
- Phase 2 uses each fold's own detector weights at `conf = 0.10` and
  **excludes fold 0** (globally less aggressive — do not ship).
- The pad geometry in `crops.pad_and_crop` matches
  `predict_with_old_classifier.pad_and_crop` so Phase 2 numbers are
  apples-to-apples with §12.

## Round 1 verdict (DINOv2-S)

Best cell at Phase 2: train_pad=0.4, serve_pad=0.0 → conditional disease
accuracy 0.576 ± 0.050, system accuracy 0.521. Below the §12 baseline
(0.627 / 0.583 with ~90 % leakage), suggesting a backbone-capacity
bottleneck rather than crop-geometry mismatch. Round 2 (B0, CNN inductive
bias, full fine-tune) tests that hypothesis.
