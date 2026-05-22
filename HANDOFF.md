# HANDOFF — Active Implementation Brief

> Read order before starting:
> **`README.md`** (current headline, architecture, metrics) →
> **`Experiment_Results.md`** (everything we have measured, most recent
> first) → **`CLAUDE.md`** (durable invariants and landmines) →
> **`instructions.md`** (original architecture intent + the three
> abandoned approaches) → this file.

---

## Status — MVP complete

Both stages of the two-stage pipeline are concluded and locked. Headline
numbers in `README.md`; full chronological detail in
`Experiment_Results.md` (most recent: §17, TTA adopted).

### Locked production config

| component | choice |
|---|---|
| detector | `Experimenting/results/kfold5_geom_no_color_binary/fold_2/train/weights/best.pt` |
| detector conf | 0.10 |
| classifier | `Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/fold_2/best.pt` |
| classifier serve_pad | 0.20 |
| TTA | 4 views (identity, hflip, rot +10°, rot −10°), post-softmax mean |
| box merging | **off** (no measurable lift on top of TTA — see §16) |
| aggregation | mean softmax across TTA views → mean across detector boxes → argmax |

### Headline (folds 1-4 Phase 2, TTA on, leak-free)

- conditional disease accuracy: **0.660 ± 0.041**
- system-level accuracy: **0.597**
- catch rate (detector image-level recall): 0.907
- negative false-alarm rate: 0.059

Clears the §12 borrowed-OCD baseline (0.627 / 0.583, which had ~90 %
source-image leakage) by **+3.3 pp cond_acc / +1.4 pp sys_acc** in a
stricter no-leakage evaluation.

---

## Active brief — productionising the MVP in a NEW repo

Demo backend lives in a separate repo (`dentiligence-api`), **NOT** in
this research repo. Goal: a thin FastAPI service the client's GCP
engineer can deploy without reading this codebase. WordPress site
calls the GCP Cloud Run endpoint.

### Proposed layout

```
dentiligence-api/
├── main.py            FastAPI app, route + response shaping only
├── inference.py       detect → crop → TTA classify → result (importable)
├── models.py          B0 classifier class
├── weights/
│   ├── detector.pt    yolov8n geom_no_color fold 2 (~6 MB)
│   └── classifier.pt  B0 aug fold 2 (~17 MB)
├── Dockerfile
├── requirements.txt
├── config.py          single source of truth (paths, conf, serve_pad)
├── .env.example
├── examples/
│   ├── client.html    minimal upload form, calls /predict
│   └── curl.sh
├── tests/
│   └── test_inference.py     smoke test on Experimenting/internet_images/
└── README.md          deployment steps for the GCP engineer
```

### Endpoints

- `POST /predict` — multipart image upload →
  `{detected: bool, disease: string|null, confidence: float, boxes: [...], processing_ms: int}`
- `GET /health` — `{status: "ok", model_loaded: bool}`
- CORS enabled for the WordPress site's origin

### Weights hosting

Push directly to the new repo. Both files together ~23 MB, well under
GitHub's 100 MB per-file limit. No LFS (quota cliffs), no Drive (auth
complications, no version-lock). Dockerfile copies them in; the image
is fully self-contained for Cloud Run.

If a 5-fold ensemble is ever shipped (~125 MB total), use a GitHub
release attachment — free, 2 GB per file, code stays clean.

---

## What stays in THIS repo (research artefacts)

- All experiment scripts under `Experimenting/`
- All results, weights, and per-fold metrics under
  `Experimenting/results/` and
  `Experimenting/classifier_experiments/results/`
- Documentation (this file, `Experiment_Results.md`, `CLAUDE.md`,
  `README.md`, `instructions.md`)
- The `inspect_pipeline.py` visualiser + `internet_images/` test
  set + `full_pipeline_test/` outputs (useful for the demo repo's
  smoke tests)

## What this repo no longer drives

`src/` and `scripts/` are **archived**. They implement the original
two-stage architecture (EfficientNet-B2 trained on tight human-bbox
crops — Phase 0 mistake #2 from `instructions.md`). The numbers under
`Experiment_Results.md` §1 were retracted by §7. Kept as runnable
history; not the path to the MVP. Do not port from there.

## Out of scope (deferred)

- **5-fold ensemble in production.** Each test image was in 4/5 folds'
  training sets — the only honest ensemble eval would need a fresh
  holdout we don't have. Single-fold (fold 2) shipping number is
  honest. See §17 leakage discussion.
- **Re-training on YOLO crops directly** (the originally-planned
  Round-2 fallback). TTA already cleared the §12 bar; train/serve
  crop-distribution mismatch isn't the bottleneck (§16).
- **More original-domain data.** The 0.66 cond_acc ceiling on the
  classifier is consistent with the 400-box data scale; getting past
  it requires more original phone photos, not architectural moves
  (§15 verdict).
- **Binary-only "OPMD vs not" head.** Would lift trivially to ~0.85+
  but changes the product (no per-disease prediction). Reserved for
  if 5-class accuracy proves insufficient in user testing.
- **Grad-CAM.** Dropped from the MVP scope in Round 1's original
  brief. Re-add if the demo specifically requests heatmap
  visualisation.

---

## Working rule (unchanged)

User runs all GPU training; assistant writes code and hands back
copy-pasteable commands. CPU-side scripts (materialise, inspect)
the assistant may run directly.
