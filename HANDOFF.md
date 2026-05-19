# HANDOFF — IMPLEMENTATION BRIEF

> You are a fresh Claude Code instance. **Read `Experimenting/RESULTS.md`
> first** — it is the current authority. Then this file's active brief
> (§NEXT). The rest of this file and `instructions.md` are historical design
> rationale; parts are now empirically contradicted (see the audit note).
> Do NOT re-derive history from git.

---

## ✅ exp8 CONCLUDED (2026-05-19) — detection revived. 🔴 NEXT = augmentation + k-fold

**Authority for everything below: `Experimenting/RESULTS.md` §7 (full numbers)
and §6 (next plan). Read that first.**

exp8a/exp8b ran: exp1/exp2 byte-identical on the positive side, only variable =
resolution-normalized negatives (Normal resized to the positives' median long
side = **276 px**) folded into train + a 1:1 37-img slice into test (no-skill
screening = 0.50). `Number A` = old exp1/exp2 weights re-scored on the same
fair test, no retrain (`eval_fair_negatives.py`).

**Result — the "detector is below trivial / detection is dead" verdict is
RETRACTED.** It was a base-rate artifact (570 negatives → trivial = 0.939) on
top of a real bug: exp1/exp2 trained on **zero negatives**. The fair ruler
alone moved false-alarm only 0.621→0.568 (resolution confound real but ~5pp);
**negative training did the work** — exp8b false-alarm **21→1 /37**, det_rate
≈ held (31→28, 37-img noise), **screening_acc 0.865** (no-skill 0.50); exp8a
13→2 /37, 0.784, and exp1's two dead classes (Leuk/OSMF) revived. Binary >
5-class confirmed with a *usable* number. The one confound-free survivor:
**confidence calibration** (conf-0.001 still false_alarm 0.946).

Match-rule decision (`eval_match_rules.py` sweep, no retrain): the IoU≥0.5 gate
was relabelling well-placed offset boxes as FP — under IoG, precision AND
recall rise together. IoP floor inert (model doesn't balloon boxes) → dropped.
**Adopted `iog>=0.5` as the headline localisation metric** (≥half the lesion
covered; what the padded crop needs), `iou>=0.5` kept for exp1–8 continuity,
`iog>=0.3` looser secondary. Wired into `metrics.py` (`match_rule_sweep` in
every run's output). NB: this does **not** move det_rate (geometry-free) — the
screening recall stays ≈0.75.

### 🔴 ACTIVE NEXT (separate conversation, to save tokens) — two things, untested

1. **Augmentation** — none ever applied (exp1–8 = YOLO built-in only). With
   ~277 thumbnails this is the most likely real lift.
2. **k-fold cross-validation** — every number so far is one 85/15 split on a
   37-img test (±2–3 = noise). At ~277 imgs k-fold buys a *trustworthy*
   number, not a bigger one.

Build both as new `Experimenting/` experiments in the exp8 pattern (binary
first — it is the better front-end). Keep the fair 1:1 resolution-normalized
negatives and the `iog>=0.5` headline rule. After that the real target is
**confidence calibration** (the only confound-free failure left). The
whole-image classification pivot (EfficientNet-B2 / DINOv2 frozen, Normal-class
resolution trap) is **deferred, not dead** — a later controlled comparison arm
on the same seeded split, no longer the escape hatch. User runs all GPU
training; you write code + exact commands. This supersedes the "yolov8s sweep /
classifier-phase" next-steps in the body below.

---

## ✅ STATUS — `plus_roboflow` IMPLEMENTED & CONCLUDED (2026-05-18); AUDITED & OVERTURNED (2026-05-19)

The 2026-05-18 conclusion ("Roboflow data clearly helps the detector") was
shown by the `Experimenting/` audit to be an **in-domain measurement
artifact** — it was judged on `web_holdout` (Roboflow-derived). Controlled, on
the original domain (`Experimenting` exp2 vs exp7, single variable = +955
unique Roboflow photos): F1/mAP50 flat, recall down, only specificity up. The
loose-box premise behind this whole change is empirically false (loc-on-hits
IoU 0.63–0.69 every run). Treat the body below as **historical rationale**,
not validated fact. Details: `Experimenting/RESULTS.md`.

## ✅ STATUS — IMPLEMENTED & EXPERIMENT CONCLUDED (2026-05-18)

This brief was executed in full. The body below is preserved as the
**rationale record** (why the change exists, the landmines, the invariants).
Current state lives in `README.md`; this header is the outcome.

**Result — adding the tight-box Roboflow data clearly helps the detector:**

| Metric | control `original_only` | treatment `plus_roboflow` |
|---|---|---|
| web_holdout mAP50 (higher-N, decisive) | 0.0185 | **0.263** (~14×) |
| web_holdout recall | 0.054 | **0.298** (~5.5×) |
| locked-37 accuracy (headline) | 0.5405 | **0.5676** (+2.7 pp) |
| Leukoplakia recall (locked-37) | 0.00 | **0.50** (dead class revived) |

The decisive signal is web_holdout (less noisy than 37 imgs); locked-37
moves the same direction. Build verified: `splits.json` byte-identical
across arms; control train ⊂ treatment train (+1996 RF pos, +149 RF neg);
**dedup caught 11 exact duplicates (Hamming 0) of locked test** and excluded
them; 59 pool near-dupes reported only; web_holdout = 142 imgs; sidecar =
6107 entries (deterministic, identical both arms).

**Open problems (next work, now unblocked):**
- detector `recommend_conf` still pins at **0.001** — confidence/calibration
  is now the live bottleneck, not box tightness.
- treatment classifier-val collapsed (0.726→0.431): a crop-distribution
  artifact, *not* the headline; addressed by the deliberate imbalance-aware
  classifier phase using `disease_sidecar.json` (still future).
- the deferred **`yolov8s` model-size sweep** is now sensible to run.

---

## 0. Project in one line
Two-stage oral-lesion screening: **YOLOv8n binary lesion detector → shared
crop → EfficientNet-B2 5-class classifier → Grad-CAM**. System is fully
integrated and works end-to-end. The only weakness is **detector recall**.

## 1. Environment / working agreements (unchanged)
- Conda env **`ai_env`**: `eval "$(conda shell.bash hook)" && conda activate ai_env`. GPU: RTX 3050 **6 GB**.
- **User runs all GPU training.** You write code + give exact commands; you do NOT launch training.
- User strongly prefers **clean modular code** — many small, independently-editable files.
- `data/test/` (37 imgs) is the **locked, sole headline metric**. Never train/tune on it.
- `ORAL_SMOKE=1` slashes epochs for a fast end-to-end plumbing check — must still pass for BOTH arms.

## 2. Why this change exists
Baseline locked-test accuracy is stuck ~54%. Diagnosed root cause: the
**loose-box training problem** — original annotator GT boxes are huge/sloppy,
so the detector learns mushy, low-confidence localization (`recommend_conf`
pins at 0.001 every run). We now have ~2,300 extra Roboflow images whose
**polygon** labels yield *tight* boxes — a direct antidote. The experiment:
**does adding this data make `yolov8n@640` better, measured cleanly?**

---

## 3. THE EXPERIMENT (this is the whole point — keep it clean)

A **controlled, single-variable** comparison. Two detector training runs,
**identical code**, differing **only in training data**:

| Arm | Detector data |
|---|---|
| **control** = `original_only` | original `pool/` (single-class) + ~570 negatives + locked test |
| **treatment** = `plus_roboflow` | everything in control **+** converted Roboflow train/valid |

Both: `yolov8n@640`, `bs8` (the only non-confounded recipe). The old 54.1% is
a loose sanity reference ONLY — it predates IoP/run-dir code, so it is NOT the
comparison baseline. The valid comparison is **control vs treatment**.

Classifier is **untouched** in both arms: Step 3 still builds classifier data
by cropping **original `pool/` images** with the current arm's detector (IoP
match → inherit GT disease class). **No Roboflow image ever reaches the
classifier.** The classifier differs between arms only because the detector
that crops for it differs — that is the intended pipeline-level effect, not a
confound. This preserves: no class-imbalance contamination, no train/serve
mismatch, locked test stays honest.

---

## 4. DATA — preserve originals, everything generated under `data/new_data/`

### Read-only, never mutate
`data/pool/`, `data/test/`, `data/Normal/`, `data/Normal/NON CANCER/`,
`data/additional/`.

### Generated layout (you create all of this)
```
data/new_data/
  det_original_only/      # YOLO tree: images/{train,val,test} labels/{...} data.yaml
  det_plus_roboflow/      # same, with Roboflow folded into TRAIN only
  web_holdout/            # YOLO tree (eval only) — Roboflow TEST splits, single-class
  disease_sidecar.json    # provenance for a FUTURE classifier phase (unused now)
  dedup_report.json       # phash collisions found/excluded
```
`config.DETECTOR_DATASET` currently points at `artifacts/detector_dataset`.
Add `NEW_DATA_ROOT = DATA_ROOT / "new_data"` and make the dataset builder
write per-arm under it. `data.yaml` + `splits.json` are **per-arm**.

### Negatives (~570) — `list_images()` is NON-recursive (landmine)
`src/common/io.py::list_images` uses `glob(f"*{ext}")`, so it sees
`data/Normal/` (120 imgs) but **NOT** `data/Normal/NON CANCER/` (450 imgs).
You must source negatives from **both** explicitly (don't blindly switch
list_images to rglob — other call sites depend on its current behavior).
Add all ~570 to the **train** split of **both** arms with empty `.txt`.

### Roboflow conversion — use the POLYGON exports, not `-obb`
Sources and class→`0` collapse (detector is binary):

| Dataset | Label format | Action |
|---|---|---|
| `data/additional/Leukoplakia.v2i.yolov8` | real YOLO boxes | pass through, class→0 |
| `data/additional/OPMD-SEG.v1i.yolov8` | **polygon seg** | poly→bbox, class→0 |
| `data/additional/OSMF DETECTION.v1i.yolov8` | **polygon seg** | poly→bbox, class→0 |

Ignore `OPMD-SEG.v1i.yolov8-obb` entirely (polygon min/max is tighter).

**poly→bbox:** line = `cls x1 y1 x2 y2 … xn yn` (normalized). `xs=coords[0::2]`,
`ys=coords[1::2]`; `cx=(min+max)/2`, `cy=(min+max)/2`, `w=max-min`, `h=max-min`;
clamp [0,1]; drop degenerate (`w<=1e-4 or h<=1e-4`). **Do NOT reuse
`read_yolo_label()` — it hard-drops any line where `len(parts)!=5`, i.e. it
silently eats every polygon.** Write a dedicated polygon parser.

Roboflow **train+valid → treatment TRAIN**. Roboflow **test splits → `web_holdout/`**
(NOT used for training either arm). Leukoplakia.v2 has no test split.

**Filename namespacing (landmine):** Roboflow stems can collide with each other
and with `pool/`. Prefix every generated file: `opmdseg__<stem>`,
`osmf__<stem>`, `leukov2__<stem>`. No silent `shutil.copy2` overwrites.

### Split stability (critical for a clean comparison)
The `pool/` train/val split MUST be **byte-identical between both arms** (same
`SPLIT_SEED`, same stratification) and locked test identical. The ONLY
difference between arms is the extra Roboflow images appended to treatment's
train. Persist the pool split once; reuse for both arms and for Step 3.

### Disease sidecar (record now, use later — NOT this run)
`disease_sidecar.json`: list of
`{generated_image, box_index, source_dataset, original_class, mapped_class}`
using `Leukoplakia→Leukoplakia, Oral Lichen Planus→Lichen_Planus,
erythroplakia→Erythroplakia, osmf→OSMF` (NH_Ulcers: none anywhere). This is
provenance for a future, deliberate classifier phase with imbalance handling
(class weights / focal / resampling / CV). **Do not wire it into the
classifier now.**

### Dedup before trusting numbers (`imagehash` NOT installed)
phash every Roboflow image vs the locked 37 `data/test/` images; exclude any
with Hamming distance ≤ threshold; also report (don't necessarily exclude)
Roboflow-vs-`pool/` near-dupes. Write `dedup_report.json`. `imagehash` is not
in `ai_env` — either add it (`pip install imagehash` in ai_env) or implement a
small 8×8 DCT/average pHash by hand. **See clarifying Q2 for threshold/policy.**

---

## 5. CODE CHANGES (clean, modular — match existing style)

Existing relevant files: `config.py`, `src/common/run_dir.py`,
`src/common/io.py`, `src/detector/build_dataset.py`,
`scripts/01_build_detector_dataset.py` … `05_evaluate_pipeline.py`.

1. **`config.py`**: add `NEW_DATA_ROOT`; make detector-dataset / splits / yaml
   paths arm-aware (e.g. a helper that takes `arm` and returns its root).
   Don't break existing static roots used elsewhere.
2. **New `src/detector/convert_roboflow.py`**: polygon/box parser + poly→bbox
   + namespacing + sidecar emission. Pure, unit-testable, no training deps.
3. **New `src/common/dedup.py`**: pHash + collision report.
4. **`src/detector/build_dataset.py`**: extend `build()` to take
   `arm ∈ {original_only, plus_roboflow}`; source negatives from both Normal
   dirs; in treatment arm fold converted Roboflow train+valid into train;
   build `web_holdout/` once; keep pool split stable across arms.
5. **`src/common/run_dir.py`**: `new_run(tag, name: str | None = None)` —
   if `name` given use it verbatim (sanitize: strip whitespace, replace `/`;
   KEEP `@`); else `f"{tag}_{ts}"` (tag is already model-first, e.g.
   `yolov8n_640`). `CURRENT_RUN`/`latest` plumbing unchanged.
6. **`scripts/01_…` + `scripts/02_…`**: add `argparse` `--arm
   {original_only,plus_roboflow}` (required) and `--out_dir` (optional, →
   `run_dir.new_run` name). Steps 03/04/05 keep reading `CURRENT_RUN`.
7. **New `src/common/console.py`**: `phase(title)` (loud banner) + `kv(dict)`
   pretty-printer. Every script prints a START banner naming **model + arm +
   out_dir** and an END banner. Detector training: ensure ultralytics
   `verbose=True` and final mAP/precision/recall + the conf sweep print.
   Classifier: per-epoch loss/val-acc line. Step 05: print a consolidated
   report (locked-37 acc + per-class recall + `recommend_conf` +
   `web_holdout` detector mAP/recall) AND persist it.
8. **Step 05 / eval**: additionally evaluate the **detector only** on
   `data/new_data/web_holdout/` → add `web_holdout_detector: {map50, recall}`
   to `pipeline_report.json` (higher-N detector signal vs the noisy 37).
9. **Optional `scripts/06_compare_runs.py <runA> <runB>`**: side-by-side table
   (locked-37 acc, per-class recall, web-holdout mAP, `recommend_conf`).

Keep `ORAL_SMOKE=1` working for both arms (conversion/dedup run fast always).

---

## 6. COMMANDS THE USER WILL RUN (document these exactly in your final summary)

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env

# CONTROL
python scripts/01_build_detector_dataset.py --arm original_only
python scripts/02_train_detector.py --arm original_only --out_dir yolov8n@640_original_only
python scripts/03_build_classifier_data.py
python scripts/04_train_classifier.py
python scripts/05_evaluate_pipeline.py

# TREATMENT
python scripts/01_build_detector_dataset.py --arm plus_roboflow
python scripts/02_train_detector.py --arm plus_roboflow --out_dir yolov8n@640_plus_roboflow
python scripts/03_build_classifier_data.py
python scripts/04_train_classifier.py
python scripts/05_evaluate_pipeline.py

# Compare
python scripts/06_compare_runs.py artifacts/runs/yolov8n@640_original_only artifacts/runs/yolov8n@640_plus_roboflow
```
Smoke first: prefix the two short chains with `ORAL_SMOKE=1`.

---

## 7. INVARIANTS — do not violate
- Originals in §4 are read-only. All generated data under `data/new_data/`.
- `data/test/` (37) is the locked headline; `web_holdout/` is a **secondary,
  eval-only** detector metric — never trained/tuned on.
- Pool train/val split byte-identical across both arms; only Roboflow train
  additions differ.
- No Roboflow image ever enters the classifier in this change.
- Detector labels are single-class `0`. Disease classes only in the sidecar.
- You do not launch GPU training; you deliver code + the command list.

---

## 8. CLARIFYING QUESTIONS — RESOLVED (user decisions, as built)
1. **`web_holdout` scope:** → web_holdout = Roboflow test splits **+ a
   seeded 30% slice of Roboflow valid** (`WEB_HOLDOUT_VALID_FRAC=0.30`,
   `SPLIT_SEED`); remaining 70% valid → treatment train.
2. **Dedup policy:** → `imagehash` **installed into `ai_env`**; Hamming
   **≤5** excludes Roboflow-vs-locked-test (both arms); Roboflow-vs-`pool`
   near-dupes **reported only**. (`PHASH_HAMMING_THRESH=5`.)
3. **`06_compare_runs.py`:** → **built** (`scripts/06_compare_runs.py`).
4. **non-osmf** (new Q): OSMF DETECTION class 0 `non-osmf` is **not** a
   lesion — those become empty-label **negatives** in treatment train
   (class 1 `osmf` → lesion). User accepted this adds a negatives-count
   difference between arms. Classifier/pipeline otherwise **untouched** —
   strictly the detector-data experiment as scoped.

## 9. Status snapshot (post-experiment)
- System works end-to-end (detector, classifier, pipeline, eval, Streamlit).
- Shipped pre-this-change: IoP containment matching (`MATCH_IOP_THRESH=0.70`),
  proportional crop padding (`CROP_PAD_FRAC=0.20`), config-gated no-detection
  fallback (default `healthy`), run-dir storage (`CURRENT_RUN`, `latest`;
  MLflow off).
- Shipped by this change: arm-aware dataset (`--arm`), `data/new_data/`
  layout, pHash dedup, Roboflow poly→bbox conversion + sidecar, web_holdout
  detector metric, `--out_dir` run naming, console banners, `06_compare`.
- **Experiment answered "does data help" → YES** (see status header).
- Now unblocked / next: `yolov8s` model-size sweep; detector confidence
  calibration (conf-floor pinning); deliberate imbalance-aware classifier
  phase using `disease_sidecar.json`.
- Eval is small (37 test / 49 val): ±2% accuracy is noise — that is exactly
  why `web_holdout` (142 imgs) exists as a higher-N detector-only check.
