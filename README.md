# OPMD_Detection — Oral Potentially Malignant Diseases Detection
---

## Current Headline — Full pipeline MVP (Exp Results §17, 2026-05-23)

**EfficientNet-B0 fine-tune (Round 3 strong aug) + 4-view TTA on
detector-emitted crops at conf = 0.10. Paired 5-fold CV, folds 1-4
(fold 0 excluded), no source-image leakage.**

| metric | value |
|---|---|
| conditional disease accuracy | **0.660 ± 0.041** |
| system-level accuracy | **0.597** |
| catch rate (detector image-level recall) | 0.907 |
| negative false-alarm rate | 0.059 |

**Comparison vs §12 borrowed-OCD baseline** (EfficientNet-B2 trained on
tight human-bbox crops of the same `pool/` images, ~90 % source-image
leakage on its test slice): 0.627 / 0.583. The MVP beats it by
**+3.3 pp cond_acc, +1.4 pp sys_acc** in a stricter no-leakage eval.

## Detector — Exp 11 (paired 5-fold CV, binary YOLOv8n, `geom_no_color`)

| metric | value |
|---|---|
| screening_acc * | **0.917 ± 0.031** |
| det_rate_pos * | **0.882** |
| false_alarm_neg * | **0.047** |
| loc IoG on hits * | **0.833 ± 0.017** |

**Test setup per fold:** ~290 positive images train + val, ~72 positive
images + ~72 fair (resolution-normalized) negative images in the blackbox
test slice (1:1 ratio → no-skill baseline 0.50). Across the 5 folds every
one of the 362 original positives lands in test exactly once.

\* See [Metrics — how we evaluate](#metrics--how-we-evaluate) below for
what each number means and why we report these specifically (and not the
usual box precision / box recall).

The full chronological record of every experiment we ran — Rounds 1-3 of
the classifier, the merge investigation, the TTA win, plus the detector
retractions, dead ends, threshold sweep, OOD sanity check, and YOLO →
old-classifier MVP measurement — lives in
[Experiment_Results.md](Experiment_Results.md).

---

## Problem Statement

Oral cancer is the **leading cancer in Indian men** and one of the top
three cancers overall in India. The dominant drivers are **betel-nut /
areca chewing, gutka, paan, and tobacco / cigar use** — habits that
expose the oral mucosa to chronic carcinogens for years before any
malignancy appears.

That long lead-in window is the opportunity. Cancer in the mouth usually
does not appear out of nowhere; it is preceded by **Oral Potentially
Malignant Disorders (OPMDs)** — Leukoplakia (white patches),
Erythroplakia (red patches), Oral Submucous Fibrosis (OSMF), Lichen
Planus, and non-healing ulcers. A patient who reaches a dentist while
still at the OPMD stage has dramatically better outcomes than one who
waits until symptoms force the issue.

The screening problem is access. Phone cameras are everywhere; dentists
trained in oral oncology are not. A reliable phone-photo triage — "see a
dentist" / "looks fine" — has real value.

## Objective

From one raw oral photograph:

1. **Detect** whether a potentially malignant lesion is present.
2. **Classify** it among the five OPMD classes (Leukoplakia /
   Erythroplakia / OSMF / Lichen Planus / NH_Ulcers).
3. **Recommend** a dentist visit, with a
   plain-language explanation.

---

## Proposed Architecture

Two stages, each doing one job:

```
full image
   │
   ▼
YOLOv8n binary lesion detector       "is there a lesion, roughly where?"
   │
   ├── no detection ─────────────────► "looks fine — no visit needed"
   │
   └── lesion box(es)
           │
           ▼
   shared crop function (pad + crop) — serve_pad = 0.20
           │
           ▼
   EfficientNet-B0 5-class classifier, fine-tuned end-to-end
   4-view test-time augmentation (identity, hflip, rot ±10°)
   mean post-softmax across views → across boxes → argmax
           │
           ▼
   disease + confidence + recommendation
```

### Why two stages?

Our first attempt was a **YOLO → YOLO cascade** (binary YOLO → 5-class
YOLO). Accuracy landed at ~20–30%. YOLO is optimised for mAP and
bounding-box recall, not classification — asking it to do both tanks
both. The fix is structural: keep YOLO at what it is good at ("is there
a lesion?") and hand the cropped lesion to a real classifier.

The exp1 → exp2 controlled comparison on identical data measured the gap
directly: collapsing the 5 disease classes into a single `lesion` class
**roughly tripled the detector's image-level recall** (det_rate 0.595 →
0.838 on the same test set). That is the price of asking YOLO to
classify, measured rather than asserted.

The classifier was originally specified to train on **the detector's
own crops** (an earlier attempt that trained EfficientNet on tight
human-annotation crops scored ~75 % in eval and **~0 % in production**
— total train / serve crop-distribution mismatch). The shipped MVP
relaxes this to GT crops with pad = 0.4 because Phase 2 measurements
(Experiment_Results.md §13–§17) showed train/serve crop-distribution
mismatch is no longer the bottleneck on the B0 fine-tune, and TTA at
inference time closes the residual geometric gap. A shared crop
function (`pad_and_crop`) is still used by both the data builder and
the inference path so the geometry stays consistent.

If the detector is silent on an image, the answer is "healthy" — there
is **no Normal class in the classifier**.

---

## Metrics — how we evaluate

For a phone-photo screening tool, the usual object-detection metrics
(**box precision, box recall, mAP**) are misleading.

- They reward how *tightly* the predicted box matches the annotator's
  box. On this dataset, the annotator boxes are deliberately rough
  ("roughly where the lesion is") — not pixel-precise ground truth.
- Clinically, we do not care whether the box is perfect. We care
  whether, **at the image level**, the lesion was caught and whether
  the crop sent to the classifier actually contains the lesion.

So we report **image-level screening metrics** as the headline:

| metric | what it measures | why we care |
|---|---|---|
| **screening_acc** | image-level: did we correctly fire on a positive image AND stay silent on a healthy image? | the screening verdict. No-skill baseline is 0.50 (1:1 test). |
| **det_rate_pos** | of the positive images, what fraction did the detector fire on? | image-level recall. A miss = a patient who walks away when they shouldn't. |
| **false_alarm_neg** | of the healthy images, what fraction did the detector fire on? | image-level specificity. False alarms erode trust in the screener. |
| **loc IoG on hits** | when the detector did hit, what fraction of the annotated lesion does the predicted box cover? | the predicted box is padded then sent to the classifier — we want it to actually contain the lesion. |

**Why `loc IoG` instead of `loc IoU`?** Because the annotated boxes are
loose. A tight, well-placed prediction sitting inside a much bigger GT
box will score low IoU even though it is "correct" for our purposes.
IoG (Intersection over Ground-truth) asks the question we actually care
about: *of the annotated lesion, how much did our prediction cover?*

**What `0.833` means in practice.** When the detector commits to a
lesion, its predicted box covers ~83% of the annotated lesion on
average. The pipeline then pads this crop by ~20% before passing it to
the classifier, so the lesion is effectively always within the
classifier's view.

Box-P, box-R, and mAP are still computed and reported per experiment in
[Experiment_Results.md](Experiment_Results.md) for transparency — they
just are not the headline.

### Operating point

The detector emits per-box scores in [0, 1]; a confidence threshold
turns those into a decision. We adopted **conf = 0.10** as the default
operating point. Other settings on the **same model**, no retraining:

| conf | screening | det_rate | false_alarm | use case |
|---|---|---|---|---|
| 0.05 | 0.917 ± 0.019 | 0.932 | 0.097 | recall-first |
| **0.10 (adopted)** | **0.917 ± 0.031** | **0.882** | **0.047** | balanced |
| 0.15 | 0.899 ± 0.021 | 0.838 | 0.039 | specificity-leaning |
| 0.25 (YOLO default) | 0.842 ± 0.041 | 0.703 | 0.020 | conservative |

---

## Directory Structure

```
.
├── README.md                   ← you are here
├── Experiment_Results.md       ← chronological record of every experiment
├── HANDOFF.md                  ← current implementation brief (active scope)
├── CLAUDE.md                   ← project-level instructions for AI assistants
├── instructions.md             ← original design + the abandoned approaches
│
├── data/                       ← originals (READ-ONLY)
│   ├── pool/                   325 lesion images + YOLO labels
│   ├── test/                   37 locked lesion images (now dissolved into kfold)
│   ├── Normal/                 570 healthy phone photos
│   ├── additional/             Roboflow exports (Leukoplakia, OPMD-SEG, OSMF)
│   └── new_data/               generated, gitignored
│
├── src/                        ← original two-stage pipeline (runnable history)
│   ├── common/                 shared crop fn, geometry, io, run_dir, dedup
│   ├── detector/  classifier/  build · train · evaluate · gradcam
│   └── pipeline.py             full image → result
│
├── scripts/                    ← thin CLI entrypoints, 01..06_*.py
│
├── Experimenting/              ← clean-baseline + cross-validated experiments
│   ├── exp_readme.md           how to run the experiment harness
│   ├── common/                 settings · datasets · negatives · metrics · kfold
│   ├── exp1..exp11_*.py        one script per experiment
│   ├── sweep_conf_threshold.py post-hoc operating-point sweep
│   ├── predict_bulk_sanity.py  OOD sanity check driver
│   ├── predict_with_old_classifier.py  YOLO → OCD-classifier MVP
│   └── results/<run>/          metrics · weights · plots (gitignored)
│
├── artifacts/                  ← per-run storage for src/ pipeline (gitignored)
├── app.py                      ← Streamlit demo
└── config.py                   ← single source of truth for paths + knobs
```

---

## Setup

```bash
eval "$(conda shell.bash hook)" && conda activate ai_env
pip install -r requirements.txt
```

GPU: RTX 3050 6 GB — batch sizes are tuned for it. Ultralytics' AMP
self-check is disabled (`DET_AMP = False`) because its helper-model
download 404s in this environment.

For how to run individual experiments see
[Experimenting/exp_readme.md](Experimenting/exp_readme.md). For the
current active brief (what is being worked on right now) see
[HANDOFF.md](HANDOFF.md).
