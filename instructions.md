# Oral Lesion Screening — Clean Rebuild Instructions

> Self-contained build spec. This is the **single source of truth** — there is
> no other doc to read. Goal: a simple, reliable, demoable baseline.
> No k-fold, no smart cropping, no augmentation zoo, no LAB/CLAHE — those are
> deliberately deferred until a working baseline exists.

---

> **Current-state note (2026-05-18).** This spec remains the authoritative
> record of the **two-stage architecture and the three abandoned approaches**
> — that part is unchanged and still binding. What it predates: IoP
> containment matching (replaced IoU for Step-3 selection), proportional crop
> padding, per-run `artifacts/runs/` storage, and the `original_only` vs
> `plus_roboflow` single-variable **data experiment** (concluded: the extra
> tight-box Roboflow data clearly helps the detector). For the current
> pipeline, commands, and results see **`README.md`**; for the experiment's
> rationale and outcome see **`HANDOFF.md`**. Do not re-derive history from
> git — these three docs are the source of truth.

> **Audit note (2026-05-19) — supersedes the above on two points.** A clean
> controlled investigation (`Experimenting/RESULTS.md`, now the top authority)
> found: (1) **the loose-box premise in §1/§4 is empirically false** —
> localisation-on-hits IoU is 0.63–0.69 every run; predictions are not tight
> boxes lost inside huge loose GT, so the IoU→IoP switch and "train classifier
> on detector crops because GT is loose" reasoning is solving a non-problem.
> (2) The 2026-05-18 "Roboflow clearly helps" was an **in-domain artifact**
> (measured on Roboflow-derived `web_holdout`); controlled on the original
> domain it does not transfer. **Still binding and in fact reinforced:** the
> two-stage *structure* and the three abandoned approaches — binary YOLO ≈ 3×
> the recall of 5-class YOLO on identical data, so "don't let YOLO classify"
> holds. Right structure, wrong reason.
>
> **exp8 correction (2026-05-19, supersedes the audit's gloomiest reading).**
> The audit's "detector is below a trivial baseline / detection is dead" was
> **retracted**: it was a 570-negative base-rate artifact on top of a real bug
> — exp1/exp2 trained on **zero negatives**. Give the binary detector that
> signal and measure it fairly (resolution-normalized 1:1 negatives) and it is
> a credible screener: **28/37 lesion images caught, 1/37 false alarm,
> screening_acc 0.865** (no-skill 0.50). Detection is NOT abandoned.
>
> **k-fold + augmentation update (2026-05-21, supersedes exp8b's single-fold
> headline AND validates §3 step 4 below).** The exp8b 0.865 was cross-
> validated to **0.842 ± 0.041 paired 5-fold CV** (`Experimenting/RESULTS.md`
> §8). On the way, exp10 swept augmentation and exp11 paired-tested the
> winner: **`geom_no_color` (= YOLO defaults but `hsv_h=hsv_s=hsv_v=0`) beats
> the default recipe by +0.101 paired screening, 3.6× tighter std, winning
> 4/5 folds** — the §3 step 4 "no HSV / colour is diagnostic" rule below is
> now empirically validated, not a design assertion. Note that exp1–8
> silently violated this rule (YOLO defaults include HSV jitter); exp11 is
> the first run where it was actually enforced. On fold 0 of exp11, default
> caught 5/72 lesions; geom_no_color caught 47/72 — HSV jitter appears to
> destabilise training on small medical data, not just corrupt a feature.
> The remaining confound-free failure is **confidence calibration**
> (FA@conf-0.001 = 0.702 ± 0.068 across the paired CV, slightly better than
> exp8b's single-fold 0.946 tail but still bad). Active plan is now
> **yolov8 transfer-learning sweep on the same kfold5 splits + geom_no_color
> aug**, then calibration — see `HANDOFF.md` §NEXT. Whole-image
> classification pivot stays deferred.

## 0. The Core Idea

Two stages, each doing one job:

```
full image ──► YOLOv8-nano (binary "lesion" detector) ──► crop ──► EfficientNet-B2 (5-class disease) ──► Grad-CAM
                         │                                                    │
                  no detection                                          disease + confidence
                         ▼
                  "Looks fine — no visit needed"
```

**The non-negotiable rule:** EfficientNet is trained on *YOLO-produced crops*,
never on the raw human annotations. This is the one thing that makes eval
numbers match real-world behaviour.

---

## 1. Lessons From Prior Attempts (do not repeat — there were three)

You are rebuilding because earlier approaches failed. The *only* reason this
history is here is so you don't re-derive these mistakes:

1. **YOLO→YOLO cascade** (binary YOLO → 5-class YOLO): ~20–30% accuracy.
   YOLO optimises mAP, not classification accuracy; the cascade also tanked
   recall. → Use a real *classifier* for the disease stage.
2. **EfficientNet on "smart crops"**: looked decent in eval (~75% test) but got
   **~0% in production**. It was trained on tight human-annotation crops and
   served full images — total train/serve distribution mismatch. → Train the
   classifier on the *detector's own crops*, not annotations.
3. **LAB/CLAHE preprocessing**: hurt accuracy and was applied inconsistently
   between train and serve. → Raw RGB + ImageNet normalisation only.

Also retired: k-fold, Mixup/CutMix, heavy augmentation, Gradio.
Kept deliberately minimal: YOLOv8-nano, EfficientNet-B2, Grad-CAM, Streamlit.

---

## 2. Starting Assets (already in this directory — that's all you get)

```
project/
├── instructions.md            # this file — the only doc
├── pool/                       # 325 lesion images + 325 YOLO .txt labels
│   ├── images/
│   └── labels/
├── test/                       # 37 lesion images + 37 labels — LOCKED held-out
│   ├── images/
│   └── labels/
├── normal/                     # 120 full healthy-mouth images (.jpeg, NO labels)
└── data_engine.py              # REFERENCE ONLY — see note below
```

Facts about what you're given:

- **`pool/` and `test/` are already split.** `pool/` = your train+val data.
  `test/` = the locked held-out test set. **Do not write a split script and do
  not touch `test/` until final evaluation.** Split `pool/` into train/val
  yourself (image-level, e.g. 85/15, stratified by disease class).
- **Labels are YOLO format** `class_id cx cy w h`. `class_id` ∈ {0..4} encodes
  the disease and **is reliable**. Box *coordinates* are noisy — trust them
  only for "roughly where the lesion is", never as tight ground truth.
- **`normal/`** = 120 full, uncropped healthy mouths. These are your detector
  **background negatives** (not classifier data). They are NOT smart-cropped —
  correct as-is.
- **Filenames are messy** (`Erythoplakia_2_25.jpg`, `OSFM_*`, `non_healing_*`).
  You do **not** need to rename anything — the disease comes from the label
  `class_id`, not the filename. Mapping, for reference only:

  | class_id | Disease         |
  |----------|-----------------|
  | 0        | Leukoplakia     |
  | 1        | Erythroplakia   |
  | 2        | OSMF            |
  | 3        | Lichen_Planus   |
  | 4        | NH_Ulcers       |

- **`data_engine.py` is reference only.** Reuse small helpers if useful
  (`YOLO_ID_TO_CLASS`, the `yolo_to_pixel` math, the filename normaliser).
  **Ignore its LAB/CLAHE, k-fold, and crop-extraction logic** — those embody
  the abandoned approach. Do not import it as the pipeline backbone.

You build everything else from scratch.

---

## 3. Build Steps (in order)

### Step 1 — Build the binary detector dataset

No renaming, no test split (already done). Produce a YOLO dataset:

1. Split `pool/` **image-level**, stratified by disease class, ~85/15 →
   train/val. Image-level = an image and all its boxes stay in one split.
2. For each `pool/` image, write a detector label: every annotation line
   copied but with **`class_id` forced to `0`** (single class `lesion`).
   Keep multiple boxes per image.
3. Add every image in `normal/` to the **train** split with an **empty `.txt`**
   (background negative — teaches YOLO not to fire on healthy tissue).
4. Convert the locked `test/` set the same way (binary labels, class 0) for
   final detector evaluation — but don't use it for tuning.
5. Persist the train/val image-stem lists to `splits.json` (so Step 3 reuses
   the exact same split and never leaks).
6. Write `data.yaml`:
   ```yaml
   path: <abs path to detector dataset root>
   train: images/train
   val:   images/val
   test:  images/test
   nc: 1
   names: [lesion]
   ```

### Step 2 — Train the detector (`yolov8n.pt`)

- `imgsz=640`, `epochs≈100`, `patience≈30`. Light geometric aug only
  (flips, small scale/translate). **No HSV/colour distortion** — colour is
  diagnostic signal. **(VALIDATED 2026-05-21 by exp11 paired 5-fold CV:
  YOLO defaults with HSV jitter cost −0.101 paired screening_acc, 3.6× wider
  std, and on fold 0 caused near-total training collapse — 5/72 caught vs
  47/72 with HSV off. `Experimenting/RESULTS.md` §8c.)** Note: also references
  the §3 step 4 "no HSV" rule. Throughout exp1–8 this rule was silently
  violated by `train_eval.run` using YOLO defaults; from `exp10` / `aug.py`
  onward, `geom_no_color = {hsv_h:0, hsv_s:0, hsv_v:0}` is the adopted
  augmentation level. The rest of the YOLO defaults (mosaic, fliplr,
  scale/translate, erasing) are kept — the exp10 sweep confirmed they are
  the load-bearing piece (off → 0.595, default → 0.865 on the same split).
- **Metric that matters: recall, not mAP.** A missed lesion is far worse than
  a false box for screening. Evaluate on val and report **lesion recall at a
  low confidence threshold (≈0.15–0.25)**. Pick the threshold for high recall;
  accept extra false positives — the classifier filters them.
- Save to `runs/detector/best.pt`.

### Step 3 — Build classifier data from the trained detector (critical)

This is what fixes mistake #2. For every `pool/` image (using its `splits.json`
split):

1. Run `runs/detector/best.pt` on the full image at the chosen low conf.
2. For each predicted box, compute IoU vs the image's ground-truth boxes.
3. If `IoU ≥ 0.5` with a GT box → **correct detection**:
   - Crop the predicted box with a small fixed padding (e.g. 12 px) using a
     **single shared crop function** (you will import the *same* function in
     `pipeline.py` — identical crop logic everywhere or mistake #2 returns).
   - Label it with the disease `class_id` of the matched GT box.
   - Save to `classifier_data/<split>/<Disease>/{stem}_d{n}.jpg`.
4. False positives (match no GT) → discard for the baseline.
5. Images with no detection → contribute nothing (acceptable for now).

Result: 5 disease folders of crops drawn from the detector's real output
distribution. **No `Normal` class in the classifier** — "healthy" is decided
upstream by "detector found nothing".

### Step 4 — Train the classifier (EfficientNet-B2, 5-class)

- `timm.create_model("efficientnet_b2", pretrained=True, num_classes=5)`.
- 260×260, raw RGB, ImageNet norm (`mean=[0.485,0.456,0.406]`,
  `std=[0.229,0.224,0.225]`). **No LAB/CLAHE.**
- Minimal aug: horizontal flip + mild rotation only.
- 2-phase fine-tune: freeze backbone ~5 epochs (lr 1e-3), unfreeze ~25 epochs
  (lr 2e-4). AdamW. CrossEntropy with inverse-frequency class weights
  (imbalanced; Erythroplakia/OSMF are the smallest).
- Train on `classifier_data/train`, validate on `…/val`.
- **Final headline number = the locked `test/` set**, run through the full
  pipeline (Step 5), not classifier-val.

### Step 5 — Pipeline + Streamlit demo

`pipeline.py` — one function, full image in → result out:

1. Run detector at the low conf threshold.
2. No detection → `"Looks fine — no dentist visit needed"`.
3. Highest-confidence detection → crop (the **shared** crop fn from Step 3) →
   EfficientNet → disease + softmax confidence → Grad-CAM on the crop.
4. Return: recommendation (`"Visit a dentist — possible {disease}"`), the
   image with the box drawn, the Grad-CAM overlay, the confidence bar.

`app.py` (Streamlit, one page): upload image → show original+box, Grad-CAM
overlay, predicted disease + confidence, plain-language recommendation. Cache
both models at module load.

**Final evaluation:** run every `test/` image through `pipeline.py`; report
end-to-end accuracy, per-class recall, and confusion matrix. That is the only
number that counts.

---

## 4. Design Decisions (recurring questions, pre-answered)

- **Why not YOLO 5-class directly?** Boxes are bad, accuracy is a
  classification metric, and a classifier gives Grad-CAM for free. The detector
  only answers "is there a lesion and roughly where".
- **Normal images:** detector background negatives only; **never** a classifier
  class in the baseline.
- **Crop padding identical** in Step 3 and `pipeline.py` — one shared function,
  imported in both. This is the single highest-risk spot for regressing to
  mistake #2.
- **Headline metric = locked `test/` via full pipeline.** Not val, not
  classifier-only accuracy.
- **Detector tuned for recall at low conf.** mAP is irrelevant to the product.

## 5. Definition of Done (baseline)

- [ ] Step 1: detector dataset built from `pool/` + `normal/`, `splits.json` saved, `test/` untouched
- [ ] Step 2: detector trained; **val lesion recall reported at chosen conf**
- [ ] Step 3: 5 disease folders of detector-emitted crops; split honoured
- [ ] Step 4: classifier trained
- [ ] Step 5: `pipeline.py` works end-to-end on a raw full image
- [ ] Streamlit app: upload → box + Grad-CAM + disease + recommendation
- [ ] **Locked `test/` evaluated through the full pipeline; accuracy + per-class recall + confusion matrix reported**

Only after all of the above: revisit k-fold, augmentation ablations, a
Normal-as-class backstop, and public-data scale-up.
