# Original Design — Two-Stage Architecture and the Three Abandoned Approaches

> Purpose of this file. This is the **original architecture intent**
> for the project, plus the three earlier approaches that failed and
> the lessons they leave behind. It is short on purpose.
>
> For current numbers see `README.md`. For everything we have
> measured (and the audit / retraction notes that used to live in this
> file) see `Experiment_Results.md`. For the active brief see
> `HANDOFF.md`.

---

## The core architecture

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
   shared crop function (pad + crop)
           │
           ▼
   5-class disease classifier (backbone chosen per experiment)
           │
           ▼
   disease + confidence + Grad-CAM + recommendation
```

**The non-negotiable rule:** EfficientNet is trained on the
*detector's own crops*, never on raw human annotations. This is the
one thing that makes eval numbers match real-world behaviour. See
abandoned approach #2 below for why this rule exists.

If the detector is silent, the answer is "healthy" — there is **no
`Normal` class in the classifier**.

---

## The three abandoned approaches (do not repeat)

You are rebuilding because three earlier approaches failed. The *only*
reason this history is documented is so you do not re-derive these
mistakes.

### 1. YOLO → YOLO cascade

A binary YOLO followed by a 5-class YOLO for the disease stage.
Accuracy landed at ~20–30 %. YOLO optimises mAP, not classification
accuracy; the cascade also tanked recall.

**Lesson:** use a real *classifier* for the disease stage, not a
second YOLO.

(Later measured directly: collapsing 5 disease classes into 1 binary
class roughly tripled YOLO's image-level recall on identical data —
`Experiment_Results.md` #3.)

### 2. EfficientNet-B2 on tight human-annotation crops

Trained on tight human-bbox crops of the annotated lesions. Looked
good in eval (~75 % accuracy) but got **~0 % in production**. It was
served full images while having only ever seen tight crops — total
train / serve distribution mismatch.

**Lesson:** train the classifier on the *detector's own crops*, not
on raw annotations. One shared crop function imported by both the
training-data builder and the live pipeline. This is the single
highest-risk regression spot in the whole system; treat the shared
crop function as load-bearing.

### 3. LAB / CLAHE preprocessing

Hurt accuracy and was applied inconsistently between train and serve.

**Lesson:** raw RGB + ImageNet normalisation only.

### Also retired in this rebuild

- k-fold for the production pipeline (k-fold was later reintroduced
  *only* for clean-baseline experiments, never for the served model)
- Mixup / CutMix (re-tested in the exp10 aug sweep — still bad on
  this data scale)
- Heavy augmentation (same)
- Gradio (Streamlit is the demo surface)

---

## Class mapping

YOLO label `class_id` ∈ {0..4} encodes the disease in the original
annotations and is reliable. Box coordinates are noisy — trust them
only for "roughly where the lesion is".

| class_id | Disease |
|---|---|
| 0 | Leukoplakia |
| 1 | Erythroplakia |
| 2 | OSMF |
| 3 | Lichen_Planus |
| 4 | NH_Ulcers |

For the binary detector, every annotation is collapsed to
`class_id = 0` (`lesion`). Disease identity flows to the classifier via
the matched GT box, never via the detector head.
