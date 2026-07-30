# Silicon Wafer Defect Classification

**A Comparative Study of CNN, VGG16, VGG16+CBAM, and YOLOv8s-cls on the WM-811K dataset**

Comparing four deep learning approaches for classifying silicon wafer defect patterns, with a focus on how each one handles extreme class imbalance. Overall accuracy turns out to be a weak discriminator — the real story is in the per-class F1 on rare defect types.

*Rihani, Y. · Haggag, S. · Gomaa, A. · Tamer, A. — Deggendorf Institute of Technology (DIT)*

---

## Overview

In semiconductor fabs, every die on a wafer is electrically probed and its pass/fail result stored as a 2D **wafer bin map**. Defect patterns on these maps (rings, clusters, scratches) point back to specific process problems — but manual inspection doesn't scale to fabs producing tens of thousands of wafers a week.

This project builds and benchmarks four classifiers on the **WM-811K** dataset to answer one question: *which imbalance-handling strategy actually works for the rare, safety-critical defect classes?*

**Key finding:** a lightweight custom CNN (~400K params) beats a pretrained VGG16 (~134M params) on the hardest defect class by **23.2 pp** — because the oversampling strategy matters more than model size.

---

## Dataset

The **WM-811K** dataset (Wu et al., 2015) contains 811,457 raw wafer maps from 46,393 production lots, of which **172,950** are expert-labelled. Each map uses a 3-value pixel vocabulary: `0 = background`, `1 = pass`, `2 = fail`.

The task is a **9-class** problem: 8 spatial defect patterns plus a defect-free class. The core challenge is extreme imbalance — the `none` class alone makes up 85.2% of all samples.

| Class      | Samples  |
|------------|----------|
| none       | 147,431  |
| Edge-Ring  | 9,680    |
| Edge-Loc   | 5,189    |
| Center     | 4,294    |
| Loc        | 3,593    |
| Scratch    | 1,193    |
| Random     | 866      |
| Donut      | 555      |
| Near-full  | 149      |

The rarest defect class (Near-full) is outnumbered by the dominant class by roughly **three orders of magnitude**.

---

## Models Compared

| Model | Type | Backbone | Params | Imbalance strategy |
|-------|------|----------|--------|--------------------|
| **A — Custom CNN** | Own model, trained from scratch | MathWorks reference arch. | ~400K | 6× geometric oversampling |
| **B — VGG16** | Benchmark | ImageNet pretrained | ~134M | WeightedRandomSampler |
| **C — VGG16 + CBAM** | Benchmark | ImageNet pretrained + attention | ~134M | WeightedRandomSampler + attention |
| **D — YOLOv8s-cls** | Benchmark | CSPDarknet (ImageNet) | ~12M | Built-in augmentation only |

> **Custom CNN** was designed and trained by our team from scratch. **VGG16 / +CBAM / YOLOv8s-cls** are built on pretrained backbones and serve as benchmarks.

---

## Preprocessing

One dataset, three preprocessing branches — each model family expects a different input format.

| Branch | Resize | Channels | Normalisation |
|--------|--------|----------|---------------|
| **CNN** | 48×48 bilinear | 1 (grayscale) | divide by 2 → {0, 0.5, 1} |
| **VGG16 / +CBAM** | 224×224 nearest-neighbour | 3 (replicated) | scale to [0, 1] |
| **YOLOv8s-cls** | 32×32 RGB PNGs on disk | 3 | handled internally by YOLO |

Photometric augmentations (blur, colour jitter) were deliberately avoided — wafer pixel values are categorical die states, and any transform producing intermediate values would corrupt the representation.

---

## Training Configuration

| Parameter | CNN (own) | VGG16 | +CBAM | YOLOv8s-cls |
|-----------|-----------|-------|-------|-------------|
| Framework | TF / Keras | PyTorch | PyTorch | Ultralytics |
| Input | 48×48×1 | 224×224×3 | 224×224×3 | 32×32×3 |
| Pretrained | No | ImageNet | ImageNet | ImageNet |
| Imbalance handling | 6× oversampling | WeightedRandomSampler | WeightedRandomSampler | Built-in aug. only |
| Batch size | 128 | 256 | 256 | 64 |
| Learning rate | 1e-3 | 1e-5 | 1e-5 | Auto |
| Max epochs | 30 | 15 | 15 | 100 (early-stopped @17) |
| Split | 90 / 5 / 5 | 90 / 5 / 5 | 90 / 5 / 5 | 90 / 5 / 5 |

All models used the Adam optimiser and sparse categorical cross-entropy. CNN and VGG16 splits were stratified so even Near-full (149 samples) appeared in every subset.

---

## Results

### Head-to-head

| Model | Accuracy | Scratch F1 | Macro-F1 | Params | Imbalance strategy |
|-------|----------|------------|----------|--------|--------------------|
| **Custom CNN** (own) | 95.57% | **82.7%** | ~93% | ~400K | 6× geometric oversampling |
| **VGG16** (benchmark) | 95.59% | 59.5% | 84.0% | ~134M | WeightedRandomSampler |
| **VGG16 + CBAM** (benchmark) | **95.81%** | 72.5% | 85.3% | ~134M | WeightedRandomSampler + attention |
| **YOLOv8s-cls** (benchmark) | ~75%\* | ~13%\* | — | ~12M | None (built-in aug. only) |

<sub>\* YOLO figures estimated from its normalised confusion matrix — no per-class precision/recall table was produced.</sub>

### What each result means

- **Custom CNN** — 95.57% accuracy, and Scratch F1 jumped from the 59% MathWorks baseline to **82.7%** after adding BatchNorm + L2 and deterministic 6× oversampling. *Caveat: oversampling happened before the split, so treat 82.7% as an optimistic upper bound.*
- **VGG16** — matched the CNN on accuracy but Scratch **precision collapsed to 43.7%**: 55 true positives against 59 false positives leaking from the `none` class. Pretrained depth does not fix an imbalance problem.
- **VGG16 + CBAM** — a 66K-parameter attention module (0.05% of total) bought **+13.0 pp** Scratch F1 by suppressing background false alarms (none→Scratch errors: 59 → 22).
- **YOLOv8s-cls** — fastest, smoothest convergence (17 epochs) and nails structurally distinct classes (none 1.00, Edge-Ring 0.93). But with no custom balancing it defaults uncertain predictions to `none` — Scratch recall just **0.13**.

### CBAM: targeted rare-class gains

| Metric | VGG16 → +CBAM | Change |
|--------|---------------|--------|
| Scratch F1 | 59.5% → 72.5% | **+13.0 pp** |
| Center F1 | 86.1% → 90.5% | +4.4 pp |
| Macro-F1 | 84.0% → 85.3% | +1.3 pp |
| Edge-Loc F1 | 77.7% → 74.2% | −3.5 pp |

Attention *reallocates* focus rather than eliminating confusion everywhere — Edge-Loc and Random regress slightly as Scratch improves.

---

## Key Takeaways

1. **Oversampling strategy dominates rare-class performance.** The 400K-param CNN beats the 134M-param VGG16 on Scratch F1 by 23.2 pp — training signal matters more than representational depth.
2. **CBAM is a cheap, targeted fix.** +66K params → +13.0 pp Scratch F1 on top of VGG16.
3. **YOLO's speed is real, but so is its weakness.** Fastest training, yet weakest on safety-critical rare classes without imbalance handling.
4. **Per-class F1 beats accuracy for imbalanced data.** Three of four models cluster at ~95.6% accuracy while their rare-class behaviour differs enormously.

---

## Limitations

- **Oversample-before-split (CNN):** augmented copies of a wafer could land in both train and test sets, so CNN figures are an optimistic upper bound.
- **Unequal training budgets:** models differ in epochs, batch size, and learning rate, so gaps can't be attributed to architecture alone.
- **YOLO ran without custom balancing:** its rare-class numbers would likely improve substantially under the same 6× treatment.
- **Grad-CAM explainability** was planned but not completed within the reporting period.

---

## Real-World Deployment

The target setting is **inline yield monitoring** — wafer bin maps are generated automatically during electrical testing, so a classifier slots into the existing data path with no extra hardware.

- **Custom CNN** (~400K params) — compact enough for CPU-only edge inference on test-floor equipment.
- **VGG16 + CBAM** — best rare-defect detection, suited to centralised GPU inspection.
- **YOLOv8s-cls** — fastest retraining, good for adapting to new defect types.

Because a missed defect can propagate across thousands of wafers, any rollout should be gated on **rare-class recall**, not headline accuracy, and run as a human-in-the-loop system with engineer review of low-confidence predictions.

---

## Team

Rihani, Y. · Haggag, S. · Gomaa, A. · Tamer, A.
Applied Natural Sciences and Industrial Engineering — Deggendorf Institute of Technology

## References

Key references: Wu et al. (WM-811K, IEEE T-SM 2015); Simonyan & Zisserman (VGG16, 2014); Woo et al. (CBAM, ECCV 2018); MathWorks wafer-map reference CNN (2023). Full list in the report.
