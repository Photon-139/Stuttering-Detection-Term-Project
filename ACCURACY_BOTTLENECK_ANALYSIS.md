# Accuracy Bottleneck Analysis — Stuttering Detection Project
**Date**: April 10, 2026  
**Team**: 18 | CS204T (Artificial Intelligence)  
**Status**: Pending TA discussion before implementation

---

## Problem Statement

All models plateau at ~70–72% accuracy regardless of architecture:

| Model | Accuracy | Notebook |
|---|---|---|
| KNN (K=5) | 59.9% | `KNN_Distance_Analysis.ipynb` |
| Decision Tree | 58.6% | `Tree_Ensemble_Deep_Dive.ipynb` |
| Random Forest | 68.3% | `Tree_Ensemble_Deep_Dive.ipynb` |
| Advanced MLP (ReLU/Adam/BN/Dropout) | ~70.5% | `Neural_Network_Deep_Dive.ipynb` |
| Tuned SVM (RBF, GridSearchCV) | ~72.5% | `SVM_Kernel_Deep_Dive.ipynb` |
| Tuned LDA (shrinkage=0.5) | ~72.5% | `Probabilistic_Models_Deep_Dive.ipynb` |

Six fundamentally different algorithms converging at the same ceiling confirms the bottleneck is **upstream of the models** — in the data itself.

---

## Root Cause: Ambiguous Labels

### How labels are currently generated

In `src/data/data_manager.py` (line 184):
```python
master_df['target'] = (master_df['NoStutteredWords'] < 2).astype(int)
```

The `NoStutteredWords` column is a **crowd-sourced annotator agreement score** (0–3 scale), NOT a clean binary label. Each clip was rated by 3 annotators, and this column counts how many said "no stuttered words."

### The breakdown (after quality filtering — 28,383 clips)

| `NoStutteredWords` | Meaning | Count | Current Label |
|---|---|---|---|
| **0** | All 3 annotators agree: HAS stutter | 7,448 | Stutter |
| **1** | 1 of 3 said "no stuttered words" — **disagreement** | 5,139 | Stutter |
| **2** | 2 of 3 said "no stuttered words" — **disagreement** | 7,135 | Fluent |
| **3** | All 3 annotators agree: NO stutter | 8,661 | Fluent |

**40.8% of "stutter" labels and 45.2% of "fluent" labels come from clips where annotators disagreed.** Models cannot learn a clean boundary when ~43% of the ground truth is ambiguous.

### Empirical confirmation

- Cosine similarity between fluent and stutter class centroids: **0.985** (nearly identical)
- The ambiguous samples are dragging both centroids together, making the classes overlap

---

## Proposed Fix

### What changes
**Only 1 file**: `src/data/data_manager.py`  
**Only 2 additions** (backward-compatible — nothing breaks for existing code):

#### 1. Add `strict` parameter to `generate_label_dict()`
When `strict=True`, keep only high-agreement samples:
- `NoStutteredWords == 0` → Stutter (label 1)
- `NoStutteredWords == 3` → Fluent (label 0)
- `NoStutteredWords == 1 or 2` → **Discarded**

Default is `strict=False` so all existing notebook code works unchanged.

#### 2. Add `label_dict` parameter to `load_from_folders()`
When provided, each `.npy` filename is checked against the dict. Files not in the dict are skipped. Labels come from the dict, not the folder name.

Default is `label_dict=None` so all existing notebook code works unchanged.

### What does NOT change
- **No re-extraction needed** — the `.npy` files on disk are fine
- **No files need to be moved** — files we keep are already in the correct folders
- **No model code changes** — `src/models/` is untouched
- **No notebook changes required** — it's opt-in via `strict=True`

### How to use it (after implementation)
```python
# In any notebook:
label_dict = DataManager.generate_label_dict(CSV_PATHS, filter_quality=True, strict=True)
X, y = manager.load_from_folders(fluent_dir, disfluent_dir, limit=SAMPLE_LIMIT, label_dict=label_dict)
```

### Data impact
- Before: ~25,800 samples (43% ambiguous)
- After: ~16,100 samples (0% ambiguous, all high-agreement)
- Still well above our operational `SAMPLE_LIMIT` of 10,000

---

## Other Observations (Lower Priority)

### WavLM Mean-Pooling
The extractor (`src/extractors/wavlm_extractor.py`, line 40) uses `mean(dim=1)` which averages across all time steps, potentially smoothing out temporal stutter signals. A future improvement could concatenate mean + std pooling for a 1536-dim vector that preserves variance information.

### Notebook Inconsistencies
- SVM, LDA, Neural Net notebooks use `SAMPLE_LIMIT = 10000` with balanced loading
- Tree and KNN notebooks load all data without a limit and use `balance_data(strategy="oversample")` separately
- Minor issue, does not affect conclusions, but should be standardized eventually

---

## Action Items

- [ ] **Discuss with TA** — Confirm the label filtering approach is acceptable for the project
- [ ] **Implement fix** — Add `strict` flag to `data_manager.py` (2 small additions)
- [ ] **Re-run best models** — SVM and LDA with strict labels to measure improvement
- [ ] **Continue hypertuning** — Current hypertuning work is still valid and should continue
- [ ] **Consider feature fusion** — If strict labels alone don't hit 85%, add MFCCs + spectral features
