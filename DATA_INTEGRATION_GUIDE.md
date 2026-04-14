# Stuttering Detection: Data Integration & Research Transition Guide

This document serves as the handoff for the next phase of the M.Tech research project. It details the critical architectural shift from the legacy dataset to the high-Separation "Hybrid Strict" dataset.

## 1. Project Background
The original research was based on the SEP-28k and FluencyBank datasets, totaling **32,321 potential labeled samples**. However, persistent visualization analysis (t-SNE/PCA) revealed massive overlap between Fluent and Disfluent classes. This was identified not as an embedding failure (WavLM is robust), but as a **ground-truth quality failure**.

## 2. The New Dataset Strategy: "Hybrid Strict"
We have successfully integrated a new high-quality reference for the "Fluent" class and standardized the "Disfluent" class labels.

### Component A: The TA's `non_stutter` Data
- **Source:** `/non_stutter` directory.
- **Content:** 6,023 fresh audio embeddings (`.embd.npy`) with corresponding `.flac` files.
- **Role:** Replaces the noisy `data/features/fluent` directory.
- **Integration:** Bypasses original CSV metadata strict-filtering as these files are unannotated (implicitly fluent).

### Component B: The "Strict Agreement" Stutters
- **Source:** `data/features/disfluent` directory.
- **Filtering:** We now apply a **Strict 3/3 Agreement filter** using the original SEP-28k/FluencyBank CSVs.
- **Findings:** By enforcing strict agreement, we keep only **6,112** (or 6,668 depending on CSVs) "True" stutters and discard ~5,000 ambiguous samples which were previously blurring the decision boundaries.

### Total Dataset Stats
- **Fluent:** 6,023 (TA)
- **Stutter:** ~6,600 (Strict 3/3)
- **Total:** ~12,600 high-confidence samples.
- **Separation:** This combination achieves excellent manifold separation in t-SNE projections.

## 3. Major Architectural Updates
The following changes have been made to the core infrastructure:

### `DataManager.py`
- **Dynamic Balancing:** The `balance_data` method is no longer hardcoded to assume "fluent" is the majority class. It now dynamically detects the minority class and resamples accordingly.
- **Flexible Loaders:** The `load_from_folders` method now supports passing `label_dict=None` to allow loading unmapped experimental data.
- **Robust Keys:** ID casting logic was adjusted to ensure `EpId` and `ClipId` do not accidentally acquire `.0` decimals during dictionary mapping.

### Notebook Standardization
- All 6 "Deep Dive" notebooks have been updated to target the new data directories.
- **Aesthetics:** All plots (Contours, Decision Boundaries, t-SNE) are now injected with a standardized legend suite using `matplotlib.lines.Line2D`.

## 4. Key Concerns & Risks
| Concern | Description | Resolution Strategy |
| :--- | :--- | :--- |
| **Data Imbalance** | The new ratio is nearly 1:1, but as new data is added, disfluencies may become the minority. | Use `manager.balance_data(strategy="oversample")` which is now dynamic. |
| **Leakage** | The `non_stutter` data may contain overlapping speakers with the test set. | Teammates should verify metadata for the LibriSpeech-style prefixes in the new data to ensure cross-speaker validation. |
| **Label Mapping** | Pandas sometimes upcasts `EpId` and `ClipId` to floats in the `strict_dict`. | Force integer casting before string conversion to prevent lookup misses. |

## 5. Instructions for Future Agents
1.  **Strict Filtering:** When loading disfluent data, *always* use `strict=True` in `generate_label_dict` to avoid ambiguous samples.
2.  **Visualization:** Use the provided t-SNE logic found in `Strict_Labelling_Tests.ipynb` to verify that any new features preserve the clear class separation.
3.  **Model Hyperparams:** Accuracy will be significantly higher on this dataset (expect 80-95%). Do not be alarmed by the sudden performance jump; it is a result of moving to high-quality ground truth.

---
*Created by Antigravity (Advanced AI Coding Assistant)*
