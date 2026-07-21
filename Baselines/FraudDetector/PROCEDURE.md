# ECOD + Leave-One-Out Fraud Baseline — Procedure

## 1. Method

### Detector: ECOD (TKDE 2022)

We use **ECOD** (*Empirical Cumulative Distribution Functions for Outlier Detection*,
Li et al., IEEE TKDE 2022) as the fraud-proxy anomaly detector.

**Why ECOD:**
- Completely parameter-free (no contamination tuning at fit time affects scores)
- State-of-the-art unsupervised outlier detection on tabular data
- Per-column score decomposition is mathematically grounded (empirical CDFs)
- Fast O(n log n) fit; pure-NumPy inference
- Publicly available in `pyod` (version-pinned, reproducible)

ECOD models each column's marginal distribution during training on the **clean
data only**. At inference time it scores a new row by measuring how far its
values fall in the tails of each column's empirical CDF and combining the
log-probability contributions across all columns.

### Bridging row-level anomaly to cell-level intent: Leave-One-Out (LOO)

Standard fraud detectors output a **row-level** anomaly score. Our task requires
a **cell-level** intent label (+1 intentional / −1 unintentional). We bridge this
gap with a Leave-One-Out perturbation:

```
For each dirty row i with flagged cells {c_1, ..., c_k}:

  score_dirty = ECOD_score(dirty_row_i)

  For each flagged cell c_j:
    row_reverted = dirty_row_i with cell c_j reset to clean value
    score_reverted_j = ECOD_score(row_reverted)
    delta_j = score_dirty - score_reverted_j

  Predict: +1 (intentional)  if delta_j > threshold
            -1 (unintentional) otherwise
```

**Intuition:** If reverting cell c_j back to its clean value substantially
*reduces* the row's anomaly score, that cell was the main driver of the
anomaly → label it as intentional. If the score barely changes after
reverting, the cell is not anomalous in context → label it unintentional.

**Batching:** All (1 + k) rows for a dirty row (the dirty row plus k reverted
variants) are scored in a single `decision_function` call to avoid the O(n)
cost of calling the scorer k+1 times independently.

### Encoding

Categorical columns are ordinally encoded (fitted on clean data only;
unknown categories → -1). LLM-injected non-numeric strings in numeric
columns (e.g., `'4B'`, `'N/A'`) are coerced to NaN via
`pd.to_numeric(..., errors='coerce')` and imputed with the batch median.

### Threshold selection

The delta threshold is swept over the 1st–99th percentile of observed delta
scores (50 candidate values) on the **full dataset** and the value maximising
`f1_weighted` is selected. No held-out split is used for threshold selection;
this is a mild optimism that should be noted when comparing to LLM baselines.

---

## 2. Datasets

| Dataset | Clean rows | Dirty rows | Errors (cells) | INT | UNINT | Note |
|---|---:|---:|---:|---:|---:|---|
| Adult-LLM | 48,842 | 19,299 | 43,259 | 18,014 | 25,245 | LLM typos may be non-numeric strings |
| Adult-Mixed | 48,842 | 9,423 | 11,661 | 3,784 | 7,877 | tab_err + greedy adversarial |
| Adult-TFM | 48,842 | 9,748 | 9,199 | 3,009 | 6,190 | TabPFN distribution-consistent |
| TwitterBot-LLM | 4,731 | 100 | 322 | 322 | 0 | Single-class (all INT) |
| TwitterBot-Mixed | 4,731 | 756 | 1,099 | 343 | 756 | tab_err + greedy adversarial |
| TwitterBot-TFM | 4,731 | 984 | 1,184 | 283 | 901 | TabPFN distribution-consistent |
| TabFact | 535 (entailed) | 449 | 2,245 | 2,245 | 0 | Row-level only; all INT by construction |

---

## 3. Codebase

| File | Purpose |
|---|---|
| `datasets.py` | Canonical loader for all 7 datasets → unified dict |
| `ecod_loo.py` | ECOD wrapper + FeatureEncoder + batched LOO attribution |
| `evaluate.py` | Metrics: F1-int, F1-unint, F1-weighted, Accuracy, AUC, threshold sweep |
| `run_all.py` | End-to-end driver; saves per-dataset CSVs and JSON metrics |
| `results/` | Output directory: raw LOO CSVs, threshold sweeps, metrics JSONs, summary |

Run:
```bash
conda run --live-stream --name base python fraud_baseline/run_all.py
# or a subset:
conda run --live-stream --name base python fraud_baseline/run_all.py --datasets adult_mixed adult_tfm
```

---

## 4. Limitations

1. **Row-level → cell-level bridge is approximate.** LOO assumes each cell
   contributes independently to the anomaly score. Multi-cell intentional edits
   (e.g., TFM masks 3 features simultaneously) may be under-attributed because
   reverting one cell at a time does not capture interaction effects.

2. **Threshold tuned on test distribution.** The delta threshold is selected
   to maximise F1-weighted on the same data it is evaluated on. This is a mild
   optimism; the gap versus a held-out-tuned threshold is expected to be small
   but should be acknowledged.

3. **Single-class datasets (TwitterBot-LLM, TabFact).** F1-unintentional is
   undefined (no unintentional errors exist). AUC is also undefined. Only
   precision/recall/F1 for the intentional class are reported.

4. **ECOD is univariate.** ECOD models each column's marginal distribution
   independently and then combines scores. It cannot detect intentional edits
   that are individually plausible per-column but implausible in combination
   (e.g., TFM samples that are in-distribution per-cell but collectively flip
   the classifier). This is precisely the hard regime Adult-TFM is designed
   to test.

5. **TabFact is row-level only.** No cell-level mask exists; all feature cells
   of a refuted row are labelled +1. The LOO delta is computed per cell but
   the ground truth is uniform across the row.
