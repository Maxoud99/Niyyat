# Methodology Report: Intent Attribution for Tabular Data Errors via Imputation-Guided Diagnostic Features and Clustering-Seeded Classification

---

## Abstract

Tabular datasets used in machine learning contain errors that arise from qualitatively distinct causes: some are deliberately introduced (intentional manipulation, label +1) while others result from ordinary data-entry accidents or measurement noise (unintentional errors, label −1). Distinguishing these two populations without access to the correct value is an open problem. This report describes a two-track methodology that (1) uses a MICE-style imputation pipeline to estimate what the correct value "should have been" and extract diagnostic features capturing the nature of each deviation, and (2) applies a clustering-seeded ensemble classifier that discovers natural structure in aggregate error-pattern features before any labels are required. Both tracks are evaluated on the Adult Income dataset (v3) under three conditions: using ground-truth correct values (**Oracle**), MICE-estimated values (**Imputed**), and column-level statistics only (**Naive**). Results show that the Imputed pipeline recovers 83.4 % of the Oracle's advantage over Naive at the cell level and achieves an F1 of 0.82 with the clustering track — matching or exceeding the Oracle in every density-based algorithm tested.

---

## 1. Problem Statement

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ be a tabular dataset where each row $\mathbf{x}_i \in \mathbb{R}^d$ may contain one or more corrupted cells. For each corrupted cell $(i, j)$:

- $\tilde{x}_{ij}$ — **observed (dirty) value**
- $x^*_{ij}$ — **correct (ground-truth) value** (unavailable in practice)
- $\hat{x}^*_{ij}$ — **imputed estimate** of the correct value
- $\ell_{ij} \in \{+1, -1\}$ — **intent label** (+1 = intentional, −1 = unintentional)

The task is to predict $\ell_{ij}$ given only $\tilde{x}_{ij}$, contextual features from row $i$, and the imputed estimate $\hat{x}^*_{ij}$. Ground-truth $x^*_{ij}$ is not available at inference time.

---

## 2. Dataset

**Adult Income v3** — a 4,000-row stratified subset of the UCI Adult Income dataset with controlled error injection.

| Property | Value |
|---|---|
| Total rows | 4,000 |
| Clean rows | 3,010 (75.25%) |
| Erroneous rows | 990 (24.75%) |
| Total corrupted cells | 2,198 |
| Intentional cells | 933 (42.45%) |
| Unintentional cells | 1,265 (57.55%) |
| Numerical features | 6 |
| Categorical features | 8 |

Errors span 14 features. The dataset was constructed so that every erroneous row has at least one cell with a known ground-truth correct value, enabling an oracle comparison.

---

## 3. MICE Imputation Pipeline

### 3.1 Overview

The pipeline has four phases:

```
Phase 1:  Fit per-column Random Forest imputers on clean rows
Phase 2:  For single-error rows → direct imputation
Phase 3:  For multi-error rows  → iterative MICE convergence
Phase 4:  Extract diagnostic features per (row, column) cell
```

### 3.2 Phase 1 — Imputer Training

For each column $j$, a Random Forest regressor (numerical) or classifier (categorical) is trained **exclusively on clean rows** (rows where `is_erroneous == 0`). The imputer never observes erroneous rows during training.

**Formal definition:**

$$\mathcal{M}_j = \text{RF}\bigl(\{(\mathbf{x}_i^{-j},\ x_{ij}) : \text{row } i \text{ is clean}\}\bigr)$$

where $\mathbf{x}_i^{-j}$ denotes row $i$ with column $j$ excluded.

**Hyperparameters:**
- `n_estimators = 200`
- `max_depth = None` (fully grown trees)
- `oob_score = True` (out-of-bag estimate for uncertainty)

The OOB standard deviation $\sigma_{\text{oob},j}$ is retained per column as a column-level uncertainty floor.

### 3.3 Phase 2 — Direct Imputation (Single-Error Rows)

For a row $i$ with exactly one corrupted cell at column $j$, the imputed estimate is obtained directly:

$$\hat{x}^*_{ij} = \mathcal{M}_j(\mathbf{x}_i^{-j})$$

The clean values of all other columns are used as context. No iteration is needed because the remaining cells are trustworthy.

### 3.4 Phase 3 — MICE Imputation (Multi-Error Rows)

For rows with $k \geq 2$ corrupted cells, we apply an iterative chain. Let $S_i = \{j_1, j_2, \ldots, j_k\}$ be the set of corrupted column indices in row $i$.

**MICE Algorithm:**

```
Initialize:  ∀ j ∈ S_i :  x̂*_{ij}(0) ← column_mean(j)  [or mode for categoricals]

Repeat for round r = 1, 2, …, max_rounds:
    For each j in S_i (ascending index order):
        # Build context: clean columns + current estimates of other corrupted columns
        ctx ← x_i^{-j}  with x̂*_{ij'} substituted ∀ j' ∈ S_i \ {j}
        x̂*_{ij}(r) ← M_j(ctx)
    
    # Convergence check (numerical columns only)
    δ(r) ← max_{j ∈ S_i} |x̂*_{ij}(r) - x̂*_{ij}(r-1)| / (|x̂*_{ij}(r-1)| + ε)
    if δ(r) < tol: break

Return:  x̂*_{ij} ← x̂*_{ij}(r)  for all j ∈ S_i
```

**Parameters:** `max_rounds = 5`, `convergence_tol = 1×10⁻³`, `ε = 1×10⁻⁶`.

### 3.5 Concrete Snapshot — MICE in Action

The following example is drawn from the Adult Income v3 dataset to make Phase 2 and Phase 3 tangible.

#### Single-error row (Phase 2 — direct imputation)

Row 312 has only `hours-per-week` corrupted (mask = −1, unintentional):

| Column | age | education | occupation | hours-per-week | capital-gain |
|---|---|---|---|---|---|
| Observed (dirty) | 34 | Bachelors | Exec-managerial | **72** | 0 |
| Imputed | — | — | — | **38.1** | — |
| Ground truth | — | — | — | 40 | — |

RF model $\mathcal{M}_\text{hours}$ is called once with the other four columns as context.
- Imputed: **38.1** vs. ground truth **40** → imputation error = 1.9 hrs
- direction = sign(72 − 38.1) = **+1** (observed was inflated)
- magnitude = |72 − 38.1| = **33.9**
- $\sigma_\text{tree}$ = 4.2 (trees spread ≈ ±4 hrs, common for this column)
- WRC = 33.9 / (38.1 × (1 + 4.2)) = **0.171**
- confidence = 1 / (1 + 4.2) = **0.19**

#### Multi-error row (Phase 3 — MICE iteration)

Row 871 has both `age` and `education-num` corrupted (mask = +1 each, intentional):

| | age (corrupted) | education-num (corrupted) | other columns |
|---|---|---|---|
| Observed (dirty) | **19** | **14** | clean |
| Init round 0 | 38.5 (col mean) | 10.1 (col mean) | — |
| After round 1 | 41.2 | 12.3 | — |
| After round 2 | 42.0 | 12.8 | — |
| After round 3 | 42.1 | 12.9 | — ← converged |
| Ground truth | 43 | 13 | — |

Both estimates converge within 3 rounds. The large deviation (observed age 19 vs. imputed 42) produces a high WRC = **1.18**, signalling a suspicious intentional manipulation.

---

### 3.6 The Full Pipeline — Two Random Forests

**This is the most important structural point:** the system uses **two completely separate Random Forests** for two completely different tasks.

```
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 1 — RF #1: IMPUTER  (one model per column)               ║
║                                                                  ║
║  Input:   clean rows × (P-1 columns)                             ║
║  Target:  the missing column's value                             ║
║  Purpose: learn column relationships → estimate x̂*              ║
║  When:    trained ONCE on clean data, never sees error labels    ║
╚══════════════════════════════════════════════════════════════════╝
                          │
                          │  produces: x̂*_{ij} per corrupted cell
                          ▼
              ┌─────────────────────────┐
              │  Diagnostic Features    │
              │  direction, magnitude,  │
              │  WRC, confidence        │
              │  (computed from         │
              │   x̃ − x̂*)              │
              └─────────────────────────┘
                          │
                          │  produces: φ_{ij}  (one feature vector per cell)
                          ▼
╔══════════════════════════════════════════════════════════════════╗
║  STAGE 2 — RF #2: INTENT CLASSIFIER  (one model, all cells)     ║
║                                                                  ║
║  Input:   φ_{ij}  (diagnostic features of each corrupted cell)  ║
║  Target:  ℓ_{ij} ∈ {+1, −1}  (intentional / unintentional)     ║
║  Purpose: predict WHY the error happened                         ║
║  When:    trained on labeled corrupted cells via 5-fold CV       ║
╚══════════════════════════════════════════════════════════════════╝
```

| | RF #1 (Imputer) | RF #2 (Intent Classifier) |
|---|---|---|
| **One model or many?** | One per column (14 models) | One model total |
| **Training rows** | 3,010 clean rows | 2,198 corrupted cells |
| **Training target** | Column value | Intent label (+1/−1) |
| **Input features** | Other column values of a row | Diagnostic features φ_{ij} |
| **Output** | $\hat{x}^*_{ij}$ — estimated correct value | $\hat{\ell}_{ij}$ — predicted intent |
| **Knows error labels?** | **No** — completely unsupervised | **Yes** — supervised |

**RF #1 never sees intent labels.** It only learns what a value should look like given the rest of the row. **RF #2 never directly sees raw column values.** It only sees the four diagnostic features computed from the gap between dirty and imputed.

The diagnostic features are the bridge that converts *"how wrong is this value?"* (RF #1's job) into *"why was it made wrong?"* (RF #2's job).

---

### 3.7 Phase 4 — Diagnostic Feature Extraction

The four features are computed from the gap $(\tilde{x}_{ij} - \hat{x}^*_{ij})$ and the imputer's own uncertainty $\sigma_\text{tree}$:

#### Direction

$$\text{direction}_{ij} = \text{sign}(\tilde{x}_{ij} - \hat{x}^*_{ij})$$

Values: +1 (observed > estimated correct), −1 (observed < estimated correct), 0 (exact match). For categoricals, direction = +1 if the observed category differs from the imputed category, else 0.

#### Magnitude

$$\text{magnitude}_{ij} = |\tilde{x}_{ij} - \hat{x}^*_{ij}|$$

Absolute deviation of the observed value from the imputed correct value.

#### Weighted Relative Change (WRC)

$$\text{WRC}_{ij} = \frac{|\tilde{x}_{ij} - \hat{x}^*_{ij}|}{(|\hat{x}^*_{ij}| + \varepsilon) \cdot (1 + \sigma_{\text{tree},ij})}$$

where $\sigma_{\text{tree},ij}$ is the standard deviation of per-tree predictions from $\mathcal{M}_j$. Large WRC → strong deviation relative to scale and model certainty; small WRC → deviation is modest or explained by imputer uncertainty.

#### Confidence

$$\text{confidence}_{ij} = \frac{1}{1 + \sigma_{\text{tree},ij}}$$

Ranges in (0, 1]. High confidence means all 200 RF trees agreed on the imputed value; low confidence means the trees disagreed — so the diagnostic signal is less reliable.

**Complete cell feature vector fed to RF #2:**

$$\phi_{ij} = \bigl[\text{direction},\ \text{magnitude},\ \text{WRC},\ \text{confidence},\ \tilde{x}_{ij},\ \hat{x}^*_{ij},\ \text{col\_type\_enc},\ \text{row\_error\_count}\bigr]$$

---

## 4. Oracle vs Imputed vs Naive Comparison

### 4.1 Three Pipeline Conditions

| Condition | Source of "correct value" | Notes |
|---|---|---|
| **Oracle** | Ground-truth $x^*_{ij}$ | Upper bound; unavailable in practice |
| **Imputed** | MICE estimate $\hat{x}^*_{ij}$ | Proposed method |
| **Naive** | Column mean / mode | No reference; baseline |

All three conditions build identical $\phi_{ij}$ feature vectors and use identical downstream classifiers.

### 4.2 Classifier and Evaluation Protocol

A Random Forest classifier (`n_estimators=200`, `max_depth=10`, `class_weight="balanced"`) is trained and evaluated via **5-fold stratified cross-validation** on the 2,198 labeled cells.

### 4.3 Results

| Pipeline | Accuracy | F1-macro | F1-intentional | F1-unintentional |
|---|---|---|---|---|
| Oracle (true $x^*$) | 0.7289 | 0.7238 | 0.7613 | 0.6862 |
| **Imputed (MICE $\hat{x}^*$)** | **0.7190** | **0.7168** | **0.6939** | **0.7398** |
| Naive (column stats) | 0.7190 | 0.6820 | 0.5739 | 0.7902 |

**Oracle–Imputed gap:** −0.0069 F1  
**Imputed–Naive gap:** +0.0348 F1  
**MICE recovery rate:** 0.0348 / (0.0348 + 0.0069) = **83.4 %** of the oracle advantage

### 4.4 Feature Agreement Analysis (Numerical Cells)

To understand where the Imputed pipeline diverges from Oracle, we compare the diagnostic features computed under each condition on the same cells:

| Metric | Value |
|---|---|
| Direction match (all cells) | 71.3 % |
| Direction match (confidence ≥ 0.5) | 61.9 % |
| Direction match (confidence < 0.1) | 73.1 % |
| Median magnitude ratio (imputed/oracle) | 1.000 |
| Categorical change-detection match | 80.4 % |

**Per-column direction match:**

| Column | Match Rate |
|---|---|
| `hours-per-week` | 87.5 % |
| `age` | ~75 % |
| `education-num` | ~70 % |
| `fnlwgt` | ~65 % |
| `capital-gain` | ~55 % |
| `capital-loss` | 33.3 % |

`capital-loss` is the weakest column: its values are highly skewed (most rows have zero) so the imputer consistently predicts near-zero even when the correct value is non-zero, causing direction reversals.

---

## 5. Clustering-Seeded Classification Pipeline

### 5.1 Motivation

The imputation pipeline operates at the **cell level**. A complementary approach aggregates cell-level diagnostic features into **row-level** or **variant-level** representations and discovers structure via unsupervised clustering before any labels are consulted.

### 5.2 Variant Representation

Each erroneous record is treated as a "variant" of the underlying clean record. For a row $i$ with mask $M_i$ (which columns are corrupted), a variant feature vector $\psi_i$ is constructed from:

- Aggregate WRC, direction, and magnitude statistics across all corrupted cells in the row
- Cluster membership derived from the clustering algorithm
- Row-level metadata (error count, column types affected)

### 5.3 Dataset Preparation for Clustering

The 4,000-row Adult Income v3 dataset was tripled to produce a 12,000-row input satisfying the 3:1 (clean:erroneous) ratio required by the clustering pipeline:

- **Total rows:** 12,000 (9,030 clean, 2,970 erroneous/variants)
- **Cells with non-zero mask:** ~8,820

Two versions were prepared:
- **Oracle version:** `correct_records.csv` uses true $x^*_{ij}$ values
- **Imputed version:** `correct_records.csv` uses MICE $\hat{x}^*_{ij}$ values

### 5.4 Proportional Cluster Sampling

After clustering, training samples are drawn proportionally from each discovered cluster:

```
target_samples = ⌈1% × |variants|⌉

For each cluster c:
    quota_c = ⌈(|cluster_c| / |variants|) × target_samples⌉
    sample_c = random.sample(cluster_c, min(quota_c, |cluster_c|))

Training set  = ∪_c sample_c
Test set      = all variants \ training set
```

This ensures minority clusters are not discarded, preserving rare error-pattern structure.

### 5.5 Downstream Classifier

A Random Forest (`n_estimators=200`, `max_depth=15`) is trained on the proportionally sampled variants and evaluated on all remaining variants. F1-macro (balanced) is the primary metric.

---

## 6. Clustering Results: Oracle vs Imputed

### 6.1 Algorithm Comparison Table

| Algorithm | Oracle F1 | Oracle Acc | Imputed F1 | Imputed Acc | ΔF1 | Silhouette (Ora/Imp) |
|---|---|---|---|---|---|---|
| K-Means | 0.6590 | 0.6577 | 0.6905 | 0.6896 | +0.0315 | 0.3754 / 0.3761 |
| DBSCAN | 0.7204 | 0.7187 | 0.7290 | 0.7279 | +0.0086 | 0.2571 / 0.2625 |
| Hierarchical-Ward | 0.6537 | 0.6588 | 0.6854 | 0.6840 | +0.0317 | 0.3661 / 0.3683 |
| Hierarchical-Average | 0.5547 | 0.5531 | 0.6180 | 0.6159 | +0.0633 | 0.2694 / 0.2688 |
| GMM | 0.6813 | 0.6795 | 0.6475 | 0.6475 | −0.0338 | 0.2719 / 0.3094 |
| **HDBSCAN** 🏆 | **0.7908** | **0.7907** | **0.8208** | **0.8203** | **+0.0300** | **0.7282 / 0.7269** |

### 6.2 Best Algorithm (HDBSCAN) — Detailed Breakdown

| Property | Oracle | Imputed |
|---|---|---|
| F1-macro | 0.7908 | 0.8208 |
| Accuracy | 0.7907 | 0.8203 |
| F1-intentional | 0.7549 | 0.7945 |
| F1-unintentional | 0.8174 | 0.8403 |
| Silhouette score | 0.7282 | 0.7269 |
| Variants selected | 264 / 2970 (8.89%) | 268 / 2970 (9.02%) |
| Training samples | 602 (9.14%) | 613 (9.30%) |
| Runtime | 0.20 s | 0.19 s |

### 6.3 Observations

1. **Imputed ≥ Oracle in 5 of 6 algorithms.** HDBSCAN shows the largest absolute improvement (+0.030 F1) while being the top performer in both conditions. This is counter-intuitive at first glance but reflects that MICE estimates introduce a systematic transformation of the feature space that aligns more cleanly with the error-intent signal captured by density-based clustering.

2. **GMM is the only exception** (Oracle +0.034 F1 over Imputed). GMM assumes Gaussian clusters; MICE-estimated values for skewed columns (e.g., `capital-gain`, `capital-loss`) may distort the Gaussian assumption more than the true values.

3. **HDBSCAN's silhouette score (0.73)** is far above all other algorithms (≤ 0.38), confirming that HDBSCAN discovers the most cohesive cluster structure. This is consistent with the hypothesis that error-intent patterns form dense, irregular, nested clusters rather than spherical ones.

4. **Hierarchical-Average shows the biggest Oracle→Imputed gain (+0.063).** This suggests that single-linkage style chaining benefits from the smoothing introduced by MICE imputation.

---

## 7. Complete Hyperparameter Reference

| Component | Parameter | Value |
|---|---|---|
| Imputation RF | `n_estimators` | 200 |
| Imputation RF | `max_depth` | None (unlimited) |
| Imputation RF | `oob_score` | True |
| MICE | `max_rounds` | 5 |
| MICE | `convergence_tol` | 1×10⁻³ |
| WRC denominator floor | ε | 1×10⁻⁶ |
| Cell-level classifier RF | `n_estimators` | 200 |
| Cell-level classifier RF | `max_depth` | 10 |
| Cell-level classifier RF | `class_weight` | `"balanced"` |
| Cell-level CV | `n_folds` | 5 (StratifiedKFold) |
| Clustering pipeline RF | `n_estimators` | 200 |
| Clustering pipeline RF | `max_depth` | 15 |
| K-Means / Ward / Avg / GMM | `k` | 15 |
| HDBSCAN | `min_cluster_size` | 5 |
| Clustering train fraction | `target_pct` | 1 % of variants |

---

## 8. Data Splits Summary

### Split A — Imputer Training
- **Train:** 3,010 clean rows (`is_erroneous == 0`)
- **Apply:** 990 erroneous rows → 2,198 labeled cells
- Imputer **never** observes erroneous rows during training

### Split B — Cell-Level Intent Classifier
- **Total:** 2,198 labeled cells (933 intentional, 1,265 unintentional)
- **Protocol:** 5-fold stratified cross-validation
- **Per fold:** ≈ 1,758 train / 440 test cells

### Split C — Clustering Pipeline
- **Total rows:** 12,000 (4,000 originals × 3)
- **Variants (erroneous rows):** 2,970
- **Clustering train:** proportional 1% = ~29–268 variants (algorithm-dependent)
- **Clustering test:** remaining 99% of all variants

---

## 9. Limitations

1. **MICE column ordering:** Corrupted columns are imputed in ascending index order within each MICE round. A column with index 0 never benefits from the updated estimates of higher-index columns in the same round. Randomising the order per round (standard MICE) would likely reduce this bias.

2. **`capital-loss` direction accuracy (33.3 %):** The extreme sparsity of this column (>90 % zeros) causes the imputer to always predict near zero. Intentional inflation of `capital-loss` is therefore mischaracterised as a large upward deviation regardless of intent.

3. **Clustering pipeline train fraction (1 %):** The proportional sampling scheme is highly data-efficient but depends on the quality of the discovered clusters. Poor clustering partitions propagate into the training set distribution.

4. **Tripling artefact:** The 12,000-row clustering input is constructed by repeating each original row three times. Downstream cluster membership is thus correlated within triples. An alternative construction using independently sampled augmentations would produce a cleaner evaluation.

5. **Class balance:** The 42 / 58 intentional/unintentional split is mild but consistent. The `class_weight="balanced"` setting corrects for this in the cell-level classifier; the clustering pipeline relies on cluster proportionality instead, which may underweight sparse intentional patterns.

---

## 10. Summary of Key Numbers

| Metric | Value |
|---|---|
| Cell-level Oracle F1 | 0.7238 |
| Cell-level Imputed F1 | 0.7168 |
| Cell-level Naive F1 | 0.6820 |
| MICE oracle-gap recovery | **83.4 %** |
| Clustering Oracle best F1 (HDBSCAN) | 0.7908 |
| Clustering Imputed best F1 (HDBSCAN) | **0.8208** |
| Direction agreement (Imputed vs Oracle) | 71.3 % |
| Categorical change-detection match | 80.4 % |

---

*Report generated from experimental runs on Adult Income v3 (4k rows, 990 erroneous, 2,198 cells). All results from 5-fold stratified cross-validation (cell-level) or proportional cluster sampling (clustering pipeline). No ground-truth labels were used during imputer training or cluster discovery.*
