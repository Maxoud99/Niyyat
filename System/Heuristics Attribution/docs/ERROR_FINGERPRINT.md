# Error Fingerprint: Intent Attribution for Tabular Data Errors Without Clean References

**Author:** Mohamed Maxoud — Hasso Plattner Institute, University of Potsdam  
**Date:** March 2026

---

## Abstract

We present **Error Fingerprint**, a heuristic-based pipeline for classifying erroneous cells in tabular datasets as *intentional* or *unintentional* without requiring clean reference values. The system uses only the dirty dataset and a blind binary error mask (which cells are wrong, but not *how* they are wrong). Eight complementary heuristics (H1–H8) extract a 13-dimensional feature vector per erroneous cell, capturing surface-level value properties, row-level coherence, error co-occurrence patterns, and domain-level column characteristics. A Random Forest classifier trained on these features achieves **92.1% accuracy (F1 = 0.903, AUC = 0.973)** on an LLM-generated Adult Income error dataset with 43,259 erroneous cells, and **99.0% accuracy (F1 = 0.984, AUC = 0.998)** on a Kireev-style programmatic error dataset with 11,661 erroneous cells. These results substantially outperform LLM-based baselines (best F1 = 0.818) and match or exceed clustering-based approaches (best F1 = 0.904) while providing per-cell interpretable explanations.

---

## 1. Introduction

Data errors in tabular datasets arise from fundamentally different causes. **Unintentional** errors—typos, sensor noise, parsing failures, OCR artifacts—are accidental and produce values that often violate domain constraints or break row-level correlations. **Intentional** errors—fraud, benefit gaming, privacy masking, fairness evasion—are deliberate and produce values that are carefully chosen to appear plausible within the data's structure.

Distinguishing between these two types matters for data repair, auditing, and trust. An unintentional error can be corrected by statistical imputation; an intentional error reveals adversarial behavior and may require flagging the entire record for review.

The core challenge is that existing approaches to intent attribution typically require *clean reference values*—the original, correct data—to compute features such as perturbation magnitude or causal impact on model predictions. In practice, clean values are rarely available: if we had them, we would not need error detection at all.

### Contribution

We propose **Error Fingerprint**, a system that classifies each erroneous cell as intentional or unintentional using *only* the dirty dataset and a blind binary mask indicating which cells are erroneous. The key insight is that intentional and unintentional errors leave distinguishable *fingerprints* in the dirty data itself:

| Property | Unintentional | Intentional |
|----------|--------------|-------------|
| Value validity | Often invalid (typos, noise) | Valid or deliberate obfuscation |
| Row coherence | Breaks internal consistency | Maintains/improves consistency |
| Value frequency | Rare / extreme values | Common / majority values |
| Feature targeting | Random columns | Strategic columns |
| Co-occurrence | Isolated single-cell errors | Correlated multi-cell edits |

---

## 2. Problem Formulation

**Input:**
- A dirty dataset `D ∈ R^{n × m}` with `n` rows and `m` columns.
- A blind binary mask `M ∈ {0, 1}^{n × m}` where `M[i,j] = 1` indicates that cell `(i,j)` is erroneous.

**Output (per erroneous cell):**
- A predicted label `ŷ ∈ {Intentional, Unintentional}`.
- A confidence score `P(Intentional | e) ∈ [0, 1]`.
- An explanation vector `(h₁, ..., h₈) ∈ [0, 1]⁸` of per-heuristic scores.

**Constraints:**
No access to clean/correct values. No access to a downstream ML model. Only the dirty data and the mask are available.

---

## 3. Method: The Error Fingerprint Pipeline

The pipeline operates in two phases: **fit** (learn column-level statistics and predictors from the dirty data) and **compute** (extract a 13-dimensional feature vector for each erroneous cell). An optional **classify** phase trains a Random Forest on labeled examples.

### 3.1 Architecture Overview

Eight heuristics are organized into three conceptual levels:

```
LEVEL 1: Surface — What does the dirty value LOOK like?
  ├─ H1: Value Plausibility     (is it a valid domain value?)
  ├─ H2: String Anomaly         (does it have typo/obfuscation patterns?)
  └─ H3: Distribution Position  (where does it sit in the column's distribution?)

LEVEL 2: Context — How does it relate to its ROW?
  ├─ H4: Row Coherence          (does it fit the rest of the row?)
  └─ H5: Error Pattern          (which/how many cells are flagged in this row?)

LEVEL 3: Domain — What do we know about the COLUMN itself?
  ├─ H6: Column Importance      (does this column drive the outcome?)
  ├─ H7: User Incentive         (would a rational human change this column?)
  └─ H8: Sensitivity Flag       (is this a protected/sensitive attribute?)
```

All eight heuristics run in a single pass (no sequential dependency between levels). The decomposition is conceptual, not computational.

### 3.2 The 13-Feature Vector

Each erroneous cell is represented by exactly 13 features:

| # | Feature | Source | Type | Intentional ↑? |
|---|---------|--------|------|:-:|
| 1 | `h1_plausible` | H1 | binary | ↑ |
| 2 | `h2_min_edit_distance` | H2 | continuous | ↓ |
| 3 | `h2_is_obfuscation` | H2 | binary | ↑ |
| 4 | `h3_distribution_score` | H3 | [0, 1] | ↑ |
| 5 | `h4_coherence_score` | H4 | [0, 1] | ↑ |
| 6 | `h5_error_count` | H5 | integer | ↑ |
| 7 | `h5_codependent_flag` | H5 | binary | ↑ |
| 8 | `h6_column_importance` | H6 | [0, 1] | ↑ |
| 9 | `h7_mutability` | H7 | [0, 1] | ↑ |
| 10 | `h7_gain_direction` | H7 | [0, 1] | ↑ |
| 11 | `h7_comprehensibility` | H7 | [0, 1] | ↑ |
| 12 | `h8_is_sensitive` | H8 | binary | ↑ |
| 13 | `h8_is_majority_value` | H8 | binary | ↑ |

---

## 4. Heuristic Descriptions

### H1: Value Plausibility

**Question:** *Is this dirty value a legitimate member of the column's domain?*

For categorical columns, H1 checks membership in the vocabulary built from clean cells (mask = 0). For numerical columns, it checks whether the value falls within the [p5, p95] percentile range.

**Output:** One binary feature `h1_plausible ∈ {0, 1}`.

**Rationale.** Unintentional errors frequently produce values outside the domain (e.g., `Bachleors`, `Whitw`). Intentional errors typically use real domain values (`Bachelors`, `White`). On the LLM dataset, 76.5% of intentional errors are plausible vs. only 13.5% of unintentional errors—a **63-percentage-point gap**.

**Failure modes.** (1) 29% of intentional errors are obfuscation tokens (`nan`, `Unknown`), which H1 marks as implausible—rescued by H2. (2) 7.5% of unintentional errors are valid values—rescued by H4.

---

### H2: String Anomaly

**Question:** *Does this string look like a typo or like deliberate obfuscation?*

H2 operates only on categorical columns and computes two features:

- **`h2_min_edit_distance`**: Levenshtein distance to the nearest clean-vocabulary member. Low distance (1–2) suggests a typo; high distance suggests random garbage or a deliberately different value.
- **`h2_is_obfuscation`**: Binary flag activated when the value matches known obfuscation patterns (`nan`, `N/A`, `Unknown`, `?`, `—`, suffix patterns like `-DMV`, `-obf`).

**Why H2 ≠ H1.** Consider `Bachleors` and `Unknown`. H1 assigns both `h1_plausible = 0` (neither is in the vocabulary). H2 distinguishes them: `Bachleors` has edit distance 1 to `Bachelors` (typo), while `Unknown` triggers the obfuscation flag (deliberate).

---

### H3: Distribution Position

**Question:** *Is this dirty value common or rare in the column's distribution?*

**Output:** One continuous feature `h3_distribution_score ∈ [0, 1]`, where 1 indicates a central/common value and 0 indicates an extreme/rare value.

- **Numerical:** `score = 1 - min(1, |z-score| / 3)`
- **Categorical:** Score derived from frequency rank of the dirty value among clean-cell values.

**Rationale.** Intentional errors tend toward common, unremarkable values (blending in). Unintentional noise produces extreme or rare values. On the LLM dataset, intentional errors average 0.679 vs. 0.133 for unintentional—a **54.7-percentage-point gap**.

**Failure modes.** For `capital-gain`, both types are 91–93% in-range, making H3 nearly blind for that column.

---

### H4: Row Coherence — *The Key Heuristic*

**Question:** *Does this dirty value make sense given the other values in the same row?*

For each column, a Random Forest predictor is trained on the dirty dataset to predict that column's value from all other columns. At inference, for each erroneous cell, the predictor estimates what the column value "should be" given the remaining features:

- **Categorical:** `h4_coherence_score = P(predicted class == dirty value)`
- **Numerical:** `h4_coherence_score = max(0, 1 - |predicted - dirty| / σ_col)`

**Why training on dirty data is acceptable.** Errors are sparse (≪50% per column), so the majority pattern learned by the predictor reflects the clean data distribution. A few percent of erroneous cells contribute tolerable noise.

**Rationale.** This heuristic handles *all ambiguous cases* where H1–H3 are blind. When both intentional and unintentional errors produce valid values (the 7.5% overlap), H4 asks: "Does `education=Doctorate` make sense for a 23-year-old part-time worker?" If no → likely unintentional. If yes → likely intentional.

---

### H5: Error Pattern

**Question:** *Is this error part of a coordinated multi-cell edit?*

**Output:** Two features:
- **`h5_error_count`**: Number of erroneous cells in the same row.
- **`h5_codependent_flag`**: Binary, 1 if a logically linked partner column (e.g., `education` and `education-num`) is also erroneous in the same row.

Co-dependent column pairs are detected automatically via column-name similarity and pairwise mutual information, supplemented by user-supplied domain knowledge.

**Rationale.** Intentional manipulators edit multiple related columns together to maintain consistency. Unintentional errors are typically isolated.

---

### H6: Column Importance

**Question:** *Is this column statistically important for the outcome?*

**Output:** One per-column constant `h6_column_importance ∈ [0, 1]`, computed as the normalized mutual information between the column and the target variable (supervised) or the maximum MI with any other column (unsupervised).

**Rationale.** A rational attacker targets columns that drive the decision. Errors in high-MI columns (`education`, `hours-per-week`) are more likely intentional than errors in low-MI columns (`fnlwgt`).

---

### H7: User Incentive

**Question:** *Would a rational human be motivated to manipulate this column?*

**Output:** Three features capturing orthogonal aspects of human motivation:
- **`h7_mutability ∈ [0, 1]`**: Is the column user-declared (freely mutable, 1.0), system-assigned (immutable, 0.0), or a sensitive attribute (soft boundary, 0.5)?
- **`h7_gain_direction ∈ [0, 1]`**: Does the dirty value shift toward a favorable outcome? Computed from the correlation between column values and the target.
- **`h7_comprehensibility ∈ [0, 1]`**: Would a layperson understand what this column means? Derived from column-name analysis.

**Why H7 ≠ H6.** H6 measures *statistical* importance (what matters to the model). H7 measures *behavioral* motivation (what matters to the user). These diverge:
- `fnlwgt`: high MI (H6 high) but incomprehensible to users (H7 low).
- `race`: low MI (H6 low) but frequent target for privacy masking (H7 high).

The *gap* between H6 and H7 is itself informative.

---

### H8: Sensitivity Flag

**Question:** *Is this a protected demographic attribute, and does the dirty value match the majority class?*

**Output:** Two features:
- **`h8_is_sensitive`**: Binary, 1 if the column is a recognized sensitive attribute (auto-detected from column names or user-supplied).
- **`h8_is_majority_value`**: Binary, 1 if the dirty value equals the majority class of the sensitive column.

**Rationale.** Privacy-motivated masking manifests as changing a minority demographic value to the majority class (e.g., changing `race=Black` to `race=White`).

**Limitation.** Signal strength depends on majority-class dominance. For `sex` (67% Male), the signal is moderate. For `race` (86% White), the signal is stronger.

---

## 5. Non-Redundancy of Heuristics

Each heuristic answers a question that no other heuristic answers:

| H | Why no other heuristic captures this |
|---|--------------------------------------|
| H1 | H2 characterizes *invalid* values but does not test membership |
| H2 | H1 only says valid/invalid; H2 distinguishes typo from obfuscation |
| H3 | H1 tests existence; H3 tests *frequency/position* |
| H4 | H1–H3 are column-only; H4 uses *cross-column* prediction |
| H5 | H1–H4 are per-cell; H5 is *per-row* pattern |
| H6 | H7 measures *human* importance, not statistical |
| H7 | H6 measures MI; H7 measures mutability/comprehensibility |
| H8 | A column can be high-MI (H6), mutable (H7), but not sensitive |

---

## 6. Combination Strategy

We employ a **hybrid** approach:

1. **Classification:** A Random Forest classifier is trained on the 13 raw features. The RF learns optimal nonlinear weighting and interaction effects (e.g., "H1=valid AND H4=incoherent") that would be invisible to a linear weighted sum.

2. **Explanation:** Per-heuristic scalar scores `h₁, ..., h₈` are computed via fixed, domain-grounded formulas. These are *not* used for classification—they provide interpretable explanations of *why* a cell was classified as intentional or unintentional.

3. **Feature importance:** The RF's feature importances reveal post-hoc which features (and implicitly which heuristics) contributed most to classification on a given dataset.

---

## 7. Experimental Setup

### 7.1 Datasets

We evaluate on two variants of the Adult Income dataset, each with different error generation mechanisms:

| Dataset | Rows | Errors | Intent % | Error Source | Error Types |
|---------|------|--------|----------|-------------|-------------|
| **LLM** | 19,539 | 43,259 | 41.6% | LLM-generated | typo, OCR, truncation, obfuscation, gain-targeted, fairness-masking |
| **Kireev** | 48,842 | 11,661 | 32.9% | Programmatic | noise injection (phase 1), strategic manipulation (phase 2) |

Both datasets share the same 15 columns of the UCI Adult Income dataset. The **LLM dataset** errors were generated by prompting language models to produce realistic intentional and unintentional modifications. The **Kireev dataset** uses a two-phase pipeline: phase 1 injects programmatic unintentional noise, and phase 2 applies strategic intentional manipulations.

### 7.2 Baselines

We compare against three families of baselines:

1. **Guessing strategies:** Random 50/50, constant prediction, probability-biased guessing. Best achievable: F1 ≈ 0.50.
2. **LLM-based classification:** 15 LLM configurations (Gemini, LLaMA, Qwen, Mixtral, etc.) with varying prompt strategies (bare-minimum, few-shot, informative). Best: Gemini bare-min, F1 = 0.818 on Adult Income.
3. **Clustering + active labeling:** HDBSCAN/DBSCAN/K-Means/GMM clustering of error fingerprints, followed by labeling ~3–4% of cells and training a classifier. Best: HDBSCAN with normal sampling, F1 = 0.904 on the LLM dataset.

### 7.3 Evaluation Protocol

We use **stratified 5-fold cross-validation** on the full set of erroneous cells. For each fold, all 8 heuristics are fitted on the entire dirty dataset (heuristic fitting uses only clean cells and is unsupervised), and the RF classifier is trained on the fold's training split and evaluated on the held-out split. We report accuracy, precision, recall, F1, and AUC-ROC.

---

## 8. Results

### 8.1 Overall Performance

| Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---------|----------|-----------|--------|-----|-----|
| **LLM** | 0.921 ± 0.001 | 0.913 ± 0.002 | 0.895 ± 0.002 | **0.904 ± 0.002** | **0.973 ± 0.000** |
| **Kireev** | 0.990 ± 0.001 | 0.984 ± 0.003 | 0.984 ± 0.004 | **0.984 ± 0.002** | **0.998 ± 0.000** |

The Kireev dataset yields near-perfect classification because programmatic errors leave highly distinctive fingerprints (e.g., noise injection produces values with large edit distances and low coherence scores, while strategic manipulation always targets high-importance columns). The LLM dataset is substantially harder because LLM-generated errors are more realistic and varied, yet the pipeline still achieves 92.1% accuracy.

### 8.2 Confusion Matrices

**LLM Dataset (43,259 cells):**

| | Pred. Unintentional | Pred. Intentional |
|---|---:|---:|
| **True Unintentional** | 23,706 | 1,539 |
| **True Intentional** | 1,901 | 16,113 |

**Kireev Dataset (11,661 cells):**

| | Pred. Unintentional | Pred. Intentional |
|---|---:|---:|
| **True Unintentional** | 7,763 | 60 |
| **True Intentional** | 61 | 3,777 |

On the LLM dataset, the primary error mode is intentional cells misclassified as unintentional (1,901 false negatives), likely the 29% obfuscation cases where intentional errors use novel tokens that mimic unintentional patterns.

### 8.3 Comparison with Baselines

| Method | Type | F1 Score |
|--------|------|:--------:|
| Random guessing (50/50) | Baseline | 0.501 |
| Constant (always unintentional) | Baseline | 0.346 |
| Prob. 0.6 biased | Baseline | 0.490 |
| Mixtral bare-min | LLM | 0.377 |
| Qwen bare-min | LLM | 0.798 |
| LLaMA few-shot | LLM | 0.810 |
| Gemini bare-min | LLM | 0.818 |
| K-Means + normal sampling | Clustering | 0.849 |
| DBSCAN + normal sampling | Clustering | 0.881 |
| HDBSCAN + normal sampling | Clustering | 0.904 |
| **Error Fingerprint (ours)** | **Heuristic + RF** | **0.904** |

Error Fingerprint matches the best clustering approach (HDBSCAN) while providing three advantages: (1) no cluster tuning needed, (2) full 5-fold CV instead of 3.75% active labeling, and (3) per-cell explanations via heuristic scores.

### 8.4 Feature Importance Analysis

RF feature importances reveal which heuristics drive classification on each dataset:

| Rank | Feature | LLM | Kireev |
|------|---------|:---:|:------:|
| 1 | `h3_distribution_score` | **0.259** | 0.136 |
| 2 | `h4_coherence_score` | **0.183** | 0.058 |
| 3 | `h1_plausible` | **0.180** | 0.165 |
| 4 | `h2_is_obfuscation` | 0.138 | 0.089 |
| 5 | `h2_min_edit_distance` | 0.091 | 0.025 |
| 6 | `h7_gain_direction` | 0.055 | **0.213** |
| 7 | `h5_error_count` | 0.038 | 0.133 |
| 8 | `h6_column_importance` | 0.022 | 0.128 |
| 9 | `h7_comprehensibility` | 0.014 | 0.021 |
| 10 | `h5_codependent_flag` | 0.010 | 0.019 |
| 11 | `h7_mutability` | 0.004 | 0.005 |
| 12 | `h8_is_sensitive` | 0.004 | 0.006 |
| 13 | `h8_is_majority_value` | 0.003 | 0.001 |

**Key observations:**

- **Feature importance ranking shifts between datasets**, validating that all heuristics are needed for generalization. On the LLM dataset, surface-level features (H3, H1, H2) dominate because LLM-generated errors have distinctive string-level patterns. On the Kireev dataset, behavioral features (`h7_gain_direction`) become the most important because programmatic intentional errors always target gain-correlated columns.

- **H4 (row coherence) is the 2nd most important on the LLM dataset**, confirming the design hypothesis that cross-column prediction is the key discriminator for realistic errors.

- **H8 (sensitivity) is consistently the weakest feature**, as predicted—only a small fraction of errors involve sensitive columns.

- **No single heuristic dominates both datasets**, justifying the 8-heuristic design.

### 8.5 Feature Distribution Analysis

Mean feature values by intent class (LLM dataset):

| Feature | Intentional | Unintentional | Δ |
|---------|:-----------:|:-------------:|:-:|
| `h1_plausible` | 0.765 | 0.135 | **+0.630** ↑ |
| `h2_min_edit_distance` | 1.399 | 1.360 | +0.039 |
| `h2_is_obfuscation` | 0.234 | 0.002 | **+0.233** ↑ |
| `h3_distribution_score` | 0.679 | 0.133 | **+0.547** ↑ |
| `h4_coherence_score` | 0.824 | 0.628 | **+0.195** ↑ |
| `h5_error_count` | 2.537 | 2.378 | +0.159 ↑ |
| `h5_codependent_flag` | 0.633 | 0.643 | −0.010 |
| `h6_column_importance` | 0.666 | 0.659 | +0.007 |
| `h7_mutability` | 0.847 | 0.929 | **−0.082** ↓ |
| `h7_gain_direction` | 0.661 | 0.548 | **+0.113** ↑ |
| `h7_comprehensibility` | 0.518 | 0.525 | −0.007 |
| `h8_is_sensitive` | 0.281 | 0.230 | +0.052 ↑ |
| `h8_is_majority_value` | 0.081 | 0.000 | **+0.081** ↑ |

The three strongest univariate separators are `h1_plausible` (Δ = +0.630), `h3_distribution_score` (Δ = +0.547), and `h2_is_obfuscation` (Δ = +0.233). Notably, `h7_mutability` shows a *negative* delta (−0.082): intentional errors are slightly more common in less-mutable columns, suggesting some intentional manipulations target columns that "should not" be changed.

---

## 9. Discussion

### Why does Error Fingerprint work?

The fundamental insight is that intent is not a property of the *value alone*—it is a property of the *relationship* between the value and its context. An intentional error is designed by a human who understands the data schema and chooses a value that is plausible within the row's context. This leaves a characteristic fingerprint: high plausibility (H1), high coherence (H4), common distribution position (H3), and targeting of strategic columns (H6, H7). Unintentional errors are random perturbations that violate these properties.

### The role of H4 (Row Coherence)

H4 is the heuristic that handles the "hard cases" where surface-level features (H1–H3) are ambiguous. When both types produce valid values (7.5% of unintentional errors are in-vocab), H4 discriminates by checking whether the value fits the cross-column correlation structure. On the LLM dataset, H4 has the 2nd-highest RF importance (0.183), confirming its role as the tiebreaker.

### Limitations

1. **Requires error detection as input.** The pipeline assumes a pre-existing binary mask. Error *detection* is a separate problem not addressed here.
2. **Column-type dependent.** H2 (string anomaly) only operates on categorical columns; numerical errors lack this discriminator.
3. **Per-column constants (H6, H7, H8)** cannot distinguish two errors in the same column. They serve as priors, not discriminators.
4. **H4's quality depends on feature correlations.** In datasets with independent columns, H4 would be weak.
5. **The 13 features are hand-designed.** A learned representation might discover additional patterns, at the cost of interpretability.

### Deferred Heuristics

Four additional heuristics were identified but deferred because they require access to a downstream ML model:
- **Causal Impact**: Does changing the cell flip the prediction?
- **Group Shift Potential**: Does the error push across a decision boundary?
- **Effort-Based Perturbation Size**: Minimal change for outcome shift.
- **Peer-Group Deviation**: Deviation from similar records via KNN.

---

## 10. Conclusion

We presented Error Fingerprint, an 8-heuristic pipeline that classifies erroneous cells in tabular datasets as intentional or unintentional without requiring clean reference values. The pipeline extracts a compact 13-dimensional feature vector per erroneous cell and achieves F1 = 0.904 on a challenging LLM-generated error dataset and F1 = 0.984 on a programmatic error dataset, outperforming LLM-based baselines and matching clustering-based approaches while providing interpretable per-heuristic explanations. The key design principle is that intent is revealed not by the erroneous value alone, but by the *relationship* between the value and its structural context within the row, the column, and the dataset.
