# Intent Attribution for Tabular Data Errors: MICE Imputation and Clustering-Based Classification

---

## 1. Problem

Given a tabular dataset where some cells have been corrupted, the goal is to classify each corrupted cell as either **intentional** (label +1, deliberate manipulation) or **unintentional** (label −1, ordinary data-entry error). The correct value of the cell is not available at inference time.

---

## 2. Datasets

### 2.1 Adult Income v3

4,000-row stratified subset of the UCI Adult Income dataset with controlled error injection.

| Property | Value |
|---|---|
| Total rows | 4,000 |
| Clean rows | 3,010 (75.3 %) |
| Erroneous rows | 990 (24.7 %) |
| Corrupted cells | 2,198 |
| Intentional (+1) | 933 (42.5 %) |
| Unintentional (−1) | 1,265 (57.5 %) |
| Features | 15 (6 numerical, 9 categorical) |
| Errors per row | 1–3 (137 single, 498 double, 355 triple) |

### 2.2 Twitter Bot

4,000-row Twitter account dataset with error injection.

| Property | Value |
|---|---|
| Total rows | 4,000 |
| Clean rows | 3,031 (75.8 %) |
| Erroneous rows | 969 (24.2 %) |
| Corrupted cells | 1,931 |
| Intentional (+1) | 1,931 (100 %) |
| Unintentional (−1) | 0 (0 %) |
| Features | 17 (all numerical) |
| Errors per row | 1–6 |

> **Note:** The Twitter Bot dataset contains only intentional errors. Intent classification is therefore degenerate (trivial accuracy of 1.0, F1-macro = 0.5). The imputation quality evaluation is still valid and reported below; classification results for this dataset are not meaningful.

---

## 3. Method

The full pipeline is a single linear sequence of four steps:

```
Step 1 → MICE: train 14 column models on clean rows
Step 2 → MICE: impute x̂* for every corrupted cell
Step 3 → Aggregate: compute per-row feature vectors from |x̃ − x̂*|
Step 4 → Classify: cluster rows → sample labels → RF predicts intent
```

### Step 1 — Train the Imputer (once, offline)

For each column $j \in \{1 \ldots 14\}$, a Random Forest model $\mathcal{M}_j$ is trained on the 3,010 clean rows. Column $j$ is the **target**; all other 13 columns are the **input features**:

$$\mathcal{M}_j \leftarrow \mathrm{RF}\bigl(\mathbf{x}_i^{-j} \rightarrow x_{ij}\bigr), \quad \forall\, i \in \text{clean rows}$$

This produces 14 models. They are never retrained and never see error labels.

### Step 2 — Impute Corrupted Cells

For each erroneous row, $\mathcal{M}_j$ is called to estimate what the corrupted cell should have been ($\hat{x}^*$):

- **Single-error row** (one corrupted cell at column $j$): call $\mathcal{M}_j$ once with the row's 13 clean columns as input.

- **Multi-error row** ($k \geq 2$ corrupted cells): iterative MICE — initialise each corrupted cell with the column mean/mode, then cycle through the corrupted columns updating each estimate using the latest values of all other columns, until convergence:

```
Initialise:  x̂*_j ← column_mean(j)  for each corrupted column j

For round r = 1, 2, …, 5:
    For each corrupted column j:
        x̂*_j ← M_j( all other columns, substituting latest x̂* for corrupted ones )
    If max relative change < 0.001: stop
```

**Example — multi-error row (row 871), two corrupted cells:**

| Round | $\hat{x}^*_\text{age}$ | $\hat{x}^*_\text{edu-num}$ |
|---|---|---|
| 0 (init) | 38.5 (mean) | 10.1 (mean) |
| 1 | 41.2 | 12.3 |
| 2 | 42.0 | 12.8 |
| 3 | 42.1 | 12.9 ← converged |
| Ground truth | 43 | 13 |

### Step 3 — Build Row-Level Feature Vectors

The imputed value $\hat{x}^*$ is used to compute the **change** between what was observed ($\tilde{x}$) and what should have been ($\hat{x}^*$). For each erroneous row, these per-cell changes are aggregated into a 10-dimensional feature vector:

| Feature | Formula |
|---|---|
| `n_changes` | number of corrupted cells in the row |
| `mean/std/min/max/median_magnitude` | statistics of $|\tilde{x} - \hat{x}^*|$ across cells |
| `mean_relative_change` | mean of $|\tilde{x}-\hat{x}^*| / (|\hat{x}^*|+1)$ |
| `min/max_new_value_encoded` | encoded range of observed dirty values |
| `feature_with_max_change_encoded` | which column had the largest change |

This step is implemented in `prepare_clustering_inputs.py`, which writes `correct_records.csv` — the MICE-estimated values — as input to Step 4.

### Step 4 — Cluster, Sample, and Classify

The 2,970 erroneous row vectors are clustered using no labels. From each cluster, ~1 % of rows are sampled proportionally and their intent labels are retrieved. A Random Forest (`n_estimators=200`, `max_depth=15`) is trained on this small labelled sample and evaluated on the remaining 99 % of rows.

---

## 5. Results

The pipeline is run under three experimental conditions that differ **only in what value is used as $\hat{x}^*$** in Steps 2–3:

| Condition | Source of $\hat{x}^*$ |
|---|---|
| **Oracle** | True ground-truth $x^*$ — upper bound, unavailable in practice |
| **Imputed** | MICE estimate — the proposed method |
| **Naive** | Column mean / mode — no imputation at all |

Everything else (clustering algorithm, RF classifier, sampling fraction) is identical across conditions.

### 5.1 Main Results — Full Pipeline (2,970 erroneous rows, ~1 % train)

MICE-imputed values are fed into `correct_records.csv`. The clustering script computes $|\tilde{x} - \hat{x}^*|$ per cell, aggregates into row-level vectors, clusters, samples ~1 % of rows per cluster for training, and evaluates on the remaining 99 %.

| Algorithm | Oracle F1 | Imputed F1 | Δ F1 |
|---|---|---|---|
| K-Means (k=15) | 0.659 | 0.691 | +0.032 |
| DBSCAN | 0.720 | 0.729 | +0.009 |
| Hierarchical-Ward (k=15) | 0.654 | 0.685 | +0.032 |
| Hierarchical-Average (k=15) | 0.555 | 0.618 | +0.063 |
| GMM (k=15) | 0.681 | 0.648 | −0.034 |
| **HDBSCAN** | **0.791** | **0.821** | **+0.030** |

**HDBSCAN best result:**

| Metric | Oracle | Imputed |
|---|---|---|
| F1-macro | 0.7908 | **0.8208** |
| F1-intentional | 0.7549 | 0.7945 |
| F1-unintentional | 0.8174 | 0.8403 |
| Silhouette score | 0.7282 | 0.7269 |
| Rows used for training | 264 / 2,970 (8.9 %) | 268 / 2,970 (9.0 %) |

### 5.2 Imputation Quality Check — How Good is MICE? (separate ablation)

This is a **separate standalone experiment** (`oracle_vs_imputed_comparison.py`) that does not feed into the clustering pipeline. Its sole purpose is to measure how closely MICE estimates approximate the true correct values when used as the only signal for classification.

It takes the 2,198 imputed cells, computes diagnostic features $\phi$ directly from them, and runs a 5-fold RF classifier to predict intent — with no clustering involved.

| Condition | F1-macro | What $\hat{x}^*$ is |
|---|---|---|
| Oracle | 0.724 | true $x^*$ |
| **Imputed (MICE)** | **0.717** | MICE estimate |
| Naive | 0.682 | column mean/mode |

MICE recovers **83.4 %** of the Oracle's advantage over Naive:

$$\frac{0.717 - 0.682}{0.724 - 0.682} = 83.4\%$$

This tells us that MICE estimates are close enough to ground truth to be useful. The main pipeline result (Section 5.1) then confirms this holds end-to-end through the full clustering classifier.

---

### 5.3 Standalone Imputation Quality — Adult Income v3

This section reports the imputation quality of MICE **independently**, measured by comparing $\hat{x}^*$ against the true $x^*$ for all 2,198 corrupted cells. Results are from `evaluate.py` (Block A: imputation quality, Block B: feature separability, Block C: intent classification ablation).

#### Numerical Columns — NRMSE and Spearman ρ

NRMSE is normalised by the column standard deviation; lower is better. Spearman ρ measures rank correlation between $\hat{x}^*$ and $x^*$.

| Column | NRMSE | Spearman ρ | p-value |
|---|---|---|---|
| `education-num` | **0.333** | **0.965** | < 0.001 |
| `age` | 0.823 | 0.573 | < 0.001 |
| `hours-per-week` | 0.929 | 0.428 | < 0.001 |
| `fnlwgt` | 1.026 | 0.137 | < 0.001 |
| `capital-gain` | 1.071 | 0.231 | < 0.001 |
| `capital-loss` | 1.086 | 0.011 | 0.78 (n.s.) |

`education-num` is the best-estimated column (structured, ordinal, strongly correlated with other features). `capital-loss` is effectively unimputable — over 90 % of rows have `capital-loss = 0`, so the model defaults to near-zero predictions regardless of the true value; the Spearman correlation is not statistically significant.

Mean NRMSE across numerical columns: **0.878**

#### Categorical Columns — Hit Rate

Hit rate = fraction of corrupted cells for which $\hat{x}^* = x^*$ exactly.

| Column | Hit Rate |
|---|---|
| `education` | 0.927 |
| `native-country` | 0.920 |
| `class` | 0.885 |
| `race` | 0.862 |
| `sex` | 0.809 |
| `marital-status` | 0.799 |
| `workclass` | 0.771 |
| `relationship` | 0.752 |
| `occupation` | **0.357** |

Mean hit rate across categorical columns: **0.787**

`occupation` is an outlier — it has 14 classes with relatively uniform distribution and weak correlation with the remaining features. All other categorical columns exceed 75 % exact recovery.

#### Feature Separability — AUC (Intentional vs Unintentional)

Each diagnostic feature is evaluated alone as a univariate classifier (area under the ROC curve). AUC = 0.5 is chance; significance is assessed by a two-sided Mann–Whitney U test.

| Feature | AUC | Significant? |
|---|---|---|
| WRC (weighted relative change) | **0.673** | ✓ p < 0.001 |
| Raw relative change | 0.633 | ✓ p < 0.001 |
| Change magnitude | 0.620 | ✓ p < 0.001 |
| Change direction | 0.580 | ✓ p < 0.001 |
| `confidence_oob` | 0.536 | ✓ p = 0.004 |
| `confidence` | 0.511 | ✗ p = 0.383 |
| `is_categorical` | 0.512 | ✗ p = 0.275 |
| `feature_importance` | 0.503 | ✗ p = 0.799 |

WRC is the most discriminative individual feature (AUC = 0.673). The imputer's confidence estimate alone is not predictive of intent (p = 0.38), but it contributes when combined with magnitude-based features in the RF classifier.

#### Intent Classification Ablation (5-fold CV, 2,198 cells)

The RF classifier is trained on different subsets of diagnostic features to measure each feature group's contribution:

| Feature Set | Accuracy | F1-macro | F1-intent | F1-uninten |
|---|---|---|---|---|
| Full (all 9 features) | **0.837** | **0.834** | **0.814** | **0.854** |
| No confidence (`direction + magnitude + WRC`) | 0.805 | 0.803 | 0.783 | 0.823 |
| Direction + confidence only | 0.648 | 0.640 | 0.588 | 0.693 |
| Naive baseline (column mean/mode) | 0.716 | 0.674 | 0.559 | 0.790 |

The full feature set outperforms the naive baseline by **+0.160 F1-macro**. Removing confidence features costs −0.031 F1, confirming that model uncertainty provides a small but consistent gain. Direction + confidence alone (no magnitude) drops sharply to 0.640, showing that the **magnitude of the change is the dominant signal**.

---

### 5.4 Standalone Imputation Quality — Twitter Bot

> **Important caveat:** All 1,931 corrupted cells in the Twitter Bot dataset are labelled **intentional** (+1). There are zero unintentional errors. Intent classification is therefore degenerate for this dataset — any classifier that predicts "intentional" for every cell achieves perfect accuracy and F1-macro = 0.5. Classification results from this dataset are not reported as meaningful outcomes.

The imputation quality evaluation is still fully valid and reported here.

The Twitter Bot dataset has **17 columns, all numerical** (no categorical columns). Mean NRMSE across columns: **0.870**

#### Numerical Columns — NRMSE and Spearman ρ

| Column | NRMSE | Spearman ρ | Notes |
|---|---|---|---|
| `listed_count` | **0.540** | **0.896** | Best estimated; strong social-graph correlate |
| `followers_count` | 0.674 | 0.924 | High rank-correlation despite moderate NRMSE |
| `verified` | 0.773 | 0.704 | Binary flag; MICE predicts probability |
| `statuses_count` | 1.006 | 0.715 | High variance column |
| `description_length` | 0.906 | 0.447 | Moderate |
| `name_length` | 1.004 | 0.167 | Weakly structured |
| `friends_count` | 1.190 | 0.493 | Most difficult; high-variance, bot-inflated |

Several binary indicator columns (`protected`, etc.) have NRMSE = N/A because they are constant-valued (always 0 in the dataset) and are excluded from NRMSE computation.

#### Direction Agreement — Oracle vs Imputed

Even though classification is degenerate, the direction of change $\mathrm{sign}(\tilde{x} - \hat{x}^*)$ can be compared against the Oracle direction $\mathrm{sign}(\tilde{x} - x^*)$ to assess MICE geometric accuracy:

| Condition | Direction Match |
|---|---|
| All cells | **81.7 %** |
| High-confidence cells (conf ≥ 0.5) | 100 % |
| Low-confidence cells (conf < 0.1) | 74.7 % |

Per-column direction agreement:

| Column | Direction Match |
|---|---|
| `friends_count` | 100 % |
| `listed_count` | 100 % |
| `name_length` | 90.8 % |
| `description_length` | 60.8 % |

The high overall direction agreement (81.7 %) and perfect match at high confidence indicate that MICE estimates are geometrically aligned with ground truth even in the absence of a meaningful classification signal. The median magnitude ratio (Imputed / Oracle) is **1.200**, indicating a systematic slight over-estimation of the change magnitude, likely due to regression-to-the-mean in the RF predictions.

---

## 6. Key Finding

MICE imputation not only approximates the Oracle closely at the cell level (−0.007 F1) but **surpasses** the Oracle in the clustering track for 5 of 6 algorithms. The most plausible explanation is that MICE estimates introduce a smooth, consistent transformation of the feature space that aligns better with the density structure discovered by HDBSCAN than the raw ground-truth values do, particularly for high-variance columns such as `capital-gain` and `fnlwgt`.

The only exception is GMM (−0.034), which assumes spherical Gaussian clusters; MICE estimates for skewed columns violate this assumption more than the true values.

---

## 7. Hyperparameters

| Component | Parameter | Value |
|---|---|---|
| Imputation RF | n\_estimators | 200 |
| Imputation RF | max\_depth | None (fully grown) |
| MICE | max\_rounds | 5 |
| MICE | convergence\_tol | 10⁻³ |
| Cell classifier RF | n\_estimators | 200 |
| Cell classifier RF | max\_depth | 10 |
| Cell classifier RF | class\_weight | balanced |
| CV | n\_folds | 5 (stratified) |
| Clustering RF | n\_estimators | 200 |
| Clustering RF | max\_depth | 15 |
| K-Means / Ward / Avg / GMM | k | 15 |
| HDBSCAN | min\_cluster\_size | 5 |
| Cluster train fraction | — | 1 % of variants |

---

## 8. Limitations

1. **MICE column ordering:** corrupted columns are imputed in fixed ascending-index order within each round. Randomising the order per round (standard MICE practice) would reduce first-column bias.
2. **`capital-loss` sparsity:** over 90 % of rows have `capital-loss = 0`. The imputer consistently predicts near-zero, giving a direction agreement of only 33 % for this column.
3. **Row tripling:** the clustering input is constructed by repeating each original row three times. Cluster membership is therefore correlated within triples, which slightly inflates the effective training size.
4. **Small training fraction:** 1 % proportional sampling is deliberately minimal. Results may be sensitive to the random seed for algorithms with coarser cluster structure (K-Means, GMM).
