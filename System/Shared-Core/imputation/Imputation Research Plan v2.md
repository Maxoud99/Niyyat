# Research Plan v2: MICE-Based Correct Value Estimation for Intent Attribution
> *"Towards Identifying Intent of Data Errors"* — GuideAI 2025 Follow-up Research Plan
> 
> **Advisor:** Tenured Professor of Data Science and Machine Learning
> **Student:** PhD Candidate — Data Quality, Adversarial ML, Responsible AI
> **Date:** March 2, 2026
> **Revision:** v2 — Incorporates MICE (Perini & Nikolic, SIGMOD 2024), Uncertainty-Driven Imputation (Wang et al., SIGMOD 2024), and Influence Functions (Miao et al., VLDB 2022)

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Literature Grounding: Three Key Papers](#2-literature-grounding-three-key-papers)
3. [Core Insight: Why MICE, Not Independent Imputation](#3-core-insight-why-mice-not-independent-imputation)
4. [Modified MICE for Corrupted Data](#4-modified-mice-for-corrupted-data)
5. [Dual Confidence Estimation](#5-dual-confidence-estimation)
6. [Confidence-Weighted Feature Engineering](#6-confidence-weighted-feature-engineering)
7. [The Reliability Hierarchy](#7-the-reliability-hierarchy)
8. [Ablation Study Design](#8-ablation-study-design)
9. [Defending Against Skeptical Reviewers](#9-defending-against-skeptical-reviewers)
10. [Implementation Steps](#10-implementation-steps)
11. [Key Equations Summary](#11-key-equations-summary)
12. [Timeline](#12-timeline)

---

## 1. Motivation & Problem Statement

### The Core Gap

The intent attribution pipeline currently depends on a **latent variable** unavailable in deployment:

$$x^* = \text{the true correct value of the corrupted cell}$$

In development, this is read from `correct_records.csv`. In the real world — a bank flagging a suspicious `income` field — **there is no `correct_records.csv`.** The pipeline collapses.

### The Formal Problem

Given:
- A flagged erroneous cell $\tilde{x}_j$ in column $j$ of row $\mathbf{r}$
- All non-corrupted features in the same row: $\mathbf{r}_{\setminus j}$
- A dataset $\mathcal{D}$ of clean (non-flagged) rows
- Possibly **multiple corrupted cells** in the same row

Estimate:
$$\hat{x}^*_j = \mathbb{E}[x_j \mid \mathbf{r}_{\setminus j}]$$

Without access to ground truth, and use $\hat{x}^*_j$ to compute features for intent classification.

### Assumption
> **Perfect Detection Assumption:** We know *where* the error is (the cell is flagged), but not *what* the true value was. We need to estimate the direction and magnitude of corruption.

### Expected Dataset Contract

The system is **dataset-agnostic**. Any dataset that follows this contract can be used:

| Required Input | Format | Description |
|---|---|---|
| **Data file** | CSV with $N$ rows × $P$ columns | Mix of clean + corrupted rows. Columns can be numerical or categorical. |
| **Mask file** | CSV with same shape ($N \times P$) | Cell-level labels: `0` = clean, `1` = intentional error, `-1` = unintentional error |
| **Row labels** (optional) | CSV with $N$ rows × ($P$+1) columns | Same data + `is_erroneous` binary label per row |

**Derived quantities (computed at runtime, not hardcoded):**

| Symbol | Definition |
|---|---|
| $\mathcal{D}_{\text{clean}}$ | Rows where ALL mask values = 0 — **training data for imputer** |
| $\mathcal{D}_{\text{dirty}}$ | Rows where ANY mask value ≠ 0 — **need imputation** |
| $K_r$ | Number of corrupted cells in erroneous row $r$ |
| $\mathcal{C}_{\text{num}}$ | Set of numerical columns (auto-detected) |
| $\mathcal{C}_{\text{cat}}$ | Set of categorical columns (auto-detected) |

> **Critical observation:** When erroneous rows have **multiple corrupted cells** ($K_r > 1$), independent per-column imputation is insufficient — iterative MICE is necessary. The system must handle arbitrary $K_r$ values.

---

## 2. Literature Grounding: Three Key Papers

### Paper 1: Perini & Nikolic — "In-Database Data Imputation" (SIGMOD 2024)

**Core concept adopted:** MICE (Multivariate Imputation by Chained Equations) — iterative per-column imputation where each column's model benefits from previously imputed values in other columns.

**What we take:**
- Per-column model training on clean rows: $f_j(\mathbf{r}_{\setminus j}) \rightarrow x_j$
- Iterative round-robin imputation for rows with multiple corrupted cells
- Stochastic residual variance $\sigma^2$ as a principled uncertainty measure

**What we do NOT take:**
- In-database SQL implementation (irrelevant — we are not inside a DBMS)
- Linear regression / Gaussian discriminant analysis (we use RF for non-linear relationships)
- Cofactor ring computation (specific to their SQL optimization)

**Key difference from their work:** They impute `NULL` (missing) values. We impute **corrupted** values — the erroneous value $\tilde{x}_j$ exists but must be masked before imputation.

---

### Paper 2: Wang et al. — "NOMI: Missing Data Imputation with Uncertainty-Driven Network" (SIGMOD 2024)

**Core concept adopted:** Use the model's own uncertainty to improve downstream decisions. High-uncertainty imputations should have less influence.

**What we take:**
- The principle that uncertainty should **propagate** to downstream tasks — not just improve imputation
- Motivation for decomposing uncertainty into epistemic (not enough similar training data) vs. aleatoric (inherently noisy column)

**What we do NOT take:**
- The NNGPI neural architecture (we use RF)
- Their retrieval module (HNSW-based neighbor search)

**Key insight for our system:** Their uncertainty-based calibration loop down-weights unreliable imputations during the iterative process. We adopt this same principle — but instead of improving the imputation, we use uncertainty to **dampen diagnostic features** via the WRC formula.

---

### Paper 3: Miao et al. — "EDIT: Efficient and Effective Data Imputation with Influence Functions" (VLDB 2022)

**Core concept adopted:** Not all training rows contribute equally to a prediction. Measuring "data support" gives a second, independent confidence signal.

**What we take:**
- The idea that prediction reliability depends on **how many diverse training rows support it**
- Adapted to RF: use **out-of-bag (OOB) prediction accuracy** as a proxy for influence concentration
- OOB is native to RF and free — no Hessian computation needed

**What we do NOT take:**
- Direct influence function computation (fragile for non-parametric models — Basu et al., ICLR 2021 showed this)
- Their representative sample selection (RSS) module
- Their weighted loss function (designed for parametric models)

**Key insight for our system:** OOB prediction error gives us a second confidence signal — **was this row well-represented in training?** — independent of tree variance.

---

## 3. Core Insight: Why MICE, Not Independent Imputation

### The Multi-Corruption Problem

When erroneous rows have multiple corrupted cells ($K_r > 1$), imputing each cell independently uses other corrupted values as inputs. Consider a row with 3 corrupted cells in columns $A$, $B$, $C$:

```
True row:      [A=correct_a, B=correct_b, C=correct_c, D=clean_d, ...]
Corrupted row: [A=wrong_a,   B=wrong_b,   C=wrong_c,   D=clean_d, ...]
Mask:          [±1,          ±1,          ±1,          0,          ...]
```

**Independent imputation (naive):**
```
Impute A using B=wrong_b(WRONG), C=wrong_c(WRONG), D=clean_d → biased
Impute B using A=wrong_a(WRONG), C=wrong_c(WRONG), D=clean_d → biased
Impute C using A=wrong_a(WRONG), B=wrong_b(WRONG), D=clean_d → biased
```

**MICE iterative imputation:**
```
Round 1: Mask all 3 → impute each using initial guesses     → rough estimates
Round 2: Impute each using Round 1 estimates                 → better estimates
Round 3: Impute each using Round 2 estimates                 → converged
```

Each round, imputed values from other columns improve the inputs for the current column's model. This is exactly the MICE procedure from Paper 1, applied to our corrupted-data setting.

### When MICE is NOT needed

For rows with only 1 corrupted cell ($K_r = 1$): a single RF prediction suffices. No iteration needed — all other features in the row are clean.

---

## 4. Modified MICE for Corrupted Data

### Standard MICE vs. Our Modified MICE

| Aspect | Standard MICE (Paper 1) | Our Modified MICE |
|---|---|---|
| Input | Dataset with `NULL` cells | Dataset with **corrupted** cells |
| Which cells | All missing across dataset | Only **flagged** cells in erroneous rows |
| Original value | Does not exist | Exists ($\tilde{x}_j$) — must be **masked** before imputation |
| Training data | Observed (non-NULL) rows | Clean rows ($\mathcal{D}_{\text{clean}}$: all mask = 0) |
| Goal | Fill data for analysis | Estimate **reference point** for intent features |
| Post-imputation | Use imputed value | Compare $\tilde{x}_j$ against $\hat{x}^*_j$ |

### The Algorithm

```
INPUT:
  D     = dataset (N rows × P columns, mix of clean + corrupted)
  M     = masks   (N rows × P columns, values in {-1, 0, 1})

STEP 1: IDENTIFY TRAINING DATA AND COLUMN TYPES
  D_clean = {r ∈ D : ∀j, M[r][j] = 0}                    # all-clean rows
  D_dirty = {r ∈ D : ∃j, M[r][j] ≠ 0}                    # erroneous rows
  C_num   = {j : column j is numerical}                     # auto-detected
  C_cat   = {j : column j is categorical}                   # auto-detected

STEP 2: ENCODE CATEGORICAL FEATURES
  For each column j in C_cat:
    Fit LabelEncoder_j on D_clean[column j]                 # encode on clean data only

STEP 3: TRAIN PER-COLUMN MODELS
  For each column j in {1, ..., P}:
    X_train = D_clean[all columns except j]   (encoded)
    y_train = D_clean[column j]
    If j ∈ C_num:
      RF_j = RandomForestRegressor(n_estimators=100, oob_score=True)
    If j ∈ C_cat:
      RF_j = RandomForestClassifier(n_estimators=100, oob_score=True)
    RF_j.fit(X_train, y_train)

STEP 4: IMPUTE CORRUPTED CELLS
  For each row r in D_dirty:
    corrupted_cols = {j : M[r][j] ≠ 0}
    
    If |corrupted_cols| == 1:
      j = the single corrupted column
      x̂*_j = RF_j.predict(r[all columns except j])
      σ_j = std of per-tree predictions
    
    If |corrupted_cols| > 1:
      # MICE iterative loop
      Initialize: set corrupted cells to column median (num) / mode (cat)
      For round = 1 to MAX_ROUNDS (default 5):
        For each j in corrupted_cols:
          Mask column j → use current estimates for other corrupted cols
          x̂*_j = RF_j.predict(r[all columns except j])
        Check convergence: if estimates stable → break

STEP 5: EXTRACT CONFIDENCE
  For each imputed cell (r, j):
    σ_j^tree = std of individual tree predictions for (r, j)
    σ_j^OOB  = RF_j OOB error for column j
    c_j = 1 / (1 + σ_j^tree)

OUTPUT:
  For each corrupted cell: (x̂*_j, σ_j^tree, σ_j^OOB, c_j)
```

---

## 5. Dual Confidence Estimation

### Why Two Confidence Signals?

Inspired by Papers 2 and 3, we extract **two independent** measures of imputation reliability:

| Signal | Source | What It Measures | When It's High |
|---|---|---|---|
| $\sigma_j^{\text{tree}}$ | Variance of individual RF tree predictions | **Ensemble disagreement** — do the trees agree? | Row is unusual (outlier in feature space) |
| $\sigma_j^{\text{OOB}}$ | RF out-of-bag prediction error for column $j$ | **Column predictability** — can this column be predicted from others? | Column has weak correlations with other features |

### The Critical Distinction

$\sigma_j^{\text{tree}}$ is **row-specific** — it varies per prediction.
$\sigma_j^{\text{OOB}}$ is **column-specific** — it's the same for all rows in that column.

| Scenario | $\sigma_j^{\text{tree}}$ | $\sigma_j^{\text{OOB}}$ | Interpretation |
|---|---|---|---|
| Low | Low | Reliable imputation — trust all features | |
| High | Low | This specific row is unusual — but the column is generally predictable | |
| Low | High | The column is inherently hard to predict — but this row is typical | |
| High | High | Unreliable imputation — suppress magnitude-based features | |

### Why This Matters for WRC

If we only used $\sigma_j^{\text{tree}}$, a column that is inherently noisy (e.g., a sparse financial column — often 0, sometimes very high) would have high $\sigma_j^{\text{tree}}$ for ALL rows. The WRC formula would suppress the feature globally — even for obvious gain-seeking cases.

By separating the two signals, we can make a smarter decision:
- If $\sigma_j^{\text{OOB}}$ is high → the column is just hard to predict → lower the weight of WRC but don't zero it
- If $\sigma_j^{\text{tree}}$ is high AND $\sigma_j^{\text{OOB}}$ is low → this specific prediction is fragile → suppress WRC aggressively

This distinction is motivated by Paper 2's epistemic vs. aleatoric uncertainty decomposition, adapted to RF.

---

## 6. Confidence-Weighted Feature Engineering

### The Diagnostic Tuple

For each flagged erroneous cell, compute:

$$\phi(\tilde{x}_j, \mathbf{r}) = \left[d_j, \; m_j, \; \text{WRC}_j, \; c_j, \; c_j^{\text{OOB}}, \; \text{FI}_j, \; \text{incentive}_j\right]$$

| Symbol | Name | Formula |
|---|---|---|
| $d_j$ | Change Direction | $\text{sign}(\tilde{x}_j - \hat{x}^*_j) \in \{-1, +1\}$ |
| $m_j$ | Change Magnitude | $\|\tilde{x}_j - \hat{x}^*_j\|$ |
| $\text{WRC}_j$ | Weighted Relative Change | $\frac{m_j}{\|\hat{x}^*_j\| \cdot (1 + \alpha \cdot \sigma_j^{\text{tree}})}$ |
| $c_j$ | Per-Cell Confidence | $\frac{1}{1 + \sigma_j^{\text{tree}}} \in (0, 1]$ |
| $c_j^{\text{OOB}}$ | Column Predictability | $1 - \text{OOB\_error}_j \in [0, 1]$ |
| $\text{FI}_j$ | Feature Importance | From RF model metadata |
| $\text{incentive}_j$ | Domain-Specific Gain Direction | From dataset schema |

### Handling Categorical Features

For categorical columns ($j \in \mathcal{C}_{\text{cat}}$):
- $d_j$: Not applicable as a sign. Instead: `changed = 1` if $\tilde{x}_j \neq \hat{x}^*_j$, else `0`
- $m_j$: Use semantic distance (e.g., edit distance for typos, ordinal distance for ordered categories)
- $\text{WRC}_j$: Not applicable. Use **prediction probability** from RF classifier as confidence instead

---

## 7. The Reliability Hierarchy

### Feature Robustness Under Imputation Error

| Feature | Formula | Robustness | Reason |
|---|---|---|---|
| `change_direction` | $d = \text{sign}(\tilde{x} - \hat{x}^*)$ | ⬆ **High** | Only flips if imputation error > corruption magnitude |
| `change_magnitude` | $m = \|\tilde{x} - \hat{x}^*\|$ | ⬆ **Medium** | Off by exactly $\|x^* - \hat{x}^*\|$ — linear bias |
| `relative_change` | $\frac{\|\tilde{x} - \hat{x}^*\|}{\|\hat{x}^*\|}$ | ⬇ **Low** | Error in both numerator AND denominator |

### Illustrative Example

Suppose for some numerical column: true value $x^* = 100{,}000$. Corrupted to $\tilde{x} = 105{,}000$ (small typo). Imputation predicts $\hat{x}^* = 40{,}000$ (the row is unusual — far from the training distribution).

| Metric | Oracle ($x^*$) | Imputed ($\hat{x}^*$) | Effect |
|---|---|---|---|
| Direction | Inflated ✓ | Inflated ✓ | **Safe** — direction preserved |
| Magnitude | 5,000 (typo) | 65,000 (looks strategic) | **Biased** — but WRC dampens this |
| Relative Change | 0.05 → typo | 1.625 → looks strategic | **Catastrophic** — without WRC |
| WRC ($\sigma = 30K$) | N/A | 0.052 → typo-like | **Rescued** — confidence dampening works |

> The WRC formula rescues the relative change from catastrophic failure by dampening with $\sigma_j$.

---

## 8. Ablation Study Design

### Experiment A: Oracle vs. Imputation Degradation

| Pipeline | Source of $x^*$ | Features |
|---|---|---|
| **Oracle** | `correct_records.csv` (ground truth) | Exact $d_j, m_j, \text{RC}_j$ |
| **Imputed** | MICE + RF estimate | $d_j, m_j, \text{WRC}_j, c_j, c_j^{\text{OOB}}$ |

**Metrics:** Per-class F1, confusion matrix, stratified F1 by confidence level.

### Experiment B: Feature Ablation

| Config | Features | Purpose |
|---|---|---|
| **Full** | $d_j + m_j + \text{WRC}_j + c_j + c_j^{\text{OOB}}$ | Proposed method |
| **Direction-only** | $d_j + c_j$ | Lower bound — most robust features |
| **No Confidence** | $d_j + m_j + \text{raw RC}$ | Shows value of confidence weighting |
| **Oracle** | Exact $d_j + m_j + \text{RC}_j$ | Upper bound |
| **Naive (no imputation)** | $\tilde{x}_j$ + column statistics (mean, std, quantiles) | **Critical baseline — see Section 9** |

### Experiment C: Imputation Model Comparison

| Model | Confidence Source | Notes |
|---|---|---|
| **Random Forest (MICE)** | Tree variance + OOB | Our primary approach |
| **Quantile Regression Forest** | Prediction intervals $[q_{0.05}, q_{0.95}]$ | Full conditional distribution |
| **Bayesian Ridge Regression** | Posterior variance | Analytical confidence |
| **KNN Imputation** | Neighbor distance variance | Instance-based |
| **MissForest** | OOB error per feature | State-of-the-art imputation baseline |

### Experiment D: MICE Convergence Study

| Configuration | Purpose |
|---|---|
| 1 round (no iteration) | Baseline — independent imputation |
| 3 rounds | Typical convergence |
| 5 rounds | Paper 1 default |
| 10 rounds | Overkill check |

Measure: imputation RMSE against oracle AND downstream intent F1 at each round. **If 1 round ≈ 5 rounds → MICE iteration is unnecessary. If 5 rounds >> 1 round → MICE is essential.**

---

## 9. Defending Against Skeptical Reviewers

### Anticipated Concern 1: "Imputation error propagates and corrupts everything"

**Defense:** 
- The reliability hierarchy (Section 7) shows $d_j$ is robust even under large imputation error
- The WRC formula (Section 6) self-suppresses when confidence is low
- Experiment A empirically measures the degradation
- The dual confidence signal (Section 5) provides a principled way to know *when* to trust the estimate

### Anticipated Concern 2: "The circularity problem — you need clean data to find clean data"

**Defense:**
- We explicitly state the Perfect Detection Assumption — errors are pre-flagged by existing detectors
- This decouples detection from attribution — standard in the data cleaning literature
- The system requires a sufficient proportion of clean rows in $\mathcal{D}_{\text{clean}}$ for imputer training — we verify this holds for each dataset

### Anticipated Concern 3: "The WRC formula is just regularized relative change — not novel"

**Defense:**
- The formula is intentionally simple — the novelty is the **semantic connection** between imputation uncertainty and intent discriminability
- No prior work has ever used $\sigma_j$ to modulate features for **intent** classification
- The dual confidence (tree + OOB) gives richer information than standard regularization

### Anticipated Concern 4: "Why not just use the corrupted value directly? No imputation needed."

**Defense — THIS IS THE MOST CRITICAL BASELINE:**
- We MUST include a **Naive Baseline** in Experiment B that uses $\tilde{x}_j$ directly with column statistics
- If the naive baseline works just as well → our imputation is unnecessary → paper is rejected
- We expect the naive baseline to **fail** because without a reference point, you cannot compute direction or magnitude of corruption — only that the value exists
- But we must **prove this empirically**, not just argue it

### Anticipated Concern 5: "Evaluation is self-referential"

**Defense:**
- Two-level ground truth: (1) synthetic data with known intent labels, (2) oracle vs. imputed comparison
- Experiment A measures degradation from losing $x^*$
- Main evaluation uses ground truth mask labels (`1` = intentional, `-1` = unintentional)

---

## 10. Implementation Steps

### Week 1: Build Imputation Module

- [ ] Create `imputation_estimator.py` in `src/attribution/`
- [ ] Auto-detect numerical vs. categorical columns from the data
- [ ] Encode categorical features (label encoding fitted on clean rows only)
- [ ] Train per-column RF models on $\mathcal{D}_{\text{clean}}$
- [ ] Implement single-cell imputation (for rows with $K_r = 1$)
- [ ] Implement MICE iterative loop (for rows with $K_r > 1$)
- [ ] Extract per-tree predictions → $\sigma_j^{\text{tree}}$
- [ ] Extract OOB scores → $\sigma_j^{\text{OOB}}$
- [ ] Unit test: verify imputation accuracy on held-out clean rows

### Week 2: Build Diagnostic Feature Module

- [ ] Create `diagnostic_features.py` in `src/attribution/`
- [ ] Compute diagnostic tuple for all corrupted cells in $\mathcal{D}_{\text{dirty}}$
- [ ] Handle numerical vs. categorical features separately (auto-detected)
- [ ] Implement WRC formula with dual confidence
- [ ] Validate: compare imputed features against oracle features for correctness

### Week 3: Run Ablation Studies

- [ ] Run **Experiment A**: Oracle vs. Imputed pipeline on same data
- [ ] Run **Experiment B**: Feature ablation including **naive baseline**
- [ ] Run **Experiment C**: Imputation model comparison
- [ ] Run **Experiment D**: MICE convergence study
- [ ] Generate confusion matrices, per-class F1, confidence-stratified metrics

### Week 4: Propagate to LLM Baselines

- [ ] Modify LLM prompts to include $\hat{x}^*_j$ and $\sigma_j$ as context
- [ ] Add confidence language to prompts:
  ```
  "The estimated correct value is {x_hat} (confidence: {high/medium/low}).
   The observed value {x_tilde} is {higher/lower} than expected by {magnitude}."
  ```
- [ ] Compare LLM performance: with oracle $x^*$ vs. with imputed $\hat{x}^*$ vs. without any reference

### Week 5: Write Up

- [ ] Draft paper section: *"Estimating the Correct Value Without Ground Truth"*
- [ ] Position contribution clearly: not a new imputation method, but a new **use** of imputation uncertainty for intent classification
- [ ] Cite Papers 1-3 appropriately: what we adopt, what we adapt, what we extend

---

## 11. Key Equations Summary

### Imputation Estimate (MICE with RF)
$$\hat{x}^*_j = \frac{1}{T}\sum_{t=1}^T f_t(\mathbf{r}_{\setminus j})$$

### Dual Confidence
$$\sigma_j^{\text{tree}} = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} \left(f_t(\mathbf{r}_{\setminus j}) - \hat{x}^*_j\right)^2}$$
$$\sigma_j^{\text{OOB}} = \text{OOB prediction error of } f_j$$
$$c_j = \frac{1}{1 + \sigma_j^{\text{tree}}} \in (0, 1]$$

### Change Direction (Most Robust)
$$d_j = \text{sign}(\tilde{x}_j - \hat{x}^*_j) \in \{-1, +1\}$$

### Change Magnitude (Medium Robust)
$$m_j = |\tilde{x}_j - \hat{x}^*_j|$$

### Confidence-Weighted Relative Change (Stabilized)
$$\text{WRC}_j = \frac{|\tilde{x}_j - \hat{x}^*_j|}{|\hat{x}^*_j| \cdot (1 + \alpha \cdot \sigma_j^{\text{tree}})}, \quad \alpha = 1.0$$

### Full Diagnostic Tuple
$$\phi(\tilde{x}_j, \mathbf{r}) = \left[d_j, \; m_j, \; \text{WRC}_j, \; c_j, \; c_j^{\text{OOB}}, \; \text{FI}_j, \; \text{incentive}_j\right]$$

---

## 12. Timeline

| Week | Task | Deliverable |
|---|---|---|
| **Week 1** | Build `imputation_estimator.py` (MICE + RF + dual confidence) | Working imputation module |
| **Week 2** | Build `diagnostic_features.py` (diagnostic tuple + WRC) | Feature engineering pipeline |
| **Week 3** | Run Experiments A, B, C, D | Ablation results + confusion matrices |
| **Week 4** | Propagate to LLM baselines | Updated prompts + comparative results |
| **Week 5** | Write up | Draft paper section |

---

## Notes

- **The contribution is NOT a new imputation method.** It is the first framework that transforms imputation uncertainty into discriminative features for classifying the intent behind data errors — making intent attribution deployable without ground truth.
- **The naive baseline (Experiment B) is non-negotiable.** If it performs well, the entire imputation approach is unjustified. Run it first.
- **The MICE convergence study (Experiment D) is important.** If 1 round ≈ 5 rounds, the multi-corruption problem is less severe than expected and the method simplifies.
- **Framing for the paper:** We adopt MICE's iterative imputation (Paper 1), adapt uncertainty-driven calibration for downstream intent features (Paper 2), and use OOB-based data support as a second confidence signal (Paper 3). The novel contribution is the diagnostic tuple and the WRC formula that bridges imputation quality to intent attribution.

---

*Research Plan v2 authored during advising session — March 2, 2026*
*Based on discussion of: Perini & Nikolic (SIGMOD 2024), Wang et al. (SIGMOD 2024), Miao et al. (VLDB 2022)*
