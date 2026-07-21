# Research Plan: Imputation-Based Correct Value Estimation for Intent Attribution
> *"Towards Identifying Intent of Data Errors"* — GuideAI 2025 Follow-up Research Plan
> 
> **Advisor:** Tenured Professor of Data Science and Machine Learning
> **Student:** PhD Candidate — Data Quality, Adversarial ML, Responsible AI
> **Date:** February 26, 2026

---

## Table of Contents

1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Core Insight: The Reliability Hierarchy](#2-core-insight-the-reliability-hierarchy)
3. [Core Algorithm](#3-core-algorithm)
4. [Confidence-Weighted Feature Engineering](#4-confidence-weighted-feature-engineering)
5. [Ablation Study Design](#5-ablation-study-design)
6. [Implementation Steps](#6-implementation-steps)
7. [Papers to Explore](#7-papers-to-explore)
8. [Key Equations Summary](#8-key-equations-summary)
9. [Timeline](#9-timeline)

---

## 1. Motivation & Problem Statement

### The Core Gap

Your entire intent attribution pipeline — both the **Clustering + Classification** approach (`clustering-organized/`) and the **LLM-based baselines** (`llm-based/`) — currently depends on a **latent variable** that does not exist in deployment:

$$x^* = \text{the true correct value of the corrupted cell}$$

In development, you read this directly from `correct_records.csv`:

```python
# Current implementation — NOT deployable
original_value = correct_df.iloc[original_record_idx][feature]
magnitude = abs(new_numeric - orig_numeric)
```

In the real world — say a bank flagging a suspicious `income` field in a loan application — **there is no `correct_records.csv`**. Your pipeline collapses.

### The Formal Problem

Given:
- A flagged erroneous cell $\tilde{x}_j$ in column $j$ of row $\mathbf{r}$
- All non-corrupted features in the same row: $\mathbf{r}_{\setminus j}$
- A dataset $\mathcal{D}$ of clean (non-flagged) rows

Estimate:
$$\hat{x}^*_j = \mathbb{E}[x_j \mid \mathbf{r}_{\setminus j}]$$

Without access to ground truth, and use $\hat{x}^*_j$ to compute features for intent classification.

### Assumption
> **Perfect Detection Assumption:** We know *where* the error is (the cell is flagged), but not *what* the true value was. We need to estimate the direction and magnitude of corruption.

---

## 2. Core Insight: The Reliability Hierarchy

Not all features degrade equally under imputation error. This is a **key contribution** of this work.

### Feature Robustness Analysis

| Feature | Formula | Sensitivity to Imputation Error | Reason |
|---|---|---|---|
| `change_direction` | $d = \text{sign}(\tilde{x} - \hat{x}^*)$ | ⬆ **High Robustness** | Only flips if imputation error > corruption magnitude |
| `change_magnitude` | $m = \|\tilde{x} - \hat{x}^*\|$ | ⬆ **Medium Robustness** | Off by exactly $\|x^* - \hat{x}^*\|$ — linear bias |
| `relative_change` | $\frac{\|\tilde{x} - \hat{x}^*\|}{\|\hat{x}^*\|}$ | ⬇ **Low Robustness** | Error in both numerator AND denominator — catastrophic for outliers |

### Illustrative Example

Alice is a loan applicant. True income $x^* = \$100,000$. Corrupted to $\tilde{x} = \$105,000$ (small typo). Imputation predicts $\hat{x}^* = \$40,000$ (Alice is unusual).

| Metric | Using $x^*$ (oracle) | Using $\hat{x}^*$ (imputed) | Conclusion |
|---|---|---|---|
| Direction | Inflated ✓ | Inflated ✓ | **Safe** |
| Magnitude | \$5,000 (typo-like) | \$65,000 (manipulation-like) | **Biased** |
| Relative Change | 0.05 → typo | 1.625 → strategic | **Catastrophic** |

> **Insight:** When imputation confidence is low, `relative_change` can flip your intent classification entirely. This must be modeled explicitly.

---

## 3. Core Algorithm

### Step 1: Train Imputation Model on Clean Rows

For each column $j$ that may contain errors, train a separate model:

$$\hat{x}^*_j = f_j(\mathbf{r}_{\setminus j}; \theta_j)$$

Where $f_j$ is:
- **Random Forest Regressor** — for numerical features (income, age, credit score)
- **Random Forest Classifier** — for categorical features (job title, zip code, education)

**Training data:** All rows in $\mathcal{D}$ where column $j$ is **not flagged** as erroneous.

```python
# Pseudocode
clean_rows = dataset[dataset['col_j_flagged'] == False]
X_train = clean_rows.drop(columns=['col_j'])
y_train = clean_rows['col_j']

imputer = RandomForestRegressor(n_estimators=100)
imputer.fit(X_train, y_train)
```

### Step 2: Extract Point Estimate AND Confidence

Do not just use the mean prediction. Extract individual tree predictions:

$$\hat{x}^*_j = \frac{1}{T} \sum_{t=1}^{T} f_t(\mathbf{r}_{\setminus j})$$

$$\sigma_j = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} \left(f_t(\mathbf{r}_{\setminus j}) - \hat{x}^*_j\right)^2}$$

```python
# Extract per-tree predictions
tree_predictions = np.array([
    tree.predict(alice_features)[0] 
    for tree in imputer.estimators_
])

x_hat = tree_predictions.mean()
sigma = tree_predictions.std()
confidence = 1 / (1 + sigma)
```

### Step 3: Interpret Confidence

| $\sigma_j$ Value | Interpretation | Action |
|---|---|---|
| Small | Alice's peers are homogeneous. Estimate is trustworthy. | Use all features |
| Large | Alice is a true outlier. Peers are diverse. Estimate is fragile. | Downweight `relative_change` |

---

## 4. Confidence-Weighted Feature Engineering

### The Diagnostic Tuple

For each flagged erroneous cell, compute the **Diagnostic Tuple**:

$$\phi(\tilde{x}_j, \mathbf{r}) = \left[d_j, \; m_j, \; \text{WRC}_j, \; c_j, \; \text{FI}_j, \; \text{incentive}_j\right]$$

Where:

| Symbol | Name | Formula |
|---|---|---|
| $d_j$ | Change Direction | $\text{sign}(\tilde{x}_j - \hat{x}^*_j) \in \{-1, +1\}$ |
| $m_j$ | Change Magnitude | $\|\tilde{x}_j - \hat{x}^*_j\|$ |
| $\text{WRC}_j$ | Weighted Relative Change | $\frac{m_j}{\|\hat{x}^*_j\| \cdot (1 + \alpha \cdot \sigma_j)}$ |
| $c_j$ | Confidence Score | $\frac{1}{1 + \sigma_j} \in (0, 1]$ |
| $\text{FI}_j$ | Feature Importance of column $j$ on target | From model metadata |
| $\text{incentive}_j$ | Domain-specific gain direction | From dataset schema |

### The Key Innovation: Confidence-Weighted Relative Change

$$\text{WRC}_j = \frac{|\tilde{x}_j - \hat{x}^*_j|}{|\hat{x}^*_j| \cdot (1 + \alpha \cdot \sigma_j)}$$

- When $\sigma_j \to 0$ (high confidence): $\text{WRC}_j \approx \text{relative\_change}$ (standard formula)
- When $\sigma_j \to \infty$ (low confidence): $\text{WRC}_j \to 0$ (feature self-suppresses)

The hyperparameter $\alpha$ controls dampening aggressiveness. **Start with $\alpha = 1.0$.**

> **This is a novel contribution:** No existing work in intent attribution explicitly models the trustworthiness of the estimated correct value as a feature.

---

## 5. Ablation Study Design

### Experiment A: Oracle vs. Imputation Degradation

The fundamental validation experiment.

| Pipeline | Source of $x^*$ | Features |
|---|---|---|
| **Oracle Pipeline** | `correct_records.csv` (ground truth) | Exact values |
| **Imputed Pipeline** | Random Forest estimate | $\hat{x}^*_j$ + confidence-weighted features |

**Procedure:**
1. Run both pipelines on the **same dataset**
2. Collect $Y_{\text{oracle}}$ and $Y_{\text{imputed}}$ — intent predictions for all flagged cells
3. Treat $Y_{\text{oracle}}$ as "ground truth" for this comparison

**Metrics:**

| Metric | Purpose |
|---|---|
| Per-class F1 between $Y_{\text{oracle}}$ and $Y_{\text{imputed}}$ | Measures **how much** intent attribution quality is lost |
| Confusion matrix between $Y_{\text{oracle}}$ and $Y_{\text{imputed}}$ | Reveals **which classes** get confused with which |
| Stratified F1 by confidence level (high $c$ vs. low $c$) | Validates that confidence score is meaningful |

**Expected Finding:** The confusion matrix will show `gain-seeking` ↔ `typo` as the most frequent confusion pair, because both magnitude and relative change are most critical for separating these two classes — and these are exactly the features most sensitive to imputation error.

---

### Experiment B: Feature Ablation

| Configuration | Features Used | Purpose |
|---|---|---|
| **Full** | $d_j + m_j + \text{WRC}_j + c_j$ | Baseline of proposed method |
| **Direction-only** | $d_j + c_j$ | Lower bound — most robust features only |
| **No Confidence** | $d_j + m_j + \text{raw relative change}$ | Shows value of confidence weighting |
| **Oracle** | Exact $d_j + m_j + \text{relative\_change}$ | Upper bound |

Compare intent classification F1 across configurations to quantify the **value of confidence weighting**.

---

### Experiment C: Imputation Model Comparison

| Model | Confidence Source | Notes |
|---|---|---|
| **Random Forest** | Tree variance $\sigma_j$ | Recommended starting point |
| **Quantile Regression Forest** | Prediction intervals $[q_{0.05}, q_{0.95}]$ | Full conditional distribution |
| **Bayesian Ridge Regression** | Posterior variance | Analytical confidence — no ensemble needed |
| **KNN Imputation** | Neighbor distance variance | Instance-based — interpretable |
| **MissForest** | OOB error per feature | State-of-the-art imputation baseline |

---

## 6. Implementation Steps

### Week 1: Build Imputation Module

- [ ] Create `imputation_estimator.py` in `src/attribution/`
- [ ] Train per-column Random Forest imputers on clean rows
- [ ] Extract per-tree predictions and compute $\hat{x}^*_j$ and $\sigma_j$
- [ ] Unit test: verify imputation accuracy on held-out clean rows

### Week 2: Modify Feature Engineering

- [ ] Modify `cell_level_attributor.py` — replace `correct_df.iloc[...]` lookups with $\hat{x}^*_j$
- [ ] Add confidence-weighted features: $\text{WRC}_j$ and $c_j$ to feature vector
- [ ] Add feature reliability flags based on confidence threshold $\tau$

### Week 3: Run Ablation Studies

- [ ] Run **Experiment A**: Oracle vs. Imputation pipelines on same dataset
- [ ] Generate per-class F1 scores and confusion matrix
- [ ] Run **Experiment B**: Feature ablation configurations
- [ ] Run **Experiment C**: Imputation model comparison

### Week 4: Propagate to LLM Baselines

- [ ] Modify prompts in `llm-based/` pipelines to include $\hat{x}^*_j$ and $\sigma_j$
- [ ] Add confidence context to prompts:
  ```
  "The estimated correct value is {x_hat} with uncertainty {sigma}. 
   The model is {'confident' if sigma < tau else 'uncertain'} about this estimate."
  ```
- [ ] Compare LLM performance with and without imputation context

---

## 7. Papers to Explore

### Core Imputation Methods

| Paper | Venue | Relevance |
|---|---|---|
| Stekhoven & Bühlmann (2012) — *"MissForest: Non-parametric missing value imputation using Random Forest"* | Bioinformatics | **Direct implementation reference** — RF-based imputation |
| Muzellec et al. (2020) — *"Missing Data Imputation using Optimal Transport"* | NeurIPS | Alternative imputation with distributional guarantees |
| Meinshausen (2006) — *"Quantile Regression Forests"* | JMLR | Prediction intervals from RF — full conditional distribution |

### Data Cleaning & Repair

| Paper | Venue | Relevance |
|---|---|---|
| Rekatsinas et al. (2017) — *"HoloClean: Holistic Data Repairs with Probabilistic Inference"* | VLDB | **Closest related work** — probabilistic framework for value repair |
| Ilyas et al. (2019) — *"Data Cleaning: Overview and Emerging Challenges"* | SIGMOD | Survey — positions your work in the broader landscape |
| Krishnan et al. (2016) — *"ActiveClean: Interactive Data Cleaning for Statistical Models"* | VLDB | Confidence-based cleaning with convergence guarantees |

### Adversarial ML & Intent

| Paper | Venue | Relevance |
|---|---|---|
| Koh & Liang (2017) — *"Understanding Black-box Predictions via Influence Functions"* | ICML | Tracing data point impact — relevant to your causal impact heuristic |
| Biggio et al. (2013) — *"Evasion Attacks against Machine Learning at Test Time"* | ECML | Adversarial manipulation patterns — validates your threat model |
| Barocas & Hardt (2016) — *"Fairness in Machine Learning"* | NIPS Tutorial | Fairness masking manipulation — theoretical grounding |

### Uncertainty Quantification

| Paper | Venue | Relevance |
|---|---|---|
| Lakshminarayanan et al. (2017) — *"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"* | NeurIPS | Ensemble-based uncertainty — generalizes your RF confidence approach |
| Gal & Ghahramani (2016) — *"Dropout as a Bayesian Approximation"* | ICML | Alternative confidence estimation without ensemble overhead |

---

## 8. Key Equations Summary

### Imputation Estimate
$$\hat{x}^*_j = \mathbb{E}[x_j \mid \mathbf{r}_{\setminus j}] \approx \frac{1}{T}\sum_{t=1}^T f_t(\mathbf{r}_{\setminus j})$$

### Imputation Confidence
$$\sigma_j = \sqrt{\frac{1}{T-1} \sum_{t=1}^{T} \left(f_t(\mathbf{r}_{\setminus j}) - \hat{x}^*_j\right)^2}, \quad c_j = \frac{1}{1 + \sigma_j}$$

### Change Direction (Most Robust)
$$d_j = \text{sign}(\tilde{x}_j - \hat{x}^*_j) \in \{-1, +1\}$$

### Change Magnitude (Medium Robust)
$$m_j = |\tilde{x}_j - \hat{x}^*_j|$$

### Confidence-Weighted Relative Change (Stabilized)
$$\text{WRC}_j = \frac{|\tilde{x}_j - \hat{x}^*_j|}{|\hat{x}^*_j| \cdot (1 + \alpha \cdot \sigma_j)}, \quad \alpha = 1.0$$

### Full Diagnostic Tuple
$$\phi(\tilde{x}_j, \mathbf{r}) = \left[d_j, \; m_j, \; \text{WRC}_j, \; c_j, \; \text{FI}_j, \; \text{incentive}_j\right]$$

### Feature Reliability Decision Rule
$$\text{use\_feature}(k) = \begin{cases} \text{True} & \text{if } k = d_j \text{ (always)} \\ \text{True} & \text{if } k = m_j \text{ and } \sigma_j < \tau \\ \text{True} & \text{if } k = \text{WRC}_j \text{ and } \sigma_j < \tau \\ \text{False} & \text{otherwise} \end{cases}$$

---

## 9. Timeline

| Week | Task | Deliverable |
|---|---|---|
| **Week 1** | Build `imputation_estimator.py` | Working imputation module with confidence scores |
| **Week 2** | Modify `cell_level_attributor.py` | Updated feature engineering pipeline |
| **Week 3** | Run Experiments A, B, C | Ablation study results + confusion matrices |
| **Week 4** | Propagate to LLM baselines | Updated LLM prompts + comparative results |
| **Week 5** | Write up | Draft section for paper: *"Estimating the Correct Value Without Ground Truth"* |

---

## Notes

- The **confidence-weighted feature engineering** is a standalone novel contribution. Frame it explicitly in the paper.
- The **Oracle vs. Imputation ablation** is your key experiment. Run it first — it tells you whether this entire approach is viable before investing more time.
- When $c_j$ is low (high uncertainty), your system should **abstain or flag** rather than make a confident but wrong intent prediction. Consider adding an "uncertain" output class.

---

*Research Plan authored during advising session — February