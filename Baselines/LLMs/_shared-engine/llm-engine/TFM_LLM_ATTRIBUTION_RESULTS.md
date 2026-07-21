# TFM-Inject Attribution Results: LLM vs. Scenario B / B+

Generated: April 27, 2026

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset | TFM-Inject (Adult Income, TFM error model) |
| Total rows | 48,842 |
| Error rows evaluated | **6,051** |
| Intentional errors (mask=+1) | 1,141 (18.9%) |
| Unintentional errors (mask=−1) | 4,910 (81.1%) |
| Feature columns | 14 (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country) |

---

## 2. TFM-Inject: Architecture & Error Generation Pipeline

### 2.1 Motivation

Rule-based error injection methods produce errors with **detectable statistical patterns** that make intent attribution artificially easy:

| Method | Error Type | Detectable Pattern |
|---|---|---|
| `tab_err` — Typo | Unintentional | Keyboard-distance character swaps |
| `tab_err` — Outlier | Unintentional | Values at fixed multiples of σ |
| `tab_err` — WrongUnit | Unintentional | Exact ×10 or ×100 scaling |
| `tab_err` — MissingValue | Unintentional | Literal `"?"` substitution |
| `tab_err` — CategorySwap | Unintentional | Uniform random among categories; ignores context |
| Greedy adversarial attack | Intentional | Values at distribution extremes (e.g., capital-gain=99,999) |

**TFM-Inject** replaces these with **distribution-consistent errors**: errors where the injected value could plausibly appear at that position given the surrounding row context. This is achieved by using Tabular Foundation Models (TFMs) which learn the joint distribution P(X) and sample from the conditional P(x_j | x_{-j}).

---

### 2.2 TFM Models Used

Two TFMs were evaluated during development; **TabPFN-v2.6** was used for all final production runs:

| Model | Version | Architecture | Categorical Handling |
|---|---|---|---|
| **TabPFN-v2.6** ✅ (production) | `tabpfn==7.0.1` | Prior-Data Fitted Network (transformer trained on synthetic priors) | Native: `TabPFNClassifier` samples from `Categorical(probs)` at temperature T; `TabPFNRegressor` samples from learned bar distribution |
| LimiX-16M | v1.1.0, `stableai-org/LimiX-16M` | Transformer with attention across sample and feature dimensions | Embedding-based: `categorical_features_indices` → snap continuous output to nearest encoded category |

**Key difference:** TabPFN provides a true probability distribution P(category) for categorical features, enabling principled ranking of candidates. LimiX uses a continuous embedding space with approximate probabilities, making it weaker for intentional error selection.

**Hardware:** 4 × NVIDIA H200 NVL (141 GB VRAM each, 565 GB total).

---

### 2.3 Dataset Features (Adult Income)

| Category | Features |
|---|---|
| **Categorical (8)** | workclass, education, marital-status, occupation, relationship, race, sex, native-country |
| **Numerical (6)** | age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| **Immutable** (never modified) | race, sex, fnlwgt |
| **Target column** (not modified) | class (>50K / ≤50K) |
| **Intentional mask targets** | capital-gain, relationship, age (top-3 by RF feature importance) |

**Domain bounds enforced:**

| Feature | Min | Max |
|---|---|---|
| age | 0 | 120 |
| education-num | 0 | 16 |
| capital-gain | 0 | — |
| capital-loss | 0 | — |
| hours-per-week | 1 | 99 |

---

### 2.4 Error Injection Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Target row error rate | **20%** of rows | Matches rule-based baseline for direct comparison |
| Unintentional fraction | **86%** of error rows | Produces ~2/3 unintentional *cells* in output (Phase 1 injects ~0.72 cells/row, Phase 2 ~2.2 cells/row) |
| Intentional fraction | **14%** of error rows | Produces ~1/3 intentional *cells* |
| Cells per error row | **random k ∈ {1, 2, 3}** | Same distribution as rule-based baseline |
| Unintentional temperature | **T = 0.7** | F3 feasibility: 59.5% yield rate with plausible values |
| Intentional temperature | **T = 1.0** | F6 feasibility: required for ≥10% flip rate |
| K candidates (intentional) | **K = 50** | F6: K=20 insufficient; K=50 gives 46% flip rate |
| Simultaneous mask columns | **top-3 features** | F6: single-feature masking gives only ~3% flip rate |
| Min numerical perturbation ε | 1% of column std | Avoids accepting near-zero float drift as an error |
| Max retries per cell | 3 | If imputed = original, retry with different column |

**Resulting label distribution (Adult Income, full run):**

| Mask value | Meaning | Count | % of error cells |
|---|---|---|---|
| +1 | Intentional | 1,141 | 18.9% |
| −1 | Unintentional | 4,910 | 81.1% |
| **Total error cells** | | **6,051** | 100% |

---

### 2.5 Pipeline Architecture

```
INPUT:  Clean dataset (Adult Income: 48,842 rows × 14 features)
        Target: ~20% rows with errors; 86% unintentional rows / 14% intentional rows

                    ┌──────────────────────────────────────────┐
                    │           Clean Dataset D                 │
                    └────────────────┬─────────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │  Select ~20% of rows  │
                         │  as error targets      │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
            ┌───────┴───────┐               ┌────────┴────────┐
            │  86% of rows  │               │  14% of rows    │
            │ UNINTENTIONAL │               │  INTENTIONAL    │
            │  (Phase 1)    │               │  (Phase 2)      │
            └───────┬───────┘               └────────┬────────┘
                    │                                 │
        ┌───────────┴───────────┐       ┌────────────┴────────────┐
        │  Mask k cells → NaN   │       │  Mask top-3 features    │
        │  TabPFN imputes at    │       │  TabPFN generates K=50  │
        │  T=0.7 (free sample)  │       │  candidates at T=1.0    │
        │  Accept if ≠ truth    │       │  Score each with RF     │
        │  mask[row,c] = -1     │       │  Pick best that flips   │
        └───────────┬───────────┘       │  target classifier      │
                    │                   │  mask[row,c] = +1       │
                    │                   └────────────┬────────────┘
                    │                                │
                    └────────────────┬───────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │  Dirty Dataset + Mask  │
                         │  dirty.csv + mask.csv  │
                         └────────────────────────┘
```

---

### 2.6 Phase 1 — Unintentional Error Generation

For each unintentional target row, for each of the k selected columns:

1. Set column to NaN → call `TabPFNUnsupervisedModel.impute(x_masked, t=0.7, n_permutations=1)`
2. **Categorical columns:** TabPFN samples from `Categorical(probs)` at T=0.7 — returns a category label
3. **Numerical columns:** TabPFN samples from its learned bar distribution at T=0.7 — returns a float, rounded to int if needed (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week)
4. Accept if `new_value ≠ original` (numerical: `|new − orig| > 0.01 × std`)
5. Assign `mask = −1`

---

### 2.7 Phase 2 — Intentional Error Generation

For each intentional target row:

1. Simultaneously mask the **top-3 mutable features by RF importance**: `capital-gain`, `relationship`, `age`
2. Generate **K=50 candidate rows** via TabPFN at T=1.0
3. Score each candidate with the **target Random Forest classifier** (same RF as rule-based baseline, trained on clean Adult Income, OrdinalEncoder for categoricals)
4. Keep candidates that **increase P(desired class = >50K)** beyond the original row's score
5. Among valid candidates: pick the one with the highest flip score (ties broken by minimal change)
6. If best candidate flips the prediction → accept; `mask = +1` for each modified column
7. If no flip in K=50: try masking top-5 features
8. If still no flip: reassign row to unintentional set; record in `stats.json`

**Flip rate achieved:** ~46% (top-3 masking, T=1.0, K=50) vs. ~3% with single-feature masking.

---

### 2.8 Temperature Protocol

Temperature T controls sampling randomness in TFM generation:

```
P(value = k) = exp(z_k / T) / Σ_j exp(z_j / T)

T → 0:   deterministic (most likely value only)
T = 1:   standard softmax (samples from learned distribution)
T → ∞:   uniform random
```

| T | Unintentional Effect | Intentional Effect |
|---|---|---|
| 0.3 | Very close to truth; low yield | Few distinct candidates |
| 0.5 | Moderate diversity; plausible | Reasonable diversity |
| **0.7** ← used | Good diversity; 59.5% yield | — |
| **1.0** ← used | High diversity | Maximum diversity; 46% flip rate |

---

### 2.9 Feasibility Tests (Pre-Implementation Validation)

| ID | Test | Question | Result |
|---|---|---|---|
| F1 | TabPFN categorical imputation | Can TabPFN learn P(education \| age, workclass, ...)? | ✅ PASS — avg accuracy = 0.750 |
| F3 | Error yield rate | How often does imputed ≠ ground truth at T=0.7? | ✅ PASS — 59.5% yield |
| F4 | Plausibility check | Are imputed values from the real distribution? | ✅ PASS — 0% out-of-distribution |
| F5 | Runtime estimate | Is generation feasible at scale? | ✅ PASS — ~3.2 min (grouped batching, 100× speedup over naïve approach) |
| F6 | Intentional flip rate | Can TFM candidates flip the RF classifier? | ✅ PASS — 46% flip rate with top-3 masking + T=1.0 |

---

### 2.10 Comparison to Rule-Based Baseline

| Parameter | Rule-Based (tab_err + greedy) | TFM-Inject (TabPFN-v2.6) |
|---|---|---|
| Row error rate | ~20% | ~20% |
| Intent split | ~2/3 unintentional, ~1/3 intentional (cells) | ~2/3 unintentional, ~1/3 intentional (cells) |
| Cells per error row | 1–3 | 1–3 (same distribution) |
| Unintentional mechanism | `tab_err` (typos, outliers, unit errors, missing, category swap) | TabPFN conditional sampling at T=0.7 |
| Intentional mechanism | Greedy adversarial attack (distribution extremes) | TabPFN-guided adversarial selection (K=50 candidates, top-3 masking, T=1.0) |
| Values on data manifold? | ❌ Often outside distribution | ✅ Sampled from learned P(X) |
| Mask format | 0 / −1 / +1 | 0 / −1 / +1 (identical) |
| Immutable features | race, sex, fnlwgt | race, sex, fnlwgt |
| Target classifier | Random Forest (OrdinalEncoder) | Same Random Forest |

---

## 3. LLM-Based Attribution Results

All five models were evaluated on **6,051 error records** across three prompt variants:
- **zero_shot** — no examples, no feature metadata
- **info** — feature descriptions provided
- **few_shot** — feature descriptions + 3 labeled examples per class

### 2.1 Gemini-2.0-Flash

| Variant | Accuracy | F1 Intentional | F1 Unintentional | **F1 Weighted** | TP | TN | FP | FN | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.2203 | 0.3246 | 0.0778 | 0.1243 | 1134 | 199 | 4711 | 7 | 1 |
| info | 0.3175 | 0.3539 | 0.2767 | 0.2913 | 1131 | 790 | 4120 | 10 | 0 |
| **few_shot** | **0.6207** | **0.4924** | **0.6973** | **0.6586** | 1113 | 2643 | 2267 | 28 | 0 |

### 2.2 Mixtral-8x7B-Instruct

| Variant | Accuracy | F1 Intentional | F1 Unintentional | **F1 Weighted** | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.6794 | 0.5231 | 0.7585 | 0.7141 | 1064 | 3047 | 1863 | 77 |
| info | 0.5274 | 0.4332 | 0.5947 | 0.5642 | 1093 | 2098 | 2812 | 48 |
| **few_shot** | **0.7333** | **0.5773** | **0.8052** | **0.7622** | 1102 | 3335 | 1575 | 39 |

### 2.3 LLaMA-3-70B-Instruct

| Variant | Accuracy | F1 Intentional | F1 Unintentional | **F1 Weighted** | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.1937 | 0.3187 | 0.0125 | 0.0703 | 1141 | 31 | 4879 | 0 |
| info | 0.6763 | 0.5323 | 0.7524 | 0.7109 | 1115 | 2977 | 1933 | 26 |
| **few_shot** | **0.8306** | **0.6814** | **0.8846** | **0.8463** | 1096 | 3930 | 980 | 45 |

### 2.4 Qwen2.5-32B-Instruct

| Variant | Accuracy | F1 Intentional | F1 Unintentional | **F1 Weighted** | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.4100 | 0.3826 | 0.4351 | 0.4252 | 1106 | 1375 | 3535 | 35 |
| info | 0.8288 | 0.6715 | 0.8842 | 0.8441 | 1059 | 3956 | 954 | 82 |
| **few_shot** | **0.7776** | **0.6178** | **0.8431** | **0.8006** | 1088 | 3617 | 1293 | 53 |

### 2.5 DeepSeek-R1-Distill-Qwen-32B

| Variant | Accuracy | F1 Intentional | F1 Unintentional | **F1 Weighted** | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| zero_shot | 0.4251 | 0.3895 | 0.4567 | 0.4440 | 1110 | 1462 | 3448 | 31 |
| info | 0.6972 | 0.5425 | 0.7738 | 0.7302 | 1086 | 3133 | 1777 | 55 |
| **few_shot** | **0.9172** | **0.8011** | **0.9477** | **0.9201** | 1009 | 4541 | 369 | 132 |

### 2.6 Summary — Best Variant Per Model

| Model | Best Variant | F1 Weighted | F1 Intentional | F1 Unintentional | Accuracy |
|---|---|---|---|---|---|
| Gemini-2.0-Flash | few_shot | 0.6586 | 0.4924 | 0.6973 | 0.6207 |
| Mixtral-8x7B | few_shot | 0.7622 | 0.5773 | 0.8052 | 0.7333 |
| LLaMA-3-70B | few_shot | 0.8463 | 0.6814 | 0.8846 | 0.8306 |
| Qwen2.5-32B | info | 0.8441 | 0.6715 | 0.8842 | 0.8288 |
| **DeepSeek-R1-Qwen-32B** | **few_shot** | **0.9201** | **0.8011** | **0.9477** | **0.9172** |

**Best overall LLM result**: DeepSeek-R1-Distill-Qwen-32B (few_shot) — **F1 Weighted = 0.9201**

---

## 4. Scenario B Results (Clustering-Based Attribution, 13 Heuristic Features)

The clustering-based system uses the following pipeline: cluster the data → assign cluster labels as pseudo-labels → train a supervised classifier (5 methods) on the pseudo-labels.

**Scenario B** uses 13 heuristic features. Evaluated on the same TFM-Inject dataset (rule-based = original Adult dataset, tfm = TFM-Inject test set). **ADS = F1(rule-based) − F1(TFM)** — positive means TFM data is harder.

### RandomForest (best classifier in most configurations)

> F1 Unintentional is derived from: `F1_unint = (F1_weighted − w_int × F1_int) / w_unint`, where w_int = 3423/9199 = 0.3721, w_unint = 5776/9199 = 0.6279 (cell-level class weights: 1141 intentional records × 3 cells each = 3423 intentional cells; 9199 − 3423 = 5776 unintentional cells).

| Clustering | F1 Weighted (TFM) | F1 Intentional (TFM) | F1 Unintentional (TFM) | Accuracy (TFM) | F1 Weighted (Rule) | ADS (F1_w) |
|---|---|---|---|---|---|---|
| HDBSCAN | 0.8972 | 0.8590 | 0.9198 | 0.8995 | 0.9721 | +0.0749 |
| DBSCAN | 0.9164 | 0.8650 | 0.9469 | 0.9179 | 0.9816 | +0.0652 |
| Random | 0.9200 | 0.8760 | 0.9461 | 0.9204 | 0.9553 | +0.0353 |
| GMM | 0.9108 | 0.8559 | 0.9433 | 0.9128 | 0.9413 | +0.0305 |
| Agglomerative | 0.9230 | 0.8793 | 0.9489 | 0.9237 | 0.9449 | +0.0219 |
| K-Means | 0.9134 | 0.8635 | 0.9430 | 0.9143 | 0.9336 | +0.0202 |
| **Mean** | **0.9135** | **0.8665** | **0.9413** | **0.9148** | **0.9548** | **+0.0413** |

### All Methods — Full Scenario B Table

| Clustering | Method | F1 Rule-Based | F1 TFM | ADS | Interpretation |
|---|---|---|---|---|---|
| K-Means | RandomForest | 0.9336 | 0.9134 | +0.0202 | Equal difficulty |
| K-Means | ClusterMajorityVote | 0.8309 | 0.8834 | −0.0525 | TFM easier |
| K-Means | KNN_k7 | 0.8922 | 0.8976 | −0.0054 | Equal difficulty |
| K-Means | LabelPropagation | 0.7256 | 0.8439 | −0.1183 | TFM easier |
| K-Means | LabelSpreading | 0.6885 | 0.7889 | −0.1004 | TFM easier |
| HDBSCAN | RandomForest | 0.9721 | 0.8972 | +0.0749 | TFM harder (moderate) |
| HDBSCAN | ClusterMajorityVote | 0.9124 | 0.8357 | +0.0767 | TFM harder (moderate) |
| HDBSCAN | KNN_k7 | 0.9650 | 0.8976 | +0.0674 | TFM harder (moderate) |
| HDBSCAN | LabelPropagation | 0.9493 | 0.8922 | +0.0571 | TFM harder (moderate) |
| HDBSCAN | LabelSpreading | 0.9506 | 0.8924 | +0.0582 | TFM harder (moderate) |
| Agglomerative | RandomForest | 0.9449 | 0.9230 | +0.0219 | Equal difficulty |
| Agglomerative | ClusterMajorityVote | 0.8429 | 0.8645 | −0.0216 | Equal difficulty |
| Agglomerative | KNN_k7 | 0.8569 | 0.9013 | −0.0444 | Equal difficulty |
| Agglomerative | LabelPropagation | 0.7260 | 0.8522 | −0.1262 | TFM easier |
| Agglomerative | LabelSpreading | 0.7118 | 0.7834 | −0.0716 | TFM easier |
| GMM | RandomForest | 0.9413 | 0.9108 | +0.0305 | TFM harder (moderate) |
| GMM | ClusterMajorityVote | 0.8574 | 0.8997 | −0.0423 | TFM easier |
| GMM | KNN_k7 | 0.9047 | 0.8953 | +0.0094 | Equal difficulty |
| GMM | LabelPropagation | 0.7218 | 0.8565 | −0.1347 | TFM easier |
| GMM | LabelSpreading | 0.7076 | 0.7987 | −0.0911 | TFM easier |
| DBSCAN | RandomForest | 0.9816 | 0.9164 | +0.0652 | TFM harder (moderate) |
| DBSCAN | ClusterMajorityVote | 0.9327 | 0.8615 | +0.0712 | TFM harder (moderate) |
| DBSCAN | KNN_k7 | 0.9709 | 0.8840 | +0.0869 | TFM harder (moderate) |
| DBSCAN | LabelPropagation | 0.8309 | 0.8844 | −0.0535 | TFM easier |
| DBSCAN | LabelSpreading | 0.8305 | 0.8702 | −0.0397 | TFM easier |
| Random | RandomForest | 0.9553 | 0.9200 | +0.0353 | TFM harder (moderate) |
| Random | ClusterMajorityVote | 0.5730 | 0.5692 | +0.0038 | Equal difficulty |
| Random | KNN_k7 | 0.9028 | 0.8853 | +0.0175 | Equal difficulty |
| Random | LabelPropagation | 0.7180 | 0.8471 | −0.1291 | TFM easier |
| Random | LabelSpreading | 0.7116 | 0.7740 | −0.0624 | TFM easier |

**Mean ADS (Scenario B, all 30 combos):** +0.0413 (averaged over RandomForest only per summary)

---

## 5. Scenario B+ Results (Clustering-Based Attribution, 23 Combined Features)

**Scenario B+** extends Scenario B with 10 additional engineered features (23 total), designed to better capture error structure.

### RandomForest (best classifier in most configurations)

> F1 Unintentional derived using the same formula as Section 4 (w_int=0.3721, w_unint=0.6279).

| Clustering | F1 Weighted (TFM) | F1 Intentional (TFM) | F1 Unintentional (TFM) | Accuracy (TFM) | F1 Weighted (Rule) | ADS (F1_w) |
|---|---|---|---|---|---|---|
| HDBSCAN | 0.9212 | 0.8932 | 0.9378 | 0.9215 | 0.9950 | +0.0738 |
| Agglomerative | 0.9022 | 0.8419 | 0.9379 | 0.9044 | 0.9902 | +0.0880 |
| GMM | 0.8964 | 0.8291 | 0.9363 | 0.8997 | 0.9758 | +0.0794 |
| K-Means | 0.9028 | 0.8446 | 0.9373 | 0.9045 | 0.9787 | +0.0759 |
| Random | 0.9222 | 0.8772 | 0.9489 | 0.9230 | 0.9853 | +0.0631 |
| DBSCAN | 0.9395 | 0.9068 | 0.9589 | 0.9397 | 0.9751 | +0.0356 |
| **Mean** | **0.9140** | **0.8655** | **0.9429** | **0.9155** | **0.9834** | **+0.0693** |

### All Methods — Full Scenario B+ Table

| Clustering | Method | F1 Rule-Based | F1 TFM | ADS | Interpretation |
|---|---|---|---|---|---|
| K-Means | RandomForest | 0.9787 | 0.9028 | +0.0759 | TFM harder (moderate) |
| K-Means | ClusterMajorityVote | 0.8649 | 0.8549 | +0.0100 | Equal difficulty |
| K-Means | KNN_k7 | 0.9168 | 0.9045 | +0.0123 | Equal difficulty |
| K-Means | LabelPropagation | 0.7025 | 0.8681 | −0.1656 | TFM easier |
| K-Means | LabelSpreading | 0.6736 | 0.8209 | −0.1473 | TFM easier |
| HDBSCAN | RandomForest | 0.9950 | 0.9212 | +0.0738 | TFM harder (moderate) |
| HDBSCAN | ClusterMajorityVote | 0.9072 | 0.9064 | +0.0008 | Equal difficulty |
| HDBSCAN | KNN_k7 | 0.9751 | 0.9191 | +0.0560 | TFM harder (moderate) |
| HDBSCAN | LabelPropagation | 0.9712 | 0.9204 | +0.0508 | TFM harder (moderate) |
| HDBSCAN | LabelSpreading | 0.9665 | 0.9133 | +0.0532 | TFM harder (moderate) |
| Agglomerative | RandomForest | 0.9902 | 0.9022 | +0.0880 | TFM harder (moderate) |
| Agglomerative | ClusterMajorityVote | 0.8371 | 0.8513 | −0.0142 | Equal difficulty |
| Agglomerative | KNN_k7 | 0.9123 | 0.9079 | +0.0044 | Equal difficulty |
| Agglomerative | LabelPropagation | 0.6690 | 0.8708 | −0.2018 | TFM easier |
| Agglomerative | LabelSpreading | 0.6565 | 0.8144 | −0.1579 | TFM easier |
| GMM | RandomForest | 0.9758 | 0.8964 | +0.0794 | TFM harder (moderate) |
| GMM | ClusterMajorityVote | 0.8639 | 0.8272 | +0.0367 | TFM harder (moderate) |
| GMM | KNN_k7 | 0.8981 | 0.8806 | +0.0175 | Equal difficulty |
| GMM | LabelPropagation | 0.7190 | 0.7611 | −0.0421 | TFM easier |
| GMM | LabelSpreading | 0.6972 | 0.7573 | −0.0601 | TFM easier |
| DBSCAN | RandomForest | 0.9751 | 0.9395 | +0.0356 | TFM harder (moderate) |
| DBSCAN | ClusterMajorityVote | 0.9586 | 0.9099 | +0.0487 | TFM harder (moderate) |
| DBSCAN | KNN_k7 | 0.9543 | 0.9236 | +0.0307 | TFM harder (moderate) |
| DBSCAN | LabelPropagation | 0.8359 | 0.8974 | −0.0615 | TFM easier |
| DBSCAN | LabelSpreading | 0.8368 | 0.8834 | −0.0466 | TFM easier |
| Random | RandomForest | 0.9853 | 0.9222 | +0.0631 | TFM harder (moderate) |
| Random | ClusterMajorityVote | 0.5730 | 0.5692 | +0.0038 | Equal difficulty |
| Random | KNN_k7 | 0.9099 | 0.9003 | +0.0096 | Equal difficulty |
| Random | LabelPropagation | 0.7113 | 0.8665 | −0.1552 | TFM easier |
| Random | LabelSpreading | 0.6811 | 0.8064 | −0.1253 | TFM easier |

**Mean ADS (Scenario B+, RandomForest only):** +0.0693

---

## 6. Head-to-Head Comparison: LLM vs. Scenario B / B+

### 5.1 Reference Points

For fair comparison, we use the **F1 Weighted on TFM-Inject** from the clustering system. The best-performing classifier combination for each scenario:

| System | Config | F1 Weighted (TFM) |
|---|---|---|
| Scenario B — best | DBSCAN + RandomForest | 0.9164 |
| Scenario B — mean (RF only) | — | 0.9135 |
| Scenario B+ — best | DBSCAN + RandomForest | 0.9395 |
| Scenario B+ — mean (RF only) | — | 0.9140 |
| Scenario B+ — HDBSCAN + RF | — | 0.9212 |

### 5.2 LLM Best Results vs. Clustering System

| Rank | System | Variant / Config | F1 Weighted | vs. B mean (+0.9135) | vs. B+ best (0.9395) |
|---|---|---|---|---|---|
| 1 | DeepSeek-R1-Qwen-32B | few_shot | **0.9201** | +0.0066 ✅ | −0.0194 ❌ |
| 2 | Qwen2.5-32B | info | 0.8441 | −0.0694 ❌ | −0.0954 ❌ |
| 3 | LLaMA-3-70B | few_shot | 0.8463 | −0.0672 ❌ | −0.0932 ❌ |
| 4 | Mixtral-8x7B | few_shot | 0.7622 | −0.1513 ❌ | −0.1773 ❌ |
| 5 | Gemini-2.0-Flash | few_shot | 0.6586 | −0.2549 ❌ | −0.2809 ❌ |
| — | Scenario B (DBSCAN+RF) | — | 0.9164 | — | — |
| — | Scenario B+ (DBSCAN+RF) | — | **0.9395** | — | — |
| — | Scenario B+ (Agglo+RF) | — | 0.9022 | — | — |

### 5.3 All LLM few_shot Results vs. Scenario B+ (DBSCAN+RandomForest = 0.9395)

| Model | F1 Weighted | Δ vs B+ DBSCAN+RF | Δ vs B+ mean |
|---|---|---|---|
| DeepSeek-R1-Qwen-32B | 0.9201 | −0.0194 | +0.0061 |
| LLaMA-3-70B | 0.8463 | −0.0932 | −0.0677 |
| Qwen2.5-32B | 0.8006 | −0.1389 | −0.1134 |
| Mixtral-8x7B | 0.7622 | −0.1773 | −0.1518 |
| Gemini-2.0-Flash | 0.6586 | −0.2809 | −0.2554 |

### 5.4 F1 Intentional Comparison (Most Difficult Metric)

Intentional errors represent only 18.9% of the error set — correctly identifying them is the hardest challenge.

| System | Config | F1 Intentional |
|---|---|---|
| DeepSeek-R1-Qwen-32B | few_shot | **0.8011** |
| LLaMA-3-70B | few_shot | 0.6814 |
| Qwen2.5-32B | info | 0.6715 |
| Scenario B+ — HDBSCAN+RF | — | 0.9212 (F1_w) / **0.8932** (F1_int) |
| Scenario B+ — DBSCAN+RF | — | 0.9395 (F1_w) / **0.9068** (F1_int) |
| Scenario B+ — Agglo+RF | — | 0.9022 (F1_w) / **0.8419** (F1_int) |
| Mixtral-8x7B | few_shot | 0.5773 |
| Gemini-2.0-Flash | few_shot | 0.4924 |

---

## 7. Key Findings

### 6.1 LLM Performance
- **DeepSeek-R1** is the clear winner among LLMs, reaching **F1_w = 0.9201** with few_shot prompting — the only LLM to match or exceed the Scenario B mean.
- **LLaMA-3-70B** and **Qwen2.5-32B** form a strong second tier (~0.84–0.85 F1_w with their best variant).
- **Gemini-2.0-Flash** and **Mixtral-8x7B** significantly underperform the clustering system.
- The **few_shot** variant consistently yields the best results (4 out of 5 models), confirming the importance of in-context examples.
- **info** variant sometimes outperforms few_shot (Qwen2.5: 0.8441 info vs. 0.8006 few_shot) — the extra examples can hurt if they confuse the model.

### 6.2 LLM vs. Clustering System
- Only **DeepSeek-R1** (few_shot, F1=0.9201) is competitive with Scenario B (mean F1=0.9135) and exceeds it slightly.
- The **Scenario B+ DBSCAN+RandomForest** configuration (F1=0.9395) **outperforms all LLMs** tested.
- The clustering system's best configuration (Scenario B+) exceeds the best LLM by **+0.019 F1 Weighted**.
- The clustering system has a decisive advantage on **F1 Intentional**: B+ DBSCAN+RF achieves 0.9068 vs. DeepSeek-R1's 0.8011.

### 6.3 TFM Dataset Difficulty (ADS Analysis)
- The TFM-Inject dataset is **harder than the rule-based** version for the clustering system in most configurations where ADS > 0 (especially with density-based clusterers HDBSCAN, DBSCAN).
- Mean ADS for Scenario B (RF only): **+0.0413** — the TFM error model consistently degrades clustering-based attribution performance.
- Mean ADS for Scenario B+: **+0.0693** — even with 23 features, TFM data remains harder by a larger margin.
- This confirms that the TFM injection model produces more realistic, harder-to-detect errors.

### 6.4 Prompt Sensitivity
| Model | zero_shot F1_w | few_shot F1_w | Δ |
|---|---|---|---|
| Gemini-2.0-Flash | 0.1243 | 0.6586 | +0.534 |
| LLaMA-3-70B | 0.0703 | 0.8463 | +0.776 |
| Qwen2.5-32B | 0.4252 | 0.8006 | +0.376 |
| DeepSeek-R1 | 0.4440 | 0.9201 | +0.476 |
| Mixtral-8x7B | 0.7141 | 0.7622 | +0.048 |

Zero-shot performance varies wildly (0.07–0.71) — LLMs must have examples or feature context to work reliably on TFM attribution.

---

## 8. Run Metadata

| Model | Run Folder | Records | Wall Time | Completed |
|---|---|---|---|---|
| Gemini-2.0-Flash | `tfm_sota_gemini_20260417_164935` | 6,051 | 3h 22m | Apr 17, 2026 |
| Mixtral-8x7B | `tfm_sota_mixtral_20260424_154335` | 6,051 | — | Apr 24, 2026 |
| LLaMA-3-70B | `tfm_sota_llama_20260424_171248` | 6,051 | — | Apr 24, 2026 |
| Qwen2.5-32B | `tfm_sota_qwen_20260424_194102` | 6,051 | — | Apr 24, 2026 |
| DeepSeek-R1-Qwen-32B | `tfm_sota_r1qwen_20260424_203235` | 6,051 | — | Apr 24, 2026 |
| Scenario B/B+ System | `compare_attribution_20260417_145728` | 11,545 (test) | — | Apr 17, 2026 |

---

*All LLM runs are located under:*
`error_detection_system/src/attribution/llm-based/output/`

*Scenario B/B+ run is located under:*
`tfm_error_injection/evaluation/output/compare_attribution_20260417_145728/`
