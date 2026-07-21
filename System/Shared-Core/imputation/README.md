# `src/imputation` — Module Documentation

> **Purpose:** Estimate correct values for corrupted cells at deployment time (when ground truth is unavailable), and extract confidence-weighted diagnostic features for downstream intent attribution.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Structure](#2-module-structure)
3. [Data Contract](#3-data-contract)
4. [Module Reference](#4-module-reference)
   - [MICEImputer](#41-miceimputer)
   - [DiagnosticFeatureExtractor](#42-diagnosticfeatureextractor)
   - [ImputationPipeline](#43-imputationpipeline)
   - [ImputationEvaluator](#44-imputationevaluator)
5. [CLI Usage](#5-cli-usage)
6. [Output Files](#6-output-files)
7. [Evaluation Results (Adult Income v3)](#7-evaluation-results-adult-income-v3)
8. [Design Decisions & Known Limitations](#8-design-decisions--known-limitations)
9. [Literature Grounding](#9-literature-grounding)

---

## 1. Overview

The intent attribution pipeline classifies *why* a data error exists — intentional (`1`) vs. unintentional (`-1`). In development, this relies on having the true correct value `x*` for each corrupted cell. In deployment, `x*` is unknown.

This module replaces `x*` with a statistical estimate `x̂*` derived from MICE (Multivariate Imputation by Chained Equations) using Random Forest models trained on the clean rows of the dataset. It then computes a diagnostic feature tuple per corrupted cell that feeds into downstream classifiers.

**Key design principle:** This module does **not** contribute a new imputation method. It adapts MICE to corrupted (not missing) data and uniquely routes imputation uncertainty into discriminative features for intent classification.

---

## 2. Module Structure

```
src/imputation/
├── __init__.py                 # Package exports
├── imputation_estimator.py     # MICEImputer — MICE with RF + dual confidence
├── diagnostic_features.py      # DiagnosticFeatureExtractor — WRC, direction, magnitude
├── run_pipeline.py             # ImputationPipeline — end-to-end orchestrator
└── evaluate.py                 # ImputationEvaluator — quality + ablation + calibration
```

---

## 3. Data Contract

The module is **dataset-agnostic**. Any dataset matching this contract works:

| File | Shape | Description |
|---|---|---|
| **Data file** | `N × P` | Feature columns only. Can contain `is_erroneous` — it is dropped automatically. Columns can be any mix of numerical and categorical. |
| **Mask file** | `N × P` | Same column names as data file. Values: `0` = clean cell, `1` = intentional error, `-1` = unintentional error. |

**Derived at runtime (nothing hardcoded):**

| Symbol | Meaning |
|---|---|
| `D_clean` | Rows where all mask values = 0 → training data for imputer |
| `D_dirty` | Rows where any mask value ≠ 0 → need imputation |
| `C_num` | Numerical columns (auto-detected via pandas dtype) |
| `C_cat` | Categorical columns (auto-detected via pandas dtype) |

---

## 4. Module Reference

### 4.1 `MICEImputer`

**File:** `imputation_estimator.py`

Trains one Random Forest model per column on clean rows, then imputes corrupted cells. For rows with multiple corrupted cells (common in this dataset), uses iterative MICE rounds so each column's imputation benefits from updated estimates of the others.

#### Constructor

```python
MICEImputer(
    n_estimators: int = 100,      # Trees per RF model
    max_depth: int | None = None, # Max depth (None = unlimited)
    max_rounds: int = 5,          # Max MICE iterations for multi-cell rows
    convergence_tol: float = 1e-3,# Stop when relative change < tol
    random_state: int = 42,
    verbose: bool = True,
)
```

#### Methods

```python
imputer.fit(data_df, mask_df) -> MICEImputer
```
Trains one `RandomForestRegressor` (numerical) or `RandomForestClassifier` (categorical) per column on `D_clean`. Fits `LabelEncoder` per categorical column on clean data. Extracts OOB score per model.

```python
imputer.impute(data_df, mask_df) -> pd.DataFrame
```
Imputes all corrupted cells. For single-cell rows: direct RF prediction. For multi-cell rows: MICE iterative loop (initialize to column median/mode → iterate until convergence).

**Output columns:**

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index in input data |
| `column` | str | Name of the corrupted column |
| `observed_value` | any | The corrupted value (x̃) |
| `imputed_value` | any | The estimated correct value (x̂*) |
| `intent_label` | int | Ground truth from mask (1 or -1) |
| `col_type` | str | `"numerical"` or `"categorical"` |
| `sigma_tree` | float | Std of per-tree predictions (row-specific uncertainty) |
| `sigma_oob` | float | OOB error for this column (column-level uncertainty) |
| `confidence` | float | `1 / (1 + sigma_tree)` — in (0, 1] |
| `mice_rounds` | int | MICE rounds used (0 = single-cell, no iteration) |

#### Dual Confidence

| Signal | Source | Scope | Meaning |
|---|---|---|---|
| `sigma_tree` | Std of T tree predictions | Row-specific | High → trees disagree → this row is unusual |
| `sigma_oob` | `1 - OOB_score` | Column-level | High → column is inherently unpredictable |

Using both signals avoids the edge case where a column is unpredictable globally (`sigma_oob` high) but all trees happen to agree on a wrong answer (`sigma_tree` low).

---

### 4.2 `DiagnosticFeatureExtractor`

**File:** `diagnostic_features.py`

Converts raw imputation results into the diagnostic feature tuple `φ(x̃_j, r)` used by downstream intent classifiers.

#### Constructor

```python
DiagnosticFeatureExtractor(
    alpha: float = 1.0,     # Dampening factor in WRC formula
    epsilon: float = 1e-10, # Division-by-zero guard
)
```

#### Methods

```python
extractor.extract(imputation_results, feature_importances=None) -> pd.DataFrame
```

**Output columns (model features):**

| Feature | Formula | Type | Robustness |
|---|---|---|---|
| `change_direction` | `sign(x̃ - x̂*)` for numerical; `changed=1` for categorical | float | **High** — only flips if imputation error > corruption magnitude |
| `change_magnitude` | `|x̃ - x̂*|` for numerical; Levenshtein distance for categorical | float | **Medium** — linearly biased by imputation error |
| `wrc` | `|x̃ - x̂*| / (|x̂*| · (1 + α · σ_tree))` for numerical; dampened normalized edit distance for categorical | float | **Stabilized** — self-suppresses when confidence is low |
| `raw_relative_change` | `|x̃ - x̂*| / (|x̂*| + ε)` | float | **Low** — for ablation only, not model input |
| `confidence` | `1 / (1 + sigma_tree)` | float | — |
| `confidence_oob` | `1 - sigma_oob` | float | — |
| `sigma_tree` | Passed through | float | — |
| `sigma_oob` | Passed through | float | — |
| `feature_importance` | Mean importance of column across all RF models | float | — |
| `is_categorical` | `1` if categorical column, `0` if numerical | int | — |

**Output columns (metadata — for inspection, not model input):**

`row_idx`, `column`, `intent_label`, `col_type`, `mice_rounds`, `observed_value`, `imputed_value`

**WRC Formula:**

```
WRC_j = |x̃_j - x̂*_j| / (|x̂*_j| · (1 + α · σ_j^tree))
```

When `σ_tree` is high (imputer is uncertain), the denominator grows, suppressing WRC toward zero. This prevents high-uncertainty imputations from generating artificially extreme features.

**Categorical features:** Levenshtein edit distance is used as magnitude. WRC is dampened normalized edit distance. The `change_direction` becomes a binary `changed` flag.

---

### 4.3 `ImputationPipeline`

**File:** `run_pipeline.py`

End-to-end orchestrator that runs fit → impute → feature extraction → save.

#### Constructor

```python
ImputationPipeline(
    n_estimators: int = 100,
    max_rounds: int = 5,
    alpha: float = 1.0,       # WRC dampening factor
    random_state: int = 42,
    verbose: bool = True,
)
```

#### Methods

```python
pipeline.run(data_df, mask_df) -> pd.DataFrame
# Returns: diagnostic_features DataFrame

pipeline.run_from_files(data_path, mask_path, output_dir=None) -> pd.DataFrame
# Loads CSVs, runs pipeline, optionally saves results

pipeline.save_results(output_dir)
# Saves: imputation_results.csv, diagnostic_features.csv, pipeline_summary.json
```

#### Aggregated Feature Importance

The pipeline computes a per-column importance score by averaging how important each column is **as a predictor** in other columns' models. This is passed to `DiagnosticFeatureExtractor` as `feature_importances`.

---

### 4.4 `ImputationEvaluator`

**File:** `evaluate.py`

Runs four evaluation blocks on the pipeline outputs.

#### Constructor

```python
ImputationEvaluator(verbose: bool = True)
```

#### Methods

```python
evaluator.evaluate(features_df, data_df, mask_df, output_dir=None) -> dict
```

**Block A — Imputation Quality** (`ImputationQualityEvaluator`)

Holds out 20% of clean rows, imputes each column one at a time, compares to known truth.

| Metric | Columns | Interpretation |
|---|---|---|
| RMSE | Numerical | Raw prediction error |
| NRMSE | Numerical | Normalized by column std. < 1.0 = better than guessing mean |
| Spearman r | Numerical | Rank-order correlation between true and imputed |
| Hit rate | Categorical | % of cells where imputed == true category |

**Block B — Feature Separability** (`FeatureSeparabilityEvaluator`)

Tests whether each diagnostic feature alone can distinguish intentional from unintentional errors.

| Metric | Interpretation |
|---|---|
| AUC | Area under ROC curve. > 0.5 = has discriminative power |
| Mann-Whitney U p-value | Are distributions different? p < 0.01 = significant |
| Spearman r with label | Monotonic relationship with intent label |

**Block C — Intent Classification Ablation** (`IntentClassificationEvaluator`)

Trains a Random Forest on diagnostic features using 5-fold stratified CV. Runs 4 configurations:

| Config | Features | Purpose |
|---|---|---|
| `full` | All 9 diagnostic features | Proposed method |
| `direction_confidence` | direction + confidence + confidence_oob + is_categorical | Most robust subset |
| `no_confidence` | direction + magnitude + raw_relative_change + is_categorical | Shows value of WRC dampening |
| `naive_baseline` | observed_numeric + col_mean + col_std + col_median + z_score + is_categorical | No imputation; lower bound |

Reports: accuracy, F1-macro, F1 per class (intentional / unintentional), confusion matrix — all as `mean ± std` across folds.

**Block D — Confidence Calibration** (`ConfidenceCalibrationEvaluator`)

Bins cells by `confidence` and measures whether higher confidence actually correlates with lower `sigma_tree` and higher `fraction_changed` (a proxy for imputation activity). Reports Expected Calibration Error (ECE).

---

## 5. CLI Usage

### Run pipeline only

```bash
python -m src.imputation.run_pipeline \
  --data    datasets/my_dataset/labeled_dataset.csv \
  --mask    datasets/my_dataset/ground_truth_masks.csv \
  --outdir  outputs/imputation/my_dataset/

# Options:
#   --n-estimators  100     Trees per RF model
#   --max-rounds    5       Max MICE iterations
#   --alpha         1.0     WRC dampening factor
#   --seed          42      Random seed
```

### Run evaluation

```bash
python -m src.imputation.evaluate \
  --features  outputs/imputation/my_dataset/diagnostic_features.csv \
  --data      datasets/my_dataset/labeled_dataset.csv \
  --mask      datasets/my_dataset/ground_truth_masks.csv \
  --outdir    outputs/imputation/my_dataset/evaluation/
```

### Programmatic usage

```python
from src.imputation import ImputationPipeline, ImputationEvaluator
import pandas as pd

# Load data
data_df = pd.read_csv("datasets/my_dataset/labeled_dataset.csv")
mask_df = pd.read_csv("datasets/my_dataset/ground_truth_masks.csv")
if "is_erroneous" in data_df.columns:
    data_df = data_df.drop(columns=["is_erroneous"])

# Run pipeline
pipeline = ImputationPipeline(n_estimators=100, max_rounds=5)
features = pipeline.run(data_df, mask_df)
pipeline.save_results("outputs/imputation/my_dataset/")

# Evaluate
evaluator = ImputationEvaluator()
results = evaluator.evaluate(
    features_df=features,
    data_df=data_df,
    mask_df=mask_df,
    output_dir="outputs/imputation/my_dataset/evaluation/",
)
```

---

## 6. Output Files

### Pipeline outputs (`outputs/imputation/<dataset>/`)

| File | Description |
|---|---|
| `imputation_results.csv` | One row per corrupted cell: observed, imputed, sigma_tree, sigma_oob, confidence, mice_rounds |
| `diagnostic_features.csv` | One row per corrupted cell: all features from diagnostic tuple + metadata |
| `pipeline_summary.json` | Dataset stats, OOB errors per column, diagnostic feature statistics |

### Evaluation outputs (`outputs/imputation/<dataset>/evaluation/`)

| File | Description |
|---|---|
| `evaluation_results.json` | Full results for all 4 evaluation blocks |
| `ablation_results.csv` | Intent classification ablation — one row per config with mean±std metrics |

---

## 7. Evaluation Results (Adult Income v3)

Dataset: 4000 rows, 15 columns (6 numerical, 9 categorical). 3010 clean training rows, 990 dirty rows, 2198 corrupted cells.

### A. Imputation Quality

| Column | Type | Quality | NRMSE / Hit Rate | Spearman r |
|---|---|---|---|---|
| `education-num` | numerical | ✓ excellent | NRMSE = 0.333 | r = 0.965 |
| `education` | categorical | ✓ excellent | hit = 0.927 | — |
| `native-country` | categorical | ✓ excellent | hit = 0.920 | — |
| `class` | categorical | ✓ good | hit = 0.885 | — |
| `race`, `sex`, `relationship` | categorical | ✓ good | hit = 0.75–0.86 | — |
| `marital-status`, `workclass` | categorical | ✓ good | hit = 0.77–0.80 | — |
| `age`, `hours-per-week` | numerical | ✓ usable | NRMSE = 0.82–0.93 | r = 0.43–0.57 |
| `occupation` | categorical | ✗ poor | hit = 0.357 | — |
| `fnlwgt`, `capital-gain`, `capital-loss` | numerical | ✗ unpredictable | NRMSE > 1.0 | r ≈ 0.01–0.23 |

**Aggregate:** Mean NRMSE = 0.878, Mean categorical hit rate = 0.787

The 3 unpredictable columns (`fnlwgt`, `capital-gain`, `capital-loss`) have negative OOB R² — they are fundamentally unpredictable from other features. Their `sigma_oob > 1.0` flags this to downstream classifiers via `confidence_oob ≈ 0`.

### B. Feature Separability

| Feature | AUC | p-value | Significant? |
|---|---|---|---|
| `wrc` | 0.673 | 9.0e-44 | ✓ ** |
| `raw_relative_change` | 0.633 | 6.4e-27 | ✓ ** |
| `change_magnitude` | 0.620 | 4.1e-22 | ✓ ** |
| `change_direction` | 0.580 | 1.1e-16 | ✓ ** |
| `sigma_oob` / `confidence_oob` | 0.536 | 3.9e-03 | ✓ ** |
| `is_categorical`, `confidence`, `sigma_tree`, `feature_importance` | ~0.51 | >0.05 | ✗ |

All four change-based features are statistically significant (p < 0.01). The WRC formula has the highest single-feature AUC (0.673), confirming its discriminative power.

### C. Intent Classification Ablation

| Config | Accuracy | F1-macro | F1 (intent.) | F1 (unintent.) | Features |
|---|---|---|---|---|---|
| **`full`** (proposed) | **0.837 ± 0.010** | **0.834** | **0.814** | **0.855** | 9 |
| `no_confidence` | 0.805 ± 0.011 | 0.803 | 0.783 | 0.823 | 4 |
| `naive_baseline` | 0.716 ± 0.014 | 0.674 | 0.559 | 0.790 | 6 |
| `direction_confidence` | 0.648 ± 0.031 | 0.640 | 0.588 | 0.693 | 4 |

**Key observations:**
- **Full features outperform all ablations** (F1-macro 0.834 vs. best ablation 0.803) — the confidence signals add value.
- **Naive baseline (0.674) is substantially worse** than full (0.834) — this justifies the imputation approach. Without a reference point, intent classification degrades by ~16 F1 points.
- **No-confidence config (0.803)** vs. **full (0.834)** — adding WRC dampening over raw RC gives +3 F1 points. The confidence weighting matters.
- `direction_confidence` alone gives 0.640 — useful as a minimal fallback when imputation quality is very low.

### D. Confidence Calibration

**ECE = 0.421** — confidence is only weakly calibrated. Higher-confidence bins do show lower `sigma_tree` and slightly lower `fraction_changed`, meaning the calibration trend is in the right direction but not tight.

| Confidence bin | n | Mean σ_tree | Fraction changed |
|---|---|---|---|
| 0.0 – 0.1 | 822 | 182.83 | 1.00 |
| 0.5 – 0.6 | 181 | 0.76 | 0.99 |
| 0.7 – 0.8 | 392 | 0.31 | 0.82 |
| 0.8 – 0.9 | 472 | 0.18 | 0.84 |
| 0.9 – 1.0 | 273 | 0.04 | 0.85 |

Confidence bins 0.1–0.5 are empty — this bimodal split (very low vs. moderate-to-high confidence) is driven by the numerical columns with high variance vs. the categorical columns with high RF agreement.

---

## 8. Design Decisions & Known Limitations

### The `capital-loss` edge case

`capital-loss` has OOB error > 1.0 (all trees agree on ~0, but corrupted values can be large). This produces extremely high WRC values (mean ~1.67 × 10⁹). The `confidence_oob` for this column is negative (~−0.1), which signals the column is unreliable. A downstream classifier that uses `confidence_oob` as a feature can learn to discount WRC for such columns. In the ablation results, the `full` config (which includes `confidence_oob`) outperforms `no_confidence` precisely because of cases like this.

### Categorical imputation is classification, not regression

For categorical columns, sigma_tree is computed as `1 - agreement_ratio` (fraction of trees that voted for the majority class). This is bounded in [0, 1] and naturally interpretable. WRC for categoricals is normalized edit distance, dampened by this sigma.

### MICE convergence is dataset-dependent

Most rows converge in 2–3 rounds (93% in Adult Income v3). Setting `max_rounds=5` is safe overhead. Setting `max_rounds=1` degrades to independent imputation — useful for the convergence ablation study from the research plan.

### Naive baseline is the critical comparison

The naive baseline (`z_score` of observed value + column stats) achieves F1-macro = 0.674. The full imputed features achieve 0.834. This 16-point gap is the empirical justification for the imputation approach.

---

## 9. Literature Grounding

| Concept | Source | Adoption in this module |
|---|---|---|
| MICE iterative imputation | Perini & Nikolic, SIGMOD 2024 | Algorithm structure, per-column RF models, iterative multi-cell loop |
| Uncertainty propagation to downstream tasks | Wang et al., SIGMOD 2024 | Using `sigma_tree` / `sigma_oob` as features for classifier, not just to improve imputation |
| OOB as data-support proxy | Miao et al., VLDB 2022 | `sigma_oob = 1 - OOB_score` as second confidence signal independent of tree variance |
| WRC dampening formula | Novel contribution | `WRC = |Δ| / (|x̂*| · (1 + α · σ_tree))` — suppresses unstable relative-change when confidence is low |
