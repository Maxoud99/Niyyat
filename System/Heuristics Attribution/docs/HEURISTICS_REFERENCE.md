# Error Fingerprint — Heuristics Reference

This document is the complete technical reference for all eight heuristics (H1–H8) that form the **Error Fingerprint** pipeline.  
Each heuristic contributes one or more columns to a fixed **13-feature matrix** used to classify erroneous cells as **intentional** (adversarial / deliberate manipulation) or **unintentional** (noise / accident).

---

## Table of Contents

1. [Overview](#overview)
2. [Shared Conventions](#shared-conventions)
3. [H1 — Value Plausibility](#h1--value-plausibility)
4. [H2 — String Anomaly](#h2--string-anomaly)
5. [H3 — Distribution Position](#h3--distribution-position)
6. [H4 — Row Coherence](#h4--row-coherence)
7. [H5 — Error Pattern](#h5--error-pattern)
8. [H6 — Column Importance](#h6--column-importance)
9. [H7 — User Incentive](#h7--user-incentive)
10. [H8 — Sensitivity Flag](#h8--sensitivity-flag)
11. [Feature Matrix Summary](#feature-matrix-summary)

---

## Overview

The pipeline takes a **dirty dataset** and a **binary error mask** as input. The mask identifies *which cells* contain errors (1 = erroneous, 0 = clean) but does **not** reveal the error's intent. The eight heuristics then answer different facets of a single question:

> *"Does this cell's error look like something a deliberate human actor would do, or does it look like random noise?"*

Each heuristic is computed independently during a **fit** step (unsupervised, using only the dirty data and the mask) and a **compute** step (produces one row per erroneous cell). The outputs are merged into a unified `(N_errors × 13)` feature matrix, which is then fed to a classifier.

```
dirty_df + mask_df
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  pipe.fit(dirty_df, mask_df)                 │   ← unsupervised
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │  feat_df = pipe.compute_features(...)        │   ← 13 features per error cell
  └─────────────────────────────────────────────┘
        │
        ▼
  RandomForest / Label Propagation / ...         ← classifier (Scenario A or B)
```

---

## Shared Conventions

### Column type detection

All heuristics use the same rule:

> A column is **numerical** if > 90 % of its non-null values can be cast to `float`. Otherwise it is **categorical**.

### Fit uses only clean cells

Statistics (vocabulary, mean, std, frequency ranks, etc.) are always learned from **clean cells only** (`mask_df == 0`). This prevents errors from corrupting the reference distribution.

### Index

Every heuristic's `compute()` output has columns `[row_idx, col_name, feature_1, ...]`. The pipeline aligns and merges all eight outputs on `(row_idx, col_name)` to produce the final 13-column matrix.

---

## H1 — Value Plausibility

**File:** `h1_value_plausibility.py`  
**Question answered:** *Is the erroneous value even a legitimate member of this column's domain?*

### Definition

H1 checks whether the erroneous cell value falls within the **known domain** of its column, where the domain is learned entirely from clean cells:

- **Categorical columns:** domain = set of unique string values observed in clean cells.
- **Numerical columns:** domain = interval [p5, p95] of numeric values in clean cells.

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h1_plausible` | int | {0, 1} | 1 = value is in the known domain; 0 = out of domain |

> **Note — internal-only outputs (not in the final matrix):**  
> `h1.compute()` also returns `h1_in_vocab` (categorical diagnostic) and `h1_in_range` (numerical diagnostic), but the pipeline selects **only `h1_plausible`** when assembling the 13-column matrix. These two columns are never passed to the classifier.

### Calculation

**Fit step:**
```
For each column c:
  if categorical:
      vocab[c] = { str(v) for v in dirty_df[mask==0][c] if v is not null }
  if numerical:
      p5[c]  = 5th  percentile of clean numeric values in c
      p95[c] = 95th percentile of clean numeric values in c
```

**Compute step:**
```
For each erroneous cell (row_idx, col_name):
  if categorical:
      h1_plausible = 1  if str(dirty_value) ∈ vocab[col_name]
                     0  otherwise (or if value is null/empty)
  if numerical:
      float_val = float(dirty_value)
      h1_plausible = 1  if p5[col_name] ≤ float_val ≤ p95[col_name]
                     0  otherwise (or if value cannot be cast to float)
```

### Intentional vs. unintentional signal

| h1_plausible | Interpretation |
|---|---|
| **1** | Value is in-domain → consistent with **intentional** error (uses a plausible cover value) |
| **0** | Value is out-of-domain → consistent with **unintentional** error (random noise, typo, encoding corruption) |

### Examples — Adult Income dataset

**Categorical column `education`** (clean vocab: `"HS-grad"`, `"Bachelors"`, `"Masters"`, …)

| row | dirty value | h1_plausible | Reason |
|---|---|---|---|
| 3 | `"Bachelors"` | **1** | Exact match in clean vocab → in-domain, looks intentional |
| 7 | `"Bachleors"` | **0** | Typo, not in vocab → out-of-domain → unintentional |
| 12 | `"Unknown"` | **0** | Obfuscation token, not in vocab → H2 rescues this as intentional |

**Numerical column `age`** (clean p5 ≈ 22, p95 ≈ 62)

| row | dirty value | h1_plausible | Reason |
|---|---|---|---|
| 5 | `35` | **1** | 22 ≤ 35 ≤ 62 → in range → in-domain, looks intentional |
| 9 | `999` | **0** | 999 > 62 → out of range → clearly unintentional outlier |
| 14 | `7` | **0** | 7 < 22 → below p5 → unintentional (impossible working-age value) |

### Known limitation

~29% of intentional errors in the Adult Income LLM dataset use obfuscation tokens (`nan`, `Unknown`, `—`) that are out-of-vocabulary. H1 scores these as `0` (implausible), same as a random typo. **H2 is responsible for rescuing these cases** by detecting the obfuscation pattern.

---

## H2 — String Anomaly

**File:** `h2_string_anomaly.py`  
**Question answered:** *If this categorical value is wrong — does it look like a typo, or deliberate obfuscation?*

### Definition

H2 characterises the **kind** of string error in a categorical column:

- A **typo** (unintentional) has a small Levenshtein edit distance to the nearest clean vocabulary entry (e.g. `"Bachleors"` → edit distance 1 to `"Bachelors"`).
- An **obfuscation token** (intentional) matches a known placeholder pattern (e.g. `"Unknown"`, `"nan"`, `"—"`, or suffix-mangled like `"Private-DMV"`).

**Applies to categorical columns only.** Both features are set to `NaN` for numerical columns.

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h2_min_edit_distance` | float\|NaN | [0, 10] or NaN | Levenshtein distance to nearest clean-vocabulary entry. Capped at 10. NaN for numerical columns. |
| `h2_is_obfuscation` | int\|NaN | {0, 1, NaN} | 1 if dirty value matches a known obfuscation pattern. 0 otherwise. NaN for numerical columns. |

> **Internal name note:** `h2.compute()` returns a column named `h2_min_edit_dist`; the pipeline renames it to `h2_min_edit_distance` before assembling the final matrix.

### Calculation

**Fit step:**
```
For each categorical column c:
  vocab[c] = list of unique string values from clean cells
             (if |vocab| > 200, keep only the 50 most-frequent entries)
  median_len[c] = median string length of vocab entries
```

**Compute step — `h2_min_edit_distance`:**
```
min_edit = min( levenshtein(dirty_value, v) for v in vocab[col_name] )
         = 0 if dirty_value already in vocab (exact match)
         = 10 if vocab is empty or all distances ≥ 10
Capped at 10.
```

**Levenshtein implementation:** custom inline dynamic-programming (no external dependencies). Uses two rolling rows for O(min(len1,len2)) memory.

**Compute step — `h2_is_obfuscation`:**

Three detection rules — any one is sufficient:

| Rule | Condition | Example |
|---|---|---|
| **1. Exact-token match** | `lower(dirty_value)` ∈ `{"nan", "none", "n/a", "unknown", "?", "—", "null", "missing", "0", "-1", "999", ...}` | `"Unknown"`, `"nan"` |
| **2. Suffix/prefix pattern** | `dirty_value` starts with a vocab entry, followed by `{-, _, space}` + `{dmv, obf, high, low, 1, 2, x, ?, —}` | `"Private-DMV"`, `"Doctorate-obf"` |
| **3. Structural placeholder** | Length == 1, OR all characters are non-alphanumeric | `"—"`, `"??"` |

### Intentional vs. unintentional signal

| Feature value | Interpretation |
|---|---|
| `h2_is_obfuscation = 1` | Strong **intentional** signal — deliberate placeholder substitution |
| `h2_min_edit_distance` small (1–2) | **Unintentional** typo — close to a real value |
| `h2_min_edit_distance` large (≥6) + `obfuscation = 0` | Structurally wrong — likely unintentional random substitution |

### Examples — Adult Income dataset

**Categorical column `workclass`** (clean vocab: `"Private"`, `"Self-emp-not-inc"`, `"Federal-gov"`, …)

| row | dirty value | h2_min_edit_distance | h2_is_obfuscation | Reason |
|---|---|---|---|---|
| 2 | `"Private"` | **0** | **0** | Exact vocab match, edit distance = 0 |
| 5 | `"Privte"` | **1** | **0** | One-char deletion typo → unintentional |
| 8 | `"Unknown"` | **7** | **1** | Matches obfuscation token set (Rule 1) → intentional |
| 11 | `"Private-DMV"` | **4** | **1** | Suffix pattern `Private + -DMV` (Rule 2) → intentional |
| 14 | `"—"` | **10** | **1** | Single non-alphanumeric char (Rule 3) → intentional |

**Numerical column `age`** (H2 does not apply to numerical columns)

| row | dirty value | h2_min_edit_distance | h2_is_obfuscation |
|---|---|---|---|
| 9 | `999` | **NaN** | **NaN** |

### Combined score (for interpretation only, not in the feature matrix)
```
h2_score = ( h2_is_obfuscation + max(0, 1 - h2_min_edit_distance / 5) ) / 2
```
High `h2_score` → intentional; low `h2_score` with small edit distance → unintentional typo.

---

## H3 — Distribution Position

**File:** `h3_distribution_position.py`  
**Question answered:** *Is this erroneous value common/central, or extreme/rare within the column's distribution?*

### Definition

H3 measures where the erroneous value sits within its column's statistical distribution (learned from clean cells). The key insight is:

- **Intentional** manipulators choose *common, unremarkable* values so they blend in.
- **Unintentional** errors (random noise, OCR corruption) tend to land in the distribution's tails.

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h3_distribution_score` | float | [0, 1] | Score ≈ 1.0 → central/common → intentional signal; Score ≈ 0.0 → extreme/rare → unintentional signal |

### Calculation

**Fit step:**
```
For each column c:
  if numerical:
      mean[c] = mean of clean numeric values
      std[c]  = std  of clean numeric values (ddof=0)
  if categorical:
      freq_rank[c][v] = rank of value v by descending frequency (rank 1 = most common)
      vocab_size[c]   = number of unique clean values
```

**Compute step — numerical:**
```
z = (dirty_value − mean[col]) / std[col]
h3_distribution_score = max(0, 1 − min(1, |z| / 3.0))
```

| z-score | Score |
|---|---|
| 0 (exactly at mean) | 1.0 |
| ±1.5 σ | 0.5 |
| ±3 σ or beyond | 0.0 |

Edge cases:
- `std == 0` (constant column) → score = 1.0 (everything is "central")
- value cannot be cast to float → score = 0.0

**Compute step — categorical:**
```
rank = freq_rank[col][dirty_value]       if dirty_value ∈ vocab
     = vocab_size[col] + 1               if dirty_value ∉ vocab (OOV penalty)

h3_distribution_score = 1 − (rank − 1) / max(1, vocab_size − 1)
```

| rank | Score |
|---|---|
| 1 (most common value) | 1.0 |
| median rank | 0.5 |
| last rank (rarest) | 0.0 |
| OOV (not in vocab) | 0.0 |

### Intentional vs. unintentional signal

| h3_distribution_score | Interpretation |
|---|---|
| **≈ 1.0** | Common/central value → **intentional** (blending strategy) |
| **≈ 0.0** | Extreme/rare/OOV → **unintentional** (random substitution, noise) |

### Examples — Adult Income dataset

**Numerical column `age`** (mean ≈ 38.6, std ≈ 13.6)

| row | dirty value | z-score | h3_distribution_score | Interpretation |
|---|---|---|---|---|
| 3 | `40` | +0.10 | **0.97** | Near mean → blends in → intentional |
| 7 | `35` | -0.26 | **0.91** | Within normal range → intentional |
| 9 | `999` | +70.6 | **0.00** | Extreme outlier → unintentional |
| 14 | `7` | -2.32 | **0.23** | Low tail → somewhat unintentional |

**Categorical column `education`** (most common: `"HS-grad"` rank 1, `"Some-college"` rank 2, …, `"Preschool"` rank 16)

| row | dirty value | rank | h3_distribution_score | Interpretation |
|---|---|---|---|---|
| 2 | `"HS-grad"` | 1 | **1.00** | Most frequent → intentional (blends into majority) |
| 5 | `"Bachelors"` | 3 | **0.87** | Very common → intentional |
| 10 | `"Preschool"` | 16 | **0.00** | Rarest value → unintentional |
| 12 | `"XYZ-fake"` | OOV | **0.00** | Not in vocab → unintentional |

---

## H4 — Row Coherence

**File:** `h4_row_coherence.py`  
**Question answered:** *Does this erroneous value fit the context of the rest of its row?*

### Definition

H4 trains one Random Forest predictor per column, then measures how well the erroneous value is "predicted" by the other columns in the same row:

- **Intentional** manipulators understand the data and keep their changes consistent with other fields (e.g. changing `education` and `education-num` together so the row still makes sense).
- **Unintentional** errors break row-level correlation structure (the injected value is inconsistent with what other columns predict).

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h4_coherence_score` | float | [0, 1] | Score ≈ 1.0 → dirty value fits row context → intentional signal; Score ≈ 0.0 → dirty value breaks row context → unintentional signal |

### Calculation

**Fit step:**
```
For each column c:
  feature_cols = all columns except c (and except target_col for non-target columns)
  X = dirty_df[feature_cols]  (all rows, categorical → OrdinalEncoded, NaN → median/mode imputed)
  y = dirty_df[c]

  if categorical c:
      rf = RandomForestClassifier(n_estimators=50)
      rf.fit(X, LabelEncoder(y))
  if numerical c:
      rf = RandomForestRegressor(n_estimators=50)
      col_std[c] = std(y)
      rf.fit(X, y)

Note: training uses ALL rows (including erroneous ones), which is acceptable
      because errors are sparse (≪50% per column) and act as negligible noise.
```

Large datasets are subsampled to 50,000 rows for efficiency.

**Compute step — categorical target column `c`:**
```
x_row = dirty_df.loc[row_idx, feature_cols]  (same encoding as fit)
proba  = rf.predict_proba(x_row)

h4_coherence_score = P( RF predicts dirty_value | other columns )
                   = 0.0  if dirty_value was unseen by LabelEncoder
```

**Compute step — numerical target column `c`:**
```
x_row    = dirty_df.loc[row_idx, feature_cols]
predicted = rf.predict(x_row)

h4_coherence_score = max(0, 1 − |predicted − dirty_value| / col_std[c])
                   = 1.0  if col_std == 0  (constant column)
```

### Intentional vs. unintentional signal

| h4_coherence_score | Interpretation |
|---|---|
| **≈ 1.0** | RF predicts exactly this value → value fits the row → **intentional** (manipulator maintained coherence) |
| **≈ 0.0** | RF predicts a very different value → breaks row context → **unintentional** (random noise) |

### Examples — Adult Income dataset

**Categorical column `education`** — RF trained on all other columns (age, workclass, hours-per-week, …)

| row | other columns context | dirty value | h4_coherence_score | Interpretation |
|---|---|---|---|---|
| 3 | age=35, workclass=Private, hours=40 | `"Bachelors"` | **0.82** | RF gives high probability to Bachelors for this row profile → intentional |
| 7 | age=20, workclass=Private, hours=20 | `"Doctorate"` | **0.03** | RF assigns near-zero probability — 20-year-olds don't typically hold Doctorates → unintentional |

**Numerical column `education-num`** — RF predicts from `education`, `age`, etc. (col_std ≈ 2.6)

| row | `education` (other col) | dirty `education-num` | RF predicted | h4_coherence_score | Interpretation |
|---|---|---|---|---|---|
| 5 | `"Bachelors"` | `13` | `13.1` | **0.96** | Matches prediction → consistent edit → intentional |
| 9 | `"Bachelors"` | `2` | `13.1` | **0.00** | 11-unit gap = 4.2σ → contradicts education column → unintentional |

---

## H5 — Error Pattern

**File:** `h5_error_pattern.py`  
**Question answered:** *Is this error part of a coordinated multi-cell edit, or an isolated accident?*

### Definition

H5 detects whether multiple logically linked columns were modified together in the same row — a hallmark of deliberate manipulation. Random noise and OCR errors strike columns independently; intentional manipulators edit clusters of related fields to preserve internal consistency.

Two complementary signals:

1. **Row error density** (`h5_error_count`): how many erroneous cells are in the same row?
2. **Co-dependent flag** (`h5_codependent_flag`): is this cell's logically linked partner column *also* erroneous in the same row?

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h5_error_count` | int | ≥ 1 | Number of erroneous cells in the same row (including this cell). |
| `h5_codependent_flag` | int | {0, 1} | 1 if at least one logically linked partner column is also erroneous in the same row. |

### Calculation

**Fit step — co-dependent pair discovery (three sources merged):**

```
Method A — User-supplied pairs (highest priority):
  Any (col_a, col_b) pairs passed via codependent_pairs=[...] argument.

Method B / Signal 1 — Name similarity:
  For every pair of columns (i, j):
    cleaned_i = remove digits, hyphens, underscores, tokens "num/no/id" from col_i
    ratio = SequenceMatcher(cleaned_i, cleaned_j).ratio()
    if ratio ≥ 0.80:
        add {col_i, col_j} to co-dependent pairs
  Example: "education" vs "education-num" → cleaned: "education" vs "education" → ratio = 1.0 ✓

Method B / Signal 2 — Mutual information:
  Using only fully-clean rows and up to 20 columns:
    For each column pair (i, j):
      MI = mutual_info_classif(X_j, y_i)   [treating all columns as discrete]
      if MI > 0.5:
          add {col_i, col_j} to co-dependent pairs

Final: codependent_pairs_ = union of all three sources.
```

**Compute step:**
```
For each erroneous cell (row_idx, col_name):

  h5_error_count = number of columns where mask_df.loc[row_idx] == 1

  h5_codependent_flag:
    partner_cols = { b for {col_name, b} ∈ codependent_pairs_ }
                  ∪ { a for {a, col_name} ∈ codependent_pairs_ }
    h5_codependent_flag = 1  if any(mask_df.loc[row_idx, p] == 1 for p in partner_cols)
                          0  otherwise
```

### Intentional vs. unintentional signal

| Feature value | Interpretation |
|---|---|
| `h5_error_count` ≥ 3 | Multiple cells in the same row modified → **intentional** coordinated edit |
| `h5_error_count` = 1 | Isolated error → consistent with **unintentional** accident |
| `h5_codependent_flag` = 1 | Linked partner column also modified → **intentional** (manipulator updated related fields together) |
| `h5_codependent_flag` = 0 | No co-dependent partner errors → **unintentional** |

### Examples — Adult Income dataset

**Auto-detected co-dependent pair:** `{education, education-num}` (name-similarity ratio = 1.0 after cleaning)

| row | errors in row | col_name | h5_error_count | h5_codependent_flag | Interpretation |
|---|---|---|---|---|---|
| 3 | `education`, `education-num` | `education` | **2** | **1** | Partner `education-num` is also erroneous → coordinated, intentional |
| 3 | same | `education-num` | **2** | **1** | Partner `education` is also erroneous → coordinated, intentional |
| 7 | only `age` marked | `age` | **1** | **0** | Isolated single-cell error → unintentional |
| 12 | `age`, `workclass`, `education` | `workclass` | **3** | **0** | 3 errors but `workclass` has no co-dependent partner → `h5_codep=0` |

---

## H6 — Column Importance

**File:** `h6_column_importance.py`  
**Question answered:** *Is this column statistically important to the outcome — making it an attractive attack target?*

### Definition

H6 measures how strongly a column correlates with the outcome variable (via mutual information), normalised to [0, 1]. The intuition is that a model-gaming adversary will preferentially corrupt **high-MI columns** because those have the greatest effect on model predictions.

> **Critical distinction from H7:** H6 is *statistical* importance, not *human behavioral* motivation. The column `fnlwgt` (sampling weight) may have high MI with the income target, but no human would touch it because nobody understands what it means. H7 captures that complementary signal.

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h6_column_importance` | float | [0, 1] | Normalised MI-based importance. **Per-column constant** — all erroneous cells in the same column get the same score. 1.0 = most important column; 0.0 = uncorrelated. |

### Calculation

**Fit step — supervised mode (target_col provided):**
```
Using only clean rows (mask == 0 for all columns in the subset):

  For each feature column c (c ≠ target_col):
    mi[c] = mutual_info_classif(X, y_target)   if target is categorical
           = mutual_info_regression(X, y_target) if target is numerical

  mi[target_col] = max(mi.values())            # target column gets max score

  importance_scores_[c] = mi[c] / max(mi.values())   # normalise to [0, 1]
```

**Fit step — unsupervised mode (no target_col):**
```
For each column c:
  peers = all other columns (sampled to 10 random peers if table > 30 cols)
  importance_scores_[c] = max( MI(c, peer) for peer in peers ) / global_max
```

All columns are OrdinalEncoded before MI computation. NaN values are median-imputed.

**Compute step:**
```
For each erroneous cell (row_idx, col_name):
  h6_column_importance = importance_scores_[col_name]   ← per-column constant
```

### Intentional vs. unintentional signal

| h6_column_importance | Interpretation |
|---|---|
| **≈ 1.0** | High-MI column → attractive attack target → **intentional** signal |
| **≈ 0.0** | Low-MI column → not worth manipulating → **unintentional** more likely |

### Examples — Adult Income dataset (target: `class`)

Typical normalised MI scores learned from the dataset:

| Column | h6_column_importance | Notes |
|---|---|---|
| `class` (target) | **1.00** | Gets max score by definition |
| `education-num` | **0.91** | Strong predictor of income → frequent attack target |
| `capital-gain` | **0.78** | High MI; manipulated to appear wealthier |
| `occupation` | **0.62** | Moderate MI |
| `age` | **0.45** | Moderate MI |
| `fnlwgt` | **0.08** | Very low MI → unlikely deliberate target |
| `race` | **0.06** | Low MI with income → H6 weak here (H8 covers the real privacy motivation) |

**Two erroneous cells in different columns of the same row:**

| row | col_name | h6_column_importance | Interpretation |
|---|---|---|---|
| 3 | `education-num` | **0.91** | High importance → error likely intentional |
| 3 | `fnlwgt` | **0.08** | Low importance → error likely accidental |

---

## H7 — User Incentive

**File:** `h7_user_incentive.py`  
**Question answered:** *Would a rational human actually want to, understand how to, and be able to change this value?*

### Definition

H7 captures **behavioral motivation** — three orthogonal signals about whether a human actor would have reason to manipulate this specific cell:

1. **Mutability** — can the value realistically be changed by a person? (SSN = immutable; education = freely changeable)
2. **Gain direction** — does the dirty value move the outcome in a favorable direction? (per-cell lookup, not a column average)
3. **Comprehensibility** — would a typical person even understand what this column measures? (fnlwgt = opaque; education = obvious)

> **Critical distinction from H6:** H6 is *statistical*, H7 is *behavioral*. The column `race` has low MI with income prediction (H6 low) but is frequently masked for privacy reasons (H7 high). The column `fnlwgt` has high MI (H6 high) but no human would touch it (H7 low). **Both are required; they are orthogonal.**

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h7_mutability` | float | {0.0, 0.5, 1.0} | Can a user realistically change this value? 0.0 = immutable, 0.5 = soft boundary (sensitive col), 1.0 = freely mutable |
| `h7_gain_direction` | float | [0, 1] | Does the dirty value shift toward a favorable outcome? 1.0 = favorable, 0.0 = unfavorable, 0.5 = neutral |
| `h7_comprehensibility` | float | {0.0, 0.3, 0.8, 1.0} | Would a typical person understand this column? 0.0 = opaque (`col1`), 0.3 = abbreviated (`fnlwgt`), 0.8 = plain word (`education`), 1.0 = multi-word phrase (`marital-status`) |

### Calculation

**Fit step — `h7_mutability`:**

Derived from column name alone (or overridden by user-supplied scores):

```
stripped = remove{-, _, space} from lowercase(col_name)

if any immutable keyword ∈ stripped:       → 0.0
   immutable keywords: {id, num, wgt, weight, score, code, key, index,
                        ssn, hash, uuid, timestamp, date, time}

elif any sensitive keyword ∈ col_name:     → 0.5
   sensitive keywords: {race, gender, sex, nationality, religion,
                        disability, ethnicity}

else:                                      → 1.0
```

**Fit step — `h7_comprehensibility`:**

Derived from column name alone:

```
if len(col_name) < 4 OR col_name contains digit:     → 0.0  (e.g. col1, id)
if any token ≤ 3 chars OR any token has no vowels:   → 0.3  (e.g. fnlwgt, hrs-per-wk)
if multiple word tokens:                              → 1.0  (e.g. marital-status)
else (single plain word ≥ 4 chars):                  → 0.8  (e.g. education, race)
```

**Fit step — `h7_gain_direction`:**

Computed using only clean rows:

```
if target_col is None:
    h7_gain_direction[col] = 0.5  for all columns (neutral)

if col == target_col:
    h7_gain_direction[col] = 1.0  (direct outcome manipulation)

if numerical col:
    r = Spearman correlation(col_values, encoded_target)    on clean rows
    h7_gain_direction[col] = clip(0.5 + 0.5 * r, 0, 1)
    → r = +1.0 → score 1.0 (increasing col → better outcome)
    → r = -1.0 → score 0.0 (increasing col → worse outcome)
    → r =  0.0 → score 0.5 (neutral)

if categorical col:
    For each unique value v:
        mean_target[v] = mean(encoded_target) over rows where col == v
    Sort values by mean_target descending.
    Top 50% of values → gain_direction[v] = 1.0   (favorable)
    Bottom 50% of values → gain_direction[v] = 0.0  (unfavorable)
```

**Compute step:**
```
For each erroneous cell (row_idx, col_name):
  h7_mutability        = mutability_[col_name]          ← per-column constant
  h7_comprehensibility = comprehensibility_[col_name]   ← per-column constant

  if gain_direction_[col_name] is a scalar:
    h7_gain_direction = gain_direction_[col_name]       ← per-column scalar (numerical col)
  else:
    h7_gain_direction = gain_direction_[col_name].get(str(dirty_value), 0.5)
                        ← per-cell lookup (categorical col)
```

### Intentional vs. unintentional signal

| Signal | High value | Low value |
|---|---|---|
| `h7_mutability` | Freely changeable column → **intentional** more likely | Immutable column (SSN/num) → error likely accidental |
| `h7_gain_direction` | Dirty value moves outcome favorably → **intentional** | Value moves outcome unfavorably or is neutral |
| `h7_comprehensibility` | Column is understandable → human can deliberately target it → **intentional** | Opaque column (`fnlwgt`) → unlikely to be a deliberate target |

### Examples — Adult Income dataset (target: `class`, 1 = >50K income)

**`h7_mutability` per column:**

| Column | h7_mutability | Reason |
|---|---|---|
| `education` | **1.0** | No immutable/sensitive keywords → freely mutable |
| `occupation` | **1.0** | Freely mutable |
| `race` | **0.5** | Contains sensitive keyword `race` → soft boundary |
| `sex` | **0.5** | Contains sensitive keyword `sex` → soft boundary |
| `education-num` | **0.0** | Contains immutable keyword `num` → immutable |
| `fnlwgt` | **0.0** | Contains immutable keyword `wgt` → immutable |

**`h7_comprehensibility` per column:**

| Column | h7_comprehensibility | Reason |
|---|---|---|
| `marital-status` | **1.0** | Multi-word phrase (marital + status) |
| `education` | **0.8** | Single plain word ≥ 4 chars |
| `race` | **0.8** | Single plain word ≥ 4 chars |
| `fnlwgt` | **0.3** | No vowels in `wgt` token → abbreviated/opaque |
| `hrs-per-wk` | **0.3** | Tokens `hrs` and `wk` are ≤ 3 chars |
| `id` | **0.0** | Length < 4 |

**`h7_gain_direction` — categorical column `education`** (>50K income is more common with higher education):

| dirty value | mean target | h7_gain_direction | Interpretation |
|---|---|---|---|
| `"Doctorate"` | 0.74 (top 50% of values) | **1.0** | Favorable — high earner profile → intentional |
| `"Masters"` | 0.57 (top 50%) | **1.0** | Favorable → intentional |
| `"HS-grad"` | 0.17 (bottom 50%) | **0.0** | Unfavorable — lower earner profile → unintentional or counter-productive |
| `"Preschool"` | 0.02 (bottom 50%) | **0.0** | Unfavorable → unintentional |

**`h7_gain_direction` — numerical column `capital-gain`** (strong positive Spearman with income, r ≈ +0.63):

| Column | Spearman r | h7_gain_direction | Interpretation |
|---|---|---|---|
| `capital-gain` | +0.63 | **0.815** | Higher capital-gain → more likely >50K → intentional direction |
| `capital-loss` | +0.03 | **0.515** | Nearly neutral |
| `age` | +0.23 | **0.615** | Moderate positive association |

---

## H8 — Sensitivity Flag

**File:** `h8_sensitivity_flag.py`  
**Question answered:** *Does this error look like privacy-motivated demographic masking?*

### Definition

H8 detects a specific, well-known intentional manipulation pattern: a user changes a **minority demographic value** to the **majority class** in order to "blend in" with the statistical population and avoid profiling.

Example in Adult Income dataset:
- `race`: 86% White → changing any non-White value to "White" is strong evidence of privacy-motivated masking.
- `sex`: 67% Male → moderate signal (less dominant majority).

Two orthogonal binary signals:

1. **Is this column a sensitive demographic attribute?** (column-level constant)
2. **Is the dirty value the majority class of that sensitive column?** (per-cell)

### Features in the final 13-column matrix

| Feature | Type | Range | Description |
|---|---|---|---|
| `h8_is_sensitive` | int | {0, 1} | 1 if this column is a sensitive demographic attribute. Per-column constant. |
| `h8_is_majority_value` | int | {0, 1} | 1 if the dirty value equals the majority class of this sensitive categorical column. Always 0 for: non-sensitive columns, numerical sensitive columns, or when no clean data exists. |

### Calculation

**Fit step — sensitive column detection:**

```
Step 1 — User-supplied (always marked sensitive):
    Any column name in sensitive_cols=[...] argument.

Step 2 — Auto-detection (remaining columns):
    For each column c not in user-supplied set:
        if any keyword ∈ {race, gender, sex, age, nationality, religion,
                          disability, marital, ethnic, origin, orientation}
           is a case-insensitive substring of col_name:
            mark c as sensitive

sensitive_cols_ = union of user-supplied + auto-detected
```

**Fit step — majority class detection:**

```
For each column c in sensitive_cols_:
  if c is NOT numerical:
      clean_vals = dirty_df[mask_df[c] == 0][c]   (clean cells only)
      majority_class_[c] = str(mode(clean_vals))   ← most frequent value
  if c is numerical:
      majority_class_[c] = None                   ← not applicable
      add c to _numerical_cols_
```

**Compute step:**
```
For each erroneous cell (row_idx, col_name):

  h8_is_sensitive = 1  if col_name ∈ sensitive_cols_
                    0  otherwise

  h8_is_majority_value:
    if col_name ∉ sensitive_cols_:            → 0
    if col_name ∈ _numerical_cols_:           → 0  (no meaningful majority for continuous)
    if majority_class_[col_name] is None:     → 0  (no clean data)
    else:
      h8_is_majority_value = 1  if str(dirty_value) == majority_class_[col_name]
                             0  otherwise
```

### Intentional vs. unintentional signal

| Feature values | Interpretation |
|---|---|
| `h8_is_sensitive=1` + `h8_is_majority_value=1` | **Strongest intentional signal** — changed to majority class in a sensitive column → privacy masking |
| `h8_is_sensitive=1` + `h8_is_majority_value=0` | Sensitive column but not the majority → possibly unintentional or a non-masking intentional edit |
| `h8_is_sensitive=0` | Not a sensitive column → both features = 0 |

### Examples — Adult Income dataset

Auto-detected sensitive columns: `race`, `sex`, `age`, `marital-status`  
Majority classes (from clean cells): `race → "White"` (86%), `sex → "Male"` (67%)

**Column `race`:**

| row | dirty value | h8_is_sensitive | h8_is_majority_value | Interpretation |
|---|---|---|---|---|
| 5 | `"White"` | **1** | **1** | Changed to majority class → strong privacy masking signal → intentional |
| 8 | `"Asian-Pac-Islander"` | **1** | **0** | Not the majority → not a masking edit |
| 12 | `"Black"` | **1** | **0** | Not the majority → not a masking edit |

**Column `age`** (numerical, sensitive but no meaningful majority):

| row | dirty value | h8_is_sensitive | h8_is_majority_value | Reason |
|---|---|---|---|---|
| 9 | `999` | **1** | **0** | `age` is numerical → `h8_is_majority_value` always 0 |

**Column `education`** (not a sensitive column):

| row | dirty value | h8_is_sensitive | h8_is_majority_value | Reason |
|---|---|---|---|---|
| 3 | `"Bachelors"` | **0** | **0** | Not a sensitive column → both features always 0 |

### Known limitations

- Both features are **per-column constants** (or near-constants), so they cannot distinguish two different erroneous cells in the same column from each other. This makes H8 the weakest discriminator.
- The majority-class signal is only meaningful when the majority is dominant (e.g. `race` at 86% White). For `sex` at 67% Male the signal is weaker.
- `h8_is_majority_value` is always 0 for numerical sensitive columns (e.g. `age`) because "majority value" has no meaningful interpretation for continuous distributions.

---

## Feature Matrix Summary

The final pipeline output is a `(N_errors × 13)` DataFrame with `MultiIndex (row_idx, col_name)`:

| # | Feature | Source | Type | Range | Notes |
|---|---|---|---|---|---|
| 1 | `h1_plausible` | H1 | int | {0, 1} | All column types |
| 2 | `h2_min_edit_distance` | H2 | float\|NaN | [0, 10] or NaN | Categorical only; NaN for numerical |
| 3 | `h2_is_obfuscation` | H2 | int\|NaN | {0, 1, NaN} | Categorical only; NaN for numerical |
| 4 | `h3_distribution_score` | H3 | float | [0, 1] | All column types |
| 5 | `h4_coherence_score` | H4 | float | [0, 1] | All column types |
| 6 | `h5_error_count` | H5 | int | ≥ 1 | All column types |
| 7 | `h5_codependent_flag` | H5 | int | {0, 1} | All column types |
| 8 | `h6_column_importance` | H6 | float | [0, 1] | Per-column constant |
| 9 | `h7_mutability` | H7 | float | {0.0, 0.5, 1.0} | Per-column constant |
| 10 | `h7_gain_direction` | H7 | float | [0, 1] | Per-cell (categorical), per-column (numerical) |
| 11 | `h7_comprehensibility` | H7 | float | {0.0, 0.3, 0.8, 1.0} | Per-column constant |
| 12 | `h8_is_sensitive` | H8 | int | {0, 1} | Per-column constant |
| 13 | `h8_is_majority_value` | H8 | int | {0, 1} | Sensitive categorical only; 0 otherwise |

> **Internal-only outputs not in the matrix:**  
> H1 also computes `h1_in_vocab` and `h1_in_range` — these are diagnostic helpers present in `h1.compute()`'s output DataFrame but are **not selected** by the pipeline when assembling the 13-column matrix.

### NaN handling

H2 produces `NaN` for numerical columns. Before classification:

```python
X = np.nan_to_num(feat_df.values, nan=-999.0)
```

The sentinel value `-999.0` is distinguishable from any real feature value, so the Random Forest can learn to treat "this is a numerical column, H2 is not applicable" as a valid signal.

### Combined score patterns

| Pattern | Likely label |
|---|---|
| H1=1, H3≈1, H4≈1, H7_gain=1 | **Intentional** — plausible, central, coherent, outcome-favorable |
| H1=0, H3≈0, H2_edit_dist=1 | **Unintentional** — out-of-domain, extreme, one-char typo |
| H2_obf=1, H8_sensitive=1, H8_majority=1 | **Intentional** — privacy masking via obfuscation token |
| H5_count≥3, H5_codep=1 | **Intentional** — coordinated multi-column edit |
| H1=0, H3≈0, H4≈0, H5_count=1 | **Unintentional** — isolated random substitution breaking all structure |

---

*Last updated: 2026-03-23 | Pipeline: H1–H8 → 13 features → RandomForest*  
*Empirical importance scores from: `run_20260320_165116` (5-fold CV, LLM + Kireev datasets)*
