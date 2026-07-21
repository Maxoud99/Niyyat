# H4 — Row Coherence

**File:** `heuristics/h4_row_coherence.py`  
**Class:** `H4RowCoherence(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H4 answers the question:

> **Does this dirty value "make sense" given the other values in the same row?**

Intentional errors (adversarial data manipulation) are inserted by someone who
*understands* the record. The manipulated value therefore tends to remain
**contextually coherent** with the other fields in the same row — for example,
a high `education-num` is still paired with a plausible `education` label.

Unintentional errors (random substitution, OCR noise, transcription mistakes)
tend to **break row-level correlation structure** — the injected value is
inconsistent with what the rest of the row predicts.

H4 is the **key heuristic** in the system. It has the most discriminative
power for structured / correlated datasets, because it models the *joint*
relationship between columns rather than treating each column in isolation.

### Relationship to other heuristics

| Heuristic | What it checks | Signal type |
|---|---|---|
| **H1** | Is the value within the column's known domain? | Binary, per-column |
| **H2** | Does the string pattern look like obfuscation? | Binary, per-column |
| **H3** | Is the value central or extreme in its column's distribution? | Continuous, per-column |
| **H4** | Does the value fit the *rest of the row*? | Continuous, cross-column |

H4 is the only heuristic that crosses column boundaries, making it uniquely
sensitive to row-level correlation breaks caused by random noise.

---

## Output

H4 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h4_coherence_score` | float ∈ [0, 1] | Row-coherence score. **High (≈1) = dirty value fits row context → intentional signal. Low (≈0) = dirty value breaks row context → unintentional signal.** |

> **Important:** `h4_coherence_score` is the **single** output feature.
> Intermediate results (RF prediction, predicted probability, z-score from
> prediction) are never exposed as output columns.

---

## Approach

One `RandomForestClassifier` or `RandomForestRegressor` is trained **per
column** using the full dirty dataset as training data.  Training on dirty
data is acceptable because errors are sparse (≪50 % per column), so they act
as negligible noise.

For each erroneous cell `(row, c)`:
1. Load the predictor trained for column `c`.
2. Build a feature vector from all other columns in that row (applying the
   same encoding and imputation used during training).
3. Ask the predictor: *"given these other columns, what would column `c` be?"*
4. Compare the prediction to the actual dirty value to produce a score.

---

## Column Type Detection

| Rule | Classification |
|---|---|
| > 90 % of non-null values parse as `float` | `num` (numerical) |
| ≤ 90 % | `cat` (categorical) |

Override per-column via `col_types` passed to `fit()`:

```python
h4.fit(dirty_df, mask_df, col_types={"age": "num", "zip_code": "cat"})
```

---

## Scoring Formulas

### Categorical target column

```
h4_coherence_score = P(RF predicts dirty_value | other columns in the row)
```

The `RandomForestClassifier.predict_proba()` is called on the row's feature
vector. The probability assigned to the dirty value's class becomes the score.

**Edge cases:**

| Condition | Score |
|---|---|
| Dirty value was unseen by `LabelEncoder` during fit (OOV) | `0.0` |
| Dirty value's class was not present in any RF leaf | `0.0` |

### Numerical target column

```
predicted  = RF.predict(x_row)
h4_coherence_score = max(0, 1 − |predicted − dirty_value| / col_std)
```

Values whose predicted result matches the dirty value exactly score 1.0;
values farther than `col_std` away score ≤ 0.0 (clamped to 0).

**Edge cases:**

| Condition | Score |
|---|---|
| `col_std == 0` (constant column) | `1.0` — all values are equally "central" |
| Dirty value cannot be cast to `float` | `0.0` |

---

## API

### `fit(dirty_df, mask_df, target_col=None, col_types=None)`

Trains one RF predictor per column using the dirty dataset.

| Parameter | Type | Description |
|---|---|---|
| `dirty_df` | `pd.DataFrame` | The dirty dataset (sparse errors are tolerable noise). |
| `mask_df` | `pd.DataFrame` | Binary mask. Used only to check structure; all rows are used for training. |
| `target_col` | `str`, optional | If provided, this column is **excluded from the feature matrix** when training predictors for all other columns (prevents leakage from an outcome variable). The predictor for `target_col` itself is still trained using all remaining columns. |
| `col_types` | `dict`, optional | Per-column type override, e.g. `{'age': 'num', 'education': 'cat'}`. |

**Per-column training procedure:**

1. Feature columns = all columns **except** `c` (and `target_col` when `target_col != c`).
2. Categorical features → `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
3. NaN imputation: numerical → **median**; categorical → **mode**.
4. For **categorical** `c` → `RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)` on LabelEncoder-encoded `y`.
5. For **numerical** `c` → `RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)`.
6. If the dataset has **> 100 000 rows**, up to **50 000 rows** are sampled for fitting (random state 42).

After `fit()` completes:
- `self.predictors_[c]` holds the fitted RF model for each column `c`.
- `self.col_stats_[c]` holds type metadata, encoders, imputation values, and
  (for numerical columns) the column standard deviation.
- `self.is_fitted` is set to `True`.

```python
# Example col_stats_ structure after fit:
{
    "education": {
        "type": "cat",
        "label_encoder": <LabelEncoder>,
        "feature_cols": ["education-num", "age", "hours-per-week"],
        "cat_feature_cols": [],
        "num_feature_cols": ["education-num", "age", "hours-per-week"],
        "ordinal_encoder": None,        # None when no categorical features
        "impute_values": {"education-num": 9.0, "age": 30.0, "hours-per-week": 40.0},
    },
    "education-num": {
        "type": "num",
        "std": 1.643,
        "feature_cols": ["education", "age", "hours-per-week"],
        "cat_feature_cols": ["education"],
        "num_feature_cols": ["age", "hours-per-week"],
        "ordinal_encoder": <OrdinalEncoder>,
        "impute_values": {"education": "HS-grad", "age": 30.0, "hours-per-week": 40.0},
    },
    ...
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Iterates over every `(row_idx, col_name)` where `mask_df == 1` and applies
the appropriate scoring formula.

Returns a `pd.DataFrame` with columns `[row_idx, col_name, h4_coherence_score]`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`. Calls `fit()` then
`compute()` in one step.

---

## Self-test Walkthrough

The self-test uses a minimal dataset where `education` and `education-num`
are highly correlated:

| idx | education | education-num | age | hours-per-week | mask (edu-num) |
|---|---|---|---|---|---|
| 0 | HS-grad | 9 | 25 | 40 | 0 (clean) |
| 1 | HS-grad | 9 | 30 | 40 | 0 (clean) |
| 2 | Bachelors | 13 | 35 | 45 | 0 (clean) |
| 3 | HS-grad | 9 | 40 | 40 | 0 (clean) |
| 4 | **Bachelors** | **5** ← error | 28 | 40 | **1 (error)** |

Row 4 has `education = "Bachelors"` but `education-num = 5`.  
The RF predictor for `education-num`, trained on rows 0–4, learns that
`Bachelors` maps to `education-num ≈ 13`.  
When asked to predict for row 4's feature context (`Bachelors`, age=28, …),
the predicted value is ~13, while the dirty value is 5.

```
h4_coherence_score = max(0, 1 − |13 − 5| / col_std)
                   ≈ LOW  (≈ 0.0 – 0.4)
```

**Run the self-test:**

```bash
# from the project root
conda run --name base python -c "
import pandas as pd, sys; sys.path.insert(0, '.')
from error_detection_system.src.attribution.heuristics import H4RowCoherence

dirty = pd.DataFrame({
    'education':     ['HS-grad', 'HS-grad', 'Bachelors', 'HS-grad',    'Bachelors'],
    'education-num': [9,          9,          13,          9,            5],
    'age':           [25,         30,         35,          40,           28],
    'hours-per-week':[40,         40,         45,          40,           40],
})
mask = pd.DataFrame({
    'education':     [0, 0, 0, 0, 0],
    'education-num': [0, 0, 0, 0, 1],
    'age':           [0, 0, 0, 0, 0],
    'hours-per-week':[0, 0, 0, 0, 0],
})

h4 = H4RowCoherence()
h4.fit(dirty, mask)
result = h4.compute(dirty, mask)
print(result.to_string(index=False))
"
```

**Expected output:**

```
 row_idx      col_name  h4_coherence_score
       4 education-num            0.336...
```

Score is LOW (≤ 0.5) because the dirty value `5` is far from the RF's
prediction of `≈ 13` given that `education = "Bachelors"`.

---

## Numeric Scoring Walkthrough

For the numerical case, suppose the RF predicts `predicted = 13.0` for row 4,
and `col_std = 1.64` (std of `[9, 9, 13, 9, 5]`):

```
h4_coherence_score = max(0, 1 − |13.0 − 5| / 1.64)
                   = max(0, 1 − 4.88)
                   = max(0, −3.88)
                   = 0.0
```

In practice the RF does not predict exactly 13.0 with only 5 training rows,
so the score is a small positive number rather than exactly 0.0.

---

## Categorical Scoring Walkthrough

Suppose a categorical column `job` has classes `["admin", "tech", "sales"]`
and the RF predicts probabilities `[0.70, 0.20, 0.10]` for those classes.

| dirty value | predicted probability | h4_coherence_score |
|---|---|---|
| `"admin"` | 0.70 | **0.700** (high → coherent) |
| `"tech"` | 0.20 | **0.200** (medium) |
| `"sales"` | 0.10 | **0.100** (low → incoherent) |
| `"unknown"` (OOV) | — | **0.000** (unseen by LabelEncoder) |

---

## Design Constraints

- **Exactly one output column** beyond `row_idx` / `col_name`: `h4_coherence_score`.
- The predictor for column `c` **never uses column `c` as a feature** — it is
  always excluded from the feature matrix.
- `target_col` parameter allows callers to exclude an outcome/label column
  from feature matrices to prevent data leakage.
- No global correlation matrix is computed — only per-column RF predictors.
- **No column is skipped during training**, even if it has many NaN values;
  NaN is imputed (median for numerical, mode for categorical).
- Both categorical and numerical columns are handled in the same dataset.
- Large datasets (> 100 000 rows) are subsampled to 50 000 rows for fitting
  to keep training time acceptable; `random_state=42` ensures reproducibility.
- All edge cases (OOV values, zero std, non-numeric dirty values, empty
  columns) are handled gracefully and return 0.0 or 1.0 as documented.

---

## Performance Notes

| Dataset size | Fit time (approx.) | Notes |
|---|---|---|
| < 10 k rows | < 5 s | Fast for typical evaluation |
| 10 k – 100 k rows | 10 – 60 s | One RF per column, `n_jobs=-1` |
| > 100 k rows | Training capped at 50 k rows; similar to above | Subsampling applied automatically |

The main cost is the number of columns, since one RF is trained per column.
For datasets with many columns (> 50), consider profiling and possibly reducing
`n_estimators` or applying column selection.

---

## Limitations

- H4 is trained on the dirty dataset, not a clean one. With very high error
  rates (> 30 % per column) the RF may learn corrupted correlations. In
  practice, error rates are ≪ 10 % so this is a negligible effect.
- With very small datasets (< 50 rows), the RF has insufficient training
  signal and scores will be noisy. H1–H3 should be weighted more heavily
  in such cases.
- H4 does not distinguish *direction* of coherence (e.g., a value could be
  coherent with the row but still an intentional error if all correlated
  columns were simultaneously manipulated). This is an inherent limitation
  of single-column prediction.
