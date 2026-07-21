# H6 — Column Importance

**File:** `heuristics/h6_column_importance.py`  
**Class:** `H6ColumnImportance(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H6 answers the question:

> **Does this column matter statistically?  
> A column with high mutual information to the outcome is a more attractive
> target for intentional, model-gaming manipulation.**

Someone who wants to game a predictive model will target the columns that
have the most leverage over predictions — i.e. those with high **mutual
information (MI)** with the target.  Corrupting a low-MI column wastes
effort because it barely moves the model's output.

H6 captures this *statistical attractiveness* signal as a per-column
constant in [0, 1].

---

## Critical Distinction from H7 (User Incentive)

H6 is **statistical** importance, not human behavioral motivation.
Both signals are needed and are **NOT redundant**:

| Column | H6 (statistical MI) | H7 (behavioral motivation) | Reason |
|---|---|---|---|
| `fnlwgt` (Adult sampling weight) | **High** — high MI with income | **Low** — humans don't understand it | Statistically predictive but socially opaque |
| `race` | **Low** — low MI with income prediction | **High** — manipulators hide it for fairness / privacy | Socially sensitive but statistically weak |
| `education-num` | **High** — directly determines income | **High** — people inflate credentials | Both signals agree |

---

## Output

H6 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h6_column_importance` | float ∈ [0, 1] | Normalised statistical importance. **Per-column constant**: every erroneous cell in the same column gets the same score. 1.0 = most important. 0.0 = least important / uncorrelated. |

> **Key constraint:** `h6_column_importance` is a **per-column constant**.
> All erroneous cells in the same column receive the same score — it does
> not vary by row within a column.  This is by design: importance is a
> property of the *column*, not of individual cells.

---

## Scoring Modes

### Supervised mode — `target_col` provided

```python
h6.fit(dirty_df, mask_df, target_col="income")
```

1. Feature columns `X` = all columns except `target_col`.
2. Target `y` = `target_col`.
3. All categoricals are `OrdinalEncoder`-encoded; NaN is imputed with median.
4. MI function selected by target type:
   - `y` is categorical → `sklearn.feature_selection.mutual_info_classif`
   - `y` is numerical   → `sklearn.feature_selection.mutual_info_regression`
5. Result: one raw MI score per feature column.
6. `target_col` itself is assigned importance = **max MI of all feature columns** (it is
   indirectly important by definition).
7. All scores are **normalized to [0, 1]** by dividing by the maximum.

### Unsupervised mode — no `target_col`

```python
h6.fit(dirty_df, mask_df)  # no target_col
```

For each column `c`:
- Compute MI between `c` and every other column (treating each other column
  as the target in turn).
- `importance[c]` = **max MI** observed across all other columns.

**Sampling cap for wide tables**: if the table has **> 30 columns**, only
**10 randomly selected peers** are sampled per column to limit O(n²) cost.

All raw scores are then **normalized to [0, 1]**.

---

## Column Type Detection

A column is treated as **numerical** when > 90 % of its non-null values can
be cast to `float`; otherwise it is treated as **categorical**.

This rule applies to:
- Feature columns during encoding.
- The target column when choosing between `mutual_info_classif` vs
  `mutual_info_regression`.
- Peer columns during unsupervised MI computation.

---

## Clean Cells Only

`fit()` computes MI using only **fully-clean rows** — rows where
`mask_df == 0` for **every** column.  This prevents erroneous cell values
from distorting the mutual-information estimates.

If fewer than **10** fully-clean rows are available, a `UserWarning` is
issued and all importance scores are set to `0.0` (uniform fallback).

---

## API

### `fit(dirty_df, mask_df, target_col=None)`

Learns per-column importance scores.

| Parameter | Type | Description |
|---|---|---|
| `dirty_df` | `pd.DataFrame` | The dirty dataset. |
| `mask_df` | `pd.DataFrame` | Binary mask aligned with `dirty_df` (0 = clean, 1 = erroneous). Same shape and column names. |
| `target_col` | `str`, optional | Name of the outcome / label column. If provided, supervised MI is used. If `None`, unsupervised pairwise MI is used. |

After `fit()` completes:
- `self.importance_scores_` — `dict[str, float]` mapping each column name to
  its normalized importance score.
- `self.is_fitted` is set to `True`.

```python
# Example importance_scores_ after supervised fit on Adult Income dataset
{
    "education-num": 1.0,     # direct deterministic signal
    "income":        1.0,     # target column → max of others
    "age":           0.12,    # weakly correlated
    "fnlwgt":        0.0,     # random noise — MI ≈ 0
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Looks up the pre-computed importance score for each erroneous cell's column.

**No row-level computation is done at this stage** — the score is simply
read from `self.importance_scores_[col_name]` for every `(row_idx, col_name)`
where `mask_df == 1`.

Returns a `pd.DataFrame` with columns
`[row_idx, col_name, h6_column_importance]`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`.  Calls `fit()` then
`compute()` in one step.

---

## Self-test Walkthrough

The self-test uses a 200-row synthetic dataset with four columns:

| Column | Role | Expected importance |
|---|---|---|
| `education-num` | Direct cause of `income` (`income = education-num > 10`) | ≈ **1.0** |
| `income` | Target column → assigned max of others | ≈ **1.0** |
| `age` | Weakly correlated with income | ≈ **0.0–0.2** |
| `fnlwgt` | Pure random noise (uniform 100k–500k) | ≈ **0.0** |

Five errors are injected:

| Error positions | Column |
|---|---|
| rows 5, 10, 15 | `education-num` |
| rows 20, 25 | `fnlwgt` |

**Expected output:**

```
 row_idx      col_name  h6_column_importance
       5 education-num                   1.0
      10 education-num                   1.0
      15 education-num                   1.0
      20        fnlwgt                   0.0
      25        fnlwgt                   0.0
```

**Run the self-test:**

```bash
# from the heuristics/ directory
python h6_column_importance.py
```

**Or run via the package (from the project root):**

```bash
conda run --name base python -c "
import pandas as pd, numpy as np
from error_detection_system.src.attribution.heuristics import H6ColumnImportance

np.random.seed(42)
n = 200
edu = np.random.randint(5, 16, n)
dirty = pd.DataFrame({
    'education-num': edu,
    'age':           np.random.randint(20, 65, n),
    'fnlwgt':        np.random.randint(100000, 500000, n),
    'income':        (edu > 10).astype(int),
})
mask = pd.DataFrame(0, index=dirty.index, columns=dirty.columns)
mask.loc[[5,10,15], 'education-num'] = 1
mask.loc[[20,25], 'fnlwgt'] = 1

h6 = H6ColumnImportance()
print(h6.fit_compute(dirty, mask, target_col='income').to_string(index=False))
"
```

---

## Relationship to Other Heuristics

| Heuristic | What it checks | Signal type | Relationship to H6 |
|---|---|---|---|
| **H1** | Is the value within the column's known domain? | Binary, per-column | Independent — domain validity vs. statistical leverage |
| **H2** | Does the string pattern look like obfuscation? | Binary, per-column | Independent — pattern anomaly vs. MI importance |
| **H3** | Is the value central or extreme in its column distribution? | Continuous, per-cell | Complementary — H3 varies by row; H6 is constant per column |
| **H4** | Does the value fit the rest of the row (RF predictor)? | Continuous, cross-column | Complementary — row coherence vs. column attractiveness |
| **H5** | Are multiple linked columns simultaneously erroneous? | Integer + binary, cross-row | Independent — coordination pattern vs. column importance |
| **H6** | Does this column have high MI with the outcome? | Continuous, **per-column** | This heuristic |
| **H7** | Would a human be motivated to manipulate this column? | Continuous, per-column | Complementary — H6 is statistical; H7 is behavioral |

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| Fewer than 10 fully-clean rows | `UserWarning`; all scores set to `0.0` |
| Column with zero variance in clean cells | MI = 0 naturally → importance = 0.0 |
| All feature columns have MI = 0 | All scores = 0.0 (normalize of all-zeros) |
| Only `target_col` present (no feature columns) | `target_col` scores 1.0 |
| `target_col` not in `dirty_df` | `ValueError` raised immediately |
| `dirty_df` and `mask_df` shape mismatch | `ValueError` raised immediately |
| Unsupervised on a table with 1 column | No peers; importance = 0.0 |
| NaN values in clean cells | Imputed with median before MI computation |
| `compute()` called before `fit()` | `RuntimeError` via `_check_fitted()` |

---

## Performance Notes

| Dataset size | Supervised fit time | Unsupervised fit time | Notes |
|---|---|---|---|
| < 10 k rows, < 20 cols | < 0.5 s | < 1 s | Negligible |
| 10 k–100 k rows, 20 cols | 0.5–3 s | 1–10 s | MI dominates |
| > 100 k rows | Similar to above | Similar, capped | Row count has low impact on MI cost |
| > 30 cols, unsupervised | — | 10 peers sampled | O(cols × 10) instead of O(cols²) |

`compute()` is O(errors) — purely a dictionary lookup per erroneous cell.

---

## Design Constraints

- Exactly **one output feature column** beyond `row_idx` / `col_name`:
  `h6_column_importance`.
- The score is a **per-column constant** — it does NOT vary by row.
- Uses **only fully-clean rows** (`mask == 0` for all columns in that row)
  to build MI statistics — no data leakage from erroneous cells.
- No hardcoded column names or dataset-specific logic.
- `importance_scores_` is a plain Python `dict` (JSON-serialisable).
- Both categorical and numerical columns are handled automatically.
- `target_col` is excluded from the feature matrix `X` during MI computation
  (it cannot be a predictor of itself).
- All edge cases (empty columns, zero variance, few clean rows) are handled
  gracefully without raising exceptions (except for input shape/key errors).

---

## Limitations

- MI estimation via `mutual_info_classif` / `mutual_info_regression` has
  variance for small sample sizes (< ~100 rows).  The `_MIN_CLEAN_ROWS = 10`
  guard only prevents a crash; results below ~100 clean rows should be
  treated with caution.
- Unsupervised sampling (10 random peers) introduces non-determinism for
  tables with > 30 columns, though the random seed is fixed at 42.
- MI measures statistical dependence, not causal importance.  A column that
  is a *consequence* of the target (not a *cause*) will still score high.
- High H6 score for a column does **not** imply the specific error in that
  column is intentional — it only means the column *could* be an attractive
  manipulation target.  Combine with other heuristics for the full picture.
