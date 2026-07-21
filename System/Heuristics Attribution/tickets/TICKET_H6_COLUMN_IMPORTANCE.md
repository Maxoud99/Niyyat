# TICKET H6 — Column Importance

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h6_column_importance.py
```

## Class
`H6ColumnImportance(BaseHeuristic)`

---

## What this heuristic answers

> **Does this column matter statistically? A column with high mutual information
> to the outcome is a more attractive target for intentional manipulation.**

This is **statistical importance** (MI-based), not human behavioral motivation.
It asks: "would corrupting this column change prediction outcomes?" A person
trying to game a model would target high-MI columns.

**Critical distinction from H7 (User Incentive):**
- `fnlwgt` (sampling weight in Adult dataset): high MI with outcome → H6 high.
  But a human wouldn't touch it because they don't understand it → H7 low.
- `race`: low MI with income prediction → H6 low.
  But a human might mask it for fairness/privacy → H7 high.
- Both signals are needed. They are NOT redundant.

---

## Output features — EXACTLY 1 column (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h6_column_importance` | float [0, 1] | Normalized statistical importance of this column. 1.0 = most important. 0.0 = least important. Same value for all erroneous cells in the same column. |

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, target_col=None):
```

Compute one importance score per column using **clean cells only** (mask == 0
rows where the target is also clean).

### Case A — `target_col` is provided (supervised)

1. Separate features `X` (all columns except `target_col`) and target `y` (`target_col`).
2. Encode all categoricals with `OrdinalEncoder`.
3. Impute NaN: median for numerical, mode for categorical.
4. Use `sklearn.feature_selection.mutual_info_classif(X, y)` if `y` is
   categorical, else `mutual_info_regression(X, y)` if `y` is numerical.
5. Result: one MI score per feature column.
6. For `target_col` itself: set importance = max of all other columns' scores
   (it's indirectly important by definition).
7. Normalize all scores to [0, 1] by dividing by the maximum.
8. Store in `self.importance_scores_: dict[col_name → float]`.

### Case B — No `target_col` (unsupervised)

For each column `c`:
- Compute MI between `c` and every other column.
- `importance[c] = max MI over all other columns`.
- Normalize to [0, 1].
- Store in `self.importance_scores_`.

**Cap computation**: if the table has >30 columns AND no target_col, compute
MI only between each column and a random sample of 10 other columns to limit cost.

---

## compute() specification

For each erroneous cell at (row_idx, col_name):
- Look up `self.importance_scores_[col_name]`.
- Return `h6_column_importance` ∈ [0, 1].

No row-level computation. This is a per-column constant.

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
import numpy as np
from h6_column_importance import H6ColumnImportance

np.random.seed(42)
n = 200
# "education-num" strongly determines "income". "age" is weakly correlated.
# "fnlwgt" is random (not correlated with income).
education_num = np.random.randint(5, 16, n)
income = (education_num > 10).astype(int)  # direct deterministic relationship
age = np.random.randint(20, 65, n)
fnlwgt = np.random.randint(100000, 500000, n)  # noise column

dirty = pd.DataFrame({
    "education-num": education_num,
    "age":           age,
    "fnlwgt":        fnlwgt,
    "income":        income,
})
mask = pd.DataFrame(np.zeros_like(dirty.values, dtype=int), columns=dirty.columns)
# Mark a few errors in education-num and fnlwgt
mask.loc[[5, 10, 15], "education-num"] = 1
mask.loc[[20, 25], "fnlwgt"] = 1

h6 = H6ColumnImportance()
h6.fit(dirty, mask, target_col="income")
result = h6.compute(dirty, mask)
print(result.to_string())

# Expected:
# education-num: h6_column_importance ≈ 1.0  (most correlated with income)
# fnlwgt:        h6_column_importance ≈ 0.0  (random → MI ≈ 0)
# income error rows: h6_column_importance = max of other columns
```

---

## Constraints
- Output is per-column constant. All erroneous cells in the same column get the
  same score. Do NOT vary by row.
- Must handle both categorical and numerical columns in the target.
- Skip `target_col` from `X` features during MI computation (not a predictor of itself).
- If a column has zero variance in clean cells → MI = 0.0 → importance = 0.0.
