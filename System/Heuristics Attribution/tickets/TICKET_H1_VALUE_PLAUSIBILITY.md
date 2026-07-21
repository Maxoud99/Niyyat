# TICKET H1 — Value Plausibility

## Depends on
TICKET_000_SETUP.md (base.py must exist first)

## File to create
```
heuristics/h1_value_plausibility.py
```

## Class
`H1ValuePlausibility(BaseHeuristic)`

---

## What this heuristic answers

> **Is this dirty value a legitimate member of this column's domain?**

A value that doesn't belong to the column's domain is strong evidence of an
unintentional error (typo, corruption). A value that does belong is consistent
with intentional manipulation.

Known limitation (document in docstring): 29% of intentional errors in the Adult
Income dataset use obfuscation tokens (`nan`, `Unknown`, `—`) which are also
out-of-vocabulary — H1 will wrongly score them as implausible. H2 is responsible
for rescuing those cases.

---

## Output features — EXACTLY these 3 columns (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h1_plausible` | int (0 or 1) | 1 if the dirty value is a plausible domain member |
| `h1_in_vocab`  | int or NaN   | Categorical only: 1 if value is in the clean vocabulary. NaN for numerical columns. |
| `h1_in_range`  | int or NaN   | Numerical only: 1 if value is within [p5, p95] of clean column values. NaN for categorical columns. |

`h1_plausible` is the single summary:
- Categorical → equals `h1_in_vocab`
- Numerical   → equals `h1_in_range`

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, col_types=None):
```

**Parameters:**
- `dirty_df` : the dirty dataset
- `mask_df`  : binary mask (0=clean, 1=error), same shape as dirty_df
- `col_types`: optional dict e.g. `{'age': 'num', 'education': 'cat'}` to
               override auto-detection per column

**What it does:**

1. For each column in dirty_df, determine column type:
   - Auto-detection rule: try `pd.to_numeric(col.dropna(), errors='coerce')`.
     If the fraction of successfully converted values > 0.9 → numerical, else categorical.
   - If `col_types` is provided, it overrides auto-detection for named columns.

2. For **categorical** columns:
   - Extract clean cells: `dirty_df.loc[mask_df[col] == 0, col]`
   - Build vocabulary: set of unique non-null string values (cast with `str()`)
   - Store as `self.col_stats_[col] = {'type': 'cat', 'vocab': set(...)}`

3. For **numerical** columns:
   - Extract clean cells as float (coerce errors → NaN, then dropna)
   - Compute p5 = 5th percentile, p95 = 95th percentile
   - Store as `self.col_stats_[col] = {'type': 'num', 'p5': float, 'p95': float}`

4. Set `self.is_fitted = True`

---

## compute() specification

```python
def compute(self, dirty_df, mask_df) -> pd.DataFrame:
```

Iterate over every (row_idx, col_name) where `mask_df[col_name].iloc[row_idx] == 1`.

For each erroneous cell, look up `self.col_stats_[col_name]` and apply:

**Categorical column:**
```
dirty_val = str(dirty_df.loc[row_idx, col_name])
h1_in_vocab  = 1 if dirty_val in vocab and dirty_val not in ('nan', 'None', '') else 0
h1_in_range  = NaN
h1_plausible = h1_in_vocab
```

**Numerical column:**
```
dirty_val = try cast to float; if cast fails → treat as categorical (h1_in_vocab=0, h1_in_range=NaN)
if cast succeeds:
    h1_in_range  = 1 if p5 <= dirty_val <= p95 else 0
    h1_in_vocab  = NaN
    h1_plausible = h1_in_range
```

**Null/NaN value (either type):**
```
h1_plausible = 0, h1_in_vocab = 0 (cat) or NaN (num), h1_in_range = NaN (cat) or 0 (num)
```

---

## Self-test (include as `if __name__ == "__main__":` block)

```python
import pandas as pd, numpy as np
from h1_value_plausibility import H1ValuePlausibility

dirty = pd.DataFrame({
    "age":       [25, 999, 30, 42, 7,   35],
    "education": ["HS-grad", "Bachleors", "Bachelors", "Masters", "Doctrate", "HS-grad"],
})
mask = pd.DataFrame({
    "age":       [0, 1, 0, 0, 1, 0],
    "education": [0, 1, 0, 0, 1, 0],
})

h1 = H1ValuePlausibility()
h1.fit(dirty, mask)
result = h1.compute(dirty, mask)
print(result.to_string())

# Expected output (4 rows, order may vary):
# row_idx  col_name   h1_plausible  h1_in_vocab  h1_in_range
#       1  age                   0          NaN            0   (999 > p95 of [25,30,42,35])
#       4  age                   0          NaN            0   (7 < p5 of [25,30,42,35])
#       1  education             0            0          NaN   ("Bachleors" not in {'HS-grad','Bachelors','Masters'})
#       4  education             0            0          NaN   ("Doctrate" not in vocab)
```

---

## Constraints
- No hardcoded column names.
- No imports from `fingerprint/` folder.
- Must work on any dirty_df + binary mask_df pair.
- `col_stats_` must be a plain dict (serializable).
