# H1 — Value Plausibility

**File:** `heuristics/h1_value_plausibility.py`  
**Class:** `H1ValuePlausibility(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H1 answers the most fundamental plausibility question:

> **Does this dirty value even belong to the column's known valid domain?**

A value that falls outside the domain (unknown category string, or numerical
value far outside the observed range) is strong evidence of an **unintentional
error** (typo, corruption, encoding bug).  
A value that is inside the domain is consistent with an **intentional
manipulation** (the attacker replaced one valid value with another valid value).

H1 is the first filter in the intent-attribution pipeline. Later heuristics
(H2+) refine the signal for edge cases H1 cannot distinguish.

---

## Outputs

H1 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h1_plausible` | int (0 or 1) | **Single binary summary.** 1 = value is plausible; 0 = not plausible |
| `h1_in_vocab` | int \| NaN | *Categorical only.* 1 if dirty value is in the clean vocabulary. NaN for numerical columns |
| `h1_in_range` | int \| NaN | *Numerical only.* 1 if dirty value is within [p5, p95] of clean values. NaN for categorical columns |

`h1_plausible` is the single canonical signal consumed by downstream steps:
- For **categorical** columns → `h1_plausible = h1_in_vocab`
- For **numerical** columns → `h1_plausible = h1_in_range`

---

## Column Type Detection

| Rule | Classification |
|---|---|
| > 90 % of non-null values parse as `float` | `num` (numerical) |
| ≤ 90 % | `cat` (categorical) |

The threshold can be overridden per-column by passing `col_types` to `fit()`:

```python
h1.fit(dirty_df, mask_df, col_types={"age": "num", "zip_code": "cat"})
```

---

## API

### `fit(dirty_df, mask_df, col_types=None)`

Learns per-column domain statistics using **only the clean cells**
(`mask_df[col] == 0`).

**Categorical columns** — builds the *vocabulary*:  
the set of unique, non-null string values present in clean cells.  
Tokens that resolve to `nan`, `none`, or empty string are excluded from the
vocabulary.

**Numerical columns** — builds the *plausible range*:  
computes the **5th and 95th percentiles** of the clean numeric values.
Values within `[p5, p95]` are considered plausible.

After `fit()` completes, all statistics are stored in `self.col_stats_`
(a plain serialisable dict) and `self.is_fitted` is set to `True`.

```
self.col_stats_ = {
    "age":       {"type": "num", "p5": 20.0, "p95": 58.0},
    "education": {"type": "cat", "vocab": {"HS-grad", "Bachelors", "Masters", ...}},
    ...
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Iterates over every `(row_idx, col_name)` where `mask_df == 1` and applies
the scoring rules below.

#### Scoring rules

**Categorical column:**

| Condition | h1_plausible | h1_in_vocab | h1_in_range |
|---|---|---|---|
| Value is null / `nan` / `none` / `""` | 0 | 0 | NaN |
| String value **in** clean vocabulary | 1 | 1 | NaN |
| String value **not in** clean vocabulary | 0 | 0 | NaN |

**Numerical column:**

| Condition | h1_plausible | h1_in_vocab | h1_in_range |
|---|---|---|---|
| Value is null / `nan` | 0 | NaN | 0 |
| Value cannot be cast to float (string) | 0 | 0 | NaN |
| `p5 ≤ value ≤ p95` | 1 | NaN | 1 |
| Value outside `[p5, p95]` | 0 | NaN | 0 |

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`. Calls `fit()` then
`compute()` in one step.

---

## Self-test

Run the built-in smoke test (requires the package to be importable):

```bash
# from the project root  
python -c "
import math, pandas as pd, numpy as np
from error_detection_system.src.attribution.heuristics.h1_value_plausibility import H1ValuePlausibility

dirty = pd.DataFrame({
    'age':       [25, 999, 30, 42, 7,   35],
    'education': ['HS-grad', 'Bachleors', 'Bachelors', 'Masters', 'Doctrate', 'HS-grad'],
})
mask = pd.DataFrame({
    'age':       [0, 1, 0, 0, 1, 0],
    'education': [0, 1, 0, 0, 1, 0],
})

h1 = H1ValuePlausibility()
result = h1.fit_compute(dirty, mask)
print(result.to_string(index=False))
"
```

**Expected output:**

```
 row_idx  col_name  h1_plausible  h1_in_vocab  h1_in_range
       1       age             0          NaN          0.0
       4       age             0          NaN          0.0
       1 education             0          0.0          NaN
       4 education             0          0.0          NaN
```

Interpretation:

| row | col | dirty value | reason implausible |
|---|---|---|---|
| 1 | `age` | 999 | Above p95 of clean ages `[25, 30, 42, 35]` |
| 4 | `age` | 7 | Below p5 of clean ages |
| 1 | `education` | `"Bachleors"` | Typo — not in vocab `{HS-grad, Bachelors, Masters}` |
| 4 | `education` | `"Doctrate"` | Typo — not in vocab |

---

## Known Limitations

| Limitation | Impact |
|---|---|
| **Obfuscation tokens** (`nan`, `Unknown`, `—`) are out-of-vocabulary by design | ~29 % of intentional errors in the Adult Income dataset score `h1_plausible = 0`, same as unintentional typos. H2 rescues these. |
| **Numerical range is [p5, p95]** — extreme but valid values (e.g. a 90-year-old) are flagged as implausible | Small false-positive rate for rare but legitimate outliers in clean data. |
| **Vocabulary is exact-string match** — same value with different casing counts as OOV | Case-normalisation can be applied upstream if needed. |
| **No fitted range available** (empty clean column) | Returns `p5 = p95 = NaN`; all numeric errors score `h1_in_range = 0`. |

---

## Design Constraints

- No hardcoded column names or dataset paths.
- Uses only clean cells (`mask == 0`) to build statistics — no data leakage.
- `col_stats_` is a plain Python dict (JSON-serialisable values only).
- No imports from the `fingerprint/` folder.
- Compatible with any `dirty_df` + binary `mask_df` pair of identical shape.

---

## Related Heuristics

| Heuristic | Complements H1 by… |
|---|---|
| **H2** | Detecting obfuscation tokens (the blind spot of H1) |
| **H3+** | Scoring magnitude of shift, statistical rarity, and context consistency |
