# H8 — Sensitivity Flag

**File:** `heuristics/h8_sensitivity_flag.py`  
**Class:** `H8SensitivityFlag(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H8 answers the question:

> **Is this a sensitive demographic attribute?  
> And is the dirty value the majority class — a possible sign that someone
> changed a minority value to blend in statistically?**

This heuristic detects **privacy-motivated demographic masking**: the pattern
where a user replaces their true minority-class demographic value with the
dominant majority class to avoid profiling by a machine-learning model or
data analyst.

Example: In the Adult Income dataset, 86 % of individuals are classified as
"White" under `race`.  A person from a minority group who wants to avoid
race-based profiling could change their value to "White".  H8 flags that
pattern with `h8_is_majority_value = 1`.

---

## Relationship to Other Heuristics

H8 is the **privacy / fairness axis** of the heuristic suite.  It is
complementary to — but distinct from — H6 and H7:

| Heuristic | Signal | Example column | Reason they differ |
|---|---|---|---|
| **H6** | Statistical MI with target | `fnlwgt` (Adult) | Statistically predictive, but nobody understands it |
| **H7** | Human behavioral motivation | `race` | Low MI with income, but high human motivation to hide it |
| **H8** | Sensitive attribute + majority-class masking | `race`, `sex`, `marital-status` | Captures the specific *privacy-masking* direction, not just motivation |

Use H6 to rank by *model leverage*, H7 to rank by *human motivation*, and H8
to identify the specific *majority-class substitution* pattern in sensitive
columns.

---

## Output

H8 produces **one row per erroneous cell** (positions where `mask_df == 1`)
with exactly **two feature columns** (plus `row_idx` / `col_name`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h8_is_sensitive` | int ∈ {0, 1} | 1 if the column is a sensitive demographic attribute |
| `h8_is_majority_value` | int ∈ {0, 1} | 1 if the dirty value equals the majority class of that sensitive categorical column |

### Invariants

| Condition | `h8_is_majority_value` |
|---|---|
| Non-sensitive column | Always **0** |
| Sensitive **numerical** column (e.g. `age`) | Always **0** — majority value not meaningful for continuous distributions |
| Sensitive categorical column, clean set empty | Always **0** — cannot determine majority class |
| Sensitive categorical column, dirty value ≠ majority | **0** |
| Sensitive categorical column, dirty value = majority | **1** |

---

## Sensitive Column Detection

### Priority 1 — User-supplied (`sensitive_cols` argument)

```python
h8.fit(dirty_df, mask_df, sensitive_cols=["race", "sex", "zipcode"])
```

All columns in the list are **unconditionally** marked sensitive.  Columns
**not** in the list are still auto-evaluated (see Priority 2).

### Priority 2 — Auto-detection (keyword matching)

For columns not explicitly supplied, H8 checks whether the column name
contains any of the following keywords as a **case-insensitive substring**:

```python
SENSITIVE_KEYWORDS = {
    "race", "gender", "sex", "age", "nationality", "religion",
    "disability", "marital", "ethnic", "origin", "orientation",
}
```

Examples of columns that auto-detect as sensitive:

| Column name | Matched keyword |
|---|---|
| `race` | `race` |
| `gender` | `gender` |
| `age` | `age` |
| `marital-status` | `marital` |
| `country-of-origin` | `origin` |
| `religious_affiliation` | `religion` |
| `sexual_orientation` | `orientation` |

No sensitive columns detected → a `UserWarning` is issued (not an exception).
This is valid — the dataset may genuinely have no demographic attributes.

---

## Numerical vs. Categorical Column Detection

A column is classified as **numerical** when more than 90 % of its non-null
values can be parsed as `float` by `pandas.to_numeric`.  Otherwise it is
treated as **categorical**.

This rule applies **only to sensitive columns** during `fit()`.  Non-sensitive
columns are never examined for majority class.

| Column | Classification | `h8_is_majority_value` |
|---|---|---|
| `age` (integers) | Numerical | Always 0 |
| `race` (strings) | Categorical | Computed from majority class |
| `sex` (strings) | Categorical | Computed from majority class |

---

## Majority Class Computation

For each sensitive **categorical** column, `fit()` computes the majority class
from **clean cells only** (rows where `mask_df[col] == 0`):

```
majority_class_[col] = mode(str(clean_values))
```

- Values are cast to `str` before computing the mode so that mixed-type
  columns are handled uniformly.
- If multiple values tie for the mode, the alphabetically first value is
  selected (pandas `Series.mode()` behaviour).
- If no clean cells exist, `majority_class_[col] = None` and
  `h8_is_majority_value` will be 0 for all cells in that column.

Only sensitive categorical columns get a majority-class entry in
`self.majority_class_`.  Numerical sensitive columns and non-sensitive columns
are **never** examined.

---

## Known Limitation

The majority-class signal is **weak when the majority class is not dominant**:

| Column | Majority class | Fraction | Signal strength |
|---|---|---|---|
| `race` (Adult Income) | White | ~86 % | **Strong** |
| `sex` (Adult Income) | Male | ~67 % | **Moderate** |
| A balanced binary column | Either | ~50 % | **Weak / noise** |

When the majority class is only marginally more common than alternatives, a
randomly corrupted value has a near-50 % chance of happening to equal the
majority class, reducing the signal-to-noise ratio.

H8 is most informative for **highly skewed** sensitive columns.  Downstream
classifiers should weight `h8_is_majority_value` accordingly, ideally in
combination with H7 (behavioral motivation) which also flags sensitive columns
as high-motivation manipulation targets.

---

## API Reference

### `H8SensitivityFlag()`

Constructor.  No arguments.

---

### `fit(dirty_df, mask_df, sensitive_cols=None) → self`

Learn sensitive columns and their majority classes.

| Parameter | Type | Description |
|---|---|---|
| `dirty_df` | `pd.DataFrame` | The dirty dataset (contains errors) |
| `mask_df` | `pd.DataFrame` | Binary mask aligned with `dirty_df`; 0 = clean, 1 = error |
| `sensitive_cols` | `list[str]`, optional | User-supplied sensitive column names |

**Post-fit attributes:**

| Attribute | Type | Description |
|---|---|---|
| `sensitive_cols_` | `set[str]` | All sensitive columns (user + auto) |
| `majority_class_` | `dict[str, str \| None]` | Majority class per sensitive categorical column |
| `_numerical_cols_` | `set[str]` | Sensitive columns classified as numerical |

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Compute H8 features for every erroneous cell (`mask_df == 1`).

Returns a `pd.DataFrame` with columns:
`[row_idx, col_name, h8_is_sensitive, h8_is_majority_value]`

Raises `RuntimeError` if called before `fit()`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience: calls `fit()` then `compute()` in one step.
Inherited from `BaseHeuristic`.

---

## Usage Examples

### Auto-detection (typical usage)

```python
import pandas as pd
from heuristics.h8_sensitivity_flag import H8SensitivityFlag

h8 = H8SensitivityFlag()
h8.fit(dirty_df, mask_df)
features = h8.compute(dirty_df, mask_df)
```

### User-supplied sensitive columns

```python
h8 = H8SensitivityFlag()
h8.fit(dirty_df, mask_df, sensitive_cols=["race", "sex", "zipcode"])
features = h8.compute(dirty_df, mask_df)
```

### One-shot convenience

```python
features = H8SensitivityFlag().fit_compute(
    dirty_df, mask_df, sensitive_cols=["race", "sex"]
)
```

### Downstream use — combining with H7

```python
from heuristics.h7_user_incentive import H7UserIncentive
from heuristics.h8_sensitivity_flag import H8SensitivityFlag
import pandas as pd

h7_feat = H7UserIncentive().fit_compute(dirty_df, mask_df, target_col="income")
h8_feat = H8SensitivityFlag().fit_compute(dirty_df, mask_df)

combined = pd.merge(h7_feat, h8_feat, on=["row_idx", "col_name"])
# Cells where h7_mutability=0.5 AND h8_is_majority_value=1 are strong candidates
# for privacy-motivated masking.
```

---

## Self-test

Running the module directly executes 4 self-tests:

```bash
python h8_sensitivity_flag.py
```

| Test | Scenario | Checks |
|---|---|---|
| **1** | `age` (numerical sensitive), `race` error = "Black" (minority) | `age` → `h8_is_majority_value=0`; "Black" ≠ "White" → `h8_is_majority_value=0` |
| **2** | Same setup but `race` error = "White" (majority) | "White" == "White" → `h8_is_majority_value=1` |
| **3** | User-supplied `sensitive_cols=["zipcode"]` | `zipcode` marked sensitive despite no keyword match |
| **4** | No sensitive columns in dataset | `UserWarning` issued; all cells return `h8_is_sensitive=0`, `h8_is_majority_value=0` |

Expected output (abbreviated):

```
============================================================
H8SensitivityFlag — Self-test
============================================================

Auto-detected sensitive columns: {'race', 'age'}
Numerical sensitive columns:     {'age'}
Majority classes:                {'race': 'White'}

--- Test 1 result (dirty value for race = 'Black') ---
 row_idx col_name  h8_is_sensitive  h8_is_majority_value
       3      age                1                     0
       4     race                1                     0

✓ Test 1 passed ...

--- Test 2 result (dirty value for race = 'White', the majority) ---
 row_idx col_name  h8_is_sensitive  h8_is_majority_value
       3      age                1                     0
       4     race                1                     1

✓ Test 2 passed ...

✓ Test 3 passed ...
✓ Test 4 passed ...

============================================================
All self-tests passed ✓
============================================================
```

---

## Design Decisions

### Why substring matching instead of exact matching?

Column names in real datasets are inconsistent: `RACE`, `race_code`,
`marital_status`, `country-of-origin`.  Substring matching (after lower-casing)
handles all these variants without requiring a pre-built exhaustive list.

### Why cast values to `str` before computing the mode?

Some categorical columns may have mixed types (e.g. integer codes mixed with
string labels due to serialization).  Casting to `str` ensures uniform
comparison in both `mode()` and the per-cell equality check in `compute()`.

### Why is `h8_is_majority_value` always 0 for numerical columns?

Numerical sensitive columns (e.g. `age`) do not have a meaningful "majority
value" — values are continuous, not categorical classes.  Computing a mode
on a continuous distribution would return an arbitrary value and produce a
meaningless signal (e.g. if the most common `age` is 28, flagging any error
that happens to be 28 as "privacy masking" would be spurious).

### Why compute majority class from clean cells only?

Erroneous cells may themselves be the manipulated values.  Including them in
the majority-class computation would bias the estimate.  Using only clean
cells (`mask_df[col] == 0`) gives a ground-truth distribution of the
*true* population.

### Why `UserWarning` (not exception) for zero sensitive columns?

A dataset genuinely may have no demographic attributes (e.g. a purely
numerical sensor dataset).  Raising an exception would break pipelines
that iterate over many datasets.  A warning keeps the user informed without
interrupting the flow.

---

## File Layout

```
heuristics/
├── base.py                    # BaseHeuristic ABC
├── h8_sensitivity_flag.py     # ← this module
└── docs/
    └── H8_SENSITIVITY_FLAG.md # ← this document
```
