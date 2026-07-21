# H7 — User Incentive

**File:** `heuristics/h7_user_incentive.py`  
**Class:** `H7UserIncentive(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H7 answers the question:

> **Would a rational human actor understand this column, be able to change its
> value, and benefit from doing so?**

This is **behavioral motivation** — it models the *human* side of intentional
data manipulation, not the statistical side.  A column is an attractive
manipulation target when all three conditions are met:

1. The manipulator **understands** what the column represents.
2. They are **able** to change the value (it is not locked or immutable).
3. Changing the value in a particular direction moves the outcome in their
   **favor**.

---

## Critical Distinction from H6 (Column Importance)

H6 measures **statistical mutual information** between a column and the
outcome.  H7 measures **human behavioral motivation**.  They are
**orthogonal signals** — both are required for a complete picture.

| Column | H6 (statistical MI) | H7 (behavioral motivation) | Why they differ |
|---|---|---|---|
| `fnlwgt` (sampling weight) | **High** — strongly predictive of income | **Low** — nobody knows what it is | Statistically powerful but opaque to humans |
| `race` | **Low** — weak MI with income | **High** — masked for fairness/privacy | Socially salient but statistically weak |
| `education` / `education-num` | **High** | **High** — credential inflation target | Both signals agree |
| `ssn` (social security number) | **Medium** | **Low** — immutable, risky to change | Legal immutability overrides statistical importance |

> **Rule of thumb:** use H6 to rank columns by *model leverage*; use H7 to
> rank columns by *human motivation*.  Together they identify cells where
> both the model and the human have reason to care.

---

## Output

H7 produces **one row per erroneous cell** (positions where `mask_df == 1`)
with exactly **three feature columns** (plus `row_idx` / `col_name`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h7_mutability` | float ∈ {0.0, 0.5, 1.0} | Can a user realistically change this value? 0 = immutable, 0.5 = soft boundary (sensitive), 1 = freely mutable |
| `h7_gain_direction` | float ∈ [0, 1] | Does the dirty value shift toward a favorable outcome? **Per-cell lookup for categorical columns** — not a column average. 0.5 = neutral / no `target_col` provided |
| `h7_comprehensibility` | float ∈ [0, 1] | Would a typical human understand what this column means? 0 = opaque, 1 = obvious |

> **Key constraint:** `h7_gain_direction` for categorical columns is a
> **per-value lookup** (the actual dirty cell value is used to look up a
> pre-computed dict).  It is **not** a per-column average.  This means two
> erroneous cells in the same column can have different `h7_gain_direction`
> values if their dirty values differ.

---

## Sub-feature Details

### 1 — Mutability (`h7_mutability`)

Answers: *can a person actually change this value?*

**Priority 1 — User-supplied** via `mutability_scores` dict.  Values must be
exactly `0.0`, `0.5`, or `1.0`.  Any column present in this dict uses the
given value unchanged.

**Priority 2 — Auto-fallback** for columns not in the dict:

| Rule | Keywords (case-insensitive substring) | Score | Meaning |
|---|---|---|---|
| Immutable | `id`, `num`, `wgt`, `weight`, `score`, `code`, `key`, `index`, `ssn`, `hash`, `uuid`, `timestamp`, `date`, `time` | **0.0** | System-assigned; human cannot change |
| Sensitive | `race`, `gender`, `sex`, `nationality`, `religion`, `disability`, `ethnicity` | **0.5** | Possible but socially risky to change |
| Default | (all others) | **1.0** | Freely mutable |

The keyword check strips separators (`-`, `_`, space) before matching, so
`edu_num3` correctly matches `num` → immutable.

---

### 2 — Gain Direction (`h7_gain_direction`)

Answers: *does this specific dirty value push the outcome in the manipulator's
favor?*

#### No `target_col`
Returns `0.5` for every cell — neutral, uninformative.  Not an error.

#### `target_col` provided

**If `col == target_col`:** score = `1.0` (directly manipulating the outcome
itself is maximum gain).

**Numerical feature column:**
```
spearman_corr  =  Spearman(col_values, encoded_target)   [using clean rows only]
gain_direction =  0.5 + 0.5 × spearman_corr             [maps (−1, 1) → (0, 1)]
```
Result is a **scalar** stored in `self.gain_direction_[col]`.

**Categorical feature column:**
1. For each unique value `v` in the column, compute `mean(encoded_target)` for
   rows where `col == v` (using clean rows only).
2. Sort values by their mean target.
3. Top 50 % of values (by mean target) → `1.0` (favorable direction).
4. Bottom 50 % → `0.0` (unfavorable direction).
5. Result is stored as a **dict** `{str(value) → 0.0 | 1.0}` in
   `self.gain_direction_[col]`.

During `compute()`, the actual dirty cell value is looked up:
```python
gain_info.get(str(dirty_value), 0.5)   # 0.5 = unseen value → neutral
```

**Target encoding** (for both numerical and categorical features):
- Numerical target → cast to `float`.
- Categorical target → sorted unique values mapped to `0, 1, 2, …` (ordinal).

---

### 3 — Comprehensibility (`h7_comprehensibility`)

Answers: *would a non-expert human understand what this column means?*

**Priority 1 — User-supplied** via `comprehensibility_scores` dict.  Any
`float` value in `[0, 1]` is accepted.

**Priority 2 — Auto-fallback** via name analysis:

| Rule | Condition | Score | Example |
|---|---|---|---|
| **Opaque code** | `len < 4` **or** name contains a digit | `0.0` | `col1`, `id2`, `edu1` |
| **Abbreviation** | Any token ≤ 3 chars **or** any token has no vowels | `0.3` | `fnlwgt`, `hrs-per-wk`, `education-num` |
| **Multi-word phrase** | Multiple tokens after splitting on `-`/`_`/space | `1.0` | `marital-status`, `native-country`, `hours-per-week` |
| **Single plain word** | Single token ≥ 4 chars with vowels | `0.8` | `education`, `occupation`, `race` |

> **Note on `fnlwgt`:** length is 6 (passes rule 1), but the token `fnlwgt`
> contains no vowels, so rule 2 fires and returns **0.3** (abbreviated /
> opaque code).  This is correct — a human recognising it as a code-like
> token is the right signal.

---

## API

### `fit(dirty_df, mask_df, target_col=None, mutability_scores=None, comprehensibility_scores=None)`

Learns per-column behavioral-incentive scores.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dirty_df` | `pd.DataFrame` | — | The dirty dataset |
| `mask_df` | `pd.DataFrame` | — | Binary mask aligned with `dirty_df` (0 = clean, 1 = erroneous) |
| `target_col` | `str` | `None` | Outcome column name.  If `None`, `h7_gain_direction = 0.5` everywhere |
| `mutability_scores` | `dict[str, float]` | `None` | Per-column override.  Values must be `0.0`, `0.5`, or `1.0` |
| `comprehensibility_scores` | `dict[str, float]` | `None` | Per-column override.  Any float in `[0, 1]` |

After `fit()` completes:

| Attribute | Type | Content |
|---|---|---|
| `self.mutability_` | `dict[str, float]` | Per-column mutability in `{0.0, 0.5, 1.0}` |
| `self.gain_direction_` | `dict[str, float \| dict[str, float]]` | Scalar for numerical cols; nested dict for categorical cols |
| `self.comprehensibility_` | `dict[str, float]` | Per-column comprehensibility in `[0, 1]` |
| `self.is_fitted` | `bool` | Set to `True` |

```python
# Example attributes after fit on Adult Income dataset:
h7.mutability_ = {
    "education-num": 0.0,   # "num" → immutable
    "fnlwgt":        0.0,   # "wgt" → immutable
    "race":          0.5,   # sensitive attribute
    "income":        1.0,   # freely mutable (default)
}

h7.comprehensibility_ = {
    "education-num": 0.3,   # abbreviated token "num"
    "fnlwgt":        0.3,   # no vowels in "fnlwgt"
    "race":          0.8,   # single plain 4-char word
    "income":        0.8,   # single plain word
}

h7.gain_direction_ = {
    "education-num": 0.93,  # Spearman ≈ 0.86 → 0.5 + 0.5×0.86 ≈ 0.93
    "fnlwgt":        0.51,  # Spearman ≈ 0.02 → ≈ 0.5 (random noise)
    "race":          {"Asian": 0.0, "Black": 1.0, "White": 0.0},
    "income":        1.0,   # target column → max incentive
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Returns H7 features for every erroneous cell (`mask == 1`).

For each `(row_idx, col_name)` where `mask == 1`:

```python
h7_mutability        = self.mutability_[col_name]

# Numerical column:
h7_gain_direction    = self.gain_direction_[col_name]          # scalar

# Categorical column:
h7_gain_direction    = self.gain_direction_[col_name].get(
                           str(dirty_df.at[row_idx, col_name]),
                           0.5                                  # unseen value → neutral
                       )

h7_comprehensibility = self.comprehensibility_[col_name]
```

Returns a `pd.DataFrame` with columns
`[row_idx, col_name, h7_mutability, h7_gain_direction, h7_comprehensibility]`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`.  Calls `fit()` then
`compute()` in one step.

---

## Self-test Walkthrough

The self-test uses a 200-row synthetic dataset (`numpy.random.seed(42)`):

| Column | Role | Mutability | Comprehensibility | Gain Direction |
|---|---|---|---|---|
| `education-num` | Numerical; direct cause of `income` | `0.0` (immutable — `num` keyword) | `0.3` (abbreviated token) | `≈ 0.93` (strong positive Spearman) |
| `fnlwgt` | Numerical; pure random noise | `0.0` (immutable — `wgt` keyword) | `0.3` (no vowels) | `≈ 0.51` (near-zero Spearman) |
| `race` | Categorical; random, weakly correlated | `0.5` (sensitive keyword) | `0.8` (single plain word) | per-value dict lookup |
| `income` | Binary target column | `1.0` (default — no keyword match) | `0.8` (single plain word) | `1.0` (target col itself) |

Errors injected:

| Position | Column |
|---|---|
| rows 5, 10 | `education-num` |
| row 20 | `fnlwgt` |
| row 30 | `race` |

**Expected output:**

```
 row_idx      col_name  h7_mutability  h7_gain_direction  h7_comprehensibility
       5 education-num            0.0           0.934785                   0.3
      10 education-num            0.0           0.934785                   0.3
      20        fnlwgt            0.0           0.509067                   0.3
      30          race            0.5           1.000000                   0.8
```

> The `race` gain_direction at row 30 is `1.0` because the dirty value at that
> row (`"Black"` in this seed) happens to be the value with the highest mean
> encoded income in the clean data.  The lookup is **per-cell**, not a column
> average.

**Run the self-test:**

```bash
# from the heuristics/ directory
python h7_user_incentive.py
```

**Or run via the package (from the project root):**

```bash
conda run --name base python -c "
import pandas as pd, numpy as np
from error_detection_system.src.attribution.heuristics import H7UserIncentive

np.random.seed(42)
n = 200
edu = np.random.randint(5, 16, n)
dirty = pd.DataFrame({
    'education-num': edu,
    'fnlwgt':        np.random.randint(100000, 500000, n),
    'race':          np.random.choice(['White', 'Black', 'Asian'], n),
    'income':        (edu > 10).astype(int),
})
mask = pd.DataFrame(0, index=dirty.index, columns=dirty.columns)
mask.loc[[5, 10], 'education-num'] = 1
mask.loc[[20], 'fnlwgt'] = 1
mask.loc[[30], 'race'] = 1

h7 = H7UserIncentive()
print(h7.fit_compute(dirty, mask, target_col='income').to_string(index=False))
"
```

---

## Clean Cells Only

`fit()` computes gain direction using only **fully-clean rows** — rows where
`mask_df == 0` for **every** column.  This prevents erroneous cell values
from distorting Spearman correlations or mean-target estimates.

If fewer than **5** fully-clean rows are available, a `UserWarning` is issued
and all `gain_direction` scores are set to `0.5` (neutral fallback).

Mutability and comprehensibility scores are derived purely from **column
names** and are therefore unaffected by row-level data quality.

---

## Relationship to Other Heuristics

| Heuristic | What it checks | Signal type | Relationship to H7 |
|---|---|---|---|
| **H1** | Is the value within the column's known domain? | Binary, per-column | Independent — domain validity vs. behavioral motivation |
| **H2** | Does the string pattern look like obfuscation? | Binary, per-column | Complementary — pattern anomaly can reinforce H7 suspicion |
| **H3** | Is the value central or extreme in the column's distribution? | Continuous, per-cell | Complementary — H3 measures distribution shift; H7 measures incentive |
| **H4** | Does the value fit the rest of the row? | Continuous, cross-column | Complementary — row incoherence may result from incentivised edits |
| **H5** | Are multiple linked columns simultaneously erroneous? | Integer + binary, cross-row | Independent — coordination pattern vs. individual motivation |
| **H6** | Does this column have high MI with the outcome? | Continuous, per-column | **Complementary** — H6 is statistical attractiveness; H7 is behavioral motivation.  They are orthogonal. |
| **H7** | Would a human be motivated to manipulate this column? | Continuous, per-column/cell | This heuristic |

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| `target_col` not provided | `h7_gain_direction = 0.5` for all cells; no error raised |
| Fewer than 5 fully-clean rows | `UserWarning`; `h7_gain_direction = 0.5` for all cells |
| Unseen categorical value in `compute()` | `gain_direction = 0.5` (neutral fallback via `.get(..., 0.5)`) |
| `col == target_col` | `gain_direction = 1.0` regardless of column type |
| `mutability_scores` override with invalid value | `ValueError` raised immediately |
| `target_col` not in `dirty_df` | `ValueError` raised immediately |
| `dirty_df` / `mask_df` shape mismatch | `ValueError` raised immediately |
| `compute()` called before `fit()` | `RuntimeError` via `_check_fitted()` |
| Constant numerical column (zero variance) | Spearman returns NaN → treated as 0.0 → `gain_direction = 0.5` |
| NaN values in clean cells | Rows with NaN in the feature or target column are excluded from Spearman / mean-target computation |
| Column name is exactly 4 chars with vowels (e.g. `race`) | Passes rule 1 (`< 4` check); single plain word → `comprehensibility = 0.8` |

---

## Performance Notes

All three sub-features (`fit()`) are O(rows × cols):

| Step | Cost | Notes |
|---|---|---|
| Mutability | O(cols) | Keyword string search only |
| Comprehensibility | O(cols) | Regex on column names only |
| Gain direction — numerical | O(rows × cols) | One Spearman per numerical column |
| Gain direction — categorical | O(rows × unique_vals × cols) | Mean-target per unique value per column |

`compute()` is O(errors) — a dictionary lookup per erroneous cell.

---

## Design Constraints

- Exactly **three output feature columns** beyond `row_idx` / `col_name`:
  `h7_mutability`, `h7_gain_direction`, `h7_comprehensibility`.
- **H7 is completely separate from H6.**  No shared state, no shared logic.
  H6 is statistical; H7 is behavioral.
- `h7_gain_direction` for categorical columns is a **per-value dict**
  (not a per-column scalar), so `compute()` must look up the actual dirty
  cell value at `(row_idx, col_name)`.
- If `target_col` is not provided, `h7_gain_direction = 0.5` everywhere —
  this is a valid fallback, not an error condition.
- `mutability_scores` overrides must use values in `{0.0, 0.5, 1.0}` only.
- Uses only **fully-clean rows** for gain direction estimation (no data
  leakage from erroneous cells).
- No hardcoded column names or dataset-specific logic anywhere.
- All attributes (`mutability_`, `gain_direction_`, `comprehensibility_`)
  are plain Python dicts (JSON-serialisable).

---

## Limitations

- Mutability and comprehensibility are inferred from **column names only**.
  A column named `score` will be marked immutable even if it is, in practice,
  editable.  Use the `mutability_scores` and `comprehensibility_scores`
  overrides to correct domain-specific exceptions.
- `h7_gain_direction` for categorical columns is binary (0.0 or 1.0 per
  value).  The `0.5` fallback for unseen values is intentionally neutral —
  it does not indicate that the value is "medium" gain; it indicates that
  the value was not seen in clean training data.
- Spearman correlation (numerical gain direction) can be noisy for columns
  with few unique values or small clean-row counts.
- H7 measures **motivation**, not **evidence of manipulation**.  A high H7
  score for a cell means *a human would have reason to change it* — it does
  not prove that the specific error was intentional.  Combine with other
  heuristics (especially H6) for a full attribution picture.
