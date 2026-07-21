# TICKET H7 — User Incentive

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h7_user_incentive.py
```

## Class
`H7UserIncentive(BaseHeuristic)`

---

## What this heuristic answers

> **Would a rational human actor be motivated to manipulate this column?**

This is **behavioral motivation**, not statistical importance. It captures
whether a real person would understand, be able to change, and benefit from
altering this particular value.

**Critical distinction from H6 (Column Importance):**

| Column | H6 (statistical MI) | H7 (user motivation) |
|--------|---------------------|----------------------|
| `fnlwgt` | HIGH — strongly predictive of income | LOW — no human understands what it is |
| `race` | LOW — weak MI with income | HIGH — human may mask for fairness/privacy |
| `education` | HIGH | HIGH — common target for forgery |
| `ssn` (sensitive ID) | MEDIUM | LOW — immutable, risky to change |

Both H6 and H7 are necessary. They capture orthogonal signals.

---

## Output features — EXACTLY 3 columns (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h7_mutability` | float [0, 1] | Can a user realistically change this value? 0=immutable, 0.5=soft, 1=freely mutable. |
| `h7_gain_direction` | float [0, 1] | Does changing this column's value in this direction correlate with a favorable outcome? |
| `h7_comprehensibility` | float [0, 1] | Would a typical human understand what this column means? |

---

## fit() specification

```python
def fit(
    self,
    dirty_df,
    mask_df,
    target_col=None,
    mutability_scores=None,
    comprehensibility_scores=None,
):
```

### Sub-feature 1: Mutability

**User-supplied** (highest priority): `mutability_scores: dict[col_name → float]`.
Values must be 0.0, 0.5, or 1.0. Any provided column uses the given value.

**Auto-fallback** for columns NOT in `mutability_scores`:
- Column name matches any of `{id, num, wgt, weight, score, code, key, index,
  ssn, hash, uuid, timestamp, date, time}` (case-insensitive substring match)
  → mutability = 0.0 (system-assigned, immutable)
- Column name contains any of `{race, gender, sex, nationality, religion,
  disability, ethnicity}` (case-insensitive) → mutability = 0.5
  (sensitive, soft boundary — possible but risky to change)
- All other columns → mutability = 1.0 (freely mutable)

### Sub-feature 2: Gain Direction

Measures: does the dirty value appear to shift toward a favorable outcome?

Requires `target_col` to be meaningful. If not provided → `h7_gain_direction = 0.5`
for all cells (neutral, uninformative).

**If target_col provided:**

For each column `c`:
1. Use clean cells only (mask == 0).
2. **Categorical target**: compute mean target class (encode as 0/1 if binary,
   or use most favorable class index) per unique value of `c`. Sort values by
   mean target → establish "favorable direction". Store `favorable_vals: set`
   (top 50% by mean target value).
3. **Numerical column `c`**: compute `spearman_corr(c, encoded_target)`.
   `gain_direction_score = 0.5 + 0.5 * spearman_corr` (maps [-1,1] → [0,1]).
4. **Categorical column `c`**: `gain_direction_score = 1.0` if dirty_value is
   in `favorable_vals`, else `0.0`. Per-value lookup.

If `c == target_col` → score = 1.0 (direct outcome manipulation → maximum gain).

Store: `self.gain_direction_[col]` — either a scalar (numerical) or a dict of
{value → 0.0|1.0} (categorical).

### Sub-feature 3: Comprehensibility

**User-supplied** (highest priority): `comprehensibility_scores: dict[col_name → float]`.

**Auto-fallback** for columns NOT in `comprehensibility_scores`:

Score based on column name analysis:
- Length ≤ 4 chars OR contains digits AND underscores/hyphens → 0.0
  (e.g., `fnlwgt`, `edu_num3`, `col1`)
- Length 5–8 chars with abbreviated look (hyphens/underscores, mixed case) → 0.3
  (e.g., `hrs-per-wk`, `ed-num`)
- Full plain English words ≤ 12 chars → 0.8 (e.g., `education`, `occupation`)
- Full plain English phrases (1–3 words separated by space/hyphen) → 1.0
  (e.g., `marital-status`, `native-country`, `hours-per-week`)

Classification is based on:
```python
import re
def _auto_comprehensibility(col_name: str) -> float:
    name = col_name.lower().replace("-", " ").replace("_", " ")
    tokens = name.split()
    # rule 1: very short or looks like code
    if len(col_name) <= 4 or re.search(r'\d', col_name):
        return 0.0
    # rule 2: abbreviation-looking (no vowels in a token, or ≤3 char tokens)
    if any(len(t) <= 3 or not re.search(r'[aeiou]', t) for t in tokens):
        return 0.3
    # rule 3: multi-word phrase
    if len(tokens) > 1:
        return 1.0
    # rule 4: single word ≥5 chars
    return 0.8
```

Store: `self.comprehensibility_: dict[col_name → float]`.

---

## compute() specification

For each erroneous cell at (row_idx, col_name):
1. `h7_mutability = self.mutability_[col_name]`
2. `h7_gain_direction`:
   - Numerical col: `self.gain_direction_[col_name]` (scalar)
   - Categorical col: `self.gain_direction_[col_name].get(str(dirty_value), 0.5)`
3. `h7_comprehensibility = self.comprehensibility_[col_name]`
4. Return all three.

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
import numpy as np
from h7_user_incentive import H7UserIncentive

np.random.seed(42)
n = 200
education_num = np.random.randint(5, 16, n)
income = (education_num > 10).astype(int)
fnlwgt = np.random.randint(100000, 500000, n)
race = np.random.choice(["White", "Black", "Asian"], n)

dirty = pd.DataFrame({
    "education-num": education_num,
    "fnlwgt":        fnlwgt,
    "race":          race,
    "income":        income,
})
mask = pd.DataFrame(np.zeros_like(dirty.values, dtype=int), columns=dirty.columns)
mask.loc[[5, 10], "education-num"] = 1
mask.loc[[20], "fnlwgt"] = 1
mask.loc[[30], "race"] = 1

h7 = H7UserIncentive()
h7.fit(dirty, mask, target_col="income")
result = h7.compute(dirty, mask)
print(result.to_string())

# Expected:
# education-num:
#   h7_mutability       = 0.0   (auto: "num" in name → immutable)
#   h7_gain_direction   ≈ 0.9   (positively correlated with income)
#   h7_comprehensibility = 0.3  (abbreviated: "education-num" → abbrev token "num")
#
# fnlwgt:
#   h7_mutability        = 0.0  ("wgt" in name → immutable)
#   h7_gain_direction    ≈ 0.5  (random → near-zero correlation)
#   h7_comprehensibility = 0.0  (6 chars but no vowels in second token → 0.0)
#
# race:
#   h7_mutability        = 0.5  (sensitive attribute)
#   h7_gain_direction    ≈ 0.5  (weak correlation with income)
#   h7_comprehensibility = 0.8  (single plain word)
```

---

## Constraints
- Do NOT merge H7 and H6. They are separate heuristics.
- `h7_gain_direction` for a categorical column must be computed per-value (not
  per-column average), so compute() must look up the actual dirty cell value.
- If `target_col` is not provided, set `h7_gain_direction = 0.5` for all cells.
  Do not raise an error — this is a valid fallback.
- Store `self.gain_direction_` as a dict of either scalar floats (numerical) or
  nested dicts (categorical) keyed by string value.
