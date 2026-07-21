# TICKET H8 — Sensitivity Flag

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h8_sensitivity_flag.py
```

## Class
`H8SensitivityFlag(BaseHeuristic)`

---

## What this heuristic answers

> **Is this a sensitive demographic attribute? And is the dirty value the
> majority class (possibly masking the real value to avoid profiling)?**

Intentional manipulation of sensitive attributes (race, gender, age, etc.) is
a known privacy-motivated pattern. Changing a minority value to the majority
value helps an individual "blend in" statistically. Unintentional errors in
sensitive columns are possible but less motivated.

**Known limitation (documented):** This signal is weak when the majority class
is not dominant. For `sex` in Adult Income: 67% Male → moderate signal.
For `race` in Adult Income: 86% White → stronger signal. Include this note
in the class docstring.

---

## Output features — EXACTLY 2 columns (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h8_is_sensitive` | int (0/1) | 1 if this column is a sensitive demographic attribute. |
| `h8_is_majority_value` | int (0/1) | 1 if the dirty value equals the majority class of this sensitive column. Only meaningful for sensitive categorical columns; 0 otherwise. |

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, sensitive_cols=None):
```

### Detecting sensitive columns

**User-supplied** (highest priority): `sensitive_cols: list[str]`.
All columns in this list are marked sensitive. Columns NOT in this list are
auto-evaluated.

**Auto-detection** for columns NOT explicitly provided:
Check each column name for case-insensitive substring match against:
```python
SENSITIVE_KEYWORDS = {
    "race", "gender", "sex", "age", "nationality", "religion",
    "disability", "marital", "ethnic", "origin", "orientation"
}
```
If any keyword appears as a substring of the column name → mark as sensitive.

Store `self.sensitive_cols_: set[str]`.

### Majority class detection

For each sensitive column:
1. Use clean cells only (mask rows where this column == 0).
2. Cast to string, find the most frequent value.
3. Store in `self.majority_class_: dict[col_name → str]`.

For non-sensitive columns: no majority class stored.

---

## compute() specification

For each erroneous cell at (row_idx, col_name):
1. `h8_is_sensitive`:
   - 1 if `col_name in self.sensitive_cols_`, else 0.
2. `h8_is_majority_value`:
   - If `h8_is_sensitive == 0` → 0.
   - If `h8_is_sensitive == 1` AND column is numerical → 0 (not meaningful for
     numeric sensitive columns like `age`).
   - If `h8_is_sensitive == 1` AND column is categorical:
     - 1 if `str(dirty_value) == self.majority_class_[col_name]`, else 0.
3. Return both values.

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
from h8_sensitivity_flag import H8SensitivityFlag

dirty = pd.DataFrame({
    "age":    [25, 30, 35, 999, 28],      # 999 = error in age (sensitive numerical)
    "race":   ["White", "White", "White", "White", "Black"],
                                          # last row Black → error → is it majority?
    "income": [0, 1, 0, 1, 0],
})
mask = pd.DataFrame({
    "age":    [0, 0, 0, 1, 0],
    "race":   [0, 0, 0, 0, 1],
    "income": [0, 0, 0, 0, 0],
})

h8 = H8SensitivityFlag()
h8.fit(dirty, mask)  # auto-detects "age" and "race" as sensitive
result = h8.compute(dirty, mask)
print(result.to_string())

# Expected:
# row=3, col=age:
#   h8_is_sensitive     = 1
#   h8_is_majority_value = 0   (numerical column → not applicable)
#
# row=4, col=race:
#   h8_is_sensitive     = 1
#   dirty value = "Black", majority class = "White"
#   h8_is_majority_value = 0   ("Black" != "White")
#
# --- Additional test: error that IS majority class ---
dirty2 = dirty.copy()
dirty2.loc[4, "race"] = "White"   # now the error is the majority class
mask2 = mask.copy()

h8b = H8SensitivityFlag()
h8b.fit(dirty2, mask2)
result2 = h8b.compute(dirty2, mask2)
# row=4, col=race: h8_is_majority_value = 1  ("White" == majority "White")
print(result2.to_string())
```

---

## Constraints
- `h8_is_majority_value` is always 0 for non-sensitive columns. Never compute
  majority class for non-sensitive columns.
- `h8_is_majority_value` is always 0 for numerical sensitive columns (age, etc.).
  The majority-direction signal is only defined for categorical sensitive columns.
- If a sensitive column has all clean cells nulled out (empty clean set) → set
  `majority_class_[col] = None` and return `h8_is_majority_value = 0`.
- Log a WARNING (not exception) if auto-detection finds 0 sensitive columns.
  This is valid (dataset may have no sensitive columns).
- Do NOT hardcode column names like "race" or "sex" into compute(). All logic
  must rely on `self.sensitive_cols_` populated in fit().
