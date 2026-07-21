# TICKET H2 — String Anomaly

## Depends on
TICKET_000_SETUP.md, TICKET_H1 (col_stats_ type detection logic can be reused as a pattern)

## File to create
```
heuristics/h2_string_anomaly.py
```

## Class
`H2StringAnomaly(BaseHeuristic)`

---

## What this heuristic answers

> **Does this string look like a typo, or does it look like deliberate obfuscation?**

H1 answers: "is this value in the vocabulary?" (binary — yes or no).
H2 answers: "what *kind* of wrong value is this?" — the characterization matters
because typos (unintentional) and obfuscation tokens (intentional) both fail H1
the same way but require opposite classifications.

Examples:
- `"Bachleors"` → edit distance 1 to `"Bachelors"` → **typo → unintentional**
- `"Unknown"` → matches obfuscation pattern → **deliberate → intentional**

**Applies to categorical columns only.** For numerical columns, output NaN for
both features (numerical errors are handled by H3/H4).

---

## Output features — EXACTLY these 2 columns (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h2_min_edit_dist` | float or NaN | Levenshtein distance to the nearest clean vocabulary entry. NaN for numerical columns. Capped at 10. |
| `h2_is_obfuscation` | int (0/1) or NaN | 1 if the dirty value matches a known obfuscation pattern. NaN for numerical columns. |

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, col_types=None):
```

For each column:
1. Auto-detect type using the same rule as H1 (pd.to_numeric fraction > 0.9 → numerical).
   Accept optional `col_types` dict to override.
2. For **categorical** columns only:
   - Build vocabulary from clean cells (mask == 0): set of non-null string values.
   - Compute `median_len`: median string length of vocabulary entries (used for length anomaly check).
3. Store: `self.col_stats_[col] = {'type': 'cat'|'num', 'vocab': set, 'median_len': float}`

---

## compute() specification

For each erroneous cell in a **categorical** column:

### Feature 1: `h2_min_edit_dist`
Compute Levenshtein distance between the dirty value (as string) and every entry
in the column vocabulary. Take the minimum. Cap at 10.

If vocabulary is empty → h2_min_edit_dist = 10 (maximum).
If dirty value IS in vocabulary → h2_min_edit_dist = 0.

Implement Levenshtein inline (do not use external libraries):
```python
def _levenshtein(s1: str, s2: str) -> int:
    # Standard DP implementation
    ...
```

### Feature 2: `h2_is_obfuscation`
Value is 1 if ANY of the following are true:

**Exact match** (case-insensitive) with known obfuscation tokens:
```python
OBFUSCATION_TOKENS = {
    'nan', 'none', 'n/a', 'na', 'unknown', 'unk', '?', '—', '-', '--',
    'null', 'nil', 'missing', 'not available', 'not applicable',
    '0', '-1', '999', '9999', '99999',
}
```

**Suffix/prefix pattern** on a valid vocabulary entry:
- dirty_value starts with a vocab entry AND has a suffix matching:
  `['-', '_', ' ']` + any of: `['dmv', 'obf', 'high', 'low', '1', '2', 'x', '?', '—']`
- Example: `"Private-DMV"` where `"Private"` is in vocab → obfuscation.

**Structural placeholder**:
- String length == 1 (single char placeholder like `"?"`, `"0"`, `"F"`)
- String consists entirely of non-alphanumeric characters

For **numerical** columns: both features = NaN.

---

## Score formula (for explanation layer, not classification)
```
h2_score = (h2_is_obfuscation + max(0, 1 - h2_min_edit_dist / 5)) / 2
```
High h2_score → evidence of INTENTIONAL error (obfuscation or close-to-valid).
Low h2_score + low edit distance → UNINTENTIONAL (typo).

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
from h2_string_anomaly import H2StringAnomaly

dirty = pd.DataFrame({
    "education": [
        "HS-grad",     # clean
        "Bachleors",   # error: typo (edit dist=1 to "Bachelors")
        "Bachelors",   # clean
        "Unknown",     # error: obfuscation token
        "—",           # error: obfuscation token
        "Doctorate-obf", # error: suffix obfuscation
    ],
    "age": [25, 30, 35, 40, 45, 50],  # numerical, should produce NaN
})
mask = pd.DataFrame({
    "education": [0, 1, 0, 1, 1, 1],
    "age":       [0, 0, 0, 0, 0, 0],
})

h2 = H2StringAnomaly()
h2.fit(dirty, mask)
result = h2.compute(dirty, mask)
print(result.to_string())

# Expected:
# row=1, education: h2_min_edit_dist=1,  h2_is_obfuscation=0  (typo)
# row=3, education: h2_min_edit_dist=7+, h2_is_obfuscation=1  (obfuscation token)
# row=4, education: h2_min_edit_dist=1,  h2_is_obfuscation=1  (single char "—")
# row=5, education: h2_min_edit_dist=3+, h2_is_obfuscation=1  (suffix pattern)
```

---

## Constraints
- No hardcoded column names.
- Levenshtein must be implemented inline — do NOT use `python-Levenshtein` or
  `editdistance` packages (not guaranteed to be installed).
- Must be fast enough for 50,000+ erroneous cells. Optimize: skip vocabulary
  entries longer than `len(dirty_value) + 5` in the edit distance loop.
- For a vocabulary of size V and average string length L, complexity is O(V * L²).
  If V > 200 for a column, restrict to the 50 most frequent vocabulary entries.
