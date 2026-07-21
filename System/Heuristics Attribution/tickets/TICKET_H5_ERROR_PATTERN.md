# TICKET H5 — Error Pattern

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h5_error_pattern.py
```

## Class
`H5ErrorPattern(BaseHeuristic)`

---

## What this heuristic answers

> **Is this cell part of a coordinated multi-cell edit? Is it co-occurring with
> errors in logically linked columns?**

Intentional manipulators tend to edit multiple cells together to maintain
consistency. A person who forges an education level will also update
`education-num`. Unintentional errors (noise, OCR) strike columns independently.

This heuristic measures two things:
1. **Row error density**: how many errors are in the same row.
2. **Co-dependent flag**: is the erroneous cell in a column that is logically
   linked to another erroneous column in the same row?

---

## Output features — EXACTLY 2 columns (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h5_error_count` | int | Number of erroneous cells in the same row (including this cell). |
| `h5_codependent_flag` | int (0/1) | 1 if this column has a co-dependent partner column that is ALSO erroneous in the same row. 0 otherwise. |

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, codependent_pairs=None):
```

Detect co-dependent column pairs. Two methods, used together:

### Method A — User-supplied pairs (highest priority)
Accept `codependent_pairs: list[tuple[str, str]]`, e.g.:
```python
codependent_pairs = [("education", "education-num"), ("native-country", "nationality")]
```
These are stored directly as trusted pairs.

### Method B — Auto-detection (fallback, always runs in addition to A)

Two auto-detection signals:

**Signal 1 — Name similarity**: columns whose names share a common prefix/root
(after stripping digits, hyphens, underscores, "num", "no", "id"). Example:
`education` and `education-num` → root `education` → flagged as co-dependent.
Use `difflib.SequenceMatcher` with threshold ≥ 0.8 on cleaned column names.

**Signal 2 — High Mutual Information**: Compute pairwise MI between all columns
(use `sklearn.feature_selection.mutual_info_classif` treating each column as
target in turn, using clean cells only). Pairs where MI > 0.5 are flagged.
Cap computation at 20 columns (all pairs) to avoid O(n²) blow-up on wide tables.

**Merging**: Union of user-supplied pairs, name-similarity pairs, and high-MI
pairs. Store as `self.codependent_pairs_: set of frozenset({col_a, col_b})`.

---

## compute() specification

For each erroneous cell at (row_idx, col_name):

1. `h5_error_count`: count number of cells in row `row_idx` where `mask == 1`.
2. `h5_codependent_flag`:
   - For each pair `{col_name, partner}` in `self.codependent_pairs_` where
     `col_name` appears in the pair:
     - If `mask.loc[row_idx, partner] == 1` → flag = 1. Break.
   - If no such partner is erroneous → flag = 0.
3. Return both values.

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
from h5_error_pattern import H5ErrorPattern

dirty = pd.DataFrame({
    "education":     ["HS-grad",   "HS-grad",  "Bachelors", "Masters"],
    "education-num": [9,            9,           13,          14],
    "age":           [25,           30,          35,          40],
})
mask = pd.DataFrame({
    "education":     [0,  0,  1,  0],   # row 2 education is error
    "education-num": [0,  0,  1,  0],   # row 2 education-num is ALSO error (coordinated)
    "age":           [0,  0,  0,  0],
})

h5 = H5ErrorPattern()
h5.fit(dirty, mask)  # should auto-detect education + education-num via name similarity
result = h5.compute(dirty, mask)
print(result.to_string())

# Expected for row=2, col=education:
#   h5_error_count = 2       (both education and education-num are errors in that row)
#   h5_codependent_flag = 1  (education-num is a co-dependent partner and is also erroneous)
#
# Expected for row=2, col=education-num:
#   h5_error_count = 2
#   h5_codependent_flag = 1
```

---

## Constraints
- `h5_error_count` includes the current cell itself (always ≥ 1 for mask==1 cells).
- Auto MI computation uses clean cells only (mask == 0). Skip MI computation if
  fewer than 100 clean rows available.
- Handle tables with <2 columns gracefully (no pairs possible → codependent_flag always 0).
- Do NOT raise an error if a user-supplied column name in `codependent_pairs` doesn't
  exist in the dataframe — warn and skip that pair.
