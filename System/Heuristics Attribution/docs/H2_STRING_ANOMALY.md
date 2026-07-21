# H2 — String Anomaly

**File:** `heuristics/h2_string_anomaly.py`  
**Class:** `H2StringAnomaly(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H2 answers the follow-up question that H1 cannot:

> **Does this out-of-vocabulary string look like a typo, or like deliberate obfuscation?**

H1 gives a binary yes/no on domain membership.  H2 *characterises* the error:

| Error kind | Pattern | Classification |
|---|---|---|
| Typo | `"Bachleors"` — edit distance 1 to `"Bachelors"` | **Unintentional** |
| Obfuscation token | `"Unknown"`, `"nan"`, `"—"` | **Intentional** |
| Suffix-mangled value | `"Private-DMV"` (where `"Private"` is in vocab) | **Intentional** |
| Structural placeholder | Single character; all non-alphanumeric | **Intentional** |

This distinction matters for intent attribution: both typos and obfuscation
tokens fail H1 (both are out-of-vocabulary), but they require **opposite**
downstream classifications.

**Applies to categorical columns only.**  For numerical columns both output
features are `NaN` (numerical errors are handled by H3/H4).

---

## Outputs

H2 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h2_min_edit_dist` | float \| NaN | Levenshtein distance to nearest clean-vocabulary entry. Capped at 10. NaN for numerical columns. |
| `h2_is_obfuscation` | int (0/1) \| NaN | 1 if the dirty value matches a known obfuscation pattern. NaN for numerical columns. |

---

## Score Formula (Explanation Layer)

```
h2_score = (h2_is_obfuscation + max(0, 1 - h2_min_edit_dist / 5)) / 2
```

| h2_score range | Interpretation |
|---|---|
| High | Strong evidence of **intentional** error (obfuscation or near-valid) |
| Low + small edit distance | Strong evidence of **unintentional** error (typo) |

This score is produced for the downstream explanation layer, not used directly
for binary classification.

---

## Column Type Detection

Identical rule to H1:

| Rule | Classification |
|---|---|
| > 90 % of non-null values parse as `float` | `num` (numerical) |
| ≤ 90 % | `cat` (categorical) |

Override per-column via `col_types` in `fit()`:

```python
h2.fit(dirty_df, mask_df, col_types={"age": "num", "zip_code": "cat"})
```

---

## API

### `fit(dirty_df, mask_df, col_types=None)`

Learns per-column vocabulary from **clean cells only** (`mask_df == 0`).

For each categorical column:
1. Collects all non-null clean cell values as the vocabulary.
2. If vocabulary size > 200 entries, caps to the 50 most-frequent entries
   (bounds compute time for high-cardinality columns).
3. Computes `median_len` (median string length across vocab entries).

Stores results in `self.col_stats_[col]`:
```python
{
    'type': 'cat' | 'num',
    'vocab': list[str],        # categorical only
    'median_len': float,       # categorical only
}
```

### `compute(dirty_df, mask_df)`

Iterates every error position (`mask_df == 1`) and returns a DataFrame.

- **Numerical columns** → both features = `NaN`.
- **Categorical columns** → compute `h2_min_edit_dist` and `h2_is_obfuscation`.

### `fit_compute(dirty_df, mask_df, **kwargs)`

Convenience wrapper: calls `fit()` then `compute()` in one step (inherited
from `BaseHeuristic`).

---

## Feature Details

### `h2_min_edit_dist`

Minimum Levenshtein distance between the dirty cell value (as string) and
**every entry in the clean vocabulary** for that column.

Special cases:
- **Empty vocabulary** → distance = 10 (maximum cap).
- **Exact match found** in vocabulary → distance = 0 (value was re-introduced
  correctly or error was in another column).
- **Cap** at 10 to prevent unbounded values.

**Optimisation applied:** vocab entries whose length differs from the dirty
string length by ≥ the current best distance are skipped — their
length-difference lower bound alone exceeds the best known distance.

### `h2_is_obfuscation`

Set to `1` if **any** of the following three rules fires:

#### Rule 1 — Exact obfuscation token (case-insensitive)

The dirty value (lowercased) matches one of:

```
nan, none, n/a, na, unknown, unk, ?, —, -, --,
null, nil, missing, not available, not applicable,
0, -1, 999, 9999, 99999
```

#### Rule 2 — Suffix/prefix pattern

The dirty value starts with a **clean vocabulary entry** (case-insensitive)
immediately followed by one of the separator characters `{'-', '_', ' '}` and
then one of the known obfuscation suffixes:

```
dmv, obf, high, low, 1, 2, x, ?, —
```

Example: `"Private-DMV"` where `"Private"` ∈ vocab → separator `-` + suffix
`dmv` → `h2_is_obfuscation = 1`.

#### Rule 3 — Structural placeholder

Either:
- String length == 1 (single-character placeholder), **or**
- String consists entirely of non-alphanumeric characters.

---

## Levenshtein Implementation

The edit distance function `_levenshtein(s1, s2)` is implemented **inline**
with standard two-row DP:

```
d(i, j) = min(
    d(i-1, j)   + 1,          # deletion
    d(i,   j-1) + 1,          # insertion
    d(i-1, j-1) + cost(i, j)  # substitution (0 if chars equal)
)
```

Memory: O(min(|s1|, |s2|)) — two rolling rows.  
Early exit: if `|len(s1) - len(s2)| ≥ _MAX_EDIT_DIST`, return cap immediately.

No external packages are used (`python-Levenshtein`, `editdistance`, etc.).

---

## Performance Characteristics

| Scenario | Complexity |
|---|---|
| Per cell, vocab size V, avg string length L | O(V × L²) |
| High-cardinality column (V > 200) | Capped at V = 50 most-frequent entries |
| Length-pruning optimisation | Skips entries where `\|len(dirty) - len(entry)\|` ≥ current best |

These measures keep the heuristic practical for datasets with 50,000+ erroneous
cells.

---

## Self-test

Run directly:

```bash
python h2_string_anomaly.py
```

Test data:

| Row | Column | Value | Mask | Expected |
|---|---|---|---|---|
| 0 | education | `"HS-grad"` | 0 (clean) | — (not in output) |
| 1 | education | `"Bachleors"` | 1 (error) | dist ≤ 2, obf = 0 (typo) |
| 2 | education | `"Bachelors"` | 0 (clean) | — (not in output) |
| 3 | education | `"Unknown"` | 1 (error) | dist = 7, obf = 1 (token) |
| 4 | education | `"—"` | 1 (error) | dist = 7, obf = 1 (structural) |
| 5 | education | `"Doctorate-obf"` | 1 (error) | dist = 4, obf = 1 (suffix) |
| 6 | education | `"Doctorate"` | 0 (clean) | — provides vocab entry |
| 0–6 | age | 25–55 | all 0 (clean) | no error rows produced |

Expected output:

```
 row_idx  col_name  h2_min_edit_dist  h2_is_obfuscation
       1 education               2.0                  0
       3 education               7.0                  1
       4 education               7.0                  1
       5 education               4.0                  1
```

---

## Design Decisions

### Why not use an external Levenshtein library?

The ticket specification explicitly forbids `python-Levenshtein` and
`editdistance` (not guaranteed to be installed in all pipeline environments).
The inline DP implementation is standard and correct.

### Why cap vocabulary at 50 most-frequent when V > 200?

For very high-cardinality columns (e.g. free-text fields accidentally treated
as categorical) an O(V × L²) scan becomes expensive. Restricting to the 50
most-frequent entries preserves the most representative vocabulary members
while bounding worst-case compute time.

### Why does the self-test add `"Doctorate"` as a clean row?

The suffix-obfuscation rule (Rule 2) requires `"Doctorate"` to be present in
the clean vocabulary so that `"Doctorate-obf"` is correctly identified as a
mangled version of a known entry.  Without a clean `"Doctorate"` row, the
prefix match cannot fire.

### Why are numerical columns passed through as NaN?

H2 is designed exclusively for string characterisation.  Numerical anomaly
detection (range outliers, distribution shifts) is the responsibility of H3
and H4.  Emitting NaN rather than 0 preserves the distinction between "not
applicable" and "no obfuscation detected" for downstream feature consumers.

---

## Dependencies

| Package | Usage |
|---|---|
| `pandas` | DataFrame I/O |
| `numpy` | NaN constants |
| Standard library only | Levenshtein, all other logic |
