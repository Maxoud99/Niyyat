# H5 — Error Pattern

**File:** `heuristics/h5_error_pattern.py`  
**Class:** `H5ErrorPattern(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H5 answers the question:

> **Is this erroneous cell part of a coordinated multi-cell edit?  
> Is it co-occurring with errors in logically linked columns?**

Intentional data manipulators edit multiple cells *together* to maintain
internal record consistency.  Someone who forges an education level will
also update `education-num` so the values agree.  Random noise (OCR errors,
transcription accidents) strikes columns *independently* — a single cell is
corrupted, with no correlated partner.

H5 captures this coordinated-edit signal with two features:

| Feature | What it measures |
|---|---|
| `h5_error_count` | How many cells in the same row are erroneous? High → coordinated edit |
| `h5_codependent_flag` | Does this column have a logically linked partner that is *also* erroneous in the same row? |

### Relationship to other heuristics

| Heuristic | What it checks | Signal type |
|---|---|---|
| **H1** | Is the value within the column's known domain? | Binary, per-column |
| **H2** | Does the string pattern look like obfuscation? | Binary, per-column |
| **H3** | Is the value central or extreme in its column's distribution? | Continuous, per-column |
| **H4** | Does the value fit the *rest of the row* (RF predictor)? | Continuous, cross-column |
| **H5** | Are multiple linked columns simultaneously erroneous (coordinated edit)? | Integer + binary, cross-row |

H5 is the only heuristic that reasons about *which rows have multiple
simultaneous errors* and whether those errors cluster in logically linked
columns — making it uniquely sensitive to the coordinated-edit signature of
intentional manipulation.

---

## Output

H5 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h5_error_count` | int ≥ 1 | Number of erroneous cells in the same row, including this cell |
| `h5_codependent_flag` | int ∈ {0, 1} | 1 if at least one co-dependent partner column is *also* erroneous in the same row; 0 otherwise |

> **Important:** `h5_error_count` always includes the current cell itself
> (minimum value is 1).  There are exactly **two heuristic output columns**
> beyond `row_idx` / `col_name`.

---

## Co-dependent Pair Discovery

The core of H5 is deciding which column pairs are *logically linked*.  Three
sources are merged (union) at fit time:

```
co-dependent pairs = user_pairs  ∪  name_similarity_pairs  ∪  high_MI_pairs
```

### Method A — User-supplied pairs (highest priority)

```python
h5.fit(dirty_df, mask_df,
       codependent_pairs=[("education", "education-num"),
                          ("native-country", "nationality")])
```

Pairs are taken as trusted ground truth.  Column names that do not exist in
the dataframe are silently skipped with a `UserWarning`.

---

### Method B / Signal 1 — Name similarity

Column names are first *cleaned*:

1. Lower-cased.
2. Hyphens and underscores replaced with spaces.
3. The tokens `num`, `no`, `id` removed (as whole words).
4. Digits stripped.
5. Whitespace collapsed.

Examples:

| Original name | Cleaned |
|---|---|
| `education` | `education` |
| `education-num` | `education ` → `education` |
| `native-country` | `native country` |
| `age2` | `age` |
| `user_id` | `user ` → `user` |

Two columns whose cleaned names score **SequenceMatcher ratio ≥ 0.80** are
flagged as co-dependent.

---

### Method B / Signal 2 — Mutual information

Pairwise mutual information (MI) between all columns is estimated using
`sklearn.feature_selection.mutual_info_classif`, treating each column as the
target in turn.

**Constraints applied:**

| Constraint | Value | Reason |
|---|---|---|
| Minimum clean rows | 100 | Fewer rows produce unreliable MI estimates |
| Column cap | 20 | Avoid O(n²) blow-up on wide tables |
| MI threshold | 0.5 | Only strongly linked pairs are flagged |

MI is computed on **clean rows only** (rows where *all* selected columns have
`mask == 0`) to avoid learning spurious correlations from errors.

All columns are encoded with `OrdinalEncoder` before MI computation so that
both categorical and numerical columns are handled uniformly.

---

## Scoring Interpretation

| `h5_error_count` | `h5_codependent_flag` | Interpretation |
|---|---|---|
| 1 | 0 | Isolated single error — more consistent with random noise |
| 1 | 1 | Isolated but in a linked column; partner is clean — weak intentional signal |
| ≥ 2 | 0 | Multiple errors in the row but not in linked columns — possible coincidence or broad noise |
| ≥ 2 | 1 | **Coordinated edit** — multiple errors AND the linked partner is also erroneous → strong intentional signal |

---

## API

### `fit(dirty_df, mask_df, codependent_pairs=None)`

Discovers co-dependent column pairs and stores them.

| Parameter | Type | Description |
|---|---|---|
| `dirty_df` | `pd.DataFrame` | The dirty dataset. |
| `mask_df` | `pd.DataFrame` | Binary mask (0 = clean, 1 = erroneous), same shape as `dirty_df`. |
| `codependent_pairs` | `list[tuple[str, str]]`, optional | User-supplied pairs of logically linked columns. |

After `fit()` completes:
- `self.codependent_pairs_` holds a `set` of `frozenset({col_a, col_b})` objects.
- `self.is_fitted` is set to `True`.

```python
# Example of codependent_pairs_ after fit on the Adult Income dataset
{
    frozenset({'education', 'education-num'}),   # name similarity
    frozenset({'age', 'hours-per-week'}),         # high MI
    frozenset({'occupation', 'income'}),          # high MI
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Iterates over every `(row_idx, col_name)` where `mask_df == 1` and computes
both features.

**Algorithm (per erroneous cell):**

1. **`h5_error_count`**: read the pre-computed row sum of `mask_df` for
   `row_idx`.
2. **`h5_codependent_flag`**:
   - Iterate over `self.codependent_pairs_`.
   - For each pair that contains `col_name`, check if the partner column is
     also erroneous in `row_idx`.
   - If any partner is erroneous → flag = 1, break.
   - Otherwise → flag = 0.

Returns a `pd.DataFrame` with columns
`[row_idx, col_name, h5_error_count, h5_codependent_flag]`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`.  Calls `fit()` then
`compute()` in one step.

---

## Self-test Walkthrough

The self-test uses a 4-row dataset with two simultaneously erroneous cells in
row 2:

| idx | education | education-num | age | mask(edu) | mask(edu-num) | mask(age) |
|---|---|---|---|---|---|---|
| 0 | HS-grad | 9 | 25 | 0 | 0 | 0 |
| 1 | HS-grad | 9 | 30 | 0 | 0 | 0 |
| **2** | **Bachelors** | **13** | 35 | **1** | **1** | 0 |
| 3 | Masters | 14 | 40 | 0 | 0 | 0 |

Row 2 has **two simultaneous errors** in `education` and `education-num`.

**Step 1 — fit():**  
Name-similarity detects `{"education", "education-num"}` because after
cleaning both reduce to `"education"` (ratio = 1.0 ≥ 0.80).

**Step 2 — compute():**

For `(row=2, col="education")`:
- `h5_error_count` = 2 (both `education` and `education-num` are errors in row 2)
- `h5_codependent_flag` = 1 (`education-num` is a co-dependent partner and is also erroneous)

For `(row=2, col="education-num")`:
- `h5_error_count` = 2
- `h5_codependent_flag` = 1 (`education` is a co-dependent partner and is also erroneous)

**Expected output:**

```
 row_idx      col_name  h5_error_count  h5_codependent_flag
       2     education               2                    1
       2 education-num               2                    1
```

**Run the self-test:**

```bash
# from the project root
conda run --name base python -m \
    error_detection_system.src.attribution.heuristics.h5_error_pattern
```

---

## Edge Cases

| Scenario | Behaviour |
|---|---|
| Table with only 1 column | No pairs possible; `h5_codependent_flag` is always 0 |
| User-supplied column name not in dataframe | `UserWarning` issued; pair skipped |
| Fewer than 100 clean rows | MI computation skipped; only name-similarity pairs used |
| Table with > 20 columns | MI computed on first 20 columns only |
| All cells in a row are clean | Row never appears in output (mask == 0 rows are excluded) |
| Co-dependent partner column missing from `mask_df` at compute time | Partner silently skipped; flag remains 0 |

---

## Design Constraints

- Exactly **two output feature columns** beyond `row_idx` / `col_name`:
  `h5_error_count` and `h5_codependent_flag`.
- `h5_error_count` counts **all** erroneous cells in the row including the
  current cell — minimum value is always **1** for any row that appears in
  the output.
- Co-dependent pair discovery is **dataset-agnostic**: no hardcoded column
  names or domain assumptions.
- MI computation uses **clean cells only** (`mask == 0` across the full row)
  to prevent errors from distorting correlation estimates.
- Column cap for MI is **20** to prevent O(n²) scaling on very wide tables.
- No error is raised for a non-existent user-supplied column name — only a
  warning, so pipelines remain robust.
- `h5_error_count` is pre-computed as a single `mask_df.sum(axis=1)` call
  before the per-cell loop for O(rows) efficiency.

---

## Performance Notes

| Dataset size | Fit time (approx.) | Notes |
|---|---|---|
| < 10 k rows | < 1 s | Name similarity O(cols²); MI fast on small data |
| 10 k – 100 k rows | 1 – 10 s | MI via `mutual_info_classif` dominates |
| > 100 k rows | Similar, capped at 20 cols | Wide tables capped; row count does not increase MI cost much |
| > 20 columns | MI capped at first 20 | O(20²) = 400 MI pairs max |

The dominant cost is MI computation.  For pipelines where fit time is
critical, pass `codependent_pairs` explicitly and rely only on name
similarity (MI will still run but contribute no new pairs if the threshold
is high enough).

---

## Limitations

- Name similarity relies on the heuristic cleaning rules (strip `num`, `id`,
  `no`, digits, hyphens).  Column naming conventions that differ greatly from
  these rules may produce missed pairs or false positives.
- MI captures statistical dependence but not semantic meaning.  Two
  numerically correlated but semantically unrelated columns will be flagged.
  Use `codependent_pairs` to override if false positives are problematic.
- H5 measures *co-occurrence* of errors, not their *semantic validity*.  A
  dataset where many columns are systematically corrupted (e.g. > 30 % error
  rate per column) will have high `h5_error_count` for all errors
  independent of intent.
- `h5_codependent_flag` does not distinguish *direction*: it fires for any
  pair member that is erroneous, regardless of which column was the "cause"
  of the coordinated edit.
