# H3 — Distribution Position

**File:** `heuristics/h3_distribution_position.py`  
**Class:** `H3DistributionPosition(BaseHeuristic)`  
**Status:** ✅ Implemented & self-tested

---

## Purpose

H3 answers the question:

> **Is this dirty value common or rare within its column's distribution?**

Intentional errors (adversarial manipulation) tend to use *common, unremarkable*
values so they blend in with surrounding data. Unintentional errors (noise,
random substitution, encoding corruption) tend to land in the tails of the
distribution — extreme numbers or vocabulary items that are rarely seen.

H3 captures this signal as a single continuous score. It is complementary to
H1 (binary in/out-of-domain) and H2 (string pattern anomalies) — while those
heuristics give binary signals, H3 provides **magnitude information** about
*how far* a value strays from the norm.

### Known Limitation

For columns where both intentional and unintentional errors cluster in similar
value ranges (e.g. `capital-gain` in the Adult Income dataset, where 91–93 %
of both kinds of errors are technically in-range), H3's discrimination power
is low. This is expected — the heuristic documents this blind spot and relies
on other heuristics to compensate.

---

## Output

H3 produces **one row per erroneous cell** (positions where `mask_df == 1`):

| Column | Type | Description |
|---|---|---|
| `row_idx` | int | Row index of the erroneous cell |
| `col_name` | str | Column name of the erroneous cell |
| `h3_distribution_score` | float ∈ [0, 1] | Distribution-position score. **High (~1) = central / common → intentional signal. Low (~0) = extreme / rare → unintentional signal.** |

> **Important:** `h3_distribution_score` is the **single** output feature.
> Intermediate computations (z-score for numerical, frequency rank for
> categorical) are used internally and are **not** exposed as output columns.

---

## Column Type Detection

| Rule | Classification |
|---|---|
| > 90 % of non-null values parse as `float` | `num` (numerical) |
| ≤ 90 % | `cat` (categorical) |

Override per-column by passing `col_types` to `fit()`:

```python
h3.fit(dirty_df, mask_df, col_types={"age": "num", "zip_code": "cat"})
```

---

## Scoring Formulas

### Numerical columns

```
zscore = (dirty_value − clean_mean) / clean_std
h3_distribution_score = 1 − min(1, |zscore| / 3.0)
```

Values near the column mean score ~1.0 (central → intentional signal).  
Values ≥ 3 standard deviations away score ~0.0 (extreme → unintentional signal).

**Edge cases:**

| Condition | Score |
|---|---|
| `clean_std == 0` (constant column) | `1.0` — every value is "central" by definition |
| Dirty value cannot be cast to `float` | `0.0` |
| Clean cells are all null / empty | `0.0` — no reference distribution available |

### Categorical columns

```
h3_distribution_score = 1 − (frequency_rank − 1) / max(1, vocab_size − 1)
```

- `frequency_rank` = rank of the dirty value in the clean vocabulary sorted by
  **descending frequency** (rank 1 = most common, rank `vocab_size` = rarest).
- If the dirty value is **not in** the clean vocabulary (out-of-vocabulary),
  it is assigned rank `vocab_size + 1` — penalised as rarer than anything seen.

| Rank | Score | Interpretation |
|---|---|---|
| 1 (most common) | 1.0 | Most likely an intentional value swap |
| vocab_size (rarest in vocab) | ~0.0 | Suspicious but in-vocabulary |
| vocab_size + 1 (OOV) | 0.0 | Almost certainly unintentional |

**Edge cases:**

| Condition | Score |
|---|---|
| Clean cells are all null / empty | `0.0` |
| `vocab_size == 1` (only one unique clean value) | `1.0` if in vocab, `0.0` if OOV |

---

## API

### `fit(dirty_df, mask_df, col_types=None)`

Learns per-column distribution statistics using **only the clean cells**
(`mask_df[col] == 0`).

**Numerical columns** — stores:
- `mean`: arithmetic mean of clean numeric values (NaN-safe coercion via
  `pd.to_numeric`)
- `std`: population standard deviation (ddof=0)

**Categorical columns** — stores:
- `freq_rank`: `dict` mapping each unique clean-cell string value → integer
  rank (1 = most frequent)
- `vocab_size`: number of unique values observed in clean cells

After `fit()` completes all statistics are stored in `self.col_stats_` and
`self.is_fitted` is set to `True`.

```python
# Example col_stats_ structure after fit:
{
    "age":       {"type": "num", "mean": 29.5, "std": 4.11},
    "education": {
        "type": "cat",
        "freq_rank": {"HS-grad": 1, "Bachelors": 2},
        "vocab_size": 2,
    },
}
```

---

### `compute(dirty_df, mask_df) → pd.DataFrame`

Iterates over every `(row_idx, col_name)` where `mask_df == 1` and applies
the appropriate scoring formula.

Returns a `pd.DataFrame` with columns `[row_idx, col_name, h3_distribution_score]`.

---

### `fit_compute(dirty_df, mask_df, **kwargs) → pd.DataFrame`

Convenience wrapper inherited from `BaseHeuristic`. Calls `fit()` then
`compute()` in one step.

---

## Self-test

A standalone self-test is embedded at the bottom of `h3_distribution_position.py`.

**Run it directly:**

```bash
# from the heuristics/ directory
python h3_distribution_position.py
```

**Or run via the package (from the project root):**

```bash
python -c "
import pandas as pd
from error_detection_system.src.attribution.heuristics import H3DistributionPosition

dirty = pd.DataFrame({
    'age':       [25, 30, 35, 200, 28],
    'education': ['HS-grad', 'HS-grad', 'Bachelors', 'XYZ-fake', 'HS-grad'],
})
mask = pd.DataFrame({
    'age':       [0, 0, 0, 1, 0],
    'education': [0, 0, 0, 1, 0],
})

h3 = H3DistributionPosition()
result = h3.fit_compute(dirty, mask)
print(result.to_string(index=False))
"
```

**Expected output:**

```
 row_idx  col_name  h3_distribution_score
       3       age                    0.0
       3 education                    0.0
```

**Interpretation:**

| row | col | dirty value | reason |
|---|---|---|---|
| 3 | `age` | `200` | ~10+ standard deviations above the mean of clean ages `[25, 30, 35, 28]` → score 0.0 |
| 3 | `education` | `"XYZ-fake"` | Not in vocabulary `{HS-grad, Bachelors}` → OOV rank → score 0.0 |

---

## Numeric Scoring Walkthrough

Given clean ages `[25, 30, 35, 28]` (mean = 29.5, std ≈ 3.84):

| dirty age | z-score | `min(1, |z|/3)` | h3_distribution_score |
|---|---|---|---|
| 30 (central) | 0.13 | 0.044 | **0.956** |
| 35 (edge) | 1.43 | 0.477 | **0.523** |
| 200 (extreme) | 44.5 | 1.0 (capped) | **0.000** |

---

## Categorical Scoring Walkthrough

Given clean education values `["HS-grad", "HS-grad", "Bachelors"]`
(HS-grad count=2 → rank 1; Bachelors count=1 → rank 2; vocab_size=2):

| dirty education | rank | `(rank−1)/max(1,vocab_size−1)` | h3_distribution_score |
|---|---|---|---|
| `"HS-grad"` | 1 | 0/1 = 0.0 | **1.000** |
| `"Bachelors"` | 2 | 1/1 = 1.0 | **0.000** |
| `"XYZ-fake"` (OOV) | 3 | 2/1 = 2.0 → capped | **0.000** |

---

## Design Constraints

- **Exactly one output column** beyond `row_idx` / `col_name`: `h3_distribution_score`.
  z-score and frequency rank are intermediate computations and are never
  exposed in the output DataFrame.
- No hardcoded column names or dataset paths.
- Uses only clean cells (`mask == 0`) to build statistics — no data leakage.
- `col_stats_` is a plain Python dict (JSON-serialisable values only).
- All edge cases (empty column, constant column, non-numeric dirty value,
  OOV categorical) are handled gracefully and score 0.0 or 1.0 as documented.

---

## Related Heuristics

| Heuristic | Relationship to H3 |
|---|---|
| **H1** | Binary in/out-of-domain check. H3 adds *magnitude*: how far outside (or how central). |
| **H2** | Detects obfuscation tokens by string pattern. H3 scores by frequency rank instead. |
| **H4+** | Further context and co-occurrence signals that refine the intentional/unintentional decision. |
