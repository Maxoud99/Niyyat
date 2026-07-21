# TICKET H3 — Distribution Position

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h3_distribution_position.py
```

## Class
`H3DistributionPosition(BaseHeuristic)`

---

## What this heuristic answers

> **Is this dirty value common or rare within its column's distribution?**

Intentional errors tend to use unremarkable, common values (to blend in).
Unintentional errors (noise, random substitution) tend to produce extreme or
rare values.

Note: This heuristic is weak for some columns. For `capital-gain` in the Adult
Income dataset, both intentional and unintentional values are 91-93% in-range —
H3 is nearly blind there. This is expected and documented.

---

## Output features — EXACTLY 1 column (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h3_distribution_score` | float | Normalized position in distribution. High (near 1) = common/central value → intentional signal. Low (near 0) = rare/extreme value → unintentional signal. |

This is a single unified feature. The computation differs by column type:

**Numerical**: `h3_distribution_score = 1 - min(1, abs(zscore) / 3.0)`
- zscore = (dirty_value - clean_mean) / clean_std
- Values near the mean score ~1.0. Values 3+ std away score ~0.
- If std == 0, score = 1.0 (constant column, value is always "central").
- If dirty_value cannot be cast to float → score = 0.0.

**Categorical**: `h3_distribution_score = 1 - (frequency_rank - 1) / max(1, vocab_size - 1)`
- frequency_rank = rank of dirty_value in clean vocabulary by frequency (rank 1 = most common).
- If dirty_value is NOT in vocabulary → frequency_rank = vocab_size + 1 (penalize as rarest).
- Result is in [0, 1]: most common value = 1.0, rarest (or out-of-vocab) = 0.0.

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, col_types=None):
```

For each column:
1. Auto-detect type (same rule as H1: pd.to_numeric fraction > 0.9 → numerical).
   Accept optional `col_types` dict to override.
2. For **numerical** columns (clean cells only):
   - Compute `mean`, `std` from clean numeric values (cast to float, drop NaN).
3. For **categorical** columns (clean cells only):
   - Build frequency table: `Counter(clean_cells.dropna().astype(str))`.
   - Compute `freq_rank`: sorted by count descending, rank 1 = most frequent.
   - Store `vocab_size` = number of unique values in clean cells.

Store: `self.col_stats_[col] = {'type': 'cat'|'num', ...stats...}`

---

## compute() specification

For each erroneous cell (mask == 1):
- Look up `self.col_stats_[col_name]`
- Apply the formula for the column type
- Return single value `h3_distribution_score` ∈ [0, 1]

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
from h3_distribution_position import H3DistributionPosition

dirty = pd.DataFrame({
    "age":       [25, 30, 35, 200, 28],   # 200 is error (extreme outlier)
    "education": ["HS-grad", "HS-grad", "Bachelors", "XYZ-fake", "HS-grad"],
                                          # "XYZ-fake" is error (out-of-vocab, rarest)
})
mask = pd.DataFrame({
    "age":       [0, 0, 0, 1, 0],
    "education": [0, 0, 0, 1, 0],
})

h3 = H3DistributionPosition()
h3.fit(dirty, mask)
result = h3.compute(dirty, mask)
print(result.to_string())

# Expected:
# row=3, age:       h3_distribution_score ≈ 0.0  (200 is ~10+ std above mean of [25,30,35,28])
# row=3, education: h3_distribution_score = 0.0  ("XYZ-fake" not in vocab → worst rank)
```

---

## Constraints
- One output column only: `h3_distribution_score`. Do NOT output z-score and
  frequency rank as separate columns — they are intermediate computations.
- No hardcoded column names.
- Handle edge cases: empty clean cells, all-null column, std == 0.
