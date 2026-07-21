# TICKET H4 — Row Coherence

## Depends on
TICKET_000_SETUP.md

## File to create
```
heuristics/h4_row_coherence.py
```

## Class
`H4RowCoherence(BaseHeuristic)`

---

## What this heuristic answers

> **Does this dirty value "make sense" given the other values in the same row?**

An intentional error is inserted by a user who understood the record. It tends
to be contextually coherent (e.g., high `education-num` is still paired with
a plausible `education` label). An unintentional error (random substitution,
OCR noise) tends to break the row-level correlation structure.

This is the **KEY heuristic** in the system. It has the most discriminative
power for structured/correlated datasets.

---

## Output features — EXACTLY 1 column (+ row_idx, col_name)

| Column | Type | Description |
|--------|------|-------------|
| `h4_coherence_score` | float [0, 1] | How well this dirty value fits the rest of the row. 1.0 = perfect fit (intentional signal). 0.0 = totally incoherent (unintentional signal). |

---

## fit() specification

```python
def fit(self, dirty_df, mask_df, target_col=None):
```

Train one predictor per column using the **dirty dataset**. Training on dirty
data is acceptable because errors are sparse (≪50% per column).

For each column `c`:
1. Build feature matrix `X` = all rows, all columns **except** `c`.
   - Encode categoricals with `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`.
   - Impute NaN with median (numerical) or mode (categorical).
   - If `target_col` is provided AND `target_col != c`: exclude `target_col` from X
     to prevent data leakage into an outcome variable.
2. Build target vector `y` = column `c`.
   - For **categorical** `c`: encode with `LabelEncoder`. → Train `RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)`.
   - For **numerical** `c`: → Train `RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)`.
3. Fit the predictor on ALL rows (even erroneous ones — sparse errors are tolerable noise).
4. Store in `self.predictors_[c]` and `self.col_stats_[c]` (type, encoders, imputer values, std for numerical).

**Performance note**: If the dirty_df has >100k rows, subsample up to 50k rows
for fitting (stratified for categorical targets). Store `self.fitted_ = True`.

---

## compute() specification

For each erroneous cell at (row_idx, col_name):
1. Load predictor for `col_name`.
2. Construct feature vector `x_row` from that row (all columns except col_name).
   - Apply same OrdinalEncoder and imputation used during fit.
3. Get prediction:
   - **Categorical**: `proba = predictor.predict_proba(x_row)`. 
     `h4_coherence_score = P(predicted class == dirty value)`.
     If dirty_value was unseen by LabelEncoder → score = 0.0.
   - **Numerical**: `predicted = predictor.predict(x_row)`.
     `h4_coherence_score = max(0, 1 - abs(predicted - dirty_value) / col_std)`.
     If `col_std == 0` → score = 1.0.
4. Return `h4_coherence_score` ∈ [0, 1].

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
from h4_row_coherence import H4RowCoherence

# "education" and "education-num" are highly correlated.
# Intentional error: change education but keep education-num consistent → coherent.
# Unintentional error: random wrong education-num → incoherent.
dirty = pd.DataFrame({
    "education":     ["HS-grad", "HS-grad", "Bachelors", "HS-grad",    "Bachelors"],
    "education-num": [9,          9,          13,          9,            5],  # 5 = error (Bachelors should be ~13)
    "age":           [25,         30,         35,          40,           28],
    "hours-per-week":[40,         40,         45,          40,           40],
})
mask = pd.DataFrame({
    "education":     [0, 0, 0, 0, 0],
    "education-num": [0, 0, 0, 0, 1],  # row 4, education-num is erroneous
})

h4 = H4RowCoherence()
h4.fit(dirty, mask)
result = h4.compute(dirty, mask)
print(result.to_string())

# Expected:
# row=4, col=education-num: h4_coherence_score should be LOW (~0.0–0.3)
# because "Bachelors" predicts education-num ≈ 13, but dirty value is 5.
```

---

## Constraints
- NO global correlation matrix. Train per-column RF predictor only.
- Do NOT skip any column from training (even if it has many NaN values — impute them).
- `target_col` parameter exists to let callers exclude the outcome label.
- Must handle both categorical and numerical columns in the same dataset.
- Store the std for each numerical column in `col_stats_` (needed for coherence score denominator).
