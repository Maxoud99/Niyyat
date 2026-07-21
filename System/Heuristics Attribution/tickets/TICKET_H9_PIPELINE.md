# TICKET H9 — Attribution Pipeline

## Depends on
- TICKET_000_SETUP.md
- TICKET_H1 through TICKET_H8 (all 8 heuristics must be implemented first)

## File to create
```
heuristics/pipeline.py
```

## Class
`AttributionPipeline`

---

## What this assembles

A single orchestrator that:
1. Fits all 8 heuristics in one call.
2. Produces the 13-feature matrix for every erroneous cell.
3. Trains a Random Forest classifier on the 13 features (if labels are available).
4. Outputs predictions and per-heuristic explanation scores.

This is the entry point for the entire system.

---

## Feature matrix (EXACTLY 13 columns)

| # | Column | Source |
|---|--------|--------|
| 1 | `h1_plausible` | H1 |
| 2 | `h2_min_edit_distance` | H2 |
| 3 | `h2_is_obfuscation` | H2 |
| 4 | `h3_distribution_score` | H3 |
| 5 | `h4_coherence_score` | H4 |
| 6 | `h5_error_count` | H5 |
| 7 | `h5_codependent_flag` | H5 |
| 8 | `h6_column_importance` | H6 |
| 9 | `h7_mutability` | H7 |
| 10 | `h7_gain_direction` | H7 |
| 11 | `h7_comprehensibility` | H7 |
| 12 | `h8_is_sensitive` | H8 |
| 13 | `h8_is_majority_value` | H8 |

All rows are `(row_idx, col_name)` pairs where `mask == 1`.

---

## Class interface

```python
class AttributionPipeline:
    def __init__(
        self,
        target_col: str = None,
        codependent_pairs: list = None,
        sensitive_cols: list = None,
        mutability_scores: dict = None,
        comprehensibility_scores: dict = None,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        ...

    def fit(self, dirty_df, mask_df, labels=None):
        """
        Fit all heuristics. If labels (pd.Series with index=MultiIndex(row, col)
        and values 0=unintentional, 1=intentional) are provided, also train the
        RF classifier.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape, 0=clean 1=error)
        labels   : pd.Series or None
            Optional. Index must be a MultiIndex of (row_idx, col_name).
            Values: 1 = intentional, 0 = unintentional.
        """
        ...

    def compute_features(self, dirty_df, mask_df) -> pd.DataFrame:
        """
        Return the 13-column feature matrix for all erroneous cells.
        Index: MultiIndex (row_idx, col_name).
        """
        ...

    def predict(self, dirty_df, mask_df) -> pd.Series:
        """
        Return predicted labels (1=intentional, 0=unintentional) for all
        erroneous cells. Requires fit() to have been called with labels.
        Index: MultiIndex (row_idx, col_name).
        """
        ...

    def predict_proba(self, dirty_df, mask_df) -> pd.DataFrame:
        """
        Return probability of intentional error for all erroneous cells.
        Columns: ['prob_unintentional', 'prob_intentional'].
        Index: MultiIndex (row_idx, col_name).
        """
        ...

    def explain(self, dirty_df, mask_df) -> pd.DataFrame:
        """
        Return per-heuristic scalar scores AND RF feature importances
        for all erroneous cells.
        
        Output columns:
          - All 13 raw feature values
          - 'rf_feature_importance_{feature}' for each of the 13 features
          - 'predicted_label' (if RF trained)
          - 'prob_intentional' (if RF trained)
        """
        ...
```

---

## fit() implementation steps

```python
def fit(self, dirty_df, mask_df, labels=None):
    # 1. Instantiate all heuristics
    self.h1_ = H1ValuePlausibility()
    self.h2_ = H2StringAnomaly()
    self.h3_ = H3DistributionPosition()
    self.h4_ = H4RowCoherence()
    self.h5_ = H5ErrorPattern(codependent_pairs=self.codependent_pairs)
    self.h6_ = H6ColumnImportance()
    self.h7_ = H7UserIncentive(
        mutability_scores=self.mutability_scores,
        comprehensibility_scores=self.comprehensibility_scores,
    )
    self.h8_ = H8SensitivityFlag(sensitive_cols=self.sensitive_cols)

    # 2. Fit all heuristics (pass target_col where applicable)
    self.h1_.fit(dirty_df, mask_df)
    self.h2_.fit(dirty_df, mask_df)
    self.h3_.fit(dirty_df, mask_df)
    self.h4_.fit(dirty_df, mask_df, target_col=self.target_col)
    self.h5_.fit(dirty_df, mask_df)
    self.h6_.fit(dirty_df, mask_df, target_col=self.target_col)
    self.h7_.fit(dirty_df, mask_df, target_col=self.target_col)
    self.h8_.fit(dirty_df, mask_df)

    # 3. Compute feature matrix
    feat_df = self.compute_features(dirty_df, mask_df)

    # 4. If labels provided: train RF classifier
    if labels is not None:
        aligned_labels = labels.loc[feat_df.index]
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.rf_.fit(feat_df.values, aligned_labels.values)
        self.feature_names_ = feat_df.columns.tolist()
        self.fitted_with_labels_ = True
    else:
        self.fitted_with_labels_ = False

    return self
```

---

## compute_features() implementation steps

```python
def compute_features(self, dirty_df, mask_df) -> pd.DataFrame:
    # Call compute() on each heuristic
    h1_out = self.h1_.compute(dirty_df, mask_df)   # cols: [row_idx, col_name, h1_plausible]
    h2_out = self.h2_.compute(dirty_df, mask_df)   # cols: [..., h2_min_edit_distance, h2_is_obfuscation]
    h3_out = self.h3_.compute(dirty_df, mask_df)   # cols: [..., h3_distribution_score]
    h4_out = self.h4_.compute(dirty_df, mask_df)   # cols: [..., h4_coherence_score]
    h5_out = self.h5_.compute(dirty_df, mask_df)   # cols: [..., h5_error_count, h5_codependent_flag]
    h6_out = self.h6_.compute(dirty_df, mask_df)   # cols: [..., h6_column_importance]
    h7_out = self.h7_.compute(dirty_df, mask_df)   # cols: [..., h7_mutability, h7_gain_direction, h7_comprehensibility]
    h8_out = self.h8_.compute(dirty_df, mask_df)   # cols: [..., h8_is_sensitive, h8_is_majority_value]

    # All outputs have MultiIndex (row_idx, col_name) — merge on index
    # Use pd.concat(axis=1) on the feature columns only
    feat_df = pd.concat([
        h1_out.set_index(["row_idx", "col_name"])[["h1_plausible"]],
        h2_out.set_index(["row_idx", "col_name"])[["h2_min_edit_distance", "h2_is_obfuscation"]],
        h3_out.set_index(["row_idx", "col_name"])[["h3_distribution_score"]],
        h4_out.set_index(["row_idx", "col_name"])[["h4_coherence_score"]],
        h5_out.set_index(["row_idx", "col_name"])[["h5_error_count", "h5_codependent_flag"]],
        h6_out.set_index(["row_idx", "col_name"])[["h6_column_importance"]],
        h7_out.set_index(["row_idx", "col_name"])[["h7_mutability", "h7_gain_direction", "h7_comprehensibility"]],
        h8_out.set_index(["row_idx", "col_name"])[["h8_is_sensitive", "h8_is_majority_value"]],
    ], axis=1)
    # Verify: exactly 13 columns
    assert feat_df.shape[1] == 13, f"Expected 13 features, got {feat_df.shape[1]}"
    return feat_df
```

**IMPORTANT**: Each heuristic's `compute()` method MUST return a DataFrame with
columns `row_idx`, `col_name`, and its own feature column(s). This contract
must be honored by all H1–H8 implementations.

---

## predict() and predict_proba() implementation

```python
def predict(self, dirty_df, mask_df) -> pd.Series:
    if not self.fitted_with_labels_:
        raise RuntimeError("Pipeline was not fitted with labels. Cannot predict.")
    feat_df = self.compute_features(dirty_df, mask_df)
    preds = self.rf_.predict(feat_df.values)
    return pd.Series(preds, index=feat_df.index, name="predicted_label")

def predict_proba(self, dirty_df, mask_df) -> pd.DataFrame:
    if not self.fitted_with_labels_:
        raise RuntimeError("Pipeline was not fitted with labels. Cannot predict.")
    feat_df = self.compute_features(dirty_df, mask_df)
    proba = self.rf_.predict_proba(feat_df.values)
    return pd.DataFrame(
        proba,
        index=feat_df.index,
        columns=["prob_unintentional", "prob_intentional"]
    )
```

---

## explain() implementation

```python
def explain(self, dirty_df, mask_df) -> pd.DataFrame:
    feat_df = self.compute_features(dirty_df, mask_df)
    result = feat_df.copy()

    if self.fitted_with_labels_:
        importances = self.rf_.feature_importances_
        for fname, imp in zip(self.feature_names_, importances):
            result[f"rf_importance_{fname}"] = imp  # constant column per feature
        preds = self.rf_.predict(feat_df.values)
        probas = self.rf_.predict_proba(feat_df.values)[:, 1]
        result["predicted_label"] = preds
        result["prob_intentional"] = probas

    return result
```

---

## Self-test (include as `if __name__ == "__main__":`)

```python
import pandas as pd
import numpy as np
from pipeline import AttributionPipeline

np.random.seed(42)
n = 300
dirty = pd.DataFrame({
    "age":           np.random.randint(20, 65, n),
    "education":     np.random.choice(["HS-grad", "Bachelors", "Masters"], n),
    "education-num": np.random.randint(5, 16, n),
    "race":          np.random.choice(["White", "Black", "Asian"], n, p=[0.86, 0.09, 0.05]),
    "income":        np.random.randint(0, 2, n),
})
mask = pd.DataFrame(np.zeros_like(dirty.values, dtype=int), columns=dirty.columns)

# Inject ~30 errors at random positions
error_indices = np.random.choice(n, 30, replace=False)
for idx in error_indices:
    col = np.random.choice(dirty.columns)
    mask.loc[idx, col] = 1

pipe = AttributionPipeline(target_col="income")
pipe.fit(dirty, mask)  # no labels: unsupervised feature extraction only

feat_df = pipe.compute_features(dirty, mask)
print(f"Feature matrix shape: {feat_df.shape}")
print(f"Feature columns: {list(feat_df.columns)}")
assert feat_df.shape[1] == 13, "Must have exactly 13 features"
print(feat_df.head(5).to_string())

explain_df = pipe.explain(dirty, mask)
print(explain_df.head(5).to_string())
print("Pipeline self-test passed.")
```

---

## Constraints
- The pipeline must NOT assume labels are available. All 8 heuristics must run
  with fit(labels=None). The RF classifier is optional.
- `compute_features()` must be callable multiple times on different datasets
  after a single fit() (e.g., train/test split scenario).
- The 13-feature column order must be fixed and deterministic (as listed in the
  table above). Do not rely on dict ordering for features.
- All 8 heuristic instances must be stored on `self` (e.g., `self.h1_`) so they
  can be inspected or re-used after fitting.
- Import from relative paths:
  ```python
  from .h1_value_plausibility   import H1ValuePlausibility
  from .h2_string_anomaly       import H2StringAnomaly
  from .h3_distribution_position import H3DistributionPosition
  from .h4_row_coherence        import H4RowCoherence
  from .h5_error_pattern        import H5ErrorPattern
  from .h6_column_importance    import H6ColumnImportance
  from .h7_user_incentive       import H7UserIncentive
  from .h8_sensitivity_flag     import H8SensitivityFlag
  from .base                    import BaseHeuristic
  ```
