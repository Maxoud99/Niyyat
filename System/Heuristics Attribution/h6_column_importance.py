"""
H6ColumnImportance — Heuristic 6
==================================
Measures the **statistical importance** of a column — i.e. how strongly it
correlates with the outcome (or with any other column when no target is
available) — as a proxy for how attractive that column is as a target for
intentional, model-gaming data manipulation.

Motivation
----------
A person who wants to game a predictive model will preferentially corrupt the
columns that matter most to the model's output: those with high mutual
information (MI) with the target.  Corrupting a low-MI column wastes effort
because it barely moves predictions.  H6 captures this attractiveness signal.

Critical distinction from H7 (User Incentive)
----------------------------------------------
H6 is **statistical** importance, NOT human behavioral motivation:

- ``fnlwgt`` (Adult Income sampling weight): high MI with outcome → H6 high.
  But a human wouldn't touch it because they don't understand it → H7 low.
- ``race``: low MI with income prediction → H6 low.
  But a human might mask it for fairness/privacy reasons → H7 high.

Both signals are needed.  They are **NOT** redundant.

Output feature (one row per error position)
--------------------------------------------
=======================  =========  ===========================================
Column                   Type       Description
=======================  =========  ===========================================
row_idx                  int        Row index of the erroneous cell
col_name                 str        Column name of the erroneous cell
h6_column_importance     float      Normalised statistical importance in [0, 1].
                                    **Per-column constant**: all erroneous
                                    cells in the same column receive the same
                                    score.  1.0 = most important column.
                                    0.0 = least important / uncorrelated.
=======================  =========  ===========================================

Scoring modes
-------------
**Supervised (target_col provided)**::

    MI(column, target_col) for each feature column
    target_col itself → max MI of all other columns
    All scores normalised to [0, 1] by dividing by the maximum.

**Unsupervised (no target_col)**::

    importance[c] = max MI between c and every other column
    (capped to 10 random peers if table has >30 columns)
    Normalised to [0, 1].

Usage example
-------------
>>> h6 = H6ColumnImportance()
>>> h6.fit(dirty_df, mask_df, target_col="income")
>>> features = h6.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import random
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Support both package import (relative) and standalone execution (absolute)
if __package__:
    from .base import BaseHeuristic
else:
    # Running directly: e.g. `python h6_column_importance.py`
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from base import BaseHeuristic  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUMERIC_THRESHOLD = 0.9          # fraction of non-null values that must parse
                                  # as float for a column to be treated numeric
_UNSUPERVISED_COL_CAP = 30        # if table has more columns than this, sample
_UNSUPERVISED_SAMPLE_K = 10       # number of random peers sampled per column
_MIN_CLEAN_ROWS = 10              # minimum clean rows needed to estimate MI


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_numeric_col(series: pd.Series) -> bool:
    """Return True if > _NUMERIC_THRESHOLD of non-null values parse as float."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    return (numeric_count / len(non_null)) > _NUMERIC_THRESHOLD


def _encode_dataframe(df: pd.DataFrame) -> np.ndarray:
    """
    Encode a DataFrame to a numeric numpy array for MI computation.

    - Numerical columns: cast to float, NaN preserved.
    - Categorical columns: OrdinalEncoder (NaN → -1 sentinel before encoding).
    """
    result = np.empty((len(df), len(df.columns)), dtype=float)
    for j, col in enumerate(df.columns):
        s = df[col]
        if _is_numeric_col(s):
            result[:, j] = pd.to_numeric(s, errors="coerce").values
        else:
            # OrdinalEncoder requires no NaN; fill with sentinel string first
            filled = s.fillna("__NaN__").astype(str).values.reshape(-1, 1)
            enc = OrdinalEncoder()
            result[:, j] = enc.fit_transform(filled).ravel()
    return result


def _impute(X: np.ndarray) -> np.ndarray:
    """
    Impute NaN values: median strategy (works for both numeric encodings and
    ordinal-encoded categoricals since they are all float after _encode).
    """
    if not np.isnan(X).any():
        return X
    imp = SimpleImputer(strategy="median")
    return imp.fit_transform(X)


def _mi_column_vs_target(
    X: np.ndarray,
    y: np.ndarray,
    y_is_categorical: bool,
) -> np.ndarray:
    """
    Compute MI between each column in X and target y.

    Parameters
    ----------
    X               : (n_samples, n_features) float array (already imputed)
    y               : (n_samples,) float array (already imputed)
    y_is_categorical: if True use mutual_info_classif, else mutual_info_regression

    Returns
    -------
    mi : (n_features,) float array of MI scores
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if y_is_categorical:
            mi = mutual_info_classif(X, y.astype(int), discrete_features=False,
                                     random_state=0)
        else:
            mi = mutual_info_regression(X, y, discrete_features=False,
                                        random_state=0)
    return mi


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] by dividing by its max. Zero array → all zeros."""
    max_val = scores.max()
    if max_val == 0.0:
        return np.zeros_like(scores, dtype=float)
    return scores / max_val


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class H6ColumnImportance(BaseHeuristic):
    """
    Heuristic 6 — Column Importance (MI-based statistical importance).

    Answers: *Does this column matter statistically?*  A column with high
    mutual information to the outcome is a more attractive target for
    intentional model-gaming manipulation.

    The score is a **per-column constant**: every erroneous cell in the same
    column receives the same ``h6_column_importance`` value.

    Parameters
    ----------
    None  (all configuration is passed to ``fit()``).

    Attributes
    ----------
    importance_scores_ : dict[str, float]
        Mapping from column name → normalized importance score in [0, 1].
        Populated after ``fit()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.importance_scores_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> "H6ColumnImportance":
        """
        Compute per-column statistical importance using clean cells only.

        Parameters
        ----------
        dirty_df   : pd.DataFrame  — the dirty dataset.
        mask_df    : pd.DataFrame  — binary mask aligned with dirty_df
                                     (0 = clean, 1 = erroneous).
        target_col : str, optional — name of the target / outcome column.
                                     If provided, supervised MI is used.
                                     If None, unsupervised pairwise MI is used.

        Returns
        -------
        self
        """
        if dirty_df.shape != mask_df.shape:
            raise ValueError("dirty_df and mask_df must have the same shape.")
        if not list(dirty_df.columns) == list(mask_df.columns):
            raise ValueError("dirty_df and mask_df must have the same columns.")

        columns = list(dirty_df.columns)

        if target_col is not None and target_col not in columns:
            raise ValueError(f"target_col '{target_col}' not found in dirty_df.")

        # ----- select clean rows (mask == 0 for the entire row) ----------
        clean_row_mask = (mask_df == 0).all(axis=1)
        clean_df = dirty_df[clean_row_mask].copy()

        if len(clean_df) < _MIN_CLEAN_ROWS:
            warnings.warn(
                f"H6ColumnImportance: only {len(clean_df)} fully-clean rows "
                f"available (minimum {_MIN_CLEAN_ROWS}). "
                "Falling back to uniform importance (all columns score 0.0).",
                UserWarning,
            )
            self.importance_scores_ = {col: 0.0 for col in columns}
            self.is_fitted = True
            return self

        if target_col is not None:
            self._fit_supervised(clean_df, columns, target_col)
        else:
            self._fit_unsupervised(clean_df, columns)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # fit helpers
    # ------------------------------------------------------------------

    def _fit_supervised(
        self,
        clean_df: pd.DataFrame,
        columns: List[str],
        target_col: str,
    ) -> None:
        """Supervised MI: compute MI of each feature column against target_col."""
        feature_cols = [c for c in columns if c != target_col]

        if not feature_cols:
            # Only the target column exists — trivial case
            self.importance_scores_ = {target_col: 1.0}
            return

        # Encode features (X) and target (y)
        X_enc = _encode_dataframe(clean_df[feature_cols])
        X_imp = _impute(X_enc)

        target_series = clean_df[target_col]
        y_is_cat = not _is_numeric_col(target_series)

        if y_is_cat:
            y_enc = OrdinalEncoder().fit_transform(
                target_series.fillna("__NaN__").astype(str).values.reshape(-1, 1)
            ).ravel()
            y_imp = y_enc  # OrdinalEncoder produces no NaN
        else:
            y_arr = pd.to_numeric(target_series, errors="coerce").values.astype(float)
            y_imp = SimpleImputer(strategy="median").fit_transform(
                y_arr.reshape(-1, 1)
            ).ravel()

        # Handle zero-variance columns: MI will be 0 naturally, but sklearn
        # may warn. Identify them and set MI=0 explicitly.
        zero_var_mask = X_imp.std(axis=0) == 0
        mi_scores = _mi_column_vs_target(X_imp, y_imp, y_is_cat)
        mi_scores[zero_var_mask] = 0.0

        # Build raw importance dict for feature columns
        raw: Dict[str, float] = {
            col: float(mi) for col, mi in zip(feature_cols, mi_scores)
        }

        # Target column itself → max MI of all feature columns
        max_feature_mi = max(raw.values()) if raw else 0.0
        raw[target_col] = max_feature_mi

        # Normalize to [0, 1]
        values = np.array([raw[c] for c in columns], dtype=float)
        normalized = _normalize(values)
        self.importance_scores_ = {
            col: float(normalized[i]) for i, col in enumerate(columns)
        }

    def _fit_unsupervised(
        self,
        clean_df: pd.DataFrame,
        columns: List[str],
    ) -> None:
        """
        Unsupervised MI: for each column c, importance = max MI(c, other column).
        If table has >_UNSUPERVISED_COL_CAP columns, sample _UNSUPERVISED_SAMPLE_K
        random peers per column to limit computation cost.
        """
        n_cols = len(columns)
        X_enc = _encode_dataframe(clean_df[columns])
        X_imp = _impute(X_enc)

        use_sampling = n_cols > _UNSUPERVISED_COL_CAP

        raw = np.zeros(n_cols, dtype=float)

        rng = random.Random(42)

        for i, col in enumerate(columns):
            # Determine which other column indices to compare against
            other_indices = [j for j in range(n_cols) if j != i]

            if use_sampling and len(other_indices) > _UNSUPERVISED_SAMPLE_K:
                other_indices = rng.sample(other_indices, _UNSUPERVISED_SAMPLE_K)

            if not other_indices:
                raw[i] = 0.0
                continue

            y = X_imp[:, i]

            # Use classif if zero-variance is the issue, but MI itself handles it
            # Determine target type: treat as categorical if column was cat-encoded
            col_is_cat = not _is_numeric_col(clean_df[col])

            # X for MI = the selected peer columns
            X_peers = X_imp[:, other_indices]  # (n_rows, n_peers)

            # sklearn expects X shape (n_samples, n_features) and y (n_samples,)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if col_is_cat:
                    mi_vals = mutual_info_classif(
                        X_peers, y.astype(int), discrete_features=False,
                        random_state=0,
                    )
                else:
                    mi_vals = mutual_info_regression(
                        X_peers, y, discrete_features=False, random_state=0
                    )

            raw[i] = float(mi_vals.max()) if len(mi_vals) > 0 else 0.0

        normalized = _normalize(raw)
        self.importance_scores_ = {
            col: float(normalized[i]) for i, col in enumerate(columns)
        }

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    def compute(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return h6_column_importance for every erroneous cell (mask == 1).

        The score is a **per-column constant**: all erroneous cells in the
        same column receive the same value regardless of which row they
        are in.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame with columns:
            row_idx              (int)
            col_name             (str)
            h6_column_importance (float in [0, 1])
        """
        self._check_fitted()

        records = []
        for col in mask_df.columns:
            error_rows = mask_df.index[mask_df[col] == 1].tolist()
            if not error_rows:
                continue
            # Per-column constant — look up once
            score = self.importance_scores_.get(col, 0.0)
            for row_idx in error_rows:
                records.append({
                    "row_idx": row_idx,
                    "col_name": col,
                    "h6_column_importance": score,
                })

        if not records:
            return pd.DataFrame(columns=["row_idx", "col_name", "h6_column_importance"])

        return pd.DataFrame(records)[["row_idx", "col_name", "h6_column_importance"]]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 200
    # "education-num" strongly determines "income". "age" is weakly correlated.
    # "fnlwgt" is random (not correlated with income).
    education_num = np.random.randint(5, 16, n)
    income = (education_num > 10).astype(int)   # direct deterministic relationship
    age = np.random.randint(20, 65, n)
    fnlwgt = np.random.randint(100000, 500000, n)  # noise column

    dirty = pd.DataFrame({
        "education-num": education_num,
        "age":           age,
        "fnlwgt":        fnlwgt,
        "income":        income,
    })
    mask = pd.DataFrame(
        np.zeros_like(dirty.values, dtype=int), columns=dirty.columns
    )
    # Mark a few errors in education-num and fnlwgt
    mask.loc[[5, 10, 15], "education-num"] = 1
    mask.loc[[20, 25], "fnlwgt"] = 1

    h6 = H6ColumnImportance()
    h6.fit(dirty, mask, target_col="income")
    result = h6.compute(dirty, mask)

    print("=" * 60)
    print("H6ColumnImportance — Self-test")
    print("=" * 60)
    print("\nRaw importance scores (all columns):")
    for col, score in sorted(h6.importance_scores_.items(),
                              key=lambda x: -x[1]):
        print(f"  {col:20s}  {score:.4f}")

    print("\nOutput (erroneous cells only):")
    print(result.to_string(index=False))

    # Assertions
    edu_score = h6.importance_scores_["education-num"]
    fnl_score = h6.importance_scores_["fnlwgt"]

    assert edu_score > 0.8, (
        f"education-num should be close to 1.0 (got {edu_score:.4f})"
    )
    assert fnl_score < 0.2, (
        f"fnlwgt should be close to 0.0 (got {fnl_score:.4f})"
    )

    # Per-column constant: all education-num rows get the same score
    edu_rows = result[result["col_name"] == "education-num"]["h6_column_importance"]
    assert edu_rows.nunique() == 1, "education-num rows should all have the same score"

    fnl_rows = result[result["col_name"] == "fnlwgt"]["h6_column_importance"]
    assert fnl_rows.nunique() == 1, "fnlwgt rows should all have the same score"

    print("\n✅ All assertions passed.")
    print(
        "\nExpected: education-num ≈ 1.0 (most correlated with income), "
        "fnlwgt ≈ 0.0 (random noise)."
    )
