"""
H4RowCoherence — Heuristic 4
==============================
Measures how well a dirty (erroneous) cell value *fits the rest of its row*,
using a trained Random-Forest predictor for each column.

Motivation
----------
Intentional errors (adversarial data manipulation) are inserted by someone who
understands the record. The manipulated value therefore tends to remain
**contextually coherent** with the other fields in the same row (e.g. a high
``education-num`` is still paired with a matching ``education`` label).
Unintentional errors (random substitution, OCR noise, transcription mistakes)
tend to **break row-level correlation structure** — the injected value is
inconsistent with what the other columns predict.

H4 quantifies this "fit" signal as a single continuous score in [0, 1]:

- **Score ≈ 1.0** → dirty value is exactly what the row context predicts →
  consistent with *intentional* manipulation.
- **Score ≈ 0.0** → dirty value is very different from what the row context
  predicts → consistent with *unintentional* noise.

Approach
--------
One ``RandomForest`` predictor is trained **per column** using the dirty
dataset as training data.  Training on dirty data is acceptable because errors
are sparse (≪50 % per column), so they act as negligible noise.  For each
erroneous cell the predictor for that column is invoked with the **other**
columns as features; the resulting probability (categorical) or distance from
predicted value (numerical) becomes the coherence score.

Column type detection
---------------------
Auto-detection: a column is treated as *numerical* when > 90 % of its non-null
values can be cast to a float; otherwise it is treated as *categorical*.
The caller may override per-column types via the ``col_types`` argument to
:py:meth:`fit`.

Output feature (one row per error position)
-------------------------------------------
===================  =======  =================================================
Column               Type     Description
===================  =======  =================================================
row_idx              int      Row index of the erroneous cell
col_name             str      Column name of the erroneous cell
h4_coherence_score   float    How well the dirty value fits the rest of the row.
                               **High (≈1) = contextually coherent → intentional
                               signal.  Low (≈0) = incoherent → unintentional
                               signal.**
===================  =======  =================================================

Scoring formulas
----------------
**Categorical target column** ``c``::

    h4_coherence_score = P(RF predicts dirty_value | other columns)

If the dirty value was unseen by the ``LabelEncoder`` during fit, score = 0.0.

**Numerical target column** ``c``::

    predicted = RF.predict(x_row)
    h4_coherence_score = max(0, 1 - abs(predicted - dirty_value) / col_std)

If ``col_std == 0`` (constant column) → score = 1.0.

Usage example
-------------
>>> h4 = H4RowCoherence()
>>> h4.fit(dirty_df, mask_df)
>>> features = h4.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .base import BaseHeuristic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUMERIC_THRESHOLD = 0.9          # fraction of non-null values that must parse as float
_MAX_FIT_ROWS = 50_000            # subsample cap for large datasets
_MAX_CAT_TARGET_CARDINALITY = 200  # skip RF training for higher-cardinality categorical targets
_RF_N_ESTIMATORS = 50
_RF_RANDOM_STATE = 42


class H4RowCoherence(BaseHeuristic):
    """
    Heuristic 4 — Row Coherence.

    Scores each erroneous cell by how well its value fits the row context,
    using a per-column Random-Forest predictor trained on the dirty dataset.

    Parameters
    ----------
    (none at construction time; configuration is passed to :py:meth:`fit`)

    Attributes
    ----------
    predictors_ : dict
        Populated after :py:meth:`fit`.  Maps column name → fitted RF model.
    col_stats_ : dict
        Populated after :py:meth:`fit`.  Maps column name → dict with keys:

        - ``'type'``: ``'cat'`` or ``'num'``
        - ``'label_encoder'``: (categorical) fitted :class:`LabelEncoder`
        - ``'std'``: (numerical) ``float`` standard deviation of the column
        - ``'feature_cols'``: ``list[str]`` columns used as features for this target
        - ``'ordinal_encoder'``: fitted :class:`OrdinalEncoder` for the feature matrix
        - ``'impute_values'``: ``dict`` mapping feature column → imputation value
    is_fitted : bool
        Set to ``True`` after :py:meth:`fit` is called.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictors_: Dict[str, object] = {}
        self.col_stats_: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        target_col: Optional[str] = None,
        col_types: Optional[Dict[str, str]] = None,
    ) -> "H4RowCoherence":
        """
        Train one Random-Forest predictor per column.

        Each predictor uses all other columns (except the target itself) as
        features to predict the values of that column.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Binary mask aligned with ``dirty_df``.  0 = clean, 1 = erroneous.
            Used to filter training rows per column: the predictor for column c
            is trained only on rows where mask_df[c] == 0, eliminating training-
            label noise regardless of the per-column error rate.
        target_col : str, optional
            If provided, this column is excluded from the feature matrix when
            training predictors for *other* columns (prevents leakage from an
            outcome variable).  The predictor for ``target_col`` itself is still
            trained using all remaining columns as features.
        col_types : dict, optional
            Per-column type override, e.g. ``{'age': 'num', 'zip_code': 'cat'}``.
            Columns not listed fall back to auto-detection.

        Returns
        -------
        self
        """
        col_types = col_types or {}
        self.predictors_ = {}
        self.col_stats_ = {}

        # ---- determine column types ----------------------------------
        col_type_map: Dict[str, str] = {}
        for col in dirty_df.columns:
            if col in col_types:
                col_type_map[col] = col_types[col]
            else:
                col_type_map[col] = self._detect_type(dirty_df[col])

        # ---- train one predictor per column --------------------------
        all_cols: List[str] = list(dirty_df.columns)

        for c in all_cols:
            ctype = col_type_map[c]

            # ---- skip near-unique identifier-like categorical columns ----
            # A categorical target with very high cardinality (free-text
            # titles, usernames, UPC/EAN/MPN identifiers) is not meaningfully
            # predictable from the rest of the row, and sklearn's RF
            # multi-class machinery scales with n_classes x n_samples --
            # with thousands of near-unique classes this blows up to tens of
            # GB of RAM for no useful signal. _score_cell() returns a
            # neutral 0.5 for any column with no trained predictor.
            if ctype == "cat" and dirty_df[c].nunique(dropna=True) > _MAX_CAT_TARGET_CARDINALITY:
                continue

            # ---- per-column clean-row filter -------------------------
            # Train the predictor for column c only on rows where column c
            # is marked clean (M[i,c] == 0).  This eliminates training-label
            # noise caused by erroneous values in c, regardless of the
            # per-column error rate.
            if c in mask_df.columns:
                clean_rows_c = mask_df[c] == 0
                fit_df = dirty_df.loc[clean_rows_c].copy()
            else:
                fit_df = dirty_df.copy()

            # Fallback: if too few clean rows, use all rows (error-rate < 5%)
            if len(fit_df) < max(50, int(0.05 * len(dirty_df))):
                fit_df = dirty_df.copy()

            if len(fit_df) > _MAX_FIT_ROWS:
                fit_df = fit_df.sample(n=_MAX_FIT_ROWS, random_state=_RF_RANDOM_STATE)

            # Feature columns: everything except c (and target_col if applicable)
            feature_cols = [
                fc for fc in all_cols
                if fc != c and not (target_col is not None and fc == target_col and fc != c)
            ]
            # Cleaner version of the exclusion rule:
            # exclude c itself; also exclude target_col when target_col != c
            feature_cols = [
                fc for fc in all_cols
                if fc != c and not (target_col is not None and target_col != c and fc == target_col)
            ]

            # ---- build feature matrix X ------------------------------
            X_raw = fit_df[feature_cols].copy()
            impute_values = self._compute_impute_values(X_raw, col_type_map)
            X_imputed = self._apply_imputation(X_raw, impute_values, col_type_map)

            # Identify categorical feature columns
            cat_feature_cols = [fc for fc in feature_cols if col_type_map[fc] == "cat"]
            num_feature_cols = [fc for fc in feature_cols if col_type_map[fc] == "num"]

            # Encode categoricals with OrdinalEncoder
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float64,
            )
            X_encoded = X_imputed.copy()
            if cat_feature_cols:
                X_encoded[cat_feature_cols] = ord_enc.fit_transform(
                    X_imputed[cat_feature_cols].astype(str)
                )
            else:
                ord_enc = None  # no categorical features to encode

            X = X_encoded.values.astype(np.float64)

            # ---- build target vector y and fit -----------------------
            y_raw = fit_df[c]

            if ctype == "cat":
                le = LabelEncoder()
                y_str = y_raw.fillna("__NaN__").astype(str)
                y = le.fit_transform(y_str)
                rf = RandomForestClassifier(
                    n_estimators=_RF_N_ESTIMATORS,
                    random_state=_RF_RANDOM_STATE,
                    n_jobs=-1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rf.fit(X, y)
                self.predictors_[c] = rf
                self.col_stats_[c] = {
                    "type": "cat",
                    "label_encoder": le,
                    "feature_cols": feature_cols,
                    "cat_feature_cols": cat_feature_cols,
                    "num_feature_cols": num_feature_cols,
                    "ordinal_encoder": ord_enc,
                    "impute_values": impute_values,
                }

            else:  # num
                y_numeric = pd.to_numeric(y_raw, errors="coerce")
                # Impute NaN in target with median (so RF can train)
                y_median = float(y_numeric.median()) if y_numeric.notna().any() else 0.0
                y = y_numeric.fillna(y_median).values.astype(np.float64)
                col_std = float(y_numeric.std(ddof=0)) if y_numeric.notna().sum() > 1 else 0.0

                rf = RandomForestRegressor(
                    n_estimators=_RF_N_ESTIMATORS,
                    random_state=_RF_RANDOM_STATE,
                    n_jobs=-1,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rf.fit(X, y)
                self.predictors_[c] = rf
                self.col_stats_[c] = {
                    "type": "num",
                    "std": col_std,
                    "feature_cols": feature_cols,
                    "cat_feature_cols": cat_feature_cols,
                    "num_feature_cols": num_feature_cols,
                    "ordinal_encoder": ord_enc,
                    "impute_values": impute_values,
                }

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the row-coherence score for every erroneous cell.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as ``dirty_df``, values in {0, 1})

        Returns
        -------
        pd.DataFrame
            One row per error position with columns:
            ``[row_idx, col_name, h4_coherence_score]``

            ``h4_coherence_score`` ∈ [0, 1]:

            - near 1 → dirty value fits row context → intentional-error signal
            - near 0 → dirty value breaks row context → unintentional-error signal
        """
        self._check_fitted()

        rows = []
        for row_idx, col_name in self._get_error_positions(mask_df):
            score = self._score_cell(dirty_df, row_idx, col_name)
            rows.append(
                {"row_idx": row_idx, "col_name": col_name, "h4_coherence_score": score}
            )

        if not rows:
            return pd.DataFrame(columns=["row_idx", "col_name", "h4_coherence_score"])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers — type detection & imputation
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_type(col: pd.Series) -> str:
        """Return ``'num'`` or ``'cat'`` by attempting numeric coercion."""
        non_null = col.dropna()
        if non_null.empty:
            return "cat"
        numeric = pd.to_numeric(non_null, errors="coerce")
        frac_numeric = numeric.notna().sum() / len(non_null)
        return "num" if frac_numeric > _NUMERIC_THRESHOLD else "cat"

    @staticmethod
    def _compute_impute_values(
        X_raw: pd.DataFrame,
        col_type_map: Dict[str, str],
    ) -> Dict[str, object]:
        """
        Compute per-column imputation values from the feature matrix.

        Numerical columns → median.
        Categorical columns → mode (most frequent value, as string).
        """
        impute_values: Dict[str, object] = {}
        for col in X_raw.columns:
            ctype = col_type_map.get(col, "cat")
            if ctype == "num":
                numeric = pd.to_numeric(X_raw[col], errors="coerce")
                impute_values[col] = float(numeric.median()) if numeric.notna().any() else 0.0
            else:
                non_null = X_raw[col].dropna().astype(str)
                if non_null.empty:
                    impute_values[col] = "__NaN__"
                else:
                    impute_values[col] = non_null.mode().iloc[0]
        return impute_values

    @staticmethod
    def _apply_imputation(
        X_raw: pd.DataFrame,
        impute_values: Dict[str, object],
        col_type_map: Dict[str, str],
    ) -> pd.DataFrame:
        """Fill NaN values using pre-computed impute_values."""
        X = X_raw.copy()
        for col in X.columns:
            ctype = col_type_map.get(col, "cat")
            fill_val = impute_values[col]
            if ctype == "num":
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_val)
            else:
                X[col] = X[col].fillna(fill_val).astype(str)
        return X

    # ------------------------------------------------------------------
    # Private helpers — scoring
    # ------------------------------------------------------------------

    def _score_cell(
        self,
        dirty_df: pd.DataFrame,
        row_idx: int,
        col_name: str,
    ) -> float:
        """Return the h4_coherence_score for a single erroneous cell."""
        if col_name not in self.predictors_:
            return 0.5  # no model trained (e.g. skipped high-cardinality target)
        stats = self.col_stats_[col_name]
        rf = self.predictors_[col_name]
        feature_cols = stats["feature_cols"]
        cat_feature_cols = stats["cat_feature_cols"]
        impute_values = stats["impute_values"]
        ord_enc: Optional[OrdinalEncoder] = stats["ordinal_encoder"]

        # ---- build feature vector for this row -----------------------
        # Detect column type map from col_stats_ for imputation
        col_type_map_local = {
            fc: ("num" if self.col_stats_.get(fc, {}).get("type", "cat") == "num" else "cat")
            for fc in feature_cols
        }

        x_series = dirty_df.loc[row_idx, feature_cols].copy()
        x_df = pd.DataFrame([x_series], columns=feature_cols)

        # Impute
        x_df = self._apply_imputation(x_df, impute_values, col_type_map_local)

        # Encode categorical features
        x_encoded = x_df.copy()
        if cat_feature_cols and ord_enc is not None:
            x_encoded[cat_feature_cols] = ord_enc.transform(
                x_df[cat_feature_cols].astype(str)
            )

        x_row = x_encoded.values.astype(np.float64)

        # ---- get prediction and compute score -----------------------
        dirty_value = dirty_df.loc[row_idx, col_name]

        if stats["type"] == "cat":
            return self._score_categorical(rf, stats["label_encoder"], dirty_value, x_row)
        else:
            return self._score_numerical(rf, stats["std"], dirty_value, x_row)

    @staticmethod
    def _score_categorical(
        rf: RandomForestClassifier,
        le: LabelEncoder,
        dirty_value,
        x_row: np.ndarray,
    ) -> float:
        """
        Score a single erroneous cell in a categorical column.

        Returns P(RF predicts dirty_value | x_row) as the coherence score.
        If dirty_value was unseen by the LabelEncoder during fit → score = 0.0.
        """
        dirty_str = str(dirty_value) if not pd.isna(dirty_value) else "__NaN__"
        try:
            # find the class index for the dirty value
            class_idx = int(np.where(le.classes_ == dirty_str)[0][0])
        except IndexError:
            # dirty_value was not seen during training
            return 0.0

        proba = rf.predict_proba(x_row)[0]  # shape (n_classes,)
        # predict_proba columns correspond to rf.classes_ (integer indices)
        # Map class_idx back to proba position
        rf_class_positions = list(rf.classes_)
        if class_idx not in rf_class_positions:
            return 0.0
        pos = rf_class_positions.index(class_idx)
        return float(proba[pos])

    @staticmethod
    def _score_numerical(
        rf: RandomForestRegressor,
        col_std: float,
        dirty_value,
        x_row: np.ndarray,
    ) -> float:
        """
        Score a single erroneous cell in a numerical column.

        Returns max(0, 1 − |predicted − dirty_value| / col_std).
        If col_std == 0 → score = 1.0.
        If dirty_value cannot be cast to float → score = 0.0.
        """
        if col_std == 0.0:
            return 1.0

        try:
            dirty_float = float(dirty_value)
        except (ValueError, TypeError):
            return 0.0

        predicted = float(rf.predict(x_row)[0])
        return float(max(0.0, 1.0 - abs(predicted - dirty_float) / col_std))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from h4_row_coherence import H4RowCoherence

    print("=" * 65)
    print("H4 Row Coherence — self-test")
    print("=" * 65)

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
        "age":           [0, 0, 0, 0, 0],
        "hours-per-week":[0, 0, 0, 0, 0],
    })

    h4 = H4RowCoherence()
    h4.fit(dirty, mask)
    result = h4.compute(dirty, mask)

    print("\nResult:")
    print(result.to_string(index=False))
    print()

    # ---- assertions --------------------------------------------------
    def get_score(df, row, col):
        row_data = df[(df["row_idx"] == row) & (df["col_name"] == col)]
        assert not row_data.empty, f"Missing entry for ({row}, {col})"
        return float(row_data.iloc[0]["h4_coherence_score"])

    score = get_score(result, 4, "education-num")
    print(f"row=4, col='education-num', dirty_value=5 (Bachelors expects ~13)")
    print(f"  h4_coherence_score = {score:.4f}  (expected LOW ≈ 0.0–0.3)")

    # Verify output structure
    assert list(result.columns) == ["row_idx", "col_name", "h4_coherence_score"], (
        f"Unexpected columns: {list(result.columns)}"
    )
    assert 0.0 <= score <= 1.0, f"Score out of [0, 1]: {score}"
    assert score <= 0.5, (
        f"Expected LOW coherence for incoherent error (Bachelors + edu-num=5), got {score:.4f}"
    )

    print("\nAll assertions passed ✓")
    print()
    print("Explanation:")
    print("  'Bachelors' (education) strongly predicts education-num ≈ 13.")
    print("  The dirty value 5 is far from what the row context expects,")
    print("  so h4_coherence_score is LOW → unintentional error signal.")
