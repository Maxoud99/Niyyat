"""
AttributionPipeline — H9
========================
Orchestrator that assembles all 8 heuristics (H1–H8) into a single callable
that produces the 12-feature matrix used for classification and explanation.

Overview
--------
The pipeline:

1. Fits all 8 heuristics in a single :py:meth:`fit` call.
2. Produces the **exactly 12-column** feature matrix for every erroneous cell
   (mask == 1) via :py:meth:`compute_features`.
3. Optionally trains a Random-Forest classifier when intent labels are
   supplied (supervised mode).
4. Exposes :py:meth:`predict`, :py:meth:`predict_proba`, and :py:meth:`explain`
   for downstream use.

Feature matrix (fixed column order)
-------------------------------------
+----+---------------------------+--------+
| #  | Column                    | Source |
+====+===========================+========+
|  1 | h1_plausible              | H1     |
|  2 | h2_min_edit_distance      | H2     |
|  3 | h2_is_obfuscation         | H2     |
|  4 | h3_distribution_score     | H3     |
|  5 | h4_coherence_score        | H4     |
|  6 | h5_error_count            | H5     |
|  7 | h5_codependent_flag       | H5     |
|  8 | h6_column_importance      | H6     |
|  9 | h7_gain_direction         | H7     |
| 10 | h7_comprehensibility      | H7     |
| 11 | h8_is_sensitive           | H8     |
| 12 | h8_is_majority_value      | H8     |
+----+---------------------------+--------+

(``h7_mutability`` is still computed by H7 but excluded from the deployed
12-feature fingerprint — paper Table 2; see the FEATURE_COLUMNS note below.)

Note
----
H2's internal column name is ``h2_min_edit_dist``; the pipeline renames it
to ``h2_min_edit_distance`` to honour the 12-column contract above.

Usage
-----
>>> pipe = AttributionPipeline(target_col="income")
>>> pipe.fit(dirty_df, mask_df)                      # unsupervised
>>> feat_df = pipe.compute_features(dirty_df, mask_df)
>>> assert feat_df.shape[1] == 12

>>> pipe.fit(dirty_df, mask_df, labels=labels_series)  # supervised
>>> preds = pipe.predict(dirty_df, mask_df)
>>> explain_df = pipe.explain(dirty_df, mask_df)
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------------------------
# Relative imports (package mode) — fall back to absolute path for standalone
# ---------------------------------------------------------------------------
if __package__:
    from .h1_value_plausibility    import H1ValuePlausibility
    from .h2_string_anomaly        import H2StringAnomaly
    from .h3_distribution_position import H3DistributionPosition
    from .h4_row_coherence         import H4RowCoherence
    from .h5_error_pattern         import H5ErrorPattern
    from .h6_column_importance     import H6ColumnImportance
    from .h7_user_incentive        import H7UserIncentive
    from .h8_sensitivity_flag      import H8SensitivityFlag
    from .base                     import BaseHeuristic
else:
    # Running as   python pipeline.py   directly
    _src_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    from attribution.heuristics.h1_value_plausibility    import H1ValuePlausibility     # type: ignore
    from attribution.heuristics.h2_string_anomaly        import H2StringAnomaly         # type: ignore
    from attribution.heuristics.h3_distribution_position import H3DistributionPosition  # type: ignore
    from attribution.heuristics.h4_row_coherence         import H4RowCoherence          # type: ignore
    from attribution.heuristics.h5_error_pattern         import H5ErrorPattern          # type: ignore
    from attribution.heuristics.h6_column_importance     import H6ColumnImportance      # type: ignore
    from attribution.heuristics.h7_user_incentive        import H7UserIncentive         # type: ignore
    from attribution.heuristics.h8_sensitivity_flag      import H8SensitivityFlag       # type: ignore
    from attribution.heuristics.base                     import BaseHeuristic           # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The 12 feature columns in their required fixed order.
FEATURE_COLUMNS: List[str] = [
    "h1_plausible",
    "h2_min_edit_distance",   # renamed from H2's internal "h2_min_edit_dist"
    "h2_is_obfuscation",
    "h3_distribution_score",
    "h4_coherence_score",
    "h5_error_count",
    "h5_codependent_flag",
    "h6_column_importance",
    # "h7_mutability",  # dropped from the deployed 12-feature fingerprint
    # (paper Table 2): RF importance <= .011 on every dataset where it is
    # defined, and removing it changes F1-macro by at most 0.0005 (see the
    # feature-importance note in Intent_Paper/chapters/7_ablation_study.tex).
    # H7 still computes the column; it is simply not selected here.
    "h7_gain_direction",
    "h7_comprehensibility",
    "h8_is_sensitive",
    "h8_is_majority_value",
]

_N_FEATURES = len(FEATURE_COLUMNS)  # must be 12


# ===========================================================================
# AttributionPipeline
# ===========================================================================

class AttributionPipeline:
    """
    Orchestrator for all 8 intent-attribution heuristics.

    The pipeline is designed to work **without labels** (unsupervised mode).
    Supplying labels to :py:meth:`fit` additionally trains a Random-Forest
    classifier whose predictions are exposed via :py:meth:`predict` and
    :py:meth:`predict_proba`.

    Parameters
    ----------
    target_col : str or None
        Name of the outcome / target column in the dataset (e.g. ``"income"``).
        Passed to H4, H6, and H7.  If ``None``, those heuristics fall back to
        their unsupervised variants.
    codependent_pairs : list of (str, str) or None
        User-supplied pairs of logically linked columns for H5.  Auto-discovery
        is always attempted in addition to these pairs.
    sensitive_cols : list of str or None
        User-supplied list of sensitive demographic columns for H8.
        Auto-detection is still performed; this list supplements it.
    mutability_scores : dict or None
        ``{col_name: float}`` override for H7 mutability scores.
    comprehensibility_scores : dict or None
        ``{col_name: float}`` override for H7 comprehensibility scores.
    n_estimators : int
        Number of trees in the Random-Forest classifier (default 100).
    random_state : int
        Random seed for the Random-Forest classifier (default 42).

    Attributes
    ----------
    h1_ … h8_ : BaseHeuristic subclasses
        Fitted heuristic instances.  Available after :py:meth:`fit`.
    rf_ : RandomForestClassifier or None
        Fitted RF classifier.  ``None`` if ``fit()`` was called without labels.
    feature_names_ : list of str
        The 12 feature column names (equals ``FEATURE_COLUMNS``).
    fitted_with_labels_ : bool
        ``True`` iff RF was trained.
    """

    def __init__(
        self,
        target_col: Optional[str] = None,
        codependent_pairs: Optional[List] = None,
        sensitive_cols: Optional[List[str]] = None,
        mutability_scores: Optional[Dict[str, float]] = None,
        comprehensibility_scores: Optional[Dict[str, float]] = None,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.target_col               = target_col
        self.codependent_pairs        = codependent_pairs
        self.sensitive_cols           = sensitive_cols
        self.mutability_scores        = mutability_scores
        self.comprehensibility_scores = comprehensibility_scores
        self.n_estimators             = n_estimators
        self.random_state             = random_state

        # Set after fit()
        self.h1_: Optional[H1ValuePlausibility]    = None
        self.h2_: Optional[H2StringAnomaly]        = None
        self.h3_: Optional[H3DistributionPosition] = None
        self.h4_: Optional[H4RowCoherence]         = None
        self.h5_: Optional[H5ErrorPattern]         = None
        self.h6_: Optional[H6ColumnImportance]     = None
        self.h7_: Optional[H7UserIncentive]        = None
        self.h8_: Optional[H8SensitivityFlag]      = None
        self.rf_: Optional[RandomForestClassifier] = None
        self.feature_names_: List[str]             = FEATURE_COLUMNS[:]
        self.fitted_with_labels_: bool             = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        labels: Optional[pd.Series] = None,
    ) -> "AttributionPipeline":
        """
        Fit all 8 heuristics.  Optionally train an RF classifier.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty (potentially erroneous) dataset.
        mask_df : pd.DataFrame
            Binary error mask aligned with ``dirty_df``.
            0 = clean cell, 1 = erroneous cell.  Same shape and column names.
        labels : pd.Series or None
            Optional intent labels indexed by MultiIndex ``(row_idx, col_name)``.
            Values: 1 = intentional, 0 = unintentional.
            If provided, an RF classifier is trained on the 12-feature matrix.

        Returns
        -------
        self
        """
        # ── 1. Instantiate all heuristics ──────────────────────────────────
        # Note: H5, H7, H8 constructors take no arguments; configuration
        # is forwarded to their respective fit() calls instead.
        self.h1_ = H1ValuePlausibility()
        self.h2_ = H2StringAnomaly()
        self.h3_ = H3DistributionPosition()
        self.h4_ = H4RowCoherence()
        self.h5_ = H5ErrorPattern()
        self.h6_ = H6ColumnImportance()
        self.h7_ = H7UserIncentive()
        self.h8_ = H8SensitivityFlag()

        # ── 2. Fit all heuristics ───────────────────────────────────────────
        self.h1_.fit(dirty_df, mask_df)
        self.h2_.fit(dirty_df, mask_df)
        self.h3_.fit(dirty_df, mask_df)
        self.h4_.fit(dirty_df, mask_df, target_col=self.target_col)
        self.h5_.fit(dirty_df, mask_df, codependent_pairs=self.codependent_pairs)
        self.h6_.fit(dirty_df, mask_df, target_col=self.target_col)
        self.h7_.fit(
            dirty_df, mask_df,
            target_col=self.target_col,
            mutability_scores=self.mutability_scores,
            comprehensibility_scores=self.comprehensibility_scores,
        )
        self.h8_.fit(dirty_df, mask_df, sensitive_cols=self.sensitive_cols)

        # ── 3. Compute the feature matrix (validates the 12-col contract) ──
        feat_df = self.compute_features(dirty_df, mask_df)

        # ── 4. Optionally train RF classifier ──────────────────────────────
        if labels is not None:
            aligned_labels = labels.loc[feat_df.index]
            self.rf_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            self.rf_.fit(feat_df.values, aligned_labels.values)
            self.feature_names_        = feat_df.columns.tolist()
            self.fitted_with_labels_   = True
        else:
            self.fitted_with_labels_   = False

        return self

    # ------------------------------------------------------------------
    # compute_features
    # ------------------------------------------------------------------
    def compute_features(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the 12-column feature matrix for all erroneous cells.

        This method is safe to call multiple times after a single
        :py:meth:`fit` (e.g. on a held-out test set).

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape, 0 = clean, 1 = error)

        Returns
        -------
        pd.DataFrame
            Shape: ``(n_errors, 12)``.
            Index: MultiIndex ``(row_idx, col_name)``.
            Columns: exactly :py:data:`FEATURE_COLUMNS` in fixed order.

        Raises
        ------
        AssertionError
            If the assembled frame does not have exactly 12 columns.
        RuntimeError
            If :py:meth:`fit` has not been called yet.
        """
        # Guard: all heuristics must be fitted
        if self.h1_ is None:
            raise RuntimeError(
                "AttributionPipeline.fit() must be called before compute_features()."
            )

        # ── Call compute() on each heuristic ───────────────────────────────
        h1_out = self.h1_.compute(dirty_df, mask_df)
        h2_out = self.h2_.compute(dirty_df, mask_df)
        h3_out = self.h3_.compute(dirty_df, mask_df)
        h4_out = self.h4_.compute(dirty_df, mask_df)
        h5_out = self.h5_.compute(dirty_df, mask_df)
        h6_out = self.h6_.compute(dirty_df, mask_df)
        h7_out = self.h7_.compute(dirty_df, mask_df)
        h8_out = self.h8_.compute(dirty_df, mask_df)

        # ── Rename H2's internal column name to match the contract ─────────
        # H2 internally uses "h2_min_edit_dist"; the 12-column contract
        # requires "h2_min_edit_distance".
        h2_out = h2_out.rename(columns={"h2_min_edit_dist": "h2_min_edit_distance"})

        # ── Set (row_idx, col_name) as the index for each output ───────────
        def _set_idx(df: pd.DataFrame) -> pd.DataFrame:
            return df.set_index(["row_idx", "col_name"])

        # ── Concatenate feature columns in the required fixed order ────────
        feat_df = pd.concat(
            [
                _set_idx(h1_out)[["h1_plausible"]],
                _set_idx(h2_out)[["h2_min_edit_distance", "h2_is_obfuscation"]],
                _set_idx(h3_out)[["h3_distribution_score"]],
                _set_idx(h4_out)[["h4_coherence_score"]],
                _set_idx(h5_out)[["h5_error_count", "h5_codependent_flag"]],
                _set_idx(h6_out)[["h6_column_importance"]],
                # h7_mutability is computed by H7 but excluded from the
                # deployed 12-feature fingerprint (see FEATURE_COLUMNS note).
                _set_idx(h7_out)[["h7_gain_direction", "h7_comprehensibility"]],
                _set_idx(h8_out)[["h8_is_sensitive", "h8_is_majority_value"]],
            ],
            axis=1,
        )

        # ── Enforce the fixed column order ─────────────────────────────────
        feat_df = feat_df[FEATURE_COLUMNS]

        # ── Hard contract assertion: exactly 12 columns ────────────────────
        assert feat_df.shape[1] == _N_FEATURES, (
            f"AttributionPipeline.compute_features(): expected {_N_FEATURES} "
            f"feature columns, got {feat_df.shape[1]}. "
            f"Columns present: {list(feat_df.columns)}"
        )

        return feat_df

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Return predicted intent labels for all erroneous cells.

        Requires :py:meth:`fit` to have been called with labels.

        Parameters
        ----------
        dirty_df, mask_df : pd.DataFrame

        Returns
        -------
        pd.Series
            Values: 1 = intentional, 0 = unintentional.
            Index: MultiIndex ``(row_idx, col_name)``.
            Name: ``"predicted_label"``.

        Raises
        ------
        RuntimeError
            If the pipeline was fitted without labels.
        """
        if not self.fitted_with_labels_:
            raise RuntimeError(
                "Pipeline was not fitted with labels.  "
                "Call fit(dirty_df, mask_df, labels=...) to enable prediction."
            )
        feat_df = self.compute_features(dirty_df, mask_df)
        preds   = self.rf_.predict(feat_df.values)
        return pd.Series(preds, index=feat_df.index, name="predicted_label")

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------
    def predict_proba(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return class probabilities for all erroneous cells.

        Parameters
        ----------
        dirty_df, mask_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns: ``["prob_unintentional", "prob_intentional"]``.
            Index: MultiIndex ``(row_idx, col_name)``.

        Raises
        ------
        RuntimeError
            If the pipeline was fitted without labels.
        """
        if not self.fitted_with_labels_:
            raise RuntimeError(
                "Pipeline was not fitted with labels.  "
                "Call fit(dirty_df, mask_df, labels=...) to enable predict_proba."
            )
        feat_df = self.compute_features(dirty_df, mask_df)
        proba   = self.rf_.predict_proba(feat_df.values)
        return pd.DataFrame(
            proba,
            index=feat_df.index,
            columns=["prob_unintentional", "prob_intentional"],
        )

    # ------------------------------------------------------------------
    # explain
    # ------------------------------------------------------------------
    def explain(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return per-heuristic feature values and (if RF trained) importance
        scores, predicted labels, and intent probabilities.

        Parameters
        ----------
        dirty_df, mask_df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Always contains the 12 raw feature columns.
            If RF was trained, additionally contains:

            * ``rf_importance_<feature>`` (constant per feature, from RF)
            * ``predicted_label``
            * ``prob_intentional``

            Index: MultiIndex ``(row_idx, col_name)``.
        """
        feat_df = self.compute_features(dirty_df, mask_df)
        result  = feat_df.copy()

        if self.fitted_with_labels_:
            importances = self.rf_.feature_importances_
            for fname, imp in zip(self.feature_names_, importances):
                result[f"rf_importance_{fname}"] = imp   # scalar broadcast

            preds  = self.rf_.predict(feat_df.values)
            probas = self.rf_.predict_proba(feat_df.values)[:, 1]
            result["predicted_label"]  = preds
            result["prob_intentional"] = probas

        return result

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        fitted = "fitted" if self.h1_ is not None else "not fitted"
        supervised = "supervised" if self.fitted_with_labels_ else "unsupervised"
        return (
            f"AttributionPipeline(target_col={self.target_col!r}, "
            f"state={fitted}, mode={supervised})"
        )


# ===========================================================================
# Self-test  (run with:  python pipeline.py)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AttributionPipeline — Self-test")
    print("=" * 60)

    import pandas as pd
    import numpy as np

    # ------------------------------------------------------------------
    # Build synthetic dataset (mirrors the ticket's self-test)
    # ------------------------------------------------------------------
    np.random.seed(42)
    n = 300
    dirty = pd.DataFrame({
        "age":           np.random.randint(20, 65, n),
        "education":     np.random.choice(["HS-grad", "Bachelors", "Masters"], n),
        "education-num": np.random.randint(5, 16, n),
        "race":          np.random.choice(
                             ["White", "Black", "Asian"], n, p=[0.86, 0.09, 0.05]
                         ),
        "income":        np.random.randint(0, 2, n),
    })
    mask = pd.DataFrame(
        np.zeros_like(dirty.values, dtype=int),
        columns=dirty.columns,
    )

    # Inject ~30 errors at random positions
    error_indices = np.random.choice(n, 30, replace=False)
    for idx in error_indices:
        col = np.random.choice(dirty.columns)
        mask.loc[idx, col] = 1

    print(f"\nDataset shape : {dirty.shape}")
    print(f"Errors injected: {int(mask.values.sum())}")

    # ------------------------------------------------------------------
    # Test 1 — Unsupervised (no labels)
    # ------------------------------------------------------------------
    print("\n--- Test 1: unsupervised mode (no labels) ---")
    pipe = AttributionPipeline(target_col="income")
    pipe.fit(dirty, mask)   # labels=None

    feat_df = pipe.compute_features(dirty, mask)
    print(f"Feature matrix shape: {feat_df.shape}")
    print(f"Feature columns ({len(feat_df.columns)}):")
    for i, col in enumerate(feat_df.columns, 1):
        print(f"  {i:2d}. {col}")

    # ── Hard assertion: exactly 12 features ──────────────────────────
    assert feat_df.shape[1] == 12, (
        f"FAILED: expected 12 features, got {feat_df.shape[1]}"
    )

    # ── Hard assertion: fixed column order ───────────────────────────
    assert list(feat_df.columns) == FEATURE_COLUMNS, (
        f"FAILED: column order mismatch.\n"
        f"  Expected: {FEATURE_COLUMNS}\n"
        f"  Got     : {list(feat_df.columns)}"
    )

    print("\nFirst 5 rows of feature matrix:")
    print(feat_df.head(5).to_string())

    # explain() should not include RF columns in unsupervised mode
    explain_df = pipe.explain(dirty, mask)
    assert list(explain_df.columns) == FEATURE_COLUMNS, (
        "explain() in unsupervised mode must return only the 12 feature columns."
    )
    print("\nexplain() (unsupervised) — first 5 rows:")
    print(explain_df.head(5).to_string())

    # predict() must raise in unsupervised mode
    try:
        pipe.predict(dirty, mask)
        raise AssertionError("FAILED: predict() should raise RuntimeError when no labels.")
    except RuntimeError as exc:
        print(f"\npredict() correctly raised RuntimeError: {exc}")

    print("\n✓ Test 1 passed — unsupervised mode, 12-column assertion OK.")

    # ------------------------------------------------------------------
    # Test 2 — Supervised (with labels)
    # ------------------------------------------------------------------
    print("\n--- Test 2: supervised mode (with labels) ---")

    # Build synthetic labels for all erroneous cells
    error_positions = [
        (row, col)
        for col in mask.columns
        for row in mask.index[mask[col] == 1]
    ]
    rng = np.random.default_rng(0)
    label_values = rng.integers(0, 2, size=len(error_positions))
    labels = pd.Series(
        label_values,
        index=pd.MultiIndex.from_tuples(error_positions, names=["row_idx", "col_name"]),
        name="intent",
    )

    pipe_sup = AttributionPipeline(target_col="income", n_estimators=50, random_state=0)
    pipe_sup.fit(dirty, mask, labels=labels)

    assert pipe_sup.fitted_with_labels_, "fitted_with_labels_ must be True."

    preds = pipe_sup.predict(dirty, mask)
    probas = pipe_sup.predict_proba(dirty, mask)
    assert preds.shape[0] == feat_df.shape[0], "predict() row count mismatch."
    assert list(probas.columns) == ["prob_unintentional", "prob_intentional"], (
        f"predict_proba() column names wrong: {list(probas.columns)}"
    )

    explain_sup = pipe_sup.explain(dirty, mask)
    rf_imp_cols = [c for c in explain_sup.columns if c.startswith("rf_importance_")]
    assert len(rf_imp_cols) == 12, (
        f"explain() must have 12 rf_importance_ columns, found {len(rf_imp_cols)}."
    )
    assert "predicted_label" in explain_sup.columns
    assert "prob_intentional" in explain_sup.columns

    print(f"predict() shape   : {preds.shape}")
    print(f"predict_proba() shape: {probas.shape}")
    print(f"explain() columns ({len(explain_sup.columns)}): "
          f"{list(explain_sup.columns)[:5]} … (truncated)")
    print("\n✓ Test 2 passed — supervised mode, RF trained and evaluated.")

    # ------------------------------------------------------------------
    # Test 3 — compute_features() is idempotent after a single fit()
    # ------------------------------------------------------------------
    print("\n--- Test 3: idempotency of compute_features() ---")
    feat_a = pipe.compute_features(dirty, mask)
    feat_b = pipe.compute_features(dirty, mask)
    pd.testing.assert_frame_equal(feat_a, feat_b)
    print("✓ Test 3 passed — compute_features() is idempotent.")

    print("\n" + "=" * 60)
    print("All self-tests passed.  AttributionPipeline is correct.")
    print("=" * 60)
