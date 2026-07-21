"""
H1ValuePlausibility — Heuristic 1
==================================
Checks whether a dirty (erroneous) cell value is a plausible member of its
column's known domain, as learned from the clean cells in the same dataset.

Column type detection
---------------------
Auto-detection: a column is treated as *numerical* when > 90 % of its non-null
values can be cast to a float; otherwise it is treated as *categorical*.
The caller can override per-column types via the ``col_types`` argument to
:py:meth:`fit`.

Output features (one row per error position)
--------------------------------------------
==============  ========  ===================================================
Column          Type      Description
==============  ========  ===================================================
row_idx         int       Row index of the erroneous cell
col_name        str       Column name of the erroneous cell
h1_plausible    int       **Single binary summary**: 1 = value belongs to the
                          column's known domain; 0 = it does not
h1_in_vocab     int|NaN   Categorical only: 1 if the dirty value is present in
                          the clean vocabulary. NaN for numerical columns.
h1_in_range     int|NaN   Numerical only: 1 if the dirty value lies within
                          [p5, p95] of the clean column values. NaN for
                          categorical columns.
==============  ========  ===================================================

Known limitation
----------------
~29 % of *intentional* errors in the Adult Income dataset use obfuscation
tokens (``nan``, ``Unknown``, ``—``) which are also out-of-vocabulary.  H1
will score those as **implausible** (h1_plausible = 0), the same as
unintentional typos.  H2 is responsible for rescuing those cases by
recognising obfuscation patterns.

Usage example
-------------
>>> h1 = H1ValuePlausibility()
>>> h1.fit(dirty_df, mask_df)
>>> features = h1.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseHeuristic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUMERIC_THRESHOLD = 0.9          # fraction of values that must parse as float
_NULL_STRINGS = {"nan", "none", ""}   # string representations treated as null
_P_LOW = 5                         # lower percentile for numerical range
_P_HIGH = 95                       # upper percentile for numerical range


class H1ValuePlausibility(BaseHeuristic):
    """
    Heuristic 1 — Value Plausibility.

    Determines whether the dirty value at each error position is a plausible
    member of its column's domain, using statistics learned exclusively from
    **clean cells** (``mask_df == 0``).

    Parameters
    ----------
    (none at construction time; configuration is passed to :py:meth:`fit`)

    Attributes
    ----------
    col_stats_ : dict
        Populated after :py:meth:`fit`.  Maps column name → dict with keys:

        - ``'type'``: ``'cat'`` or ``'num'``
        - ``'vocab'``: (categorical) ``set`` of clean string values
        - ``'p5'``, ``'p95'``: (numerical) ``float`` percentile bounds
    is_fitted : bool
        Set to ``True`` after :py:meth:`fit` is called.
    """

    def __init__(self) -> None:
        super().__init__()
        self.col_stats_: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        col_types: Optional[Dict[str, str]] = None,
    ) -> "H1ValuePlausibility":
        """
        Learn per-column domain statistics from the *clean* cells only.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Binary mask aligned with ``dirty_df``.  0 = clean, 1 = erroneous.
        col_types : dict, optional
            Per-column type override, e.g. ``{'age': 'num', 'education': 'cat'}``.
            Any column not listed falls back to auto-detection.

        Returns
        -------
        self
        """
        col_types = col_types or {}
        self.col_stats_ = {}

        for col in dirty_df.columns:
            # ---- determine column type --------------------------------
            if col in col_types:
                ctype = col_types[col]
            else:
                ctype = self._detect_type(dirty_df[col])

            # ---- extract clean cells ----------------------------------
            clean_mask = mask_df[col] == 0
            clean_vals = dirty_df.loc[clean_mask, col]

            # ---- build statistics ------------------------------------
            if ctype == "cat":
                vocab = self._build_vocab(clean_vals)
                self.col_stats_[col] = {"type": "cat", "vocab": vocab}
            else:  # num
                p5, p95 = self._build_range(clean_vals)
                self.col_stats_[col] = {"type": "num", "p5": p5, "p95": p95}

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute plausibility features for every erroneous cell.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape, values in {0, 1})

        Returns
        -------
        pd.DataFrame
            One row per error position with columns:
            ``[row_idx, col_name, h1_plausible, h1_in_vocab, h1_in_range]``
        """
        self._check_fitted()

        rows = []
        for row_idx, col_name in self._get_error_positions(mask_df):
            raw_val = dirty_df.loc[row_idx, col_name]
            stats = self.col_stats_[col_name]

            if stats["type"] == "cat":
                row = self._score_categorical(row_idx, col_name, raw_val, stats)
            else:
                row = self._score_numerical(row_idx, col_name, raw_val, stats)

            rows.append(row)

        if not rows:
            return pd.DataFrame(
                columns=["row_idx", "col_name", "h1_plausible", "h1_in_vocab", "h1_in_range"]
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
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
    def _build_vocab(clean_vals: pd.Series) -> set:
        """Return the set of unique, non-null string values from clean cells."""
        vocab = set()
        for v in clean_vals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            s = str(v)
            if s.lower() not in _NULL_STRINGS:
                vocab.add(s)
        return vocab

    @staticmethod
    def _build_range(clean_vals: pd.Series) -> tuple[float, float]:
        """Return (p5, p95) from numeric clean cells, or (NaN, NaN) if empty."""
        numeric = pd.to_numeric(clean_vals, errors="coerce").dropna()
        if numeric.empty:
            return float("nan"), float("nan")
        p5 = float(np.percentile(numeric, _P_LOW))
        p95 = float(np.percentile(numeric, _P_HIGH))
        return p5, p95

    @staticmethod
    def _is_null(val) -> bool:
        """Return True if the value is effectively null/missing."""
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        if str(val).lower() in _NULL_STRINGS:
            return True
        return False

    def _score_categorical(
        self, row_idx, col_name: str, raw_val, stats: dict
    ) -> dict:
        """Score a single erroneous cell in a categorical column."""
        if self._is_null(raw_val):
            return {
                "row_idx": row_idx,
                "col_name": col_name,
                "h1_plausible": 0,
                "h1_in_vocab": 0,
                "h1_in_range": np.nan,
            }

        dirty_str = str(raw_val)
        in_vocab = int(dirty_str in stats["vocab"] and dirty_str.lower() not in _NULL_STRINGS)
        return {
            "row_idx": row_idx,
            "col_name": col_name,
            "h1_plausible": in_vocab,
            "h1_in_vocab": in_vocab,
            "h1_in_range": np.nan,
        }

    def _score_numerical(
        self, row_idx, col_name: str, raw_val, stats: dict
    ) -> dict:
        """Score a single erroneous cell in a numerical column."""
        # Null check first
        if self._is_null(raw_val):
            return {
                "row_idx": row_idx,
                "col_name": col_name,
                "h1_plausible": 0,
                "h1_in_vocab": np.nan,
                "h1_in_range": 0,
            }

        # Try numeric cast; fall back to categorical scoring if it fails
        try:
            float_val = float(raw_val)
        except (ValueError, TypeError):
            return {
                "row_idx": row_idx,
                "col_name": col_name,
                "h1_plausible": 0,
                "h1_in_vocab": 0,
                "h1_in_range": np.nan,
            }

        p5, p95 = stats["p5"], stats["p95"]
        in_range = int(p5 <= float_val <= p95)
        return {
            "row_idx": row_idx,
            "col_name": col_name,
            "h1_plausible": in_range,
            "h1_in_vocab": np.nan,
            "h1_in_range": in_range,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from h1_value_plausibility import H1ValuePlausibility  # noqa: F401 (works when run directly)

    dirty = pd.DataFrame(
        {
            "age":       [25, 999, 30, 42, 7,   35],
            "education": ["HS-grad", "Bachleors", "Bachelors", "Masters", "Doctrate", "HS-grad"],
        }
    )
    mask = pd.DataFrame(
        {
            "age":       [0, 1, 0, 0, 1, 0],
            "education": [0, 1, 0, 0, 1, 0],
        }
    )

    h1 = H1ValuePlausibility()
    h1.fit(dirty, mask)
    result = h1.compute(dirty, mask)

    print("=" * 65)
    print("H1 Value Plausibility — self-test result")
    print("=" * 65)
    print(result.to_string(index=False))
    print()

    # ---- assertions -------------------------------------------------------
    def get(df, row, col, feature):
        row_data = df[(df["row_idx"] == row) & (df["col_name"] == col)]
        assert not row_data.empty, f"Missing row for ({row}, {col})"
        return row_data.iloc[0][feature]

    # age row 1: 999 should be out-of-range
    assert get(result, 1, "age", "h1_plausible") == 0, "age row 1 should be implausible"
    assert get(result, 1, "age", "h1_in_range")  == 0, "age row 1 should be out of range"
    assert math.isnan(get(result, 1, "age", "h1_in_vocab")), "age row 1 h1_in_vocab should be NaN"

    # age row 4: 7 should be out-of-range
    assert get(result, 4, "age", "h1_plausible") == 0, "age row 4 should be implausible"
    assert get(result, 4, "age", "h1_in_range")  == 0, "age row 4 should be out of range"

    # education row 1: typo "Bachleors" not in vocab
    assert get(result, 1, "education", "h1_plausible") == 0
    assert get(result, 1, "education", "h1_in_vocab")  == 0
    assert math.isnan(get(result, 1, "education", "h1_in_range"))

    # education row 4: typo "Doctrate" not in vocab
    assert get(result, 4, "education", "h1_plausible") == 0
    assert get(result, 4, "education", "h1_in_vocab")  == 0

    print("All assertions passed ✓")
