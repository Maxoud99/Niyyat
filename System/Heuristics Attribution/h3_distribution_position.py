"""
H3DistributionPosition — Heuristic 3
======================================
Measures how central or extreme a dirty (erroneous) cell value is within
its column's statistical distribution, as learned from clean cells only.

Motivation
----------
Intentional errors (adversarial data manipulation) tend to use *common*,
unremarkable values so they blend in with the rest of the data.  Unintentional
errors (noise, random substitution, encoding corruption) tend to land in the
tails of the distribution — extreme or rarely-seen values.

H3 captures this signal as a single continuous score in [0, 1]:

- **Score ≈ 1.0** → value is central / common → consistent with *intentional*
  blending.
- **Score ≈ 0.0** → value is extreme / rare or out-of-vocabulary →
  consistent with *unintentional* noise.

Column type detection
---------------------
Auto-detection: a column is treated as *numerical* when > 90 % of its
non-null values can be cast to a float; otherwise it is treated as
*categorical*.  The caller can override per-column types via the
``col_types`` argument to :py:meth:`fit`.

Output feature (one row per error position)
-------------------------------------------
======================  =======  ================================================
Column                  Type     Description
======================  =======  ================================================
row_idx                 int      Row index of the erroneous cell
col_name                str      Column name of the erroneous cell
h3_distribution_score   float    Normalised position in the column's distribution.
                                  High (~1) = central/common → intentional signal.
                                  Low (~0) = extreme/rare   → unintentional signal.
======================  =======  ================================================

Scoring formulas
----------------
**Numerical**::

    zscore = (dirty_value - clean_mean) / clean_std
    h3_distribution_score = 1 - min(1, abs(zscore) / 3.0)

  Edge cases:
  - ``clean_std == 0`` (constant column)  → score = 1.0
  - dirty value cannot be cast to float   → score = 0.0

**Categorical**::

    h3_distribution_score = 1 - (frequency_rank - 1) / max(1, vocab_size - 1)

  where *frequency_rank* is the rank of the dirty value in the clean
  vocabulary sorted by descending frequency (rank 1 = most common).
  If the dirty value is **not in** the clean vocabulary it is treated as
  rank ``vocab_size + 1`` (penalised as rarest).
  Result is in [0, 1]: most-common = 1.0, rarest / OOV = 0.0.

Known limitation
----------------
For columns where both intentional and unintentional errors concentrate in a
similar range (e.g. ``capital-gain`` in the Adult Income dataset, where 91-93 %
of both error kinds are in-range), H3's discrimination power is low.  This is
expected and documented — other heuristics compensate.

Usage example
-------------
>>> h3 = H3DistributionPosition()
>>> h3.fit(dirty_df, mask_df)
>>> features = h3.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Optional

import pandas as pd

from .base import BaseHeuristic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUMERIC_THRESHOLD = 0.9    # fraction of values that must parse as float
_ZSCORE_CAP = 3.0           # z-score at which the score reaches 0.0


class H3DistributionPosition(BaseHeuristic):
    """
    Heuristic 3 — Distribution Position.

    Scores each erroneous cell by how central (common) or extreme (rare) its
    value is relative to the distribution of clean values in the same column.

    Parameters
    ----------
    (none at construction time; configuration is passed to :py:meth:`fit`)

    Attributes
    ----------
    col_stats_ : dict
        Populated after :py:meth:`fit`.  Maps column name → dict with keys:

        - ``'type'``: ``'cat'`` or ``'num'``
        - (numerical) ``'mean'``: float, mean of clean numeric values
        - (numerical) ``'std'``:  float, std of clean numeric values
        - (categorical) ``'freq_rank'``: dict mapping value → int rank (1 = most common)
        - (categorical) ``'vocab_size'``: int, number of unique clean values
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
    ) -> "H3DistributionPosition":
        """
        Learn per-column distribution statistics from the *clean* cells only.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset (contains errors, but no ground-truth labels).
        mask_df : pd.DataFrame
            Binary mask aligned with ``dirty_df``.  0 = clean, 1 = erroneous.
        col_types : dict, optional
            Per-column type override, e.g. ``{'age': 'num', 'zip_code': 'cat'}``.
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
            if ctype == "num":
                mean, std = self._build_numerical_stats(clean_vals)
                self.col_stats_[col] = {"type": "num", "mean": mean, "std": std}
            else:  # cat
                freq_rank, vocab_size = self._build_categorical_stats(clean_vals)
                self.col_stats_[col] = {
                    "type": "cat",
                    "freq_rank": freq_rank,
                    "vocab_size": vocab_size,
                }

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the distribution-position score for every erroneous cell.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame
            One row per error position with columns:
            ``[row_idx, col_name, h3_distribution_score]``

            ``h3_distribution_score`` ∈ [0, 1]:
            - near 1 → central / common value → intentional-error signal
            - near 0 → extreme / rare value   → unintentional-error signal
        """
        self._check_fitted()

        rows = []
        for row_idx, col_name in self._get_error_positions(mask_df):
            raw_val = dirty_df.loc[row_idx, col_name]
            stats = self.col_stats_[col_name]

            if stats["type"] == "num":
                score = self._score_numerical(raw_val, stats)
            else:
                score = self._score_categorical(raw_val, stats)

            rows.append(
                {"row_idx": row_idx, "col_name": col_name, "h3_distribution_score": score}
            )

        if not rows:
            return pd.DataFrame(columns=["row_idx", "col_name", "h3_distribution_score"])

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers — type detection & statistics
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
    def _build_numerical_stats(clean_vals: pd.Series) -> tuple[float, float]:
        """
        Return ``(mean, std)`` from the clean numeric cells.

        Returns ``(NaN, NaN)`` if there are no valid numeric clean values.
        """
        numeric = pd.to_numeric(clean_vals, errors="coerce").dropna()
        if numeric.empty:
            return float("nan"), float("nan")
        return float(numeric.mean()), float(numeric.std(ddof=0))

    @staticmethod
    def _build_categorical_stats(
        clean_vals: pd.Series,
    ) -> tuple[dict, int]:
        """
        Build a frequency-rank mapping for a categorical column.

        Ranks are 1-based: rank 1 = most frequent value in clean cells.

        Returns
        -------
        freq_rank : dict  {str → int}
        vocab_size : int   number of unique values in clean cells
        """
        non_null = clean_vals.dropna().astype(str)
        if non_null.empty:
            return {}, 0

        counts = Counter(non_null)
        # Sort by descending count, then alphabetically for stability
        sorted_vals = sorted(counts.keys(), key=lambda v: (-counts[v], v))
        freq_rank = {val: rank + 1 for rank, val in enumerate(sorted_vals)}
        vocab_size = len(freq_rank)
        return freq_rank, vocab_size

    # ------------------------------------------------------------------
    # Private helpers — scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_numerical(raw_val, stats: dict) -> float:
        """
        Score a single erroneous cell in a numerical column.

        Returns a float in [0, 1].  Values near the mean score ~1.0;
        values ≥ 3 standard deviations away score ~0.0.
        """
        mean, std = stats["mean"], stats["std"]

        # Empty / all-null clean column → cannot score
        if math.isnan(mean):
            return 0.0

        # Constant column (std == 0) → every value is "central"
        if std == 0.0:
            return 1.0

        # Try to cast dirty value to float
        try:
            float_val = float(raw_val)
        except (ValueError, TypeError):
            return 0.0

        zscore = (float_val - mean) / std
        return max(0.0, 1.0 - min(1.0, abs(zscore) / _ZSCORE_CAP))

    @staticmethod
    def _score_categorical(raw_val, stats: dict) -> float:
        """
        Score a single erroneous cell in a categorical column.

        Returns a float in [0, 1].  Most-common value = 1.0; rarest or
        out-of-vocabulary value = 0.0.
        """
        freq_rank: dict = stats["freq_rank"]
        vocab_size: int = stats["vocab_size"]

        # Empty clean column → cannot score
        if vocab_size == 0:
            return 0.0

        dirty_str = str(raw_val)

        if dirty_str in freq_rank:
            rank = freq_rank[dirty_str]
        else:
            # Out-of-vocabulary: penalise as rank vocab_size + 1
            rank = vocab_size + 1

        # Normalise to [0, 1]
        # rank=1 (most common) → 1.0; rank=vocab_size+1 (OOV) → 0.0
        return max(0.0, 1.0 - (rank - 1) / max(1, vocab_size))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    from h3_distribution_position import H3DistributionPosition  # noqa: F401

    dirty = pd.DataFrame(
        {
            "age": [25, 30, 35, 200, 28],
            # 200 is the injected error — extreme outlier
            "education": ["HS-grad", "HS-grad", "Bachelors", "XYZ-fake", "HS-grad"],
            # "XYZ-fake" is the injected error — out-of-vocabulary
        }
    )
    mask = pd.DataFrame(
        {
            "age": [0, 0, 0, 1, 0],
            "education": [0, 0, 0, 1, 0],
        }
    )

    h3 = H3DistributionPosition()
    h3.fit(dirty, mask)
    result = h3.compute(dirty, mask)

    print("=" * 65)
    print("H3 Distribution Position — self-test result")
    print("=" * 65)
    print(result.to_string(index=False))
    print()

    # ---- assertions -------------------------------------------------------
    def get_score(df, row, col):
        row_data = df[(df["row_idx"] == row) & (df["col_name"] == col)]
        assert not row_data.empty, f"Missing row for ({row}, {col})"
        return float(row_data.iloc[0]["h3_distribution_score"])

    # age row=3: 200 is ~10+ std above the mean of [25, 30, 35, 28] → score ≈ 0.0
    age_score = get_score(result, 3, "age")
    assert age_score == 0.0, f"Expected 0.0 for extreme age=200, got {age_score}"

    # education row=3: "XYZ-fake" is OOV → worst rank → score = 0.0
    edu_score = get_score(result, 3, "education")
    assert edu_score == 0.0, f"Expected 0.0 for OOV education, got {edu_score}"

    # Verify exactly 1 output column beyond row_idx and col_name
    feature_cols = [c for c in result.columns if c not in ("row_idx", "col_name")]
    assert feature_cols == ["h3_distribution_score"], (
        f"Expected exactly ['h3_distribution_score'], got {feature_cols}"
    )

    print(f"age=200  → h3_distribution_score = {age_score:.4f}  (expected ≈ 0.0) ✓")
    print(f"education='XYZ-fake' → h3_distribution_score = {edu_score:.4f}  (expected = 0.0) ✓")
    print()
    print("All assertions passed ✓")
