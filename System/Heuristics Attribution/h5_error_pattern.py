"""
H5ErrorPattern — Heuristic 5
==============================
Detects coordinated multi-cell edits: a hallmark of *intentional* data
manipulation where a person edits logically linked columns together to
preserve internal consistency.

Motivation
----------
When someone deliberately forges a record they typically touch several
related fields at the same time — e.g. they change ``education`` *and*
``education-num`` together so the record still "makes sense".  Random noise
or OCR errors, by contrast, strike columns independently.

H5 captures this pattern with two complementary signals per erroneous cell:

1. **Row error density** (``h5_error_count``): how many erroneous cells are
   in the same row?  A high count implies a coordinated edit rather than an
   isolated accident.

2. **Co-dependent flag** (``h5_codependent_flag``): does this erroneous cell
   have a *logically linked* partner column that is *also* erroneous in the
   same row?  If ``education`` and ``education-num`` are both wrong in row 7,
   both cells raise this flag.

Co-dependent pairs are discovered automatically (name similarity + mutual
information) and can be supplemented by user-supplied pairs.

Output features (one row per error position)
--------------------------------------------
======================  ==========  ==========================================
Column                  Type        Description
======================  ==========  ==========================================
row_idx                 int         Row index of the erroneous cell
col_name                str         Column name of the erroneous cell
h5_error_count          int         Number of erroneous cells in the same row
                                    (including the current cell, always ≥ 1).
h5_codependent_flag     int (0/1)   1 if at least one co-dependent partner
                                    column is *also* erroneous in the same row.
======================  ==========  ==========================================

Usage example
-------------
>>> h5 = H5ErrorPattern()
>>> h5.fit(dirty_df, mask_df)
>>> features = h5.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import difflib
import re
import warnings
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

from .base import BaseHeuristic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NAME_SIM_THRESHOLD = 0.8          # SequenceMatcher ratio threshold
_MI_THRESHOLD = 0.5                # mutual information threshold for pairing
_MAX_MI_COLS = 20                  # cap on column count for MI computation
_MIN_CLEAN_ROWS_FOR_MI = 100       # minimum clean rows required to run MI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_col_name(name: str) -> str:
    """
    Strip digits, hyphens, underscores, and common suffixes so that
    ``education-num`` and ``education`` both reduce to ``education``.
    """
    # lower-case, then remove the suffixes 'num', 'no', 'id' and separators
    name = name.lower()
    name = re.sub(r'[-_]', ' ', name)
    # strip the tokens 'num', 'no', 'id' appearing as whole words
    name = re.sub(r'\b(num|no|id)\b', '', name)
    # remove digits
    name = re.sub(r'\d+', '', name)
    # collapse whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _name_similarity_pairs(columns: List[str]) -> Set[frozenset]:
    """
    Return pairs of column names whose cleaned representations have a
    SequenceMatcher ratio ≥ _NAME_SIM_THRESHOLD.
    """
    pairs: Set[frozenset] = set()
    cleaned = [(_clean_col_name(c), c) for c in columns]
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            clean_i, orig_i = cleaned[i]
            clean_j, orig_j = cleaned[j]
            # skip if either cleaned name is empty (would trivially match)
            if not clean_i or not clean_j:
                continue
            ratio = difflib.SequenceMatcher(None, clean_i, clean_j).ratio()
            if ratio >= _NAME_SIM_THRESHOLD:
                pairs.add(frozenset({orig_i, orig_j}))
    return pairs


def _mutual_info_pairs(
    dirty_df: pd.DataFrame,
    mask_df: pd.DataFrame,
) -> Set[frozenset]:
    """
    Compute pairwise mutual information between all columns (capped at
    _MAX_MI_COLS) using only clean cells.  Pairs where MI > _MI_THRESHOLD
    are returned.
    """
    pairs: Set[frozenset] = set()
    cols = list(dirty_df.columns)

    # ── cap at _MAX_MI_COLS columns ─────────────────────────────────────────
    if len(cols) > _MAX_MI_COLS:
        cols = cols[:_MAX_MI_COLS]

    if len(cols) < 2:
        return pairs

    # ── build a fully-clean sub-frame ───────────────────────────────────────
    # A row is "clean" only when ALL cells in the selected columns are clean
    sub_mask = mask_df[cols]
    clean_rows = (sub_mask == 0).all(axis=1)
    clean_df = dirty_df.loc[clean_rows, cols].copy()

    if len(clean_df) < _MIN_CLEAN_ROWS_FOR_MI:
        return pairs   # not enough data for reliable MI estimates

    # ── encode all columns to numeric (ordinal) ──────────────────────────────
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    try:
        X_enc = enc.fit_transform(clean_df.astype(str))
    except Exception:
        return pairs

    X_enc_df = pd.DataFrame(X_enc, columns=cols)

    # ── for each column as target, compute MI against all others ─────────────
    # We treat every column as discrete (categorical encoding).
    for i, target_col in enumerate(cols):
        y = X_enc_df[target_col].values
        feature_cols = [c for c in cols if c != target_col]
        if not feature_cols:
            continue
        X_feat = X_enc_df[feature_cols].values
        try:
            mi_scores = mutual_info_classif(
                X_feat, y, discrete_features=True, random_state=42
            )
        except Exception:
            continue
        for j, fc in enumerate(feature_cols):
            if mi_scores[j] > _MI_THRESHOLD:
                pairs.add(frozenset({target_col, fc}))

    return pairs


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class H5ErrorPattern(BaseHeuristic):
    """
    Heuristic 5 — Error Pattern (coordinated multi-cell edits).

    Measures two complementary signals for each erroneous cell:

    * ``h5_error_count``      — number of erroneous cells in the same row.
    * ``h5_codependent_flag`` — whether a logically linked partner column is
      *also* erroneous in the same row.

    Parameters
    ----------
    (none at construction time; configuration is passed to :py:meth:`fit`)

    Attributes
    ----------
    codependent_pairs_ : set of frozenset
        Set of ``frozenset({col_a, col_b})`` pairs discovered after
        :py:meth:`fit`.  Populated from (a) user-supplied pairs, (b) name
        similarity, and (c) mutual information.
    is_fitted : bool
        Set to ``True`` after :py:meth:`fit` completes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.codependent_pairs_: Set[frozenset] = set()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        codependent_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> "H5ErrorPattern":
        """
        Discover co-dependent column pairs from the dataset.

        Three sources are merged (union):

        1. **User-supplied pairs** (*Method A*, highest priority) — trusted,
           stored as-is after validating that both columns exist.
        2. **Name-similarity pairs** (*Method B / Signal 1*) — pairs whose
           cleaned column names share a SequenceMatcher ratio ≥ 0.80.
        3. **High-MI pairs** (*Method B / Signal 2*) — pairs whose mutual
           information exceeds 0.5 (computed on clean rows only; skipped when
           fewer than 100 clean rows are available or the table has > 20 cols).

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Binary mask (0 = clean, 1 = erroneous), same shape as dirty_df.
        codependent_pairs : list of (str, str), optional
            User-supplied column pairs known to be logically linked,
            e.g. ``[("education", "education-num")]``.  Column names that do
            not exist in ``dirty_df`` are silently skipped with a warning.

        Returns
        -------
        self
        """
        cols = list(dirty_df.columns)
        all_pairs: Set[frozenset] = set()

        # ── Method A: user-supplied pairs ────────────────────────────────────
        if codependent_pairs:
            for col_a, col_b in codependent_pairs:
                missing = [c for c in (col_a, col_b) if c not in dirty_df.columns]
                if missing:
                    warnings.warn(
                        f"H5ErrorPattern.fit(): column(s) {missing} not found in "
                        f"dataframe — skipping user-supplied pair ({col_a!r}, {col_b!r}).",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                all_pairs.add(frozenset({col_a, col_b}))

        # ── Method B: auto-detection ─────────────────────────────────────────
        if len(cols) >= 2:
            # Signal 1 — name similarity
            name_pairs = _name_similarity_pairs(cols)
            all_pairs.update(name_pairs)

            # Signal 2 — mutual information (may return empty set gracefully)
            mi_pairs = _mutual_info_pairs(dirty_df, mask_df)
            all_pairs.update(mi_pairs)

        self.codependent_pairs_ = all_pairs
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute H5 features for every erroneous cell (``mask_df == 1``).

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame with columns:
            ``row_idx``, ``col_name``, ``h5_error_count``,
            ``h5_codependent_flag``
        One row per erroneous cell.
        """
        self._check_fitted()

        records = []
        error_positions = self._get_error_positions(mask_df)

        # Pre-compute per-row error counts for efficiency
        row_error_counts = mask_df.sum(axis=1)  # Series: index → count

        for row_idx, col_name in error_positions:
            # ── h5_error_count ──────────────────────────────────────────────
            h5_error_count = int(row_error_counts.loc[row_idx])

            # ── h5_codependent_flag ─────────────────────────────────────────
            h5_codependent_flag = 0
            for pair in self.codependent_pairs_:
                if col_name not in pair:
                    continue
                # find the partner column
                partner = next(iter(pair - {col_name}))
                if partner not in mask_df.columns:
                    continue
                if mask_df.loc[row_idx, partner] == 1:
                    h5_codependent_flag = 1
                    break

            records.append(
                {
                    "row_idx": row_idx,
                    "col_name": col_name,
                    "h5_error_count": h5_error_count,
                    "h5_codependent_flag": h5_codependent_flag,
                }
            )

        return pd.DataFrame(
            records,
            columns=["row_idx", "col_name", "h5_error_count", "h5_codependent_flag"],
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("H5ErrorPattern — Self-test")
    print("=" * 60)

    dirty = pd.DataFrame(
        {
            "education":     ["HS-grad",   "HS-grad",  "Bachelors", "Masters"],
            "education-num": [9,            9,           13,          14],
            "age":           [25,           30,          35,          40],
        }
    )
    mask = pd.DataFrame(
        {
            "education":     [0, 0, 1, 0],   # row 2 education is error
            "education-num": [0, 0, 1, 0],   # row 2 education-num is ALSO error
            "age":           [0, 0, 0, 0],
        }
    )

    h5 = H5ErrorPattern()
    h5.fit(dirty, mask)  # should auto-detect education + education-num via name similarity

    print("\n── Discovered co-dependent pairs ──")
    for p in h5.codependent_pairs_:
        print("   ", set(p))

    result = h5.compute(dirty, mask)
    print("\n── Computed features ──")
    print(result.to_string(index=False))

    # ── Assertions ─────────────────────────────────────────────────────────
    row2_edu = result[(result["row_idx"] == 2) & (result["col_name"] == "education")]
    row2_edunum = result[(result["row_idx"] == 2) & (result["col_name"] == "education-num")]

    assert len(row2_edu) == 1, "Expected one row for (2, education)"
    assert len(row2_edunum) == 1, "Expected one row for (2, education-num)"

    assert row2_edu["h5_error_count"].item() == 2, (
        f"h5_error_count for (2, education) should be 2, got "
        f"{row2_edu['h5_error_count'].item()}"
    )
    assert row2_edunum["h5_error_count"].item() == 2, (
        f"h5_error_count for (2, education-num) should be 2, got "
        f"{row2_edunum['h5_error_count'].item()}"
    )
    assert row2_edu["h5_codependent_flag"].item() == 1, (
        "h5_codependent_flag for (2, education) should be 1 "
        "(education-num is a co-dependent partner and is also erroneous)"
    )
    assert row2_edunum["h5_codependent_flag"].item() == 1, (
        "h5_codependent_flag for (2, education-num) should be 1 "
        "(education is a co-dependent partner and is also erroneous)"
    )

    print("\n── All assertions passed ✓ ──")

    # ── Additional test: user-supplied pairs ────────────────────────────────
    print("\n── Test: user-supplied pairs ──")
    h5b = H5ErrorPattern()
    h5b.fit(
        dirty,
        mask,
        codependent_pairs=[("education", "education-num"), ("nonexistent", "col")],
    )
    assert frozenset({"education", "education-num"}) in h5b.codependent_pairs_, \
        "User-supplied pair should be stored"
    print("   User-supplied pair stored correctly ✓")
    print("   Warning for missing columns was expected above ✓")

    # ── Additional test: single-column table (edge case) ────────────────────
    print("\n── Test: single-column table (edge case) ──")
    dirty_single = pd.DataFrame({"age": [25, 30, 35]})
    mask_single = pd.DataFrame({"age": [0, 1, 0]})
    h5c = H5ErrorPattern()
    h5c.fit(dirty_single, mask_single)
    result_single = h5c.compute(dirty_single, mask_single)
    assert result_single["h5_codependent_flag"].item() == 0, \
        "Single-column table: codependent_flag must be 0"
    assert result_single["h5_error_count"].item() == 1, \
        "Single-column table: error_count must be 1"
    print("   Single-column edge case passed ✓")

    print("\n✅  Self-test complete.")
