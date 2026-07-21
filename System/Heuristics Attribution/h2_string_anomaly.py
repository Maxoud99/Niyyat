"""
H2StringAnomaly — Heuristic 2
==============================
Characterises the *kind* of wrong string value found in a categorical column.

Where H1 answers "is this value in the vocabulary?" (binary), H2 answers the
follow-up question:

    **Does this string look like a typo, or does it look like deliberate
    obfuscation?**

- A **typo** (unintentional) has a small Levenshtein distance to the nearest
  clean vocabulary entry, e.g. ``"Bachleors"`` → edit distance 1 to
  ``"Bachelors"``.
- An **obfuscation token** (intentional) matches a known placeholder pattern,
  e.g. ``"Unknown"``, ``"nan"``, ``"—"``, or a suffix-mangled value like
  ``"Private-DMV"``.

**Applies to categorical columns only.**  For numerical columns both output
features are set to ``NaN`` (numerical errors are handled by H3/H4).

Column type detection
---------------------
Auto-detection: a column is treated as *numerical* when > 90 % of its non-null
values can be cast to ``float``; otherwise it is treated as *categorical*.
The caller can override per-column types via the ``col_types`` argument to
:py:meth:`fit`.

Output features (one row per error position)
--------------------------------------------
===================  ============  ================================================
Column               Type          Description
===================  ============  ================================================
row_idx              int           Row index of the erroneous cell
col_name             str           Column name of the erroneous cell
h2_min_edit_dist     float | NaN   Levenshtein distance to nearest clean-vocab entry.
                                   NaN for numerical columns. Capped at 10.
h2_is_obfuscation    int | NaN     1 if the dirty value matches a known obfuscation
                                   pattern. 0 otherwise. NaN for numerical columns.
===================  ============  ================================================

Score formula (explanation layer, not classification)
-----------------------------------------------------
::

    h2_score = (h2_is_obfuscation + max(0, 1 - h2_min_edit_dist / 5)) / 2

- High h2_score  → evidence of **intentional** error (obfuscation / near-vocab)
- Low  h2_score + low edit distance → **unintentional** (typo)

Usage example
-------------
>>> h2 = H2StringAnomaly()
>>> h2.fit(dirty_df, mask_df)
>>> features = h2.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import math
import sys
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Support both direct execution (python h2_string_anomaly.py) and package import
if __name__ == "__main__" or not __package__:
    # Running as a standalone script – add src/ to sys.path so that
    # `attribution.heuristics.base` resolves correctly.
    # Layout: src/attribution/heuristics/h2_string_anomaly.py
    #         ^--- _src_root is here
    _src_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    from attribution.heuristics.base import BaseHeuristic  # type: ignore
else:
    from .base import BaseHeuristic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_THRESHOLD = 0.9   # fraction of non-null values that must parse as float

# Maximum edit distance cap
_MAX_EDIT_DIST = 10

# Vocabulary size cap: if a column has more than this many unique clean values,
# restrict distance computation to the most-frequent MAX_VOCAB_SIZE entries.
_MAX_VOCAB_SIZE = 50
_LARGE_VOCAB_THRESHOLD = 200

# Known obfuscation tokens (matched case-insensitively)
OBFUSCATION_TOKENS: frozenset = frozenset({
    "nan", "none", "n/a", "na", "unknown", "unk", "?", "—", "-", "--",
    "null", "nil", "missing", "not available", "not applicable",
    "0", "-1", "999", "9999", "99999",
})

# Obfuscation suffixes / infixes used with suffix-pattern detection
_OBF_SEPARATORS = {"-", "_", " "}
_OBF_SUFFIXES = {"dmv", "obf", "high", "low", "1", "2", "x", "?", "—"}


# ---------------------------------------------------------------------------
# Levenshtein (inline, no external dependencies)
# ---------------------------------------------------------------------------

def _levenshtein(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings using standard
    dynamic-programming.  Implemented inline so the heuristic has no external
    package requirement.

    Optimisation: the inner loop skips ``s2`` entries whose length differs from
    ``len(s1)`` by more than 5, since those can never produce a smaller
    distance than the cap anyway.

    Parameters
    ----------
    s1, s2 : str
        The two strings to compare.

    Returns
    -------
    int
        Edit distance (number of single-character insertions, deletions or
        substitutions needed to transform ``s1`` into ``s2``).
    """
    len1, len2 = len(s1), len(s2)

    # Early-exit: length difference alone gives a lower bound
    if abs(len1 - len2) >= _MAX_EDIT_DIST:
        return _MAX_EDIT_DIST

    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Use two rolling rows to keep memory O(min(len1, len2))
    if len1 < len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    prev = list(range(len2 + 1))
    curr = [0] * (len2 + 1)

    for i in range(1, len1 + 1):
        curr[0] = i
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,       # insertion
                prev[j] + 1,           # deletion
                prev[j - 1] + cost,    # substitution
            )
        prev, curr = curr, prev

    return min(prev[len2], _MAX_EDIT_DIST)


# ---------------------------------------------------------------------------
# Helper: column type detection (same rule as H1)
# ---------------------------------------------------------------------------

def _detect_type(col: pd.Series) -> str:
    """Return ``'num'`` or ``'cat'`` by attempting numeric coercion."""
    non_null = col.dropna()
    if non_null.empty:
        return "cat"
    numeric = pd.to_numeric(non_null, errors="coerce")
    frac_numeric = numeric.notna().sum() / len(non_null)
    return "num" if frac_numeric > _NUMERIC_THRESHOLD else "cat"


# ---------------------------------------------------------------------------
# H2StringAnomaly
# ---------------------------------------------------------------------------

class H2StringAnomaly(BaseHeuristic):
    """
    Heuristic 2 — String Anomaly Characterisation.

    For each erroneous cell in a **categorical** column, compute:

    - ``h2_min_edit_dist`` — Levenshtein distance to the nearest clean
      vocabulary entry (capped at 10).  A small distance → typo; a large
      distance → the value has no lexical resemblance to any valid entry.
    - ``h2_is_obfuscation`` — 1 when the dirty value matches a known
      obfuscation pattern (placeholder token, suffix-mangled, structural
      placeholder).

    Parameters
    ----------
    (none at construction time; configuration is passed to :py:meth:`fit`)

    Attributes
    ----------
    col_stats_ : dict
        Populated after :py:meth:`fit`.  Maps column name → dict with keys:

        - ``'type'``: ``'cat'`` or ``'num'``
        - ``'vocab'``: (categorical) ``list[str]`` of clean vocabulary entries
          (possibly capped to ``_MAX_VOCAB_SIZE`` most-frequent)
        - ``'median_len'``: (categorical) median string length across vocab
    is_fitted : bool
        ``True`` after :py:meth:`fit` is called.
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
    ) -> "H2StringAnomaly":
        """
        Learn per-column vocabulary from the *clean* cells only.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Binary mask aligned with ``dirty_df``.  0 = clean, 1 = erroneous.
        col_types : dict, optional
            Per-column type override, e.g.
            ``{'age': 'num', 'education': 'cat'}``.
            Any column not listed falls back to auto-detection.

        Returns
        -------
        self
        """
        col_types = col_types or {}
        self.col_stats_ = {}

        for col in dirty_df.columns:
            # ---- determine column type --------------------------------
            ctype = col_types.get(col) or _detect_type(dirty_df[col])

            if ctype == "num":
                self.col_stats_[col] = {"type": "num"}
                continue

            # ---- categorical: extract clean cells --------------------
            clean_mask = mask_df[col] == 0
            clean_vals = dirty_df.loc[clean_mask, col]

            vocab_counts: Dict[str, int] = {}
            for v in clean_vals:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                s = str(v)
                if s == "":
                    continue
                vocab_counts[s] = vocab_counts.get(s, 0) + 1

            # If the vocabulary is very large, keep only the most frequent
            # entries to bound compute time.
            if len(vocab_counts) > _LARGE_VOCAB_THRESHOLD:
                sorted_entries = sorted(
                    vocab_counts.items(), key=lambda kv: kv[1], reverse=True
                )
                vocab_counts = dict(sorted_entries[:_MAX_VOCAB_SIZE])

            vocab_list = list(vocab_counts.keys())

            # Median string length of vocabulary entries
            if vocab_list:
                lengths = sorted(len(w) for w in vocab_list)
                n = len(lengths)
                if n % 2 == 1:
                    median_len = float(lengths[n // 2])
                else:
                    median_len = (lengths[n // 2 - 1] + lengths[n // 2]) / 2.0
            else:
                median_len = 0.0

            self.col_stats_[col] = {
                "type": "cat",
                "vocab": vocab_list,
                "median_len": median_len,
            }

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------
    def compute(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute string-anomaly features for every erroneous cell.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape, values in {0, 1})

        Returns
        -------
        pd.DataFrame
            One row per error position with columns:
            ``[row_idx, col_name, h2_min_edit_dist, h2_is_obfuscation]``
        """
        self._check_fitted()

        rows = []
        for row_idx, col_name in self._get_error_positions(mask_df):
            raw_val = dirty_df.loc[row_idx, col_name]
            stats = self.col_stats_[col_name]

            if stats["type"] == "num":
                rows.append({
                    "row_idx": row_idx,
                    "col_name": col_name,
                    "h2_min_edit_dist": np.nan,
                    "h2_is_obfuscation": np.nan,
                })
            else:
                rows.append(
                    self._score_categorical(row_idx, col_name, raw_val, stats)
                )

        if not rows:
            return pd.DataFrame(
                columns=["row_idx", "col_name", "h2_min_edit_dist", "h2_is_obfuscation"]
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_categorical(row_idx, col_name: str, raw_val, stats: dict) -> dict:
        """Compute both H2 features for one erroneous cell in a categorical column."""
        dirty_str = "" if (
            raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val))
        ) else str(raw_val)

        vocab: list = stats["vocab"]

        # ---- Feature 1: minimum edit distance to clean vocabulary -------
        min_edit = H2StringAnomaly._min_edit_distance(dirty_str, vocab)

        # ---- Feature 2: obfuscation flag --------------------------------
        is_obf = int(H2StringAnomaly._is_obfuscation(dirty_str, vocab))

        return {
            "row_idx": row_idx,
            "col_name": col_name,
            "h2_min_edit_dist": float(min_edit),
            "h2_is_obfuscation": is_obf,
        }

    @staticmethod
    def _min_edit_distance(dirty_str: str, vocab: list) -> int:
        """
        Return the minimum Levenshtein distance from ``dirty_str`` to any
        entry in ``vocab``.

        Special cases
        -------------
        - Empty vocab  → return ``_MAX_EDIT_DIST`` (10).
        - ``dirty_str`` already in vocab → return 0 (exact match → cost 0).

        Optimisation: skip vocab entries whose length differs from
        ``len(dirty_str)`` by more than 5, because the length-difference lower
        bound alone would already exceed the cap.
        """
        if not vocab:
            return _MAX_EDIT_DIST

        dirty_lower = dirty_str.lower()
        # Build a fast membership set from vocab (case-sensitive, as stored)
        vocab_set = set(vocab)
        if dirty_str in vocab_set:
            return 0

        dirty_len = len(dirty_str)
        best = _MAX_EDIT_DIST

        for entry in vocab:
            # Length-difference early-exit optimisation
            if abs(len(entry) - dirty_len) >= best:
                continue
            dist = _levenshtein(dirty_str, entry)
            if dist < best:
                best = dist
                if best == 0:
                    break   # can't do better

        return best

    @staticmethod
    def _is_obfuscation(dirty_str: str, vocab: list) -> bool:
        """
        Return ``True`` if ``dirty_str`` matches any obfuscation pattern.

        Three detection rules (any one is sufficient):

        1. **Exact-token match** — ``dirty_str`` (case-insensitive) is in
           :data:`OBFUSCATION_TOKENS`.
        2. **Suffix/prefix pattern** — ``dirty_str`` starts with a vocab entry
           followed by one of ``{'-', '_', ' '}`` plus one of
           ``_OBF_SUFFIXES``.
        3. **Structural placeholder** — string length == 1, OR the string
           consists entirely of non-alphanumeric characters.
        """
        if not dirty_str:
            return False

        # Rule 1: exact obfuscation token (case-insensitive)
        if dirty_str.lower() in OBFUSCATION_TOKENS:
            return True

        # Rule 2: suffix/prefix pattern  (e.g. "Private-DMV")
        dirty_lower = dirty_str.lower()
        for vocab_entry in vocab:
            prefix = vocab_entry.lower()
            if not dirty_lower.startswith(prefix):
                continue
            remainder = dirty_lower[len(prefix):]
            if len(remainder) < 2:
                continue
            sep = remainder[0]
            suffix = remainder[1:]
            if sep in _OBF_SEPARATORS and suffix in _OBF_SUFFIXES:
                return True

        # Rule 3: structural placeholder
        if len(dirty_str) == 1:
            return True
        if all(not ch.isalnum() for ch in dirty_str):
            return True

        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    # H2StringAnomaly is already imported at module level via the sys.path trick above

    dirty = pd.DataFrame({
        "education": [
            "HS-grad",        # row 0: clean
            "Bachleors",      # row 1: error — typo (edit dist 1 to "Bachelors")
            "Bachelors",      # row 2: clean
            "Unknown",        # row 3: error — obfuscation token
            "—",              # row 4: error — obfuscation token (single char / non-alnum)
            "Doctorate-obf",  # row 5: error — suffix obfuscation ("Doctorate" is clean)
            "Doctorate",      # row 6: clean  ← required so "Doctorate" is in vocab
        ],
        "age": [25, 30, 35, 40, 45, 50, 55],   # numerical → NaN for both features
    })
    mask = pd.DataFrame({
        "education": [0, 1, 0, 1, 1, 1, 0],
        "age":       [0, 0, 0, 0, 0, 0, 0],
    })

    h2 = H2StringAnomaly()
    h2.fit(dirty, mask)
    result = h2.compute(dirty, mask)

    print("=" * 70)
    print("H2 String Anomaly — self-test result")
    print("=" * 70)
    print(result.to_string(index=False))
    print()

    # ---- helper ---------------------------------------------------------
    def get(df, row, col, feature):
        row_data = df[(df["row_idx"] == row) & (df["col_name"] == col)]
        assert not row_data.empty, f"No result for ({row}, {col})"
        return row_data.iloc[0][feature]

    # row 1, education: "Bachleors" → typo
    ed1_dist = get(result, 1, "education", "h2_min_edit_dist")
    ed1_obf  = get(result, 1, "education", "h2_is_obfuscation")
    assert ed1_dist <= 2, f"Expected small edit dist for typo, got {ed1_dist}"
    assert ed1_obf == 0,  f"Expected not obfuscation for typo, got {ed1_obf}"
    print(f"row 1 'Bachleors': dist={ed1_dist}, obf={ed1_obf}  ✓ (typo)")

    # row 3, education: "Unknown" → obfuscation token
    ed3_dist = get(result, 3, "education", "h2_min_edit_dist")
    ed3_obf  = get(result, 3, "education", "h2_is_obfuscation")
    assert ed3_obf == 1, f"Expected obfuscation=1 for 'Unknown', got {ed3_obf}"
    print(f"row 3 'Unknown':   dist={ed3_dist}, obf={ed3_obf}  ✓ (obfuscation token)")

    # row 4, education: "—" → single non-alnum char
    ed4_dist = get(result, 4, "education", "h2_min_edit_dist")
    ed4_obf  = get(result, 4, "education", "h2_is_obfuscation")
    assert ed4_obf == 1, f"Expected obfuscation=1 for '—', got {ed4_obf}"
    print(f"row 4 '—':         dist={ed4_dist}, obf={ed4_obf}  ✓ (structural placeholder)")

    # row 5, education: "Doctorate-obf" → suffix obfuscation
    ed5_dist = get(result, 5, "education", "h2_min_edit_dist")
    ed5_obf  = get(result, 5, "education", "h2_is_obfuscation")
    assert ed5_obf == 1, f"Expected obfuscation=1 for 'Doctorate-obf', got {ed5_obf}"
    print(f"row 5 'Doctorate-obf': dist={ed5_dist}, obf={ed5_obf}  ✓ (suffix obfuscation)")

    # age column errors: all mask==0, so no error rows for age
    age_rows = result[result["col_name"] == "age"]
    assert age_rows.empty, "Expected no error rows for numerical column 'age'"
    print("age column: no error rows (all clean)  ✓")

    print()
    print("All assertions passed ✓")
