"""
H8SensitivityFlag — Heuristic 8
==================================
Detects **privacy-motivated demographic masking**: a user changes a minority
demographic value to the majority class to avoid statistical profiling.

Motivation
----------
Intentional manipulation of sensitive attributes (race, gender, age, marital
status, etc.) follows a well-known privacy-protection pattern:

    *Change your minority-class value to the majority class so you "blend in"
    with the statistical population and are harder to profile.*

For example, in the Adult Income dataset:
- ``race``:  86 % White  → changing any non-White value to "White" is strong
  evidence of privacy-motivated masking.
- ``sex``:   67 % Male   → moderate signal (less dominant majority).

H8 captures two orthogonal binary signals for every erroneous cell:

1. **Is this a sensitive demographic attribute?**
   (column-level, constant within a column)
2. **Is the dirty value the majority class of that sensitive column?**
   (per-cell, only meaningful for categorical sensitive columns)

**Known limitation:** The majority-class signal is weak when the majority
class is not dominant (e.g. ``sex`` at 67 % Male).  For highly skewed
sensitive columns (e.g. ``race`` at 86 % White) the signal is stronger.
This signal is only defined for *categorical* sensitive columns; numerical
sensitive columns such as ``age`` always receive 0 for
``h8_is_majority_value`` because "majority value" is not meaningful for
continuous distributions.

Output features (one row per error position)
---------------------------------------------
======================  ==========  ==========================================
Column                  Type        Description
======================  ==========  ==========================================
row_idx                 int         Row index of the erroneous cell
col_name                str         Column name of the erroneous cell
h8_is_sensitive         int (0/1)   1 if this column is a sensitive demographic
                                    attribute; 0 otherwise.
h8_is_majority_value    int (0/1)   1 if the dirty value equals the majority
                                    class of this sensitive categorical column.
                                    Always 0 for:
                                    • non-sensitive columns
                                    • numerical sensitive columns (e.g. age)
                                    • sensitive columns with no clean data
======================  ==========  ==========================================

Usage example
-------------
>>> h8 = H8SensitivityFlag()
>>> h8.fit(dirty_df, mask_df)                         # auto-detect sensitive cols
>>> h8.fit(dirty_df, mask_df, sensitive_cols=["race", "sex"])  # user-supplied
>>> features = h8.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import logging
import sys
import warnings
from typing import Dict, List, Optional, Set

import pandas as pd

# Support both package import (relative) and standalone execution (absolute)
if __package__:
    from .base import BaseHeuristic
else:
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from base import BaseHeuristic  # type: ignore[no-redef]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Case-insensitive substrings used for auto-detecting sensitive columns.
SENSITIVE_KEYWORDS: frozenset = frozenset({
    "race", "gender", "sex", "age", "nationality", "religion",
    "disability", "marital", "ethnic", "origin", "orientation",
})

#: Fraction of non-null values that must parse as float for a column to be
#: considered numerical.
_NUMERIC_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_numeric_col(series: pd.Series) -> bool:
    """
    Return True when > _NUMERIC_THRESHOLD of non-null values are numeric.

    Parameters
    ----------
    series : pd.Series
        The column to test.

    Returns
    -------
    bool
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    return (numeric_count / len(non_null)) > _NUMERIC_THRESHOLD


def _auto_detect_sensitive(columns: List[str]) -> Set[str]:
    """
    Return the set of column names that contain at least one sensitive keyword
    as a case-insensitive substring.

    Parameters
    ----------
    columns : list of str
        All column names to check.

    Returns
    -------
    set of str
    """
    sensitive: Set[str] = set()
    for col in columns:
        col_lower = col.lower()
        for kw in SENSITIVE_KEYWORDS:
            if kw in col_lower:
                sensitive.add(col)
                break
    return sensitive


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class H8SensitivityFlag(BaseHeuristic):
    """
    Heuristic 8 — Sensitivity Flag.

    Detects privacy-motivated demographic masking: the pattern where a user
    replaces a minority demographic value with the majority class to avoid
    statistical profiling.

    Two binary features per erroneous cell:

    * ``h8_is_sensitive``      — column is a sensitive demographic attribute.
    * ``h8_is_majority_value`` — the dirty value equals the majority class of
                                 that sensitive categorical column.

    **Known limitation:** the majority-class signal is weak when the majority
    class is not statistically dominant.  For ``sex`` in Adult Income (~67 %
    Male) the signal is moderate; for ``race`` (~86 % White) it is stronger.
    ``h8_is_majority_value`` is always 0 for numerical sensitive columns (such
    as ``age``) because "majority value" has no meaningful interpretation for
    continuous distributions.

    Parameters
    ----------
    None — all configuration is passed through fit().

    Attributes (post-fit)
    ---------------------
    sensitive_cols_ : set of str
        Column names identified (or user-supplied) as sensitive.
    majority_class_ : dict mapping col_name → str or None
        For each sensitive *categorical* column: the most frequent value
        computed from clean cells.  None when the clean set is empty.
    is_fitted : bool
        True after fit() has been called.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sensitive_cols_: Set[str] = set()
        self.majority_class_: Dict[str, Optional[str]] = {}
        self._numerical_cols_: Set[str] = set()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        sensitive_cols: Optional[List[str]] = None,
    ) -> "H8SensitivityFlag":
        """
        Learn which columns are sensitive and what their majority class is.

        Sensitive column detection
        --------------------------
        1. **User-supplied** columns (``sensitive_cols`` list) are always
           marked sensitive — regardless of their name.
        2. Remaining columns are **auto-evaluated**: any column whose name
           contains a keyword from ``SENSITIVE_KEYWORDS`` as a
           case-insensitive substring is also marked sensitive.

        Majority-class detection
        ------------------------
        For each sensitive column that is **categorical** (i.e. NOT detected
        as numerical):
        - Use only clean cells (rows where ``mask_df[col] == 0``).
        - Cast values to ``str`` and find the mode.
        - Store in ``self.majority_class_[col]``.
        - If the clean set is empty, store ``None``.

        Numerical sensitive columns (e.g. ``age``) are tracked in
        ``self._numerical_cols_`` so that ``h8_is_majority_value`` can
        correctly return 0 for them in compute().

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Binary mask (0 = clean, 1 = error).  Same shape as dirty_df.
        sensitive_cols : list of str, optional
            User-supplied list of column names to treat as sensitive.
            Columns in this list override auto-detection (they are always
            sensitive).  Columns NOT in this list are still auto-evaluated.

        Returns
        -------
        self
        """
        user_sensitive: Set[str] = set(sensitive_cols) if sensitive_cols else set()

        # Columns not explicitly supplied → auto-evaluate
        non_user_cols = [c for c in dirty_df.columns if c not in user_sensitive]
        auto_sensitive = _auto_detect_sensitive(non_user_cols)

        self.sensitive_cols_ = user_sensitive | auto_sensitive

        if not self.sensitive_cols_:
            warnings.warn(
                "H8SensitivityFlag.fit(): no sensitive columns detected. "
                "The dataset may genuinely have no demographic attributes, "
                "or you may want to pass sensitive_cols explicitly.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "H8SensitivityFlag: 0 sensitive columns detected for columns: %s",
                list(dirty_df.columns),
            )

        # Determine which sensitive columns are numerical
        self._numerical_cols_ = {
            col for col in self.sensitive_cols_
            if col in dirty_df.columns and _is_numeric_col(dirty_df[col])
        }

        # Compute majority class for each sensitive categorical column
        self.majority_class_ = {}
        for col in self.sensitive_cols_:
            if col not in dirty_df.columns:
                # User supplied a column name that doesn't exist — skip silently
                logger.warning(
                    "H8SensitivityFlag.fit(): sensitive column '%s' not found "
                    "in dirty_df; skipping majority-class computation.",
                    col,
                )
                continue

            if col in self._numerical_cols_:
                # Skip majority-class computation for numerical sensitive cols
                continue

            # Use only clean rows for this column
            clean_mask = mask_df[col] == 0
            clean_values = dirty_df.loc[clean_mask, col].dropna()

            if len(clean_values) == 0:
                self.majority_class_[col] = None
                logger.warning(
                    "H8SensitivityFlag.fit(): no clean cells found for "
                    "sensitive column '%s'; majority_class set to None.",
                    col,
                )
            else:
                mode_result = clean_values.astype(str).mode()
                # mode() returns a Series; take first element (alphabetically
                # first if there is a tie)
                self.majority_class_[col] = str(mode_result.iloc[0])

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
        Compute H8 features for every erroneous cell (mask_df == 1).

        For each erroneous cell at (row_idx, col_name):

        * ``h8_is_sensitive``:
          1 if ``col_name in self.sensitive_cols_``, else 0.

        * ``h8_is_majority_value``:
          - Always 0 if ``h8_is_sensitive == 0``.
          - Always 0 if the column is numerical (even if sensitive).
          - Always 0 if the column has no clean data (majority_class is None).
          - 1 if ``str(dirty_value) == self.majority_class_[col_name]``.
          - 0 otherwise.

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame
            Columns: [row_idx, col_name, h8_is_sensitive, h8_is_majority_value]
            One row per erroneous cell.
        """
        self._check_fitted()

        records = []
        for row_idx, col_name in self._get_error_positions(mask_df):
            # ── h8_is_sensitive ──────────────────────────────────────────────
            is_sensitive = int(col_name in self.sensitive_cols_)

            # ── h8_is_majority_value ─────────────────────────────────────────
            is_majority = 0

            if is_sensitive:
                # Numerical sensitive columns → not applicable
                if col_name not in self._numerical_cols_:
                    maj = self.majority_class_.get(col_name)
                    if maj is not None:
                        dirty_value = dirty_df.at[row_idx, col_name]
                        is_majority = int(str(dirty_value) == maj)

            records.append({
                "row_idx":             row_idx,
                "col_name":            col_name,
                "h8_is_sensitive":     is_sensitive,
                "h8_is_majority_value": is_majority,
            })

        return pd.DataFrame(
            records,
            columns=["row_idx", "col_name", "h8_is_sensitive", "h8_is_majority_value"],
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from h8_sensitivity_flag import H8SensitivityFlag  # noqa: F811

    print("=" * 60)
    print("H8SensitivityFlag — Self-test")
    print("=" * 60)

    dirty = pd.DataFrame({
        "age":    [25, 30, 35, 999, 28],      # 999 = error in age (sensitive numerical)
        "race":   ["White", "White", "White", "White", "Black"],
                                              # last row Black → error → not majority?
        "income": [0, 1, 0, 1, 0],
    })
    mask = pd.DataFrame({
        "age":    [0, 0, 0, 1, 0],
        "race":   [0, 0, 0, 0, 1],
        "income": [0, 0, 0, 0, 0],
    })

    h8 = H8SensitivityFlag()
    h8.fit(dirty, mask)  # auto-detects "age" and "race" as sensitive

    print(f"\nAuto-detected sensitive columns: {h8.sensitive_cols_}")
    print(f"Numerical sensitive columns:     {h8._numerical_cols_}")
    print(f"Majority classes:                {h8.majority_class_}")

    result = h8.compute(dirty, mask)
    print("\n--- Test 1 result (dirty value for race = 'Black') ---")
    print(result.to_string(index=False))

    # Verify expectations
    age_row  = result[result["col_name"] == "age"].iloc[0]
    race_row = result[result["col_name"] == "race"].iloc[0]

    assert age_row["h8_is_sensitive"]      == 1, "age should be sensitive"
    assert age_row["h8_is_majority_value"] == 0, "age is numerical → majority_value must be 0"
    assert race_row["h8_is_sensitive"]     == 1, "race should be sensitive"
    assert race_row["h8_is_majority_value"] == 0, \
        f"'Black' != 'White' (majority), expected 0 got {race_row['h8_is_majority_value']}"

    print("\n✓ Test 1 passed: age numerical → h8_is_majority_value=0; "
          "'Black' ≠ majority 'White' → h8_is_majority_value=0")

    # --- Test 2: error that IS the majority class ---
    dirty2 = dirty.copy()
    dirty2.loc[4, "race"] = "White"   # now the error is the majority class
    mask2 = mask.copy()

    h8b = H8SensitivityFlag()
    h8b.fit(dirty2, mask2)
    result2 = h8b.compute(dirty2, mask2)
    print("\n--- Test 2 result (dirty value for race = 'White', the majority) ---")
    print(result2.to_string(index=False))

    race_row2 = result2[result2["col_name"] == "race"].iloc[0]
    assert race_row2["h8_is_majority_value"] == 1, \
        f"'White' == majority 'White', expected 1 got {race_row2['h8_is_majority_value']}"

    print("\n✓ Test 2 passed: 'White' == majority 'White' → h8_is_majority_value=1")

    # --- Test 3: user-supplied sensitive_cols ---
    dirty3 = pd.DataFrame({
        "zipcode": ["10001", "10002", "10003", "XXXXX", "10005"],
        "income":  [50000, 60000, 55000, 70000, 45000],
    })
    mask3 = pd.DataFrame({
        "zipcode": [0, 0, 0, 1, 0],
        "income":  [0, 0, 0, 0, 0],
    })

    h8c = H8SensitivityFlag()
    h8c.fit(dirty3, mask3, sensitive_cols=["zipcode"])  # explicitly marked sensitive
    result3 = h8c.compute(dirty3, mask3)
    print("\n--- Test 3 result (user-supplied sensitive_cols=['zipcode']) ---")
    print(result3.to_string(index=False))

    zip_row = result3[result3["col_name"] == "zipcode"].iloc[0]
    assert zip_row["h8_is_sensitive"] == 1, "zipcode should be sensitive (user-supplied)"

    print("\n✓ Test 3 passed: user-supplied 'zipcode' marked sensitive")

    # --- Test 4: no sensitive columns (warning expected) ---
    import warnings as _warnings
    dirty4 = pd.DataFrame({
        "feature_a": [1, 2, 3, 99, 5],
        "feature_b": ["x", "y", "z", "w", "v"],
    })
    mask4 = pd.DataFrame({
        "feature_a": [0, 0, 0, 1, 0],
        "feature_b": [0, 0, 0, 0, 0],
    })

    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        h8d = H8SensitivityFlag()
        h8d.fit(dirty4, mask4)
        assert len(w) == 1 and issubclass(w[0].category, UserWarning), \
            "Expected a UserWarning for 0 sensitive columns"

    result4 = h8d.compute(dirty4, mask4)
    print("\n--- Test 4 result (no sensitive columns → UserWarning issued) ---")
    print(result4.to_string(index=False))
    assert (result4["h8_is_sensitive"] == 0).all(), \
        "All cells should be non-sensitive"
    assert (result4["h8_is_majority_value"] == 0).all(), \
        "All majority_value should be 0 for non-sensitive columns"

    print("\n✓ Test 4 passed: UserWarning raised; all cells non-sensitive")

    print("\n" + "=" * 60)
    print("All self-tests passed ✓")
    print("=" * 60)
