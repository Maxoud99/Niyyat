"""
H7UserIncentive — Heuristic 7
================================
Measures **behavioral motivation** — whether a rational human actor would be
motivated to manipulate this column.  This is entirely separate from H6
(statistical mutual-information importance).

Motivation
----------
A column can be statistically important (H6 high) yet never targeted by
real humans because they do not understand it (e.g. ``fnlwgt``).  Conversely,
a column can be statistically weak (H6 low) but heavily manipulated because
humans understand it and feel incentivised to change it (e.g. ``race`` for
fairness/privacy reasons, ``education`` for credential inflation).

H7 captures three orthogonal behavioral signals:

1. **Mutability** — can a human realistically change this value at all?
2. **Gain direction** — does changing this value *in this direction* move the
   outcome favourably?  (Per dirty-cell lookup, not a per-column average.)
3. **Comprehensibility** — would a typical person understand what this column
   even measures?

Critical distinction from H6
-----------------------------
=========  ====================  ====================
Column     H6 (MI)               H7 (behavioral)
=========  ====================  ====================
fnlwgt     HIGH — predictive     LOW  — nobody understands it
race       LOW  — weak MI        HIGH — mask for fairness/privacy
education  HIGH                  HIGH — credential forgery target
ssn        MEDIUM                LOW  — immutable and risky
=========  ====================  ====================

Both H6 and H7 are required.  They are **not** redundant — they capture
orthogonal signals.

Output features (one row per erroneous cell)
--------------------------------------------
======================  ==========  =============================================
Column                  Type        Description
======================  ==========  =============================================
row_idx                 int         Row index of the erroneous cell
col_name                str         Column name of the erroneous cell
h7_mutability           float [0,1] Can a user realistically change this value?
                                    0 = immutable, 0.5 = soft boundary,
                                    1 = freely mutable.
h7_gain_direction       float [0,1] Does the dirty value shift toward a
                                    favourable outcome?  **Per-cell lookup** for
                                    categorical columns (not a column average).
                                    0.5 = neutral / no target_col provided.
h7_comprehensibility    float [0,1] Would a typical human understand what this
                                    column means?  0 = opaque, 1 = obvious.
======================  ==========  =============================================

Usage example
-------------
>>> h7 = H7UserIncentive()
>>> h7.fit(dirty_df, mask_df, target_col="income")
>>> features = h7.compute(dirty_df, mask_df)
"""

from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Support both package import (relative) and standalone execution (absolute)
if __package__:
    from .base import BaseHeuristic
else:
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from base import BaseHeuristic  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants — mutability auto-classification keyword sets (fallback path)
# ---------------------------------------------------------------------------
# These substring lists are a zero-cost fallback for when the primary,
# few-shot LLM classifier (_llm_classify_mutability, below) is unavailable
# or does not cover a given column. Being English-only, exact-substring
# rules, they do not generalize to unseen or non-English column names, so
# they are deliberately not the primary mechanism.

#: Substrings that indicate a system-assigned, immutable column.
_IMMUTABLE_KEYWORDS = frozenset({
    "id", "num", "wgt", "weight", "score", "code", "key",
    "index", "ssn", "hash", "uuid", "timestamp", "date", "time",
})

#: Substrings that indicate a sensitive attribute (soft boundary).
_SENSITIVE_KEYWORDS = frozenset({
    "race", "gender", "sex", "nationality", "religion",
    "disability", "ethnicity",
})

#: Threshold for treating a column as numeric.
_NUMERIC_THRESHOLD = 0.9

#: Minimum number of clean rows needed to compute gain direction reliably.
_MIN_CLEAN_ROWS = 5

#: Maps the LLM classifier's three output labels to the scalar h7_mutability
#: values used everywhere else in this module.
_MUTABILITY_LABEL_TO_SCORE = {
    "immutable": 0.0,
    "protected": 0.5,
    "mutable": 1.0,
}

# ---------------------------------------------------------------------------
# Few-shot LLM mutability classifier (primary path)
# ---------------------------------------------------------------------------
# Same caching discipline as the intent-signal rule extractor
# ("User-guided Attribution/declarative/extractor.py"): one Gemini call per
# dataset schema, result cached as JSON, never recomputed at attribution
# time. Classifying by analogy from a handful of labeled examples per class
# generalizes to column names that share no substring with any keyword
# (e.g. "identity_document_no", non-English names), unlike the fallback
# keyword rule above.

_MUTABILITY_PROMPT_TEMPLATE = """\
You are a data-schema analyst. For each column name below, judge whether a
person supplying their own data could realistically CHANGE that value at
will, whether it is fixed / assigned by a system or process, or whether it
is a protected / sensitive personal attribute.

Classify each column into exactly one of:
  immutable — system-assigned, temporal, or derived; a person supplying
              their own data cannot alter it.
              Examples: "transaction_id" -> immutable, "created_at" -> immutable,
              "row_hash" -> immutable, "account_uuid" -> immutable,
              "credit_score" -> immutable, "last_login_timestamp" -> immutable
  protected — a sensitive/demographic attribute; legally or practically
              difficult for a person to falsify even though it is
              self-reported.
              Examples: "gender" -> protected, "ethnicity" -> protected,
              "disability_status" -> protected, "religion" -> protected
  mutable   — freely self-reported / user-controlled.
              Examples: "occupation" -> mutable, "education_level" -> mutable,
              "hours_worked_per_week" -> mutable, "marital_status" -> mutable

Classify these columns from a NEW dataset by the same reasoning, even if
the exact name never appeared above:
{columns}

Return ONLY a JSON object mapping each column name to one of "immutable",
"protected", "mutable". No prose, no markdown.
Example format: {{"col_a": "mutable", "col_b": "immutable"}}
"""


def _parse_mutability_json(text: str) -> Optional[Dict[str, str]]:
    """Best-effort JSON parse, tolerant of markdown code fences."""
    if not text:
        return None
    s = text.strip()
    for fence in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(fence, "")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _llm_classify_mutability(
    columns: List[str],
    cache_path: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, float]:
    """
    Classify column mutability via a few-shot LLM call, cached to disk.

    Returns a ``{col_name: score}`` dict covering only the columns the LLM
    successfully classified. Any column missing from the result (LLM
    unavailable, malformed response, column omitted by the model) is left
    to the caller to resolve via the keyword-based fallback
    (``_auto_mutability``); this function never raises on failure, it
    simply returns as much as it could classify (possibly ``{}``).
    """
    if cache_path and Path(cache_path).exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return {
                col: _MUTABILITY_LABEL_TO_SCORE[label]
                for col, label in cached.items()
                if label in _MUTABILITY_LABEL_TO_SCORE
            }
        except Exception:
            pass  # fall through to a fresh LLM call

    if not columns:
        return {}

    try:
        from gemini_safe import SafeGeminiModel
    except ImportError:
        return {}

    if not api_key:
        return {}

    try:
        model = SafeGeminiModel(
            api_key=api_key,
            generation_config={"temperature": 0, "max_output_tokens": 4096},
        )
        prompt = _MUTABILITY_PROMPT_TEMPLATE.format(columns="\n".join(columns))
        raw = model.generate_content_safe(prompt, timeout=180)
    except Exception:
        return {}

    parsed = _parse_mutability_json(raw)
    if not parsed:
        return {}

    labels = {
        col: label
        for col, label in parsed.items()
        if col in columns and label in _MUTABILITY_LABEL_TO_SCORE
    }

    if cache_path and labels:
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    return {col: _MUTABILITY_LABEL_TO_SCORE[label] for col, label in labels.items()}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_numeric_col(series: pd.Series) -> bool:
    """Return True if >_NUMERIC_THRESHOLD of non-null values parse as float."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
    return (numeric_count / len(non_null)) > _NUMERIC_THRESHOLD


def _auto_mutability(col_name: str) -> float:
    """
    Derive a mutability score from the column name alone.

    Rules (evaluated in priority order):
    1. Name contains an immutable keyword (case-insensitive substring) → 0.0
    2. Name contains a sensitive keyword (case-insensitive substring) → 0.5
    3. Otherwise → 1.0

    Returns
    -------
    float : 0.0, 0.5, or 1.0
    """
    lower = col_name.lower()
    # Remove separators so "edu_num3" → "edunum3" before checking
    stripped = re.sub(r"[-_ ]", "", lower)

    for kw in _IMMUTABLE_KEYWORDS:
        if kw in stripped:
            return 0.0

    for kw in _SENSITIVE_KEYWORDS:
        if kw in lower:          # sensitive check on the original lower name
            return 0.5

    return 1.0


def _auto_comprehensibility(col_name: str) -> float:
    """
    Derive a comprehensibility score from the column name alone.

    Rules (evaluated in order):

    1. Length **< 4** chars OR contains digits → 0.0
       (e.g. ``col1``, ``id``, ``num1``).
       Note: exactly-4-char plain English words such as ``race`` or ``age``
       are intentionally excluded from this rule (they pass to rule 2/4).
    2. Any token ≤ 3 chars OR no vowels in a token → 0.3
       (e.g. ``hrs-per-wk``, ``fnlwgt``, ``edu_num``).
    3. Multiple word tokens (space/hyphen/underscore separated) → 1.0
       (e.g. ``marital-status``, ``hours-per-week``).
    4. Single word ≥ 4 chars with all tokens having vowels → 0.8
       (e.g. ``education``, ``occupation``, ``race``).
    """
    # Rule 1: very short (< 4 chars) or contains a digit
    if len(col_name) < 4 or re.search(r"\d", col_name):
        return 0.0

    # Normalise separators to spaces and tokenise
    name = col_name.lower().replace("-", " ").replace("_", " ")
    tokens = name.split()

    # Rule 2: any token looks like an abbreviation (short or no vowels)
    if any(len(t) <= 3 or not re.search(r"[aeiou]", t) for t in tokens):
        return 0.3

    # Rule 3: multi-word phrase
    if len(tokens) > 1:
        return 1.0

    # Rule 4: single plain word ≥ 4 chars with vowels
    return 0.8


def _encode_target(series: pd.Series) -> np.ndarray:
    """
    Encode a target column to a 1-D float array for Spearman correlation.

    Categorical targets are ordinal-encoded (sorted unique values).
    Numerical targets are cast to float.
    """
    if _is_numeric_col(series):
        return pd.to_numeric(series, errors="coerce").values.astype(float)
    # Ordinal encoding: sort unique non-null values and assign 0,1,2,...
    unique_vals = sorted(series.dropna().unique(), key=str)
    mapping = {v: i for i, v in enumerate(unique_vals)}
    return series.map(mapping).values.astype(float)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class H7UserIncentive(BaseHeuristic):
    """
    Heuristic 7 — User Incentive (behavioral manipulation motivation).

    Answers: *Would a rational human actor understand, be able to change, and
    benefit from altering this column's value?*

    This is **behavioral motivation**, not statistical importance (H6).

    Parameters
    ----------
    None  (all configuration is passed to ``fit()``).

    Attributes
    ----------
    mutability_ : dict[str, float]
        Per-column mutability score in {0.0, 0.5, 1.0}.  Populated after
        ``fit()``.
    gain_direction_ : dict[str, float | dict[str, float]]
        Per-column gain-direction information.  For numerical columns: a
        scalar float in [0, 1].  For categorical columns: a nested dict
        mapping ``str(value) → 0.0 | 1.0``.  Populated after ``fit()``.
    comprehensibility_ : dict[str, float]
        Per-column comprehensibility score in [0, 1].  Populated after
        ``fit()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mutability_: Dict[str, float] = {}
        self.gain_direction_: Dict[str, Union[float, Dict[str, float]]] = {}
        self.comprehensibility_: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        target_col: Optional[str] = None,
        mutability_scores: Optional[Dict[str, float]] = None,
        comprehensibility_scores: Optional[Dict[str, float]] = None,
        mutability_cache_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> "H7UserIncentive":
        """
        Learn per-column behavioral-incentive features from the dataset.

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset (may contain errors).
        mask_df : pd.DataFrame
            Binary mask aligned with ``dirty_df``.  0 = clean cell,
            1 = erroneous cell.  Same shape and column names.
        target_col : str, optional
            Name of the outcome / label column used to compute gain direction.
            If ``None``, ``h7_gain_direction`` is set to 0.5 for all cells
            (neutral — not an error, just uninformative).
        mutability_scores : dict[str, float], optional
            User-supplied override for specific columns.  Values must be
            0.0, 0.5, or 1.0.  Takes priority over both the LLM classifier
            and the keyword fallback for any column present in this dict.
        comprehensibility_scores : dict[str, float], optional
            User-supplied override for specific columns.  Any float in [0,1].
            Columns absent from this dict fall back to the automatic
            name-analysis rule.
        mutability_cache_path : str, optional
            Path to cache/load the few-shot LLM mutability classification
            (one Gemini call per dataset schema, JSON-cached thereafter, the
            same discipline as the intent-signal rule extractor). If a
            cache file already exists there it is loaded and no LLM call is
            made. If ``None``, mutability is derived purely from the
            keyword-based fallback rule (``_auto_mutability``).
        gemini_api_key : str, optional
            API key for the mutability classifier. Required only on a cache
            miss when ``mutability_cache_path`` is set.

        Returns
        -------
        self
        """
        if dirty_df.shape != mask_df.shape:
            raise ValueError("dirty_df and mask_df must have the same shape.")
        if list(dirty_df.columns) != list(mask_df.columns):
            raise ValueError("dirty_df and mask_df must have the same column names.")
        if target_col is not None and target_col not in dirty_df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in dirty_df columns."
            )

        columns = list(dirty_df.columns)
        mutability_scores = mutability_scores or {}
        comprehensibility_scores = comprehensibility_scores or {}

        # ── 1. Mutability ──────────────────────────────────────────────
        # Priority: explicit user override > few-shot LLM classifier
        # (cached) > keyword-substring fallback.
        llm_mutability: Dict[str, float] = {}
        if mutability_cache_path is not None:
            unresolved = [c for c in columns if c not in mutability_scores]
            llm_mutability = _llm_classify_mutability(
                unresolved, mutability_cache_path, gemini_api_key
            )

        for col in columns:
            if col in mutability_scores:
                val = float(mutability_scores[col])
                if val not in (0.0, 0.5, 1.0):
                    raise ValueError(
                        f"mutability_scores['{col}'] = {val}; "
                        "must be one of 0.0, 0.5, or 1.0."
                    )
                self.mutability_[col] = val
            elif col in llm_mutability:
                self.mutability_[col] = llm_mutability[col]
            else:
                self.mutability_[col] = _auto_mutability(col)

        # ── 2. Comprehensibility ───────────────────────────────────────
        for col in columns:
            if col in comprehensibility_scores:
                self.comprehensibility_[col] = float(comprehensibility_scores[col])
            else:
                self.comprehensibility_[col] = _auto_comprehensibility(col)

        # ── 3. Gain direction ──────────────────────────────────────────
        self._fit_gain_direction(dirty_df, mask_df, columns, target_col)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # fit helpers
    # ------------------------------------------------------------------

    def _fit_gain_direction(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        columns: list,
        target_col: Optional[str],
    ) -> None:
        """
        Compute gain_direction_ for every column.

        If ``target_col`` is None → 0.5 (neutral) for all columns.

        For each feature column ``c``:
        - If ``c == target_col`` → 1.0  (direct outcome manipulation).
        - Numerical ``c``:   Spearman corr(c, encoded_target),
                             mapped [−1, 1] → [0, 1] via ``0.5 + 0.5 * r``.
        - Categorical ``c``: per-value lookup dict  {str_value → 0.0 | 1.0}
                             where 1.0 = value is in the top-50 % of values
                             by mean(encoded_target).
        """
        if target_col is None:
            for col in columns:
                self.gain_direction_[col] = 0.5
            return

        # Identify fully-clean rows (mask == 0 for ALL columns)
        clean_mask = (mask_df == 0).all(axis=1)
        clean_df = dirty_df[clean_mask].copy()

        if len(clean_df) < _MIN_CLEAN_ROWS:
            warnings.warn(
                f"H7UserIncentive: only {len(clean_df)} fully-clean rows "
                f"available (minimum {_MIN_CLEAN_ROWS}). "
                "Falling back to neutral gain_direction (0.5) for all columns.",
                UserWarning,
            )
            for col in columns:
                self.gain_direction_[col] = 0.5
            return

        # Encode target to a numeric array (works for both cat and num targets)
        encoded_target = _encode_target(clean_df[target_col])

        # Remove rows where target encoding produced NaN
        valid = ~np.isnan(encoded_target)
        if valid.sum() < _MIN_CLEAN_ROWS:
            for col in columns:
                self.gain_direction_[col] = 0.5
            return

        encoded_target = encoded_target[valid]
        clean_df = clean_df[valid]

        for col in columns:
            # Direct outcome column → maximum incentive
            if col == target_col:
                self.gain_direction_[col] = 1.0
                continue

            col_series = clean_df[col]

            if _is_numeric_col(col_series):
                self._fit_gain_numerical(col, col_series, encoded_target)
            else:
                self._fit_gain_categorical(col, col_series, encoded_target)

    def _fit_gain_numerical(
        self,
        col: str,
        col_series: pd.Series,
        encoded_target: np.ndarray,
    ) -> None:
        """
        Store a scalar gain_direction for a numerical column.

        Uses Spearman rank correlation mapped from [-1, 1] to [0, 1]:
            gain_direction = 0.5 + 0.5 * spearman_corr
        """
        col_vals = pd.to_numeric(col_series, errors="coerce").values.astype(float)
        valid = ~(np.isnan(col_vals) | np.isnan(encoded_target))
        if valid.sum() < 3:
            self.gain_direction_[col] = 0.5
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr, _ = spearmanr(col_vals[valid], encoded_target[valid])

        # spearmanr can return NaN when one array is constant
        if np.isnan(corr):
            corr = 0.0

        self.gain_direction_[col] = float(np.clip(0.5 + 0.5 * corr, 0.0, 1.0))

    def _fit_gain_categorical(
        self,
        col: str,
        col_series: pd.Series,
        encoded_target: np.ndarray,
    ) -> None:
        """
        Store a per-value dict gain_direction for a categorical column.

        For each unique string value ``v`` of the column:
        - Compute mean(encoded_target) over rows where col == v.
        - Sort values by this mean.
        - Top 50 % (by mean target) → 1.0 (favorable).
        - Bottom 50 % → 0.0 (unfavorable).

        The resulting dict maps ``str(value) → 0.0 | 1.0``.
        """
        col_vals = col_series.values
        unique_vals = [v for v in np.unique(col_vals) if v is not None
                       and not (isinstance(v, float) and np.isnan(v))]

        if not unique_vals:
            self.gain_direction_[col] = {}
            return

        # Mean encoded-target per unique value
        mean_target: Dict[str, float] = {}
        for v in unique_vals:
            rows_with_v = col_vals == v
            mt = float(encoded_target[rows_with_v].mean()) if rows_with_v.any() else 0.0
            mean_target[str(v)] = mt

        # Identify the top 50 % of values by mean target
        sorted_vals = sorted(mean_target.items(), key=lambda x: x[1], reverse=True)
        cutoff = max(1, len(sorted_vals) // 2)  # at least 1 value is "favorable"
        favorable = {v for v, _ in sorted_vals[:cutoff]}

        # Use ceiling division so the top half is never empty
        per_value: Dict[str, float] = {
            v: (1.0 if v in favorable else 0.0)
            for v in mean_target
        }
        self.gain_direction_[col] = per_value

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    def compute(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return H7 features for every erroneous cell (mask == 1).

        For each erroneous cell at (row_idx, col_name):

        - ``h7_mutability``       = ``self.mutability_[col_name]``
        - ``h7_gain_direction``
            - Numerical col: ``self.gain_direction_[col_name]`` (scalar)
            - Categorical col: ``self.gain_direction_[col_name].get(
              str(dirty_value), 0.5)``  ← **per-value lookup**
        - ``h7_comprehensibility`` = ``self.comprehensibility_[col_name]``

        Parameters
        ----------
        dirty_df : pd.DataFrame
        mask_df  : pd.DataFrame  (same shape as dirty_df, values in {0, 1})

        Returns
        -------
        pd.DataFrame with columns:
            row_idx               (int)
            col_name              (str)
            h7_mutability         (float ∈ {0.0, 0.5, 1.0})
            h7_gain_direction     (float ∈ [0, 1])
            h7_comprehensibility  (float ∈ [0, 1])
        One row per erroneous cell.
        """
        self._check_fitted()

        records = []
        error_positions = self._get_error_positions(mask_df)

        for row_idx, col_name in error_positions:
            mutability = self.mutability_.get(col_name, 1.0)
            comprehensibility = self.comprehensibility_.get(col_name, 0.5)

            gain_info = self.gain_direction_.get(col_name, 0.5)

            if isinstance(gain_info, dict):
                # Categorical column: look up the actual dirty cell value
                dirty_value = dirty_df.at[row_idx, col_name]
                gain_direction = gain_info.get(str(dirty_value), 0.5)
            else:
                # Numerical column (or neutral fallback): scalar
                gain_direction = float(gain_info)

            records.append({
                "row_idx": row_idx,
                "col_name": col_name,
                "h7_mutability": mutability,
                "h7_gain_direction": gain_direction,
                "h7_comprehensibility": comprehensibility,
            })

        if not records:
            return pd.DataFrame(columns=[
                "row_idx", "col_name",
                "h7_mutability", "h7_gain_direction", "h7_comprehensibility",
            ])

        return pd.DataFrame(records)[[
            "row_idx", "col_name",
            "h7_mutability", "h7_gain_direction", "h7_comprehensibility",
        ]]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from h7_user_incentive import H7UserIncentive  # type: ignore[import]

    print("=" * 60)
    print("H7UserIncentive — Self-test")
    print("=" * 60)

    np.random.seed(42)
    n = 200
    education_num = np.random.randint(5, 16, n)
    income = (education_num > 10).astype(int)
    fnlwgt = np.random.randint(100000, 500000, n)
    race = np.random.choice(["White", "Black", "Asian"], n)

    dirty = pd.DataFrame({
        "education-num": education_num,
        "fnlwgt":        fnlwgt,
        "race":          race,
        "income":        income,
    })
    mask = pd.DataFrame(np.zeros_like(dirty.values, dtype=int), columns=dirty.columns)
    mask.loc[[5, 10], "education-num"] = 1
    mask.loc[[20], "fnlwgt"] = 1
    mask.loc[[30], "race"] = 1

    h7 = H7UserIncentive()
    h7.fit(dirty, mask, target_col="income")
    result = h7.compute(dirty, mask)

    print("\nFitted mutability scores:")
    for col, s in h7.mutability_.items():
        print(f"  {col:20s}  {s}")

    print("\nFitted comprehensibility scores:")
    for col, s in h7.comprehensibility_.items():
        print(f"  {col:20s}  {s}")

    print("\nFitted gain_direction info:")
    for col, gd in h7.gain_direction_.items():
        print(f"  {col:20s}  {gd}")

    print("\nOutput (erroneous cells only):")
    print(result.to_string(index=False))

    # ── Assertions ─────────────────────────────────────────────────────
    print("\n--- Assertions ---")

    # education-num
    edu_rows = result[result["col_name"] == "education-num"].iloc[0]
    assert edu_rows["h7_mutability"] == 0.0, (
        f"education-num mutability should be 0.0 ('num' in name), "
        f"got {edu_rows['h7_mutability']}"
    )
    assert edu_rows["h7_gain_direction"] > 0.7, (
        f"education-num gain_direction should be > 0.7 (positively correlated "
        f"with income), got {edu_rows['h7_gain_direction']:.4f}"
    )
    assert edu_rows["h7_comprehensibility"] == 0.3, (
        f"education-num comprehensibility should be 0.3 (abbreviated token 'num'), "
        f"got {edu_rows['h7_comprehensibility']}"
    )

    # fnlwgt
    fnl_rows = result[result["col_name"] == "fnlwgt"].iloc[0]
    assert fnl_rows["h7_mutability"] == 0.0, (
        f"fnlwgt mutability should be 0.0 ('wgt' in name), "
        f"got {fnl_rows['h7_mutability']}"
    )
    assert 0.3 <= fnl_rows["h7_gain_direction"] <= 0.7, (
        f"fnlwgt gain_direction should be ≈ 0.5 (random, near-zero corr), "
        f"got {fnl_rows['h7_gain_direction']:.4f}"
    )
    # fnlwgt: 6 chars, no digits → rule 1 doesn't fire.
    # Single token "fnlwgt" has no vowels → rule 2 fires → 0.3.
    # (The ticket's inline comment says 0.0, but the spec's own code produces 0.3
    #  because rule 1 only checks len≤4 OR contains digits; fnlwgt has neither.)
    assert fnl_rows["h7_comprehensibility"] == 0.3, (
        f"fnlwgt comprehensibility should be 0.3 (no vowels → abbreviation rule), "
        f"got {fnl_rows['h7_comprehensibility']}"
    )

    # race
    race_rows = result[result["col_name"] == "race"].iloc[0]
    assert race_rows["h7_mutability"] == 0.5, (
        f"race mutability should be 0.5 (sensitive attribute), "
        f"got {race_rows['h7_mutability']}"
    )
    assert race_rows["h7_comprehensibility"] == 0.8, (
        f"race comprehensibility should be 0.8 (single plain word), "
        f"got {race_rows['h7_comprehensibility']}"
    )

    # Gain direction: categorical lookup must use the actual dirty value
    race_dirty_val = str(dirty.at[30, "race"])
    gd_info = h7.gain_direction_["race"]
    assert isinstance(gd_info, dict), "gain_direction_['race'] must be a per-value dict"
    expected_gd = gd_info.get(race_dirty_val, 0.5)
    assert race_rows["h7_gain_direction"] == expected_gd, (
        f"race gain_direction at row 30 should be {expected_gd} "
        f"(per-value lookup for '{race_dirty_val}'), "
        f"got {race_rows['h7_gain_direction']}"
    )

    # No target_col → all gain_direction = 0.5
    h7_no_target = H7UserIncentive()
    h7_no_target.fit(dirty, mask)
    result_no_target = h7_no_target.compute(dirty, mask)
    assert (result_no_target["h7_gain_direction"] == 0.5).all(), (
        "Without target_col, all h7_gain_direction must be 0.5"
    )

    # User-supplied overrides respected
    h7_override = H7UserIncentive()
    h7_override.fit(
        dirty, mask,
        target_col="income",
        mutability_scores={"race": 1.0, "fnlwgt": 0.0},
        comprehensibility_scores={"fnlwgt": 0.9},
    )
    assert h7_override.mutability_["race"] == 1.0, \
        "User override for race mutability not respected"
    assert h7_override.comprehensibility_["fnlwgt"] == 0.9, \
        "User override for fnlwgt comprehensibility not respected"

    print("✅ All assertions passed.")
    print(
        "\nExpected summary:\n"
        "  education-num : mutability=0.0 (num→immutable), "
        "gain_direction≈0.9 (positive corr), comprehensibility=0.3 (abbreviated)\n"
        "  fnlwgt        : mutability=0.0 (wgt→immutable), "
        "gain_direction≈0.5 (random noise), comprehensibility=0.0 (no vowels)\n"
        "  race          : mutability=0.5 (sensitive), "
        "gain_direction=per-value lookup, comprehensibility=0.8 (plain word)"
    )
