#!/usr/bin/env python3
"""
Error Fingerprint — Feature Extraction
========================================

Extracts intent-discriminative features from ONLY:
  - The dirty dataset (with errors already in it)
  - A blind binary mask (which cells are erroneous)

NO clean/correct values are used.

Feature Groups:
  1. Value Plausibility   — is the value valid / a known vocabulary entry?
  2. Row Coherence        — does the value fit with the rest of the row?
  3. Error Pattern        — how many errors, which features, co-occurrence
  4. Value Distribution   — z-score, percentile, majority-class membership
  5. String-Level         — edit distance, typo patterns, obfuscation tokens
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings
import re

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_numeric_column(series: pd.Series) -> bool:
    """Check if a column is numeric (or can be cast to numeric)."""
    try:
        pd.to_numeric(series.dropna(), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _to_numeric_safe(val) -> Optional[float]:
    """Try to cast a value to float, return None on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            curr_row.append(min(ins, dele, sub))
        prev_row = curr_row
    return prev_row[-1]


# Common keyboard adjacency pairs for typo detection
_KEYBOARD_NEIGHBORS = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etfs", "t": "rygs",
    "y": "tuhs", "u": "yijs", "i": "uoks", "o": "ipls", "p": "o",
    "a": "qwsz", "s": "wedxza", "d": "erfcxs", "f": "rtgvcd",
    "g": "tyhbvf", "h": "yujnbg", "j": "uikmnh", "k": "iolmj",
    "l": "opk", "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb",
    "b": "vghn", "n": "bhjm", "m": "njk",
}

OBFUSCATION_TOKENS = {
    "unknown", "n/a", "na", "—", "-", "?", "none", "null", "nan",
    "not available", "not specified", "other", "unspecified",
}


def _has_typo_pattern(value: str) -> float:
    """
    Heuristic: does the string look like it contains a typo?
    Returns a score in [0, 1].
    Checks: transpositions, keyboard adjacency substitutions, repeated chars.
    """
    v = str(value).lower().strip()
    if len(v) <= 2:
        return 0.0
    score = 0.0
    # Check for adjacent-char transpositions
    for i in range(len(v) - 1):
        if i + 2 < len(v) and v[i] == v[i + 2] and v[i] != v[i + 1]:
            score += 0.3  # possible transposition
    # Check for repeated chars (stuttering)
    for i in range(len(v) - 2):
        if v[i] == v[i + 1] == v[i + 2]:
            score += 0.3
    return min(score, 1.0)


def _is_obfuscation(value: str) -> int:
    """Check if value is an obfuscation token."""
    return int(str(value).lower().strip() in OBFUSCATION_TOKENS)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Feature Extractor
# ─────────────────────────────────────────────────────────────────────────────

class ErrorFingerprintExtractor:
    """
    Extracts features for intent attribution using ONLY dirty data + blind mask.

    Parameters
    ----------
    dirty_df : pd.DataFrame
        The dirty dataset (N rows × P columns). May contain errors.
    mask_df : pd.DataFrame
        Blind binary mask (N rows × P columns). 1 = erroneous, 0 = clean.
        Must have the same column names as dirty_df (or a subset).
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.column_stats_: Dict = {}
        self.column_vocabs_: Dict = {}
        self.column_types_: Dict = {}
        self.coherence_models_: Dict = {}
        self.feature_columns_: List[str] = []
        self._fitted = False

    def fit(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame) -> "ErrorFingerprintExtractor":
        """
        Learn column statistics and coherence models from the dirty dataset.

        Uses ALL rows (both clean and erroneous) to build:
        - Column vocabularies (valid values per column)
        - Column statistics (mean, std, percentiles)
        - Coherence models (predict each column from others)
        """
        # Identify feature columns (shared between dirty and mask)
        self.feature_columns_ = [c for c in mask_df.columns
                                 if c in dirty_df.columns and c != "is_erroneous"]

        if self.verbose:
            print(f"Feature columns: {self.feature_columns_}")
            print(f"Dataset: {len(dirty_df)} rows, {len(self.feature_columns_)} features")

        # ── Step 1: Detect column types ──
        for col in self.feature_columns_:
            self.column_types_[col] = (
                "numeric" if _is_numeric_column(dirty_df[col]) else "categorical"
            )

        if self.verbose:
            n_num = sum(1 for v in self.column_types_.values() if v == "numeric")
            n_cat = sum(1 for v in self.column_types_.values() if v == "categorical")
            print(f"Column types: {n_num} numeric, {n_cat} categorical")

        # ── Step 2: Column statistics ──
        for col in self.feature_columns_:
            vals = dirty_df[col].dropna()
            if self.column_types_[col] == "numeric":
                nums = pd.to_numeric(vals, errors="coerce").dropna()
                self.column_stats_[col] = {
                    "mean": float(nums.mean()) if len(nums) > 0 else 0.0,
                    "std": max(float(nums.std()), 1e-6) if len(nums) > 1 else 1.0,
                    "median": float(nums.median()) if len(nums) > 0 else 0.0,
                    "p5": float(nums.quantile(0.05)) if len(nums) > 0 else 0.0,
                    "p95": float(nums.quantile(0.95)) if len(nums) > 0 else 0.0,
                    "min": float(nums.min()) if len(nums) > 0 else 0.0,
                    "max": float(nums.max()) if len(nums) > 0 else 0.0,
                }
            else:
                freq = vals.astype(str).str.strip().str.lower().value_counts(normalize=True)
                self.column_stats_[col] = {
                    "frequencies": freq.to_dict(),
                    "top_value": freq.index[0] if len(freq) > 0 else "",
                    "n_unique": len(freq),
                }

        # ── Step 3: Column vocabularies (for categorical) ──
        for col in self.feature_columns_:
            if self.column_types_[col] == "categorical":
                vocab = set(dirty_df[col].dropna().astype(str).str.strip().unique())
                self.column_vocabs_[col] = vocab
            else:
                self.column_vocabs_[col] = set()

        # ── Step 4: Train coherence models ──
        # For each column, train a model to predict it from ALL OTHER columns.
        # We use clean rows ONLY (mask=0 in ALL columns) to train, so the
        # model learns normal patterns.  But crucially, we're learning from
        # the dirty dataset — we just filter to rows with no flagged errors.
        self._train_coherence_models(dirty_df, mask_df)

        self._fitted = True
        return self

    def _train_coherence_models(self, dirty_df: pd.DataFrame, mask_df: pd.DataFrame):
        """
        Train one model per column to predict that column from the others.
        Trained on rows where NO cell is flagged (all-clean rows in the dirty dataset).
        """
        # Find rows where the mask is all-zero (clean rows)
        mask_cols = [c for c in self.feature_columns_ if c in mask_df.columns]
        clean_mask = (mask_df[mask_cols] == 0).all(axis=1)
        n_clean = clean_mask.sum()

        if self.verbose:
            print(f"\nTraining coherence models on {n_clean} clean rows "
                  f"(from {len(dirty_df)} total)...")

        if n_clean < 20:
            if self.verbose:
                print("  WARNING: Too few clean rows for coherence models. "
                      "Using ALL rows instead.")
            clean_mask = pd.Series(True, index=dirty_df.index)
            n_clean = len(dirty_df)

        df_train = dirty_df.loc[clean_mask, self.feature_columns_].copy()

        # Encode all columns for sklearn
        # Build a unified encoded frame
        encoded = pd.DataFrame(index=df_train.index)
        self._encoders = {}

        for col in self.feature_columns_:
            if self.column_types_[col] == "numeric":
                encoded[col] = pd.to_numeric(df_train[col], errors="coerce").fillna(0)
            else:
                le = LabelEncoder()
                encoded[col] = le.fit_transform(df_train[col].astype(str).fillna("__MISSING__"))
                self._encoders[col] = le

        # Also encode the FULL dirty dataset for inference later
        self._full_encoded = pd.DataFrame(index=dirty_df.index)
        for col in self.feature_columns_:
            if self.column_types_[col] == "numeric":
                self._full_encoded[col] = pd.to_numeric(
                    dirty_df[col], errors="coerce"
                ).fillna(0)
            else:
                le = self._encoders[col]
                vals = dirty_df[col].astype(str).fillna("__MISSING__")
                # Handle unseen labels
                known = set(le.classes_)
                vals_safe = vals.apply(lambda x: x if x in known else "__MISSING__")
                if "__MISSING__" not in known:
                    le.classes_ = np.append(le.classes_, "__MISSING__")
                self._full_encoded[col] = le.transform(vals_safe)

        # Train one model per column
        for col in self.feature_columns_:
            other_cols = [c for c in self.feature_columns_ if c != col]
            X = encoded[other_cols].values
            y = encoded[col].values

            if self.column_types_[col] == "numeric":
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=8,
                    min_samples_leaf=5, random_state=42, n_jobs=-1,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    min_samples_leaf=5, random_state=42, n_jobs=-1,
                    class_weight="balanced",
                )

            model.fit(X, y)
            self.coherence_models_[col] = model

            if self.verbose:
                oob = ""
                print(f"  {col} ({self.column_types_[col]}): trained on {len(X)} rows")

    def extract_features(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract features for every erroneous cell.

        Returns a DataFrame with one row per erroneous cell, containing:
        - Identifiers: row_idx, column_name
        - All feature groups (plausibility, coherence, pattern, distribution, string)

        Parameters
        ----------
        dirty_df : pd.DataFrame
            The dirty dataset.
        mask_df : pd.DataFrame
            Blind binary mask (same shape/columns as dirty_df subset).
        """
        assert self._fitted, "Must call fit() first."

        records = []
        mask_cols = [c for c in self.feature_columns_ if c in mask_df.columns]

        # Precompute full-dataset encoded frame for coherence scoring
        full_enc = self._full_encoded

        # Error pattern: per-row counts
        row_error_counts = mask_df[mask_cols].sum(axis=1)

        for row_idx in range(len(dirty_df)):
            for col in mask_cols:
                if mask_df.iloc[row_idx][col] != 1:
                    continue

                dirty_val = dirty_df.iloc[row_idx][col]
                dirty_str = str(dirty_val).strip() if pd.notna(dirty_val) else ""
                is_numeric = self.column_types_[col] == "numeric"

                feat = {
                    "row_idx": row_idx,
                    "column_name": col,
                    "is_numeric": int(is_numeric),
                }

                # ── Group 1: Value Plausibility ──
                feat.update(self._plausibility_features(col, dirty_val, dirty_str, is_numeric))

                # ── Group 2: Row Coherence ──
                feat.update(self._coherence_features(col, row_idx, dirty_val, is_numeric, full_enc))

                # ── Group 3: Error Pattern ──
                feat.update(self._pattern_features(row_idx, col, mask_df, mask_cols, row_error_counts))

                # ── Group 4: Value Distribution ──
                feat.update(self._distribution_features(col, dirty_val, is_numeric))

                # ── Group 5: String-Level ──
                feat.update(self._string_features(col, dirty_str, is_numeric))

                records.append(feat)

        df = pd.DataFrame(records)
        if self.verbose:
            print(f"\nExtracted {len(df)} feature vectors for {len(df)} erroneous cells")
            print(f"Features per cell: {len(df.columns) - 2}")  # minus identifiers
        return df

    # ─────────────────────────────────────────────────────────────────────
    #  Feature Group 1: Value Plausibility
    # ─────────────────────────────────────────────────────────────────────

    def _plausibility_features(
        self, col: str, dirty_val, dirty_str: str, is_numeric: bool
    ) -> Dict:
        feat = {}
        if is_numeric:
            num_val = _to_numeric_safe(dirty_val)
            stats = self.column_stats_[col]
            if num_val is not None:
                feat["plaus_in_range"] = int(stats["p5"] <= num_val <= stats["p95"])
                feat["plaus_in_minmax"] = int(stats["min"] <= num_val <= stats["max"])
            else:
                # Non-parseable value in a numeric column → strong unintentional signal
                feat["plaus_in_range"] = 0
                feat["plaus_in_minmax"] = 0
        else:
            vocab = self.column_vocabs_[col]
            feat["plaus_in_vocab"] = int(dirty_str in vocab)
            # Minimum edit distance to any valid vocab entry
            if len(vocab) > 0 and dirty_str not in vocab:
                min_dist = min(_edit_distance(dirty_str.lower(), v.lower())
                              for v in vocab)
                feat["plaus_min_edit_dist"] = min_dist
            else:
                feat["plaus_min_edit_dist"] = 0

        return feat

    # ─────────────────────────────────────────────────────────────────────
    #  Feature Group 2: Row Coherence
    # ─────────────────────────────────────────────────────────────────────

    def _coherence_features(
        self, col: str, row_idx: int, dirty_val, is_numeric: bool,
        full_enc: pd.DataFrame,
    ) -> Dict:
        feat = {}
        model = self.coherence_models_.get(col)
        if model is None:
            feat["coher_predicted_match"] = 0
            feat["coher_confidence"] = 0.0
            feat["coher_residual"] = 0.0
            return feat

        other_cols = [c for c in self.feature_columns_ if c != col]
        x = full_enc.iloc[row_idx][other_cols].values.reshape(1, -1)

        if is_numeric:
            predicted = model.predict(x)[0]
            observed = full_enc.iloc[row_idx][col]
            # How close is the dirty value to what the model expects?
            residual = abs(observed - predicted)
            col_std = self.column_stats_[col]["std"]
            feat["coher_residual_norm"] = residual / (col_std + 1e-6)
            feat["coher_predicted_match"] = int(residual < col_std)

            # Tree-level uncertainty
            tree_preds = np.array([t.predict(x)[0] for t in model.estimators_])
            feat["coher_confidence"] = 1.0 / (1.0 + float(np.std(tree_preds)))
        else:
            predicted_class = model.predict(x)[0]
            observed_class = full_enc.iloc[row_idx][col]
            feat["coher_predicted_match"] = int(predicted_class == observed_class)

            # Probability of the observed class
            proba = model.predict_proba(x)[0]
            classes = model.classes_
            if observed_class in classes:
                idx = list(classes).index(observed_class)
                feat["coher_confidence"] = float(proba[idx])
            else:
                feat["coher_confidence"] = 0.0

            feat["coher_residual_norm"] = 1.0 - feat["coher_confidence"]

        return feat

    # ─────────────────────────────────────────────────────────────────────
    #  Feature Group 3: Error Pattern
    # ─────────────────────────────────────────────────────────────────────

    def _pattern_features(
        self, row_idx: int, col: str, mask_df: pd.DataFrame,
        mask_cols: List[str], row_error_counts: pd.Series,
    ) -> Dict:
        feat = {}
        # How many errors in this row?
        feat["pat_n_errors_in_row"] = int(row_error_counts.iloc[row_idx])

        # Which features are erroneous in this row? (as a set indicator)
        row_mask = mask_df.iloc[row_idx]
        erroneous_cols = [c for c in mask_cols if row_mask[c] == 1]

        # Feature identity (which column is this error in?)
        for c in self.feature_columns_:
            feat[f"pat_col_{c}"] = int(c == col)

        # Are co-dependent features changed together?
        # This checks if related features are both flagged
        feat["pat_multi_error"] = int(len(erroneous_cols) > 1)

        return feat

    # ─────────────────────────────────────────────────────────────────────
    #  Feature Group 4: Value Distribution
    # ─────────────────────────────────────────────────────────────────────

    def _distribution_features(
        self, col: str, dirty_val, is_numeric: bool
    ) -> Dict:
        feat = {}
        if is_numeric:
            num_val = _to_numeric_safe(dirty_val)
            stats = self.column_stats_[col]
            if num_val is not None:
                feat["dist_z_score"] = (num_val - stats["mean"]) / stats["std"]
                feat["dist_abs_z"] = abs(feat["dist_z_score"])
                rng = stats["max"] - stats["min"]
                feat["dist_percentile_approx"] = (
                    (num_val - stats["min"]) / rng if rng > 0 else 0.5
                )
            else:
                feat["dist_z_score"] = 0.0
                feat["dist_abs_z"] = 0.0
                feat["dist_percentile_approx"] = 0.0
        else:
            dirty_str = str(dirty_val).strip().lower()
            freqs = self.column_stats_[col].get("frequencies", {})
            feat["dist_value_freq"] = freqs.get(dirty_str, 0.0)
            # Is this the most common value?
            top = self.column_stats_[col].get("top_value", "")
            feat["dist_is_majority"] = int(dirty_str == top)
            # Frequency rank (0 = most common)
            sorted_vals = sorted(freqs.items(), key=lambda x: -x[1])
            rank = next(
                (i for i, (v, _) in enumerate(sorted_vals) if v == dirty_str),
                len(sorted_vals),
            )
            feat["dist_freq_rank"] = rank
            feat["dist_n_unique_ratio"] = (
                1.0 / self.column_stats_[col].get("n_unique", 1)
            )

        return feat

    # ─────────────────────────────────────────────────────────────────────
    #  Feature Group 5: String-Level
    # ─────────────────────────────────────────────────────────────────────

    def _string_features(
        self, col: str, dirty_str: str, is_numeric: bool
    ) -> Dict:
        feat = {}
        if is_numeric:
            # Minimal string features for numeric columns
            feat["str_is_obfuscation"] = _is_obfuscation(dirty_str)
            feat["str_typo_score"] = 0.0
            feat["str_len_anomaly"] = 0.0
        else:
            feat["str_is_obfuscation"] = _is_obfuscation(dirty_str)
            feat["str_typo_score"] = _has_typo_pattern(dirty_str)

            # String length compared to column median length
            vocab = self.column_vocabs_[col]
            if len(vocab) > 0:
                median_len = np.median([len(v) for v in vocab])
                feat["str_len_anomaly"] = abs(len(dirty_str) - median_len) / (median_len + 1)
            else:
                feat["str_len_anomaly"] = 0.0

            # Contains digits in a text column?
            feat["str_has_digits"] = int(bool(re.search(r'\d', dirty_str)))

            # Contains special characters?
            feat["str_has_special"] = int(bool(re.search(r'[^\w\s\-\.]', dirty_str)))

        return feat


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def extract_error_fingerprints(
    dirty_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    One-call convenience: fit extractor + extract features.

    Parameters
    ----------
    dirty_df : DataFrame (N × P)
    mask_df  : DataFrame (N × P), values in {0, 1}

    Returns
    -------
    DataFrame with one row per erroneous cell and all fingerprint features.
    """
    extractor = ErrorFingerprintExtractor(verbose=verbose)
    extractor.fit(dirty_df, mask_df)
    return extractor.extract_features(dirty_df, mask_df)
