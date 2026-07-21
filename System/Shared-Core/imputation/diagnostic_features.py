#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Confidence-Weighted Diagnostic Feature Extractor
=================================================

Dataset-agnostic module that converts raw imputation results into the
diagnostic feature tuple used for intent classification.

For each corrupted cell, computes:
  - change_direction   (sign of corruption — most robust)
  - change_magnitude   (absolute difference — medium robust)
  - wrc                (weighted relative change — stabilized by sigma)
  - confidence         (1 / (1 + sigma_tree))
  - confidence_oob     (1 - OOB_error for the column)
  - feature_importance (from RF model metadata)

Handles numerical and categorical columns differently:
  - Numerical:    direction = sign(observed - imputed), magnitude = |diff|
  - Categorical:  changed = (observed != imputed), magnitude = edit distance
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class DiagnosticFeatureExtractor:
    """
    Extract confidence-weighted diagnostic features from imputation results.

    Parameters
    ----------
    alpha : float, default=1.0
        Dampening factor for sigma_tree in the WRC formula.
        Higher alpha → more aggressive suppression of uncertain imputations.
    epsilon : float, default=1e-10
        Small constant to avoid division by zero.
    """

    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10):
        self.alpha = alpha
        self.epsilon = epsilon

    def extract(
        self,
        imputation_results: pd.DataFrame,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute diagnostic features for all corrupted cells.

        Parameters
        ----------
        imputation_results : pd.DataFrame
            Output from MICEImputer.impute(). Required columns:
            row_idx, column, observed_value, imputed_value, intent_label,
            col_type, sigma_tree, sigma_oob, confidence, mice_rounds.

        feature_importances : dict, optional
            col_name -> importance score (from RF model metadata).
            If None, feature_importance will be set to 0.

        Returns
        -------
        features_df : pd.DataFrame
            One row per corrupted cell with all diagnostic features +
            metadata columns (row_idx, column, intent_label).
        """
        required_cols = [
            "row_idx", "column", "observed_value", "imputed_value",
            "intent_label", "col_type", "sigma_tree", "sigma_oob",
            "confidence", "mice_rounds",
        ]
        missing = set(required_cols) - set(imputation_results.columns)
        if missing:
            raise ValueError(f"Missing columns in imputation_results: {missing}")

        records = []

        for _, cell in imputation_results.iterrows():
            feat = self._compute_cell_features(cell, feature_importances)
            records.append(feat)

        features_df = pd.DataFrame(records)
        return features_df

    def _compute_cell_features(
        self,
        cell: pd.Series,
        feature_importances: Optional[Dict[str, float]],
    ) -> dict:
        """Compute diagnostic features for a single corrupted cell."""
        col_type = cell["col_type"]
        observed = cell["observed_value"]
        imputed = cell["imputed_value"]
        sigma_tree = cell["sigma_tree"]
        sigma_oob = cell["sigma_oob"]
        confidence = cell["confidence"]

        # Feature importance (if available)
        fi = 0.0
        if feature_importances is not None:
            fi = feature_importances.get(cell["column"], 0.0)

        # Column predictability = 1 - OOB_error
        confidence_oob = 1.0 - sigma_oob

        if col_type == "numerical":
            feat = self._numerical_features(observed, imputed, sigma_tree)
        else:
            feat = self._categorical_features(observed, imputed, sigma_tree)

        # Assemble full feature dict
        feat.update({
            # Metadata (for grouping / evaluation, not model features)
            "row_idx": cell["row_idx"],
            "column": cell["column"],
            "intent_label": cell["intent_label"],
            "col_type": col_type,
            "mice_rounds": cell["mice_rounds"],
            # Confidence features (model features)
            "sigma_tree": sigma_tree,
            "sigma_oob": sigma_oob,
            "confidence": confidence,
            "confidence_oob": confidence_oob,
            "feature_importance": fi,
            # Raw values (for inspection, not model features)
            "observed_value": observed,
            "imputed_value": imputed,
        })

        return feat

    def _numerical_features(
        self, observed, imputed, sigma_tree: float
    ) -> dict:
        """Compute direction, magnitude, WRC for a numerical cell."""
        try:
            obs_f = float(observed)
        except (ValueError, TypeError):
            obs_f = 0.0

        try:
            imp_f = float(imputed)
        except (ValueError, TypeError):
            imp_f = 0.0

        diff = obs_f - imp_f

        # Direction (most robust feature)
        direction = float(np.sign(diff)) if diff != 0 else 0.0

        # Magnitude (medium robust)
        magnitude = abs(diff)

        # Raw relative change (least robust — for ablation only)
        denom_raw = abs(imp_f) + self.epsilon
        raw_relative_change = magnitude / denom_raw

        # Weighted Relative Change (stabilized by sigma_tree)
        denom_wrc = abs(imp_f) * (1.0 + self.alpha * sigma_tree) + self.epsilon
        wrc = magnitude / denom_wrc

        return {
            "change_direction": direction,
            "change_magnitude": magnitude,
            "raw_relative_change": raw_relative_change,
            "wrc": wrc,
            "is_categorical": 0,
        }

    def _categorical_features(
        self, observed, imputed, sigma_tree: float
    ) -> dict:
        """Compute features for a categorical cell."""
        obs_str = str(observed).strip()
        imp_str = str(imputed).strip()

        # Changed or not
        changed = 0.0 if obs_str == imp_str else 1.0

        # Edit distance as magnitude proxy
        edit_dist = self._levenshtein(obs_str, imp_str)

        # Normalized edit distance
        max_len = max(len(obs_str), len(imp_str), 1)
        norm_edit_dist = edit_dist / max_len

        # WRC analog: dampened normalized edit distance
        wrc = norm_edit_dist / (1.0 + self.alpha * sigma_tree)

        return {
            "change_direction": changed,      # 1 = changed, 0 = same
            "change_magnitude": float(edit_dist),
            "raw_relative_change": norm_edit_dist,
            "wrc": wrc,
            "is_categorical": 1,
        }

    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if s1 == s2:
            return 0
        len1, len2 = len(s1), len(s2)
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1

        # Use two-row optimization
        prev = list(range(len2 + 1))
        curr = [0] * (len2 + 1)
        for i in range(1, len1 + 1):
            curr[0] = i
            for j in range(1, len2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr[j] = min(
                    curr[j - 1] + 1,      # insertion
                    prev[j] + 1,           # deletion
                    prev[j - 1] + cost,    # substitution
                )
            prev, curr = curr, prev
        return prev[len2]

    @staticmethod
    def get_model_feature_names() -> list:
        """
        Return the list of feature names that should be used as input
        to a downstream intent classifier. Excludes metadata columns.
        """
        return [
            "change_direction",
            "change_magnitude",
            "wrc",
            "confidence",
            "confidence_oob",
            "sigma_tree",
            "sigma_oob",
            "feature_importance",
            "is_categorical",
        ]
