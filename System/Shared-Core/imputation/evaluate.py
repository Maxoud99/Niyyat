#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imputation Quality & Intent Attribution Evaluator
===================================================

Evaluates the imputation pipeline from three complementary angles:

A. Imputation Quality
   - Column-level RMSE / accuracy vs. oracle (held-out clean rows)
   - Rank correlation between true and imputed values (numerical)
   - Categorical hit rate: % of cells where imputed == oracle category

B. Diagnostic Feature Separability
   - Per-feature AUC (can this feature alone separate intent labels?)
   - Per-feature Mann-Whitney U test (are distributions different?)
   - Confidence-stratified F1: does filtering by confidence improve results?

C. Intent Classification (RF on diagnostic features)
   - Accuracy, Precision, Recall, F1 (macro + per-class)
   - Confusion matrix
   - Ablation study across 5 feature configurations:
       (1) Oracle upper bound       – exact original values
       (2) Full imputed features    – all diagnostic features
       (3) Direction + confidence   – most robust subset
       (4) No confidence weighting  – raw relative change only
       (5) Naive baseline           – no imputation; column stats only

D. Confidence Calibration
   - Binned confidence vs. actual imputation accuracy (reliability diagram)
   - Expected Calibration Error (ECE)

Usage:
    python -m src.imputation.evaluate \\
        --features  outputs/imputation/adult_income_v3/diagnostic_features.csv \\
        --data      datasets/adult_income_v3/labeled_dataset_v3.csv \\
        --mask      datasets/adult_income_v3/ground_truth_masks_v3.csv \\
        --outdir    outputs/imputation/adult_income_v3/evaluation/
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ======================================================================
# A. Imputation Quality
# ======================================================================

class ImputationQualityEvaluator:
    """
    Evaluate imputation quality by holding out a subset of clean rows
    and comparing imputed vs. actual values.
    """

    def evaluate(
        self,
        data_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        imputation_results: pd.DataFrame,
        holdout_frac: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        """
        Evaluate imputation accuracy on held-out clean rows.

        Strategy: treat a fraction of clean rows as if they were corrupted
        (mask each column one-at-a-time), impute, then compare to the
        known true value.

        Parameters
        ----------
        data_df : pd.DataFrame
            Feature-only dataset (no is_erroneous).
        mask_df : pd.DataFrame
            Original mask (to identify clean rows).
        imputation_results : pd.DataFrame
            Output of MICEImputer.impute() — for the actual dirty rows.
        holdout_frac : float
            Fraction of clean rows to hold out for quality assessment.

        Returns
        -------
        quality : dict
            Per-column and aggregate quality metrics.
        """
        from .imputation_estimator import MICEImputer

        # Identify clean rows
        clean_mask = (mask_df == 0).all(axis=1)
        clean_idx = data_df.index[clean_mask].tolist()

        rng = np.random.default_rng(random_state)
        n_holdout = max(1, int(len(clean_idx) * holdout_frac))
        holdout_idx = rng.choice(clean_idx, size=n_holdout, replace=False).tolist()
        train_idx = [i for i in clean_idx if i not in holdout_idx]

        # Build train/holdout splits using only clean rows
        data_train = data_df.loc[train_idx]
        mask_train = mask_df.loc[train_idx]
        data_holdout = data_df.loc[holdout_idx]

        # Train a fresh imputer on the reduced training set
        imputer = MICEImputer(verbose=False)
        # Create a mask of zeros for the training slice
        mask_train_zeros = pd.DataFrame(
            0, index=mask_train.index, columns=mask_train.columns
        )
        imputer.fit(data_train, mask_train_zeros)

        col_quality = {}

        for col in data_df.columns:
            # Build a single-column mask
            single_mask = pd.DataFrame(
                0, index=data_holdout.index, columns=data_df.columns
            )
            single_mask[col] = 1  # treat col as "corrupted"

            # Impute on holdout using train models
            # Combine train + holdout for impute call
            combined_data = pd.concat([data_train, data_holdout])
            combined_mask = pd.concat([mask_train_zeros, single_mask])

            results = imputer.impute(combined_data, combined_mask)
            col_results = results[results["column"] == col]

            if len(col_results) == 0:
                continue

            true_vals = data_holdout[col].values
            imp_vals = col_results["imputed_value"].values

            col_type = imputer.col_types_[col]

            if col_type == "numerical":
                try:
                    true_f = true_vals.astype(float)
                    imp_f = np.array([float(v) for v in imp_vals])
                    rmse = float(np.sqrt(np.mean((true_f - imp_f) ** 2)))
                    mae = float(np.mean(np.abs(true_f - imp_f)))
                    # Normalized RMSE (relative to std of true values)
                    std = np.std(true_f)
                    nrmse = rmse / std if std > 0 else float("inf")
                    # Spearman rank correlation
                    corr, pval = stats.spearmanr(true_f, imp_f)
                    col_quality[col] = {
                        "type": "numerical",
                        "n": len(true_f),
                        "rmse": rmse,
                        "mae": mae,
                        "nrmse": nrmse,
                        "spearman_r": float(corr),
                        "spearman_p": float(pval),
                    }
                except Exception:
                    pass
            else:
                true_s = [str(v).strip() for v in true_vals]
                imp_s = [str(v).strip() for v in imp_vals]
                n = len(true_s)
                hit_rate = sum(t == p for t, p in zip(true_s, imp_s)) / n if n > 0 else 0
                col_quality[col] = {
                    "type": "categorical",
                    "n": n,
                    "hit_rate": hit_rate,
                }

        # Aggregate
        num_rmses = [v["nrmse"] for v in col_quality.values() if v["type"] == "numerical" and np.isfinite(v["nrmse"])]
        cat_hits = [v["hit_rate"] for v in col_quality.values() if v["type"] == "categorical"]

        aggregate = {
            "numerical_mean_nrmse": float(np.mean(num_rmses)) if num_rmses else None,
            "categorical_mean_hit_rate": float(np.mean(cat_hits)) if cat_hits else None,
            "n_numerical_cols": len(num_rmses),
            "n_categorical_cols": len(cat_hits),
        }

        return {"per_column": col_quality, "aggregate": aggregate}


# ======================================================================
# B. Diagnostic Feature Separability
# ======================================================================

class FeatureSeparabilityEvaluator:
    """
    Test whether each diagnostic feature can distinguish intentional
    from unintentional errors on its own.
    """

    FEATURE_COLS = [
        "change_direction",
        "change_magnitude",
        "raw_relative_change",
        "wrc",
        "confidence",
        "confidence_oob",
        "sigma_tree",
        "sigma_oob",
        "feature_importance",
        "is_categorical",
    ]

    def evaluate(self, features_df: pd.DataFrame) -> Dict:
        """
        Compute per-feature AUC, Mann-Whitney U, and Spearman correlation
        with the intent label.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of DiagnosticFeatureExtractor.extract().

        Returns
        -------
        separability : dict
            Per-feature statistics.
        """
        y = features_df["intent_label"].values.astype(float)
        # Map -1/1 to 0/1 for AUC
        y_binary = (y == 1).astype(int)

        results = {}

        for feat in self.FEATURE_COLS:
            if feat not in features_df.columns:
                continue
            x = features_df[feat].fillna(0).values.astype(float)

            # AUC
            try:
                auc = roc_auc_score(y_binary, x)
                # Flip if AUC < 0.5 (feature is inversely predictive)
                if auc < 0.5:
                    auc = 1.0 - auc
            except Exception:
                auc = None

            # Mann-Whitney U test
            group_pos = x[y_binary == 1]
            group_neg = x[y_binary == 0]
            if len(group_pos) > 0 and len(group_neg) > 0:
                mw_stat, mw_pval = stats.mannwhitneyu(
                    group_pos, group_neg, alternative="two-sided"
                )
            else:
                mw_stat, mw_pval = None, None

            # Spearman correlation with label
            corr, pval = stats.spearmanr(x, y)

            # Distribution summary per class
            def _summary(arr):
                return {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "std": float(np.std(arr)),
                }

            results[feat] = {
                "auc": float(auc) if auc is not None else None,
                "mann_whitney_stat": float(mw_stat) if mw_stat is not None else None,
                "mann_whitney_pval": float(mw_pval) if mw_pval is not None else None,
                "spearman_r": float(corr),
                "spearman_p": float(pval),
                "intentional": _summary(group_pos) if len(group_pos) > 0 else {},
                "unintentional": _summary(group_neg) if len(group_neg) > 0 else {},
            }

        return results


# ======================================================================
# C. Intent Classification + Ablation
# ======================================================================

class IntentClassificationEvaluator:
    """
    Train RF classifiers on diagnostic features and evaluate with
    stratified 5-fold cross-validation.

    Runs 5 ablation configurations:
      (1) Full features (proposed method)
      (2) Direction + confidence (most robust subset)
      (3) No confidence weighting (raw RC only)
      (4) Naive baseline (no imputation — observed value + column stats)
      (5) Oracle upper bound (if oracle data available)
    """

    CONFIGS = {
        "full": [
            "change_direction", "change_magnitude", "wrc",
            "confidence", "confidence_oob", "sigma_tree",
            "sigma_oob", "feature_importance", "is_categorical",
        ],
        "direction_confidence": [
            "change_direction", "confidence", "confidence_oob", "is_categorical",
        ],
        "no_confidence": [
            "change_direction", "change_magnitude", "raw_relative_change", "is_categorical",
        ],
        "naive_baseline": [
            "observed_numeric", "col_mean", "col_std",
            "col_median", "z_score", "is_categorical",
        ],
    }

    def evaluate(
        self,
        features_df: pd.DataFrame,
        data_df: Optional[pd.DataFrame] = None,
        mask_df: Optional[pd.DataFrame] = None,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> Dict:
        """
        Run cross-validated intent classification for all ablation configs.

        Parameters
        ----------
        features_df : pd.DataFrame
            Diagnostic features from DiagnosticFeatureExtractor.
        data_df : pd.DataFrame, optional
            Original data — needed for naive baseline column statistics.
        mask_df : pd.DataFrame, optional
            Original mask — needed for naive baseline.
        n_folds : int
            Number of stratified CV folds.

        Returns
        -------
        results : dict
            Per-config metrics.
        """
        y = features_df["intent_label"].values.astype(int)
        unique_labels = np.unique(y)
        results = {}

        # Single-class detection: classification is meaningless
        if len(unique_labels) < 2:
            label_name = "intentional" if unique_labels[0] == 1 else "unintentional"
            results["_single_class"] = {
                "note": (f"Only one intent class present ({label_name}, "
                         f"label={unique_labels[0]}). Classification ablation "
                         f"is not meaningful — skipping all configs."),
                "unique_labels": unique_labels.tolist(),
                "n_samples": len(y),
            }
            return results

        # Augment with naive baseline features if data provided
        features_aug = self._augment_naive_features(features_df, data_df, mask_df)

        for config_name, feature_list in self.CONFIGS.items():
            available = [f for f in feature_list if f in features_aug.columns]
            if len(available) < 2:
                results[config_name] = {"error": "Not enough features available"}
                continue

            X = features_aug[available].fillna(0).values.astype(float)
            metrics = self._cross_validate(X, y, n_folds, random_state)
            metrics["features_used"] = available
            results[config_name] = metrics

        return results

    def _cross_validate(
        self, X: np.ndarray, y: np.ndarray, n_folds: int, random_state: int
    ) -> Dict:
        """Run stratified K-fold CV and aggregate metrics."""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        fold_metrics = {
            "accuracy": [], "precision_intentional": [], "recall_intentional": [],
            "f1_intentional": [], "precision_unintentional": [], "recall_unintentional": [],
            "f1_unintentional": [], "f1_macro": [],
        }
        all_y_true, all_y_pred = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_metrics["f1_macro"].append(f1_score(y_test, y_pred, average="macro", labels=[-1, 1]))

            for label, name in [(-1, "unintentional"), (1, "intentional")]:
                if label in y_test:
                    fold_metrics[f"precision_{name}"].append(
                        precision_score(y_test, y_pred, pos_label=label, zero_division=0)
                    )
                    fold_metrics[f"recall_{name}"].append(
                        recall_score(y_test, y_pred, pos_label=label, zero_division=0)
                    )
                    fold_metrics[f"f1_{name}"].append(
                        f1_score(y_test, y_pred, pos_label=label, zero_division=0)
                    )

        # Aggregate confusion matrix
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        cm = confusion_matrix(all_y_true, all_y_pred, labels=[-1, 1])

        def _mean_std(lst):
            if not lst:
                return {"mean": None, "std": None}
            return {"mean": float(np.mean(lst)), "std": float(np.std(lst))}

        return {
            "accuracy": _mean_std(fold_metrics["accuracy"]),
            "f1_macro": _mean_std(fold_metrics["f1_macro"]),
            "f1_intentional": _mean_std(fold_metrics["f1_intentional"]),
            "f1_unintentional": _mean_std(fold_metrics["f1_unintentional"]),
            "precision_intentional": _mean_std(fold_metrics["precision_intentional"]),
            "recall_intentional": _mean_std(fold_metrics["recall_intentional"]),
            "precision_unintentional": _mean_std(fold_metrics["precision_unintentional"]),
            "recall_unintentional": _mean_std(fold_metrics["recall_unintentional"]),
            "confusion_matrix": cm.tolist(),
            "n_samples": len(all_y_true),
            "n_folds": n_folds,
        }

    def _augment_naive_features(
        self,
        features_df: pd.DataFrame,
        data_df: Optional[pd.DataFrame],
        mask_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Add naive baseline features: observed value + column-level statistics.
        These features do NOT use imputation — they answer:
        'Can you distinguish intent from the corrupted value alone?'
        """
        aug = features_df.copy()

        if data_df is None or mask_df is None:
            # Cannot compute naive features without original data
            return aug

        # Build column statistics from clean rows
        clean_mask = (mask_df == 0).all(axis=1)
        data_clean = data_df[clean_mask]

        col_stats = {}
        for col in data_df.columns:
            if data_df[col].dtype in ("object", "category"):
                col_stats[col] = {"mean": 0.0, "std": 1.0, "median": 0.0}
            else:
                vals = pd.to_numeric(data_clean[col], errors="coerce").dropna()
                col_stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()) if vals.std() > 0 else 1.0,
                    "median": float(vals.median()),
                }

        obs_numeric, col_mean, col_std, col_median, z_score = [], [], [], [], []

        for _, row in features_df.iterrows():
            col = row["column"]
            obs = row["observed_value"]
            try:
                obs_f = float(obs)
            except (ValueError, TypeError):
                obs_f = 0.0

            stats_c = col_stats.get(col, {"mean": 0.0, "std": 1.0, "median": 0.0})
            z = (obs_f - stats_c["mean"]) / stats_c["std"]

            obs_numeric.append(obs_f)
            col_mean.append(stats_c["mean"])
            col_std.append(stats_c["std"])
            col_median.append(stats_c["median"])
            z_score.append(z)

        aug["observed_numeric"] = obs_numeric
        aug["col_mean"] = col_mean
        aug["col_std"] = col_std
        aug["col_median"] = col_median
        aug["z_score"] = z_score

        return aug


# ======================================================================
# D. Confidence Calibration
# ======================================================================

class ConfidenceCalibrationEvaluator:
    """
    Assess whether the confidence score c_j = 1/(1+sigma_tree) is
    calibrated with respect to actual imputation correctness.

    Bins cells by confidence and measures the actual direction-correctness
    (i.e., was change_direction correct relative to oracle?) per bin.
    """

    def evaluate(
        self,
        features_df: pd.DataFrame,
        n_bins: int = 10,
    ) -> Dict:
        """
        Compute binned calibration statistics.

        We use 'direction accuracy' as a proxy for imputation correctness:
        - A cell is 'direction-correct' if its change_direction matches
          the dominant direction for its intent label in the dataset.
          For intentional (1): direction >0 is expected.
          For unintentional (-1): any direction is plausible.

        Parameters
        ----------
        features_df : pd.DataFrame
            Diagnostic features.
        n_bins : int
            Number of confidence bins.

        Returns
        -------
        calibration : dict
            Per-bin stats + Expected Calibration Error.
        """
        df = features_df.copy()

        # Bin by confidence
        df["conf_bin"] = pd.cut(
            df["confidence"],
            bins=np.linspace(0, 1, n_bins + 1),
            labels=[f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}" for i in range(n_bins)],
            include_lowest=True,
        )

        # Use sigma_tree as the primary calibration metric:
        # in each bin, compute mean sigma_tree and mean |change_direction|
        bins_stats = []
        ece = 0.0
        total = len(df)

        for bin_label in df["conf_bin"].cat.categories:
            subset = df[df["conf_bin"] == bin_label]
            if len(subset) == 0:
                continue

            mean_conf = subset["confidence"].mean()
            mean_sigma_tree = subset["sigma_tree"].mean()
            mean_magnitude = subset["change_magnitude"].mean()
            mean_wrc = subset["wrc"].mean()
            n = len(subset)
            intent_dist = subset["intent_label"].value_counts().to_dict()

            # ECE contribution: |mean_conf - (fraction of non-zero directions)|
            frac_changed = (subset["change_direction"] != 0).mean()
            ece += (n / total) * abs(mean_conf - frac_changed)

            bins_stats.append({
                "bin": str(bin_label),
                "n": n,
                "mean_confidence": float(mean_conf),
                "mean_sigma_tree": float(mean_sigma_tree),
                "mean_magnitude": float(mean_magnitude),
                "mean_wrc": float(mean_wrc),
                "fraction_changed": float(frac_changed),
                "intent_distribution": {str(k): int(v) for k, v in intent_dist.items()},
            })

        return {
            "bins": bins_stats,
            "ece": float(ece),
            "n_bins": n_bins,
        }


# ======================================================================
# Main Orchestrator
# ======================================================================

class ImputationEvaluator:
    """
    Runs all four evaluation blocks and saves results.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def evaluate(
        self,
        features_df: pd.DataFrame,
        data_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run full evaluation suite.

        Parameters
        ----------
        features_df : pd.DataFrame
            Output of DiagnosticFeatureExtractor.
        data_df : pd.DataFrame
            Feature-only dataset (no is_erroneous).
        mask_df : pd.DataFrame
            Mask file.
        output_dir : str, optional
            Where to save results.

        Returns
        -------
        all_results : dict
        """
        all_results = {"timestamp": datetime.now().isoformat()}

        # A. Imputation Quality
        if self.verbose:
            print(f"\n{'='*70}")
            print("  A. IMPUTATION QUALITY (held-out clean rows)")
            print(f"{'='*70}")
        iq_eval = ImputationQualityEvaluator()
        from .imputation_estimator import MICEImputer
        dummy_results = pd.DataFrame()  # Quality eval handles this internally
        quality = iq_eval.evaluate(data_df, mask_df, dummy_results)
        all_results["imputation_quality"] = quality
        if self.verbose:
            self._print_quality(quality)

        # B. Feature Separability
        if self.verbose:
            print(f"\n{'='*70}")
            print("  B. DIAGNOSTIC FEATURE SEPARABILITY")
            print(f"{'='*70}")
        sep_eval = FeatureSeparabilityEvaluator()
        separability = sep_eval.evaluate(features_df)
        all_results["feature_separability"] = separability
        if self.verbose:
            self._print_separability(separability)

        # C. Intent Classification Ablation
        if self.verbose:
            print(f"\n{'='*70}")
            print("  C. INTENT CLASSIFICATION ABLATION (5-fold CV)")
            print(f"{'='*70}")
        clf_eval = IntentClassificationEvaluator()
        classification = clf_eval.evaluate(features_df, data_df, mask_df)
        all_results["intent_classification"] = classification
        if self.verbose:
            self._print_classification(classification)

        # D. Confidence Calibration
        if self.verbose:
            print(f"\n{'='*70}")
            print("  D. CONFIDENCE CALIBRATION")
            print(f"{'='*70}")
        cal_eval = ConfidenceCalibrationEvaluator()
        calibration = cal_eval.evaluate(features_df)
        all_results["confidence_calibration"] = calibration
        if self.verbose:
            self._print_calibration(calibration)

        # Save
        if output_dir is not None:
            self._save(all_results, output_dir)

        return all_results

    # ------------------------------------------------------------------
    # Print helpers
    # ------------------------------------------------------------------

    def _print_quality(self, quality: dict):
        pc = quality["per_column"]
        agg = quality["aggregate"]
        print(f"\n  Aggregate:")
        if agg["numerical_mean_nrmse"] is not None:
            print(f"    Numerical   mean NRMSE:    {agg['numerical_mean_nrmse']:.4f}  "
                  f"(lower is better; 1.0 = same as guessing mean)")
        if agg["categorical_mean_hit_rate"] is not None:
            print(f"    Categorical mean hit rate: {agg['categorical_mean_hit_rate']:.4f}  "
                  f"(higher is better)")
        print(f"\n  Per-column:")
        for col, m in sorted(pc.items(), key=lambda x: x[0]):
            if m["type"] == "numerical":
                quality_tag = "✓" if m["nrmse"] < 1.0 else "✗"
                print(f"    {quality_tag} {col:30s}  RMSE={m['rmse']:12.2f}  "
                      f"NRMSE={m['nrmse']:.4f}  Spearman_r={m['spearman_r']:.4f}")
            else:
                quality_tag = "✓" if m["hit_rate"] > 0.5 else "✗"
                print(f"    {quality_tag} {col:30s}  hit_rate={m['hit_rate']:.4f}")

    def _print_separability(self, separability: dict):
        print(f"\n  {'Feature':<30s}  {'AUC':>6}  {'MW_pval':>10}  "
              f"{'Spearman_r':>10}  {'Intent mean':>12}  {'Uninten mean':>12}")
        print(f"  {'-'*84}")
        for feat, m in sorted(separability.items(), key=lambda x: -(x[1].get("auc") or 0)):
            auc = f"{m['auc']:.4f}" if m["auc"] is not None else "  N/A"
            pval = f"{m['mann_whitney_pval']:.2e}" if m["mann_whitney_pval"] is not None else "  N/A"
            sr = f"{m['spearman_r']:.4f}"
            i_mean = f"{m.get('intentional', {}).get('mean', float('nan')):.4f}"
            u_mean = f"{m.get('unintentional', {}).get('mean', float('nan')):.4f}"
            sig = "**" if (m["mann_whitney_pval"] or 1) < 0.01 else \
                  "*" if (m["mann_whitney_pval"] or 1) < 0.05 else "  "
            print(f"  {feat:<30s}  {auc:>6}  {pval:>10}  {sr:>10}  {i_mean:>12}  {u_mean:>12}  {sig}")
        print(f"\n  ** p<0.01  * p<0.05")

    def _print_classification(self, classification: dict):
        # Handle single-class datasets
        if "_single_class" in classification:
            info = classification["_single_class"]
            print(f"\n  ⚠ {info['note']}")
            print(f"    Samples: {info['n_samples']}")
            return

        print(f"\n  {'Config':<25s}  {'Acc':>6}  {'F1_macro':>8}  "
              f"{'F1_intent':>9}  {'F1_uninten':>10}  {'Features':>5}")
        print(f"  {'-'*70}")
        for config, m in classification.items():
            if "error" in m:
                print(f"  {config:<25s}  ERROR: {m['error']}")
                continue
            acc = f"{m['accuracy']['mean']:.4f}±{m['accuracy']['std']:.4f}"
            f1m = f"{m['f1_macro']['mean']:.4f}"
            f1i = f"{m['f1_intentional']['mean']:.4f}" if m['f1_intentional']['mean'] is not None else "   N/A"
            f1u = f"{m['f1_unintentional']['mean']:.4f}" if m['f1_unintentional']['mean'] is not None else "   N/A"
            nf = len(m.get("features_used", []))
            print(f"  {config:<25s}  {acc}  {f1m:>8}  {f1i:>9}  {f1u:>10}  {nf:>5}")

    def _print_calibration(self, calibration: dict):
        print(f"\n  ECE (Expected Calibration Error): {calibration['ece']:.4f}  "
              f"(lower is better; 0.0 = perfect)")
        print(f"\n  Confidence bin  |  n    | mean_conf | σ_tree    | frac_changed")
        print(f"  {'-'*65}")
        for b in calibration["bins"]:
            print(f"  {b['bin']:15s}  |  {b['n']:4d} | "
                  f"{b['mean_confidence']:.4f}    | "
                  f"{b['mean_sigma_tree']:9.2f} | "
                  f"{b['fraction_changed']:.4f}")

    def _save(self, all_results: dict, output_dir: str):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        path = out / "evaluation_results.json"
        with open(path, "w") as f:
            json.dump(self._to_serializable(all_results), f, indent=2)

        if self.verbose:
            print(f"\n  Saved: {path}")

        # Also save classification ablation as CSV for easy reading
        clf = all_results.get("intent_classification", {})
        rows = []
        for config, m in clf.items():
            if config.startswith("_") or "error" in m or "accuracy" not in m:
                continue
            rows.append({
                "config": config,
                "accuracy_mean": m["accuracy"]["mean"],
                "accuracy_std": m["accuracy"]["std"],
                "f1_macro_mean": m["f1_macro"]["mean"],
                "f1_macro_std": m["f1_macro"]["std"],
                "f1_intentional_mean": m["f1_intentional"]["mean"],
                "f1_intentional_std": m["f1_intentional"]["std"],
                "f1_unintentional_mean": m["f1_unintentional"]["mean"],
                "f1_unintentional_std": m["f1_unintentional"]["std"],
                "n_features": len(m.get("features_used", [])),
            })
        if rows:
            ablation_path = out / "ablation_results.csv"
            pd.DataFrame(rows).to_csv(ablation_path, index=False)
            if self.verbose:
                print(f"  Saved: {ablation_path}")

    @staticmethod
    def _to_serializable(obj):
        if isinstance(obj, dict):
            return {k: ImputationEvaluator._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ImputationEvaluator._to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate imputation quality and diagnostic features."
    )
    parser.add_argument("--features", required=True,
                        help="Path to diagnostic_features.csv")
    parser.add_argument("--data", required=True,
                        help="Path to data CSV (labeled_dataset or combined_dataset).")
    parser.add_argument("--mask", required=True,
                        help="Path to mask CSV (ground_truth_masks).")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for results.")
    args = parser.parse_args()

    # Load files
    features_df = pd.read_csv(args.features)
    data_df = pd.read_csv(args.data)
    mask_df = pd.read_csv(args.mask)

    # Drop is_erroneous if present
    if "is_erroneous" in data_df.columns:
        data_df = data_df.drop(columns=["is_erroneous"])

    evaluator = ImputationEvaluator(verbose=True)
    evaluator.evaluate(
        features_df=features_df,
        data_df=data_df,
        mask_df=mask_df,
        output_dir=args.outdir,
    )


if __name__ == "__main__":
    main()
