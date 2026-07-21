#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Imputation Pipeline
================================

Dataset-agnostic orchestrator that:
1. Loads any dataset that follows the (data + mask) contract
2. Runs MICE imputation with dual confidence
3. Extracts diagnostic features
4. Optionally evaluates imputation quality against oracle
5. Saves results

Usage:
    python -m src.imputation.run_pipeline \\
        --data  datasets/my_dataset/labeled_dataset.csv \\
        --mask  datasets/my_dataset/ground_truth_masks.csv \\
        --outdir outputs/imputation/my_dataset/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .imputation_estimator import MICEImputer
from .diagnostic_features import DiagnosticFeatureExtractor


class ImputationPipeline:
    """
    End-to-end pipeline: data loading → imputation → feature extraction → save.

    Parameters
    ----------
    n_estimators : int
        Trees per RF model.
    max_rounds : int
        Max MICE iterations for multi-corruption rows.
    alpha : float
        WRC dampening factor.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_rounds: int = 5,
        alpha: float = 1.0,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.imputer = MICEImputer(
            n_estimators=n_estimators,
            max_rounds=max_rounds,
            random_state=random_state,
            verbose=verbose,
        )
        self.feature_extractor = DiagnosticFeatureExtractor(alpha=alpha)
        self.verbose = verbose

        # Stored results
        self.imputation_results_: Optional[pd.DataFrame] = None
        self.diagnostic_features_: Optional[pd.DataFrame] = None
        self.summary_: Optional[dict] = None

    def run(
        self,
        data_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        correct_data_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run the full pipeline on in-memory DataFrames.

        Parameters
        ----------
        data_df : pd.DataFrame, shape (N, P)
            Feature-only data (no 'is_erroneous' column).
        mask_df : pd.DataFrame, shape (N, P)
            Mask: 0=clean, ±1=corrupted. Same columns as data_df.
        correct_data_df : pd.DataFrame, shape (N, P), optional
            Ground truth correct values aligned row-for-row with data_df.
            When provided, a ``correct_value`` column is written to
            imputation_results.csv so imputation accuracy can be measured
            directly per cell.

        Returns
        -------
        diagnostic_features : pd.DataFrame
            One row per corrupted cell with all diagnostic features.
        """
        # Step 1: Fit imputer on clean rows
        self.imputer.fit(data_df, mask_df)

        # Step 2: Impute corrupted cells
        self.imputation_results_ = self.imputer.impute(
            data_df, mask_df, correct_data_df=correct_data_df
        )

        # Step 3: Gather feature importances from all per-column models
        feature_importances = self._aggregate_feature_importances()

        # Step 4: Extract diagnostic features
        self.diagnostic_features_ = self.feature_extractor.extract(
            self.imputation_results_,
            feature_importances=feature_importances,
        )

        # Step 5: Compute summary statistics
        self.summary_ = self._compute_summary(data_df, mask_df)

        if self.verbose:
            self._print_summary()

        return self.diagnostic_features_

    def run_from_files(
        self,
        data_path: str,
        mask_path: str,
        correct_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load CSVs, run the pipeline, and optionally save results.

        Parameters
        ----------
        data_path : str
            Path to data CSV. If it contains 'is_erroneous', that column
            is dropped automatically.
        mask_path : str
            Path to mask CSV.
        correct_data_path : str, optional
            Path to a CSV with the same columns as data_path containing the
            ground truth correct values (one row per row in data_path).
            When provided, ``correct_value`` is populated in imputation_results.
        output_dir : str, optional
            If provided, save results to this directory.

        Returns
        -------
        diagnostic_features : pd.DataFrame
        """
        data_df, mask_df = self._load_data(data_path, mask_path)

        correct_data_df = None
        if correct_data_path is not None:
            correct_data_df = pd.read_csv(correct_data_path)
            # Drop is_erroneous if present
            if "is_erroneous" in correct_data_df.columns:
                correct_data_df = correct_data_df.drop(columns=["is_erroneous"])
            # Align to data columns
            correct_data_df = correct_data_df[[c for c in mask_df.columns if c in correct_data_df.columns]]
            if self.verbose:
                print(f"  Correct data: {correct_data_path}  →  {correct_data_df.shape}")

        features = self.run(data_df, mask_df, correct_data_df=correct_data_df)

        if output_dir is not None:
            self.save_results(output_dir)

        return features

    def save_results(self, output_dir: str):
        """Save imputation results, diagnostic features, and summary."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.imputation_results_ is not None:
            path = out / "imputation_results.csv"
            self.imputation_results_.to_csv(path, index=False)
            if self.verbose:
                print(f"  Saved: {path}")

        if self.diagnostic_features_ is not None:
            path = out / "diagnostic_features.csv"
            self.diagnostic_features_.to_csv(path, index=False)
            if self.verbose:
                print(f"  Saved: {path}")

        if self.summary_ is not None:
            path = out / "pipeline_summary.json"
            # Convert numpy types for JSON serialization
            summary_serializable = self._to_serializable(self.summary_)
            with open(path, "w") as f:
                json.dump(summary_serializable, f, indent=2)
            if self.verbose:
                print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self, data_path: str, mask_path: str):
        """Load and validate CSVs."""
        data_df = pd.read_csv(data_path)
        mask_df = pd.read_csv(mask_path)

        # Drop 'is_erroneous' if present (it's a label, not a feature)
        if "is_erroneous" in data_df.columns:
            data_df = data_df.drop(columns=["is_erroneous"])

        # Ensure columns match
        if list(data_df.columns) != list(mask_df.columns):
            # Try to align by using mask columns
            common = [c for c in mask_df.columns if c in data_df.columns]
            if len(common) == len(mask_df.columns):
                data_df = data_df[common]
            else:
                raise ValueError(
                    f"Cannot align data ({list(data_df.columns)}) "
                    f"with mask ({list(mask_df.columns)})"
                )

        if self.verbose:
            print(f"\n{'='*70}")
            print("LOADING DATA")
            print(f"{'='*70}")
            print(f"Data:  {data_path}  →  {data_df.shape}")
            print(f"Mask:  {mask_path}  →  {mask_df.shape}")

        return data_df, mask_df

    def _aggregate_feature_importances(self) -> Dict[str, float]:
        """
        Compute a per-column importance score.

        For each column j, its importance = mean of how important j is
        as a predictor in OTHER columns' models.
        """
        importances = {col: 0.0 for col in self.imputer.feature_columns_}
        counts = {col: 0 for col in self.imputer.feature_columns_}

        for target_col, model in self.imputer.models_.items():
            predictor_cols = [
                c for c in self.imputer.feature_columns_ if c != target_col
            ]
            fi = model.feature_importances_
            for i, col in enumerate(predictor_cols):
                importances[col] += fi[i]
                counts[col] += 1

        # Average
        for col in importances:
            if counts[col] > 0:
                importances[col] /= counts[col]

        return importances

    def _compute_summary(
        self, data_df: pd.DataFrame, mask_df: pd.DataFrame
    ) -> dict:
        """Compute pipeline summary statistics."""
        n_total = len(data_df)
        clean_mask = (mask_df == 0).all(axis=1)
        n_clean = clean_mask.sum()
        n_dirty = (~clean_mask).sum()

        corrupted_per_row = (mask_df[~clean_mask] != 0).sum(axis=1)
        total_corrupted_cells = (mask_df != 0).sum().sum()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset": {
                "total_rows": int(n_total),
                "n_columns": int(data_df.shape[1]),
                "n_clean_rows": int(n_clean),
                "n_dirty_rows": int(n_dirty),
                "clean_ratio": float(n_clean / n_total),
                "total_corrupted_cells": int(total_corrupted_cells),
                "corrupted_per_row_distribution": {
                    str(k): int(v)
                    for k, v in corrupted_per_row.value_counts().sort_index().items()
                },
            },
            "column_types": self.imputer.get_column_types(),
            "oob_errors": {
                col: round(err, 6) for col, err in self.imputer.get_oob_errors().items()
            },
            "imputer_config": {
                "n_estimators": self.imputer.n_estimators,
                "max_rounds": self.imputer.max_rounds,
                "convergence_tol": self.imputer.convergence_tol,
                "alpha": self.feature_extractor.alpha,
            },
        }

        if self.diagnostic_features_ is not None:
            df = self.diagnostic_features_
            summary["diagnostic_stats"] = {
                "n_cells": len(df),
                "mean_confidence": float(df["confidence"].mean()),
                "mean_sigma_tree": float(df["sigma_tree"].mean()),
                "mean_sigma_oob": float(df["sigma_oob"].mean()),
                "mean_wrc": float(df["wrc"].mean()),
                "intent_label_distribution": {
                    str(k): int(v)
                    for k, v in df["intent_label"].value_counts().items()
                },
            }

        return summary

    def _print_summary(self):
        """Print a human-readable summary."""
        if self.summary_ is None:
            return
        s = self.summary_
        ds = s["dataset"]

        print(f"\n{'='*70}")
        print("PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"Rows:              {ds['total_rows']} total, "
              f"{ds['n_clean_rows']} clean, {ds['n_dirty_rows']} dirty")
        print(f"Corrupted cells:   {ds['total_corrupted_cells']}")
        print(f"Corruptions/row:   {ds['corrupted_per_row_distribution']}")

        if "diagnostic_stats" in s:
            diag = s["diagnostic_stats"]
            print(f"\nDiagnostic Features ({diag['n_cells']} cells):")
            print(f"  Mean confidence:   {diag['mean_confidence']:.4f}")
            print(f"  Mean sigma_tree:   {diag['mean_sigma_tree']:.4f}")
            print(f"  Mean sigma_oob:    {diag['mean_sigma_oob']:.4f}")
            print(f"  Mean WRC:          {diag['mean_wrc']:.4f}")
            print(f"  Intent labels:     {diag['intent_label_distribution']}")

        print(f"\nOOB errors per column:")
        for col, err in s["oob_errors"].items():
            print(f"  {col:30s}  {err:.4f}")

    @staticmethod
    def _to_serializable(obj):
        """Recursively convert numpy types to Python native for JSON."""
        if isinstance(obj, dict):
            return {k: ImputationPipeline._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [ImputationPipeline._to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run MICE imputation pipeline for intent attribution."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to data CSV (labeled_dataset or combined_dataset)."
    )
    parser.add_argument(
        "--mask", required=True,
        help="Path to mask CSV (ground_truth_masks)."
    )
    parser.add_argument(
        "--correct", default=None,
        help=(
            "Path to a CSV with ground truth correct values (same shape as --data). "
            "When provided, a correct_value column is added to imputation_results.csv."
        ),
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory for results. If not set, results are printed only."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100,
        help="Number of trees per RF model (default: 100)."
    )
    parser.add_argument(
        "--max-rounds", type=int, default=5,
        help="Max MICE iterations (default: 5)."
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0,
        help="WRC dampening factor (default: 1.0)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )

    args = parser.parse_args()

    pipeline = ImputationPipeline(
        n_estimators=args.n_estimators,
        max_rounds=args.max_rounds,
        alpha=args.alpha,
        random_state=args.seed,
        verbose=True,
    )

    features = pipeline.run_from_files(
        data_path=args.data,
        mask_path=args.mask,
        correct_data_path=args.correct,
        output_dir=args.outdir,
    )

    print(f"\n✓ Done. {len(features)} diagnostic feature vectors produced.")

    if args.outdir:
        print(f"  Results saved to: {args.outdir}")


if __name__ == "__main__":
    main()
