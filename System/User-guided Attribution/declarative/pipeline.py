#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Declarative Attribution Pipeline
===================================

Runs the full intent-attribution pipeline using declarative (user-defined)
integrity constraints as features.

Supports three scenarios:
  C        — constraint features only                (Family 3 standalone)
  B+C      — fingerprint B features + constraint features  (Family 1 + 3)
  (B+)+C   — fingerprint B+ features + constraint features (Family 1 + 2 + 3)

For C-only the pipeline is entirely self-contained.
For B+C and (B+)+C, pass a pre-computed ``external_features_df`` produced by
the fingerprint pipeline (same row_idx / column_name index).

Pipeline steps (same as Scenario B / B+):
  1. Build feature matrix from constraint evaluator (+ optional external features)
  2. Cluster erroneous cells on the feature matrix
  3. Proportional 1% label sampling from clusters
  4. Train Random Forest on sampled labels
  5. Predict on remaining cells
  6. Evaluate against ground truth
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from evaluator import ConstraintEvaluator

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class DeclarativePipeline:
    """
    Intent attribution pipeline driven by declarative integrity constraints.

    Parameters
    ----------
    constraints : List[Dict]
        Extracted constraints (from extractor.extract_constraints()).
    sampling_rate : float
        Fraction of erroneous cells to label from clusters (default 0.01 = 1%).
    clustering : str
        'dbscan', 'hdbscan', or 'kmeans'.
    n_clusters : int
        Number of clusters for KMeans (ignored for HDBSCAN).
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        constraints: List[Dict],
        sampling_rate: float = 0.01,
        clustering: str = "hdbscan",
        n_clusters: int = 15,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.constraints    = constraints
        self.sampling_rate  = sampling_rate
        self.clustering     = clustering
        self.n_clusters     = n_clusters
        self.random_state   = random_state
        self.verbose        = verbose

        self.evaluator  = ConstraintEvaluator(constraints)
        self.scaler     = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        ground_truth_mask: Optional[pd.DataFrame] = None,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Execute the full pipeline.

        Parameters
        ----------
        dirty_df : DataFrame
            Dirty dataset.
        mask_df : DataFrame
            Binary error mask (1 = erroneous cell).
        ground_truth_mask : DataFrame, optional
            Intent mask (-1 / 1) for evaluation. If omitted, no metrics computed.
        external_features_df : DataFrame, optional
            Pre-computed features from the fingerprint (B or B+) pipeline.
            Must contain row_idx and column_name columns.
            When provided, these features are concatenated with constraint
            features → implements B+C or (B+)+C scenarios.

        Returns
        -------
        dict with keys: features_df, predictions, metrics, feature_importance, elapsed
        """
        np.random.seed(self.random_state)
        t0 = time.time()

        # ── Step 1: Extract constraint features ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 1: EXTRACTING DECLARATIVE CONSTRAINT FEATURES")
            print("=" * 70)

        features_df = self.evaluator.extract_features(dirty_df, mask_df)

        if len(features_df) == 0:
            print("[pipeline] No erroneous cells found.")
            return {"features_df": features_df, "predictions": None,
                    "metrics": None, "feature_importance": None, "elapsed": 0.0}

        # ── Step 2: Merge with external features (B+C / (B+)+C) ──
        if external_features_df is not None:
            if self.verbose:
                ext_feat_cols = [c for c in external_features_df.columns
                                 if c not in ("row_idx", "column_name", "intent_label")]
                print(f"\n[pipeline] Merging {len(ext_feat_cols)} external feature(s) "
                      f"(B/B+ fingerprint) with constraint features.")
            # Keep only feature columns from external (drop labels if present)
            ext_drop = [c for c in ("intent_label",) if c in external_features_df.columns]
            ext = external_features_df.drop(columns=ext_drop, errors="ignore")
            features_df = features_df.merge(ext, on=["row_idx", "column_name"], how="inner")
            if self.verbose:
                print(f"[pipeline] Combined feature matrix: {features_df.shape[1] - 2} features")

        # ── Step 3: Attach ground-truth labels ──
        if ground_truth_mask is not None:
            features_df = self._attach_labels(features_df, ground_truth_mask)

        label_col = "intent_label" if "intent_label" in features_df.columns else None

        # ── Step 4: Build feature matrix ──
        id_cols   = ["row_idx", "column_name"]
        skip_cols = id_cols + ([label_col] if label_col else [])
        feat_cols = [c for c in features_df.columns if c not in skip_cols]

        X        = features_df[feat_cols].fillna(0).values.astype(float)
        X_scaled = self.scaler.fit_transform(X)

        if self.verbose:
            print(f"\n[pipeline] Feature matrix: {X_scaled.shape[0]} cells × "
                  f"{X_scaled.shape[1]} features")
            print(f"           Features: {feat_cols}")

        # ── Step 5: Cluster ──
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"STEP 2: CLUSTERING ({self.clustering.upper()})")
            print("=" * 70)

        cluster_labels = self._cluster(X_scaled)

        # ── Step 6: Proportional sampling ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 3: PROPORTIONAL LABEL SAMPLING (1%)")
            print("=" * 70)

        train_idx, test_idx = self._proportional_sample(
            cluster_labels, features_df, label_col
        )

        # ── Step 7: Train RF ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 4: TRAINING RANDOM FOREST")
            print("=" * 70)

        X_train  = X[train_idx]
        y_train  = features_df.iloc[train_idx][label_col].values
        X_test   = X[test_idx]
        y_test   = features_df.iloc[test_idx][label_col].values if label_col else None

        self.classifier.fit(X_train, y_train)
        if self.verbose:
            print(f"  Samples — train: {len(X_train)}  test: {len(X_test)}")
            print(f"  Training distribution: {dict(Counter(y_train))}")

        # ── Step 8: Predict ──
        y_pred = self.classifier.predict(X_test)
        features_df.loc[features_df.index[test_idx], "predicted_intent"] = y_pred

        # ── Step 9: Evaluate ──
        metrics = None
        if label_col and y_test is not None:
            if self.verbose:
                print("\n" + "=" * 70)
                print("STEP 5: EVALUATION")
                print("=" * 70)
            metrics = self._evaluate(y_test, y_pred)

        # ── Feature importance ──
        importance = pd.DataFrame({
            "feature":    feat_cols,
            "importance": self.classifier.feature_importances_,
        }).sort_values("importance", ascending=False)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n[pipeline] Done in {elapsed:.1f}s")
            print("\nTop 10 features:")
            print(importance.head(10).to_string(index=False))

        return {
            "features_df":       features_df,
            "predictions":       y_pred,
            "metrics":           metrics,
            "feature_importance": importance,
            "elapsed":           elapsed,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _attach_labels(
        self,
        features_df: pd.DataFrame,
        gt_mask: pd.DataFrame,
    ) -> pd.DataFrame:
        labels = []
        for _, row in features_df.iterrows():
            r_idx = int(row["row_idx"])
            col   = row["column_name"]
            if col in gt_mask.columns and r_idx < len(gt_mask):
                labels.append(int(gt_mask.iloc[r_idx][col]))
            else:
                labels.append(0)
        features_df = features_df.copy()
        features_df["intent_label"] = labels
        features_df = features_df[features_df["intent_label"] != 0].copy()
        if self.verbose:
            dist = Counter(features_df["intent_label"])
            print(f"\n[pipeline] Ground truth: {dict(dist)}")
        return features_df

    def _cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.clustering == "dbscan":
            nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
            dists, _ = nn.kneighbors(X_scaled)
            eps = max(float(np.percentile(np.sort(dists[:, -1]), 90)), 1e-6)
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
            n_cl = len(set(labels)) - (1 if -1 in labels else 0)
            if self.verbose:
                print(f"  DBSCAN: eps={eps:.3f}, {n_cl} clusters, "
                      f"{(labels == -1).sum()} noise points")
            return labels

        if self.clustering == "hdbscan" and HDBSCAN_AVAILABLE:
            try:
                min_cs = max(5, len(X_scaled) // 200)
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cs, min_samples=3, core_dist_n_jobs=-1
                )
                labels = clusterer.fit_predict(X_scaled)
                n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
                if self.verbose:
                    print(f"  HDBSCAN: {n_cl} clusters, {(labels == -1).sum()} noise points")
                return labels
            except Exception as e:
                print(f"  HDBSCAN failed ({e}) — falling back to KMeans")

        km     = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        if self.verbose:
            print(f"  KMeans: {self.n_clusters} clusters")
        return labels

    def _proportional_sample(
        self,
        cluster_labels: np.ndarray,
        features_df: pd.DataFrame,
        label_col: Optional[str],
    ):
        n_total = len(features_df)
        target  = max(int(round(n_total * self.sampling_rate)), 2)

        valid_mask   = cluster_labels != -1
        valid_indices = np.where(valid_mask)[0]
        valid_labels  = cluster_labels[valid_mask]

        if len(valid_indices) == 0:
            idx = np.arange(n_total)
            np.random.shuffle(idx)
            return idx[:target].tolist(), idx[target:].tolist()

        unique_labels = np.unique(valid_labels)
        cluster_sizes = {lb: (valid_labels == lb).sum() for lb in unique_labels}
        total_valid   = len(valid_indices)

        # Pure proportional allocation — NO per-cluster floor. Clusters that
        # round to 0 receive no seed; this is what keeps the total budget
        # exactly at `target` regardless of how many clusters the chosen
        # algorithm discovers (e.g. HDBSCAN can find far more clusters than
        # a fixed-k method, which previously inflated the label budget well
        # past the nominal 1% via the old max(1, ...) floor).
        allocation = {lb: int(round(cluster_sizes[lb] / total_valid * target))
                      for lb in unique_labels}

        # Rounding-correction pass to hit `target` exactly.
        diff = sum(allocation.values()) - target
        for lb in sorted(unique_labels, key=lambda lb: allocation[lb], reverse=True):
            if diff == 0:
                break
            if diff > 0:
                if allocation[lb] > 0:
                    allocation[lb] -= 1
                    diff -= 1
            else:
                allocation[lb] += 1
                diff += 1

        selected = []
        for lb in unique_labels:
            cidx    = valid_indices[valid_labels == lb]
            n_samp  = allocation.get(lb, 0)
            sampled = np.random.choice(cidx, size=min(n_samp, len(cidx)), replace=False)
            selected.extend(sampled.tolist())

        train_idx = sorted(set(selected))
        test_idx  = sorted(set(range(n_total)) - set(train_idx))

        if self.verbose:
            print(f"  Sampled {len(train_idx)} / {n_total} cells "
                  f"({100 * len(train_idx) / n_total:.1f}%)")
            if label_col and label_col in features_df.columns:
                dist = Counter(features_df.iloc[train_idx][label_col].tolist())
                print(f"  Train label distribution: {dict(dist)}")

        return train_idx, test_idx

    def _evaluate(self, y_true, y_pred) -> Dict:
        labels = sorted(set(y_true) | set(y_pred))
        acc    = accuracy_score(y_true, y_pred)
        f1_w   = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)

        metrics = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_mac}

        if 1 in labels:
            metrics["f1_intentional"]        = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["precision_intentional"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["recall_intentional"]    = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if -1 in labels:
            metrics["f1_unintentional"]        = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["precision_unintentional"] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["recall_unintentional"]    = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)

        print(f"  Accuracy:         {acc:.4f}")
        print(f"  F1 Weighted:      {f1_w:.4f}")
        print(f"  F1 Macro:         {f1_mac:.4f}")
        if "f1_intentional" in metrics:
            print(f"  F1 Intentional:   {metrics['f1_intentional']:.4f}")
        if "f1_unintentional" in metrics:
            print(f"  F1 Unintentional: {metrics['f1_unintentional']:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"  Confusion Matrix (labels={labels}):\n  {cm}")

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Results persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_results(result: Dict, out_dir: str, scenario: str) -> None:
    """Save pipeline outputs to out_dir/scenario/."""
    p = Path(out_dir) / scenario
    p.mkdir(parents=True, exist_ok=True)

    if result.get("features_df") is not None:
        result["features_df"].to_csv(p / "features.csv", index=False)

    if result.get("feature_importance") is not None:
        result["feature_importance"].to_csv(p / "feature_importance.csv", index=False)

    metrics = result.get("metrics")
    if metrics:
        with open(p / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[results] Saved to {p}/")
        print(f"  F1 Weighted:    {metrics.get('f1_weighted', 'N/A'):.4f}")
        print(f"  F1 Intentional: {metrics.get('f1_intentional', 'N/A'):.4f}")
