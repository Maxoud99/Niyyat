#!/usr/bin/env python3
"""
Error Fingerprint — Intent Attribution Pipeline
=================================================

Complete pipeline:
  1. Extract fingerprint features (dirty data + blind mask only)
  2. Cluster erroneous cells by fingerprint
  3. Sample representatives from clusters
  4. Train classifier on sampled labels
  5. Evaluate on all remaining erroneous cells

Supports evaluation with ground truth labels (for benchmarking)
and also a "deployment" mode where labels come from human annotation.
"""

import numpy as np
import pandas as pd
import warnings
import time
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from feature_extraction import ErrorFingerprintExtractor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class FingerprintIntentPipeline:
    """
    End-to-end intent attribution using only dirty data + blind mask.

    Parameters
    ----------
    sampling_rate : float
        Fraction of erroneous records to sample for labeling (default 0.01 = 1%).
    clustering : str
        Clustering algorithm: 'kmeans' or 'hdbscan'.
    n_clusters : int
        Number of clusters for K-Means (ignored for HDBSCAN).
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print progress.
    """

    def __init__(
        self,
        sampling_rate: float = 0.01,
        clustering: str = "hdbscan",
        n_clusters: int = 15,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.clustering = clustering
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.verbose = verbose

        self.extractor = ErrorFingerprintExtractor(verbose=verbose)
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def run(
        self,
        dirty_df: pd.DataFrame,
        mask_df: pd.DataFrame,
        ground_truth_mask: pd.DataFrame = None,
    ) -> dict:
        """
        Run the full pipeline.

        Parameters
        ----------
        dirty_df : DataFrame
            The dirty dataset.
        mask_df : DataFrame
            Blind binary mask (0/1).
        ground_truth_mask : DataFrame, optional
            Full mask with intent labels (-1/1). Only used for evaluation.
            If provided, labels are taken from here instead of human annotation.

        Returns
        -------
        dict with keys:
            - 'features_df': extracted features
            - 'predictions': predicted intent per erroneous cell
            - 'metrics': evaluation metrics (if ground_truth provided)
            - 'feature_importance': RF feature importance
        """
        np.random.seed(self.random_state)

        t0 = time.time()

        # ── Step 1: Extract features ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 1: EXTRACTING ERROR FINGERPRINTS")
            print("=" * 70)

        self.extractor.fit(dirty_df, mask_df)
        features_df = self.extractor.extract_features(dirty_df, mask_df)

        if len(features_df) == 0:
            print("No erroneous cells found!")
            return {"features_df": features_df, "predictions": None, "metrics": None}

        # Attach ground truth labels if available
        if ground_truth_mask is not None:
            features_df = self._attach_ground_truth(features_df, ground_truth_mask)

        # ── Step 2: Prepare feature matrix ──
        id_cols = ["row_idx", "column_name"]
        label_col = "intent_label" if "intent_label" in features_df.columns else None
        feature_cols = [c for c in features_df.columns
                        if c not in id_cols + ([label_col] if label_col else [])]

        X = features_df[feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        if self.verbose:
            print(f"\nFeature matrix: {X_scaled.shape[0]} cells × {X_scaled.shape[1]} features")

        # ── Step 3: Cluster ──
        if self.verbose:
            print("\n" + "=" * 70)
            print(f"STEP 2: CLUSTERING ({self.clustering.upper()})")
            print("=" * 70)

        cluster_labels = self._cluster(X_scaled)

        # ── Step 4: Proportional sampling ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 3: PROPORTIONAL SAMPLING")
            print("=" * 70)

        train_indices, test_indices = self._proportional_sample(
            cluster_labels, features_df, label_col
        )

        # ── Step 5: Train classifier ──
        if self.verbose:
            print("\n" + "=" * 70)
            print("STEP 4: TRAINING CLASSIFIER")
            print("=" * 70)

        X_train = X[train_indices]
        y_train = features_df.iloc[train_indices][label_col].values
        X_test = X[test_indices]
        y_test = features_df.iloc[test_indices][label_col].values if label_col else None

        self.classifier.fit(X_train, y_train)

        if self.verbose:
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")
            dist = Counter(y_train)
            print(f"  Training distribution: {dict(dist)}")

        # ── Step 6: Predict ──
        y_pred = self.classifier.predict(X_test)
        features_df.loc[features_df.index[test_indices], "predicted_intent"] = y_pred

        # ── Step 7: Evaluate ──
        metrics = None
        if label_col and y_test is not None:
            if self.verbose:
                print("\n" + "=" * 70)
                print("STEP 5: EVALUATION")
                print("=" * 70)
            metrics = self._evaluate(y_test, y_pred)

        # ── Feature importance ──
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.classifier.feature_importances_,
        }).sort_values("importance", ascending=False)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\nTotal time: {elapsed:.1f}s")
            print(f"\nTop 15 features:")
            print(importance.head(15).to_string(index=False))

        return {
            "features_df": features_df,
            "predictions": y_pred,
            "metrics": metrics,
            "feature_importance": importance,
            "elapsed": elapsed,
        }

    def _attach_ground_truth(
        self, features_df: pd.DataFrame, gt_mask: pd.DataFrame
    ) -> pd.DataFrame:
        """Attach intent labels from ground truth mask to feature rows."""
        labels = []
        feat_cols_in_mask = [c for c in self.extractor.feature_columns_
                            if c in gt_mask.columns]

        for _, row in features_df.iterrows():
            r_idx = int(row["row_idx"])
            col = row["column_name"]
            if col in gt_mask.columns:
                label = gt_mask.iloc[r_idx][col]
                labels.append(int(label))
            else:
                labels.append(0)

        features_df["intent_label"] = labels
        # Filter to only cells with actual intent labels
        features_df = features_df[features_df["intent_label"] != 0].copy()
        if self.verbose:
            dist = Counter(features_df["intent_label"])
            print(f"\nGround truth labels: {dict(dist)}")
        return features_df

    def _cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """Run clustering on the feature matrix."""
        if self.clustering == "hdbscan" and HDBSCAN_AVAILABLE:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5, min_samples=3, core_dist_n_jobs=-1,
            )
            labels = clusterer.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            if self.verbose:
                print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
        else:
            if self.clustering == "hdbscan" and not HDBSCAN_AVAILABLE:
                print("  HDBSCAN not available, falling back to K-Means")
            km = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = km.fit_predict(X_scaled)
            if self.verbose:
                print(f"  K-Means: {self.n_clusters} clusters")

        return labels

    def _proportional_sample(
        self,
        cluster_labels: np.ndarray,
        features_df: pd.DataFrame,
        label_col: str,
    ):
        """
        Sample proportionally from each cluster.
        Returns (train_indices, test_indices).
        """
        n_total = len(features_df)
        target = max(1, int(n_total * self.sampling_rate))

        # Handle noise points
        valid_mask = cluster_labels != -1
        valid_indices = np.where(valid_mask)[0]
        valid_labels = cluster_labels[valid_mask]

        if len(valid_indices) == 0:
            # Fallback: random sample
            all_idx = np.arange(n_total)
            np.random.shuffle(all_idx)
            train_idx = all_idx[:target]
            test_idx = all_idx[target:]
            return train_idx.tolist(), test_idx.tolist()

        unique_labels = np.unique(valid_labels)
        cluster_sizes = {l: (valid_labels == l).sum() for l in unique_labels}
        total_valid = len(valid_indices)

        # Proportional allocation
        allocation = {}
        for label in unique_labels:
            n = max(1, int(round((cluster_sizes[label] / total_valid) * target)))
            allocation[label] = n

        # Adjust to match target
        total_alloc = sum(allocation.values())
        while total_alloc > target:
            largest = max(allocation, key=allocation.get)
            if allocation[largest] > 1:
                allocation[largest] -= 1
                total_alloc -= 1
            else:
                break

        # Sample
        selected = []
        for label in unique_labels:
            n_sample = allocation.get(label, 0)
            cluster_idx = valid_indices[valid_labels == label]
            if len(cluster_idx) >= n_sample:
                sampled = np.random.choice(cluster_idx, size=n_sample, replace=False)
            else:
                sampled = cluster_idx
            selected.extend(sampled.tolist())

        train_indices = sorted(set(selected))
        test_indices = sorted(set(range(n_total)) - set(train_indices))

        if self.verbose:
            print(f"  Sampled {len(train_indices)} cells from {len(unique_labels)} clusters")
            print(f"  Training: {len(train_indices)} ({100*len(train_indices)/n_total:.1f}%)")
            print(f"  Test: {len(test_indices)} ({100*len(test_indices)/n_total:.1f}%)")
            if label_col and label_col in features_df.columns:
                train_dist = Counter(
                    features_df.iloc[train_indices][label_col].tolist()
                )
                print(f"  Training label distribution: {dict(train_dist)}")

        return train_indices, test_indices

    def _evaluate(self, y_true, y_pred) -> dict:
        """Compute evaluation metrics."""
        # Determine unique labels
        labels = sorted(set(y_true) | set(y_pred))

        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)

        metrics = {
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_macro,
        }

        # Per-class metrics
        if 1 in labels:
            metrics["f1_intentional"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["precision_intentional"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["recall_intentional"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if -1 in labels:
            metrics["f1_unintentional"] = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["precision_unintentional"] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["recall_unintentional"] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)

        print(f"  Accuracy:        {acc:.4f}")
        print(f"  F1 Weighted:     {f1_w:.4f}")
        print(f"  F1 Macro:        {f1_macro:.4f}")
        if "f1_intentional" in metrics:
            print(f"  F1 Intentional:  {metrics['f1_intentional']:.4f}")
        if "f1_unintentional" in metrics:
            print(f"  F1 Unintentional:{metrics['f1_unintentional']:.4f}")

        print(f"\n  Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"  Confusion Matrix (labels={labels}):")
        print(f"  {cm}")

        return metrics
