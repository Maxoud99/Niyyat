#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering Attribution Pipeline
================================

The clustered counterpart of no_clustering/pipeline.py: same feature sets,
same Random Forest, same 1% label budget, but the seed cells are chosen by
cluster-proportional sampling (DBSCAN) instead of uniform random sampling.

Per project decision, this is the ONE clustering algorithm used everywhere
in the project (declarative/pipeline.py's Family C/B+C/(B+)+C scenarios use
the same DBSCAN parameterisation) -- there is no multi-algorithm sweep here
by design, unlike the legacy clustering-organized/ exploration.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class ClusteringPipeline:
    """Intent attribution pipeline with DBSCAN cluster-proportional seed sampling."""

    def __init__(self, sampling_rate: float = 0.01, random_state: int = 42, verbose: bool = True):
        self.sampling_rate = sampling_rate
        self.random_state = random_state
        self.verbose = verbose

        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        )

    def run(self, features_df: pd.DataFrame) -> Dict:
        np.random.seed(self.random_state)
        t0 = time.time()

        id_cols = ["row_idx", "column_name"]
        skip_cols = id_cols + ["intent_label"]
        feat_cols = [c for c in features_df.columns if c not in skip_cols]

        X = features_df[feat_cols].fillna(0).values.astype(float)
        y = features_df["intent_label"].values
        X_scaled = self.scaler.fit_transform(X)

        if self.verbose:
            print(f"\n[pipeline] Feature matrix: {X_scaled.shape[0]} cells x "
                  f"{X_scaled.shape[1]} features (DBSCAN clustering)")

        cluster_labels = self._cluster(X_scaled)
        train_idx, test_idx = self._proportional_sample(cluster_labels)

        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test, y_test = X_scaled[test_idx], y[test_idx]

        if self.verbose:
            print(f"  Sampled {len(train_idx)} / {len(y)} cells "
                  f"({100 * len(train_idx) / len(y):.1f}%) via cluster-proportional sampling")
            print(f"  Train label distribution: {dict(Counter(y_train))}")

        self.classifier.fit(X_train, y_train)
        if len(self.classifier.classes_) == 1:
            y_pred = np.full(len(X_test), self.classifier.classes_[0], dtype=int)
        else:
            y_pred = self.classifier.predict(X_test)

        metrics = self._evaluate(y_test, y_pred)
        importance = pd.DataFrame({
            "feature": feat_cols, "importance": self.classifier.feature_importances_,
        }).sort_values("importance", ascending=False)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"[pipeline] Done in {elapsed:.1f}s")

        features_df = features_df.copy()
        features_df.loc[features_df.index[test_idx], "predicted_intent"] = y_pred

        return {
            "features_df": features_df, "predictions": y_pred, "metrics": metrics,
            "feature_importance": importance, "elapsed": elapsed, "n_seed": len(train_idx),
            "n_clusters": int(len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),
        }

    def _cluster(self, X_scaled: np.ndarray) -> np.ndarray:
        """DBSCAN with eps = 90th percentile of the k-NN distance curve, min_samples=5
        -- identical parameterisation to declarative/pipeline.py._cluster()."""
        k = min(5, len(X_scaled) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        dists, _ = nbrs.kneighbors(X_scaled)
        eps = max(float(np.percentile(np.sort(dists[:, -1]), 90)), 1e-6)
        labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        if self.verbose:
            print(f"  DBSCAN: eps={eps:.3f}, {n_cl} clusters, {(labels == -1).sum()} noise points")
        return labels

    def _proportional_sample(self, cluster_labels: np.ndarray):
        n_total = len(cluster_labels)
        target = max(int(round(n_total * self.sampling_rate)), 2)
        rng = np.random.RandomState(self.random_state)

        clusters = sorted(set(cluster_labels))
        train_idx = []
        for c in clusters:
            members = np.where(cluster_labels == c)[0]
            n_take = max(1, int(round(len(members) / n_total * target)))
            n_take = min(n_take, len(members))
            train_idx.extend(rng.choice(members, size=n_take, replace=False).tolist())

        train_idx = sorted(set(train_idx))
        all_idx = np.arange(n_total)
        test_idx = sorted(np.setdiff1d(all_idx, train_idx).tolist())
        return train_idx, test_idx

    def _evaluate(self, y_true, y_pred) -> Dict:
        labels = sorted(set(y_true) | set(y_pred))
        acc = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        metrics = {"accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_mac}
        if 1 in labels:
            metrics["f1_intentional"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["precision_intentional"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics["recall_intentional"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        if -1 in labels:
            metrics["f1_unintentional"] = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["precision_unintentional"] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
            metrics["recall_unintentional"] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
        if self.verbose:
            print(f"  Accuracy:    {acc:.4f}")
            print(f"  F1 Weighted: {f1_w:.4f}")
        return metrics


def save_results(result: Dict, out_dir: str, scenario: str) -> None:
    p = Path(out_dir) / scenario
    p.mkdir(parents=True, exist_ok=True)
    if result.get("features_df") is not None:
        result["features_df"].to_csv(p / "features.csv", index=False)
    if result.get("feature_importance") is not None:
        result["feature_importance"].to_csv(p / "feature_importance.csv", index=False)
    metrics = result.get("metrics")
    if metrics:
        meta = dict(metrics)
        meta["n_seed"] = result.get("n_seed")
        meta["n_clusters"] = result.get("n_clusters")
        with open(p / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[results] Saved to {p}/  F1-w={metrics.get('f1_weighted', float('nan')):.4f}")
