#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-Clustering Attribution Pipeline
====================================

Same feature sets and same Random Forest as the clustering pipelines
(Scenario B / B+ in supervised_baseline, Scenario C / B+C / (B+)+C in
declarative/pipeline.py), but the clustering + cluster-proportional
sampling step is removed entirely. The 1% label budget is instead drawn
by **plain uniform random sampling** over all erroneous cells.

This isolates the effect of the clustering step: same features, same
classifier, same label budget — the only difference is how the labeled
seed set is chosen.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


class NoClusteringPipeline:
    """
    Intent attribution pipeline with random (non-clustered) seed sampling.

    Parameters
    ----------
    sampling_rate : float
        Fraction of erroneous cells to label (default 0.01 = 1%), drawn
        uniformly at random — no clustering step.
    random_state : int
    verbose : bool
    """

    def __init__(
        self,
        sampling_rate: float = 0.01,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.sampling_rate = sampling_rate
        self.random_state  = random_state
        self.verbose       = verbose

        self.scaler     = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def run(self, features_df: pd.DataFrame) -> Dict:
        """
        Parameters
        ----------
        features_df : DataFrame
            Must contain ``row_idx``, ``column_name``, ``intent_label``
            (+1 / -1) and one or more feature columns.

        Returns
        -------
        dict with keys: predictions, metrics, feature_importance, elapsed, n_seed
        """
        np.random.seed(self.random_state)
        t0 = time.time()

        id_cols   = ["row_idx", "column_name"]
        skip_cols = id_cols + ["intent_label"]
        feat_cols = [c for c in features_df.columns if c not in skip_cols]

        X = features_df[feat_cols].fillna(0).values.astype(float)
        y = features_df["intent_label"].values
        X_scaled = self.scaler.fit_transform(X)

        if self.verbose:
            print(f"\n[pipeline] Feature matrix: {X_scaled.shape[0]} cells × "
                  f"{X_scaled.shape[1]} features (no clustering)")

        train_idx, test_idx = self._random_sample(len(y))

        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_test,  y_test  = X_scaled[test_idx],  y[test_idx]

        if self.verbose:
            print(f"  Sampled {len(train_idx)} / {len(y)} cells "
                  f"({100 * len(train_idx) / len(y):.1f}%) uniformly at random")
            print(f"  Train label distribution: {dict(Counter(y_train))}")

        self.classifier.fit(X_train, y_train)

        if len(self.classifier.classes_) == 1:
            y_pred = np.full(len(X_test), self.classifier.classes_[0], dtype=int)
        else:
            y_pred = self.classifier.predict(X_test)

        metrics = self._evaluate(y_test, y_pred)

        importance = pd.DataFrame({
            "feature":    feat_cols,
            "importance": self.classifier.feature_importances_,
        }).sort_values("importance", ascending=False)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"[pipeline] Done in {elapsed:.1f}s")

        features_df = features_df.copy()
        features_df.loc[features_df.index[test_idx], "predicted_intent"] = y_pred

        return {
            "features_df":        features_df,
            "predictions":        y_pred,
            "metrics":            metrics,
            "feature_importance": importance,
            "elapsed":            elapsed,
            "n_seed":             len(train_idx),
        }

    def _random_sample(self, n_total: int):
        target = max(int(round(n_total * self.sampling_rate)), 2)
        rng = np.random.RandomState(self.random_state)
        all_idx = np.arange(n_total)
        train_idx = rng.choice(all_idx, size=min(target, n_total), replace=False)
        test_idx = np.setdiff1d(all_idx, train_idx)
        return sorted(train_idx.tolist()), sorted(test_idx.tolist())

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

        if self.verbose:
            print(f"  Accuracy:    {acc:.4f}")
            print(f"  F1 Weighted: {f1_w:.4f}")

        return metrics


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
        meta = dict(metrics)
        meta["n_seed"] = result.get("n_seed")
        with open(p / "metrics.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[results] Saved to {p}/  F1-w={metrics.get('f1_weighted', float('nan')):.4f}")
