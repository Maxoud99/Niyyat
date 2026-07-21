#!/usr/bin/env python
"""
Scenario B — Cluster → Sample 1% → Label → Train/Propagate → Apply to 99%

Strategy:
  1. Fit the AttributionPipeline on (dirty, mask) — fully unsupervised.
  2. Compute the 12-feature matrix for every erroneous cell.
  3. Cluster the feature matrix with 6 different algorithms.
  4. Proportionally sample ~1% of cells per cluster as the "labeled" set
     (ground-truth labels are revealed for these cells only).
  5. Train / propagate labels from the 1% sample to the remaining 99%.
     Five label methods are tested per clustering:
       A. Random Forest classifier
       B. Cluster Majority Vote
       C. KNN (k=7, distance-weighted)
       D. Label Propagation  (sklearn)
       E. Label Spreading    (sklearn)
  6. Evaluate all combinations against ground truth.

Datasets tested:
  • LLM  — adult_income_dataset/tenth-trial/...
  • Kireev — mixed_error_pipeline/output/...

Output per run:
  heuristics/output/run_YYYYMMDD_HHMMSS/
    {llm|kireev}/
      {algorithm}/
        cluster_assignments.csv
        sampled_cells.csv
        metrics_rf.csv
        metrics_majority_vote.csv
        metrics_knn.csv
        metrics_label_propagation.csv
        metrics_label_spreading.csv
    comparison_table.csv
    summary.csv
    run_log.txt
"""

import matplotlib
matplotlib.use("Agg")

import sys
import os
import io
import time
import json
import warnings
import datetime
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    silhouette_score,
)

warnings.filterwarnings("ignore")

# Optional HDBSCAN
try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# -- Add the heuristics package to path --
HEURISTIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
if HEURISTIC_ROOT not in sys.path:
    sys.path.insert(0, HEURISTIC_ROOT)

from attribution.heuristics.pipeline import AttributionPipeline

# ===========================================================================
# Paths & Config
# ===========================================================================

BASE = "/home/mohamed/error_injector/llms_baseline"
LLM_DIR = os.path.join(
    BASE, "adult_income_dataset/tenth-trial/data/raw/run_v2_20260617_173016"
)
KIREEV_DIR = os.path.join(BASE, "mixed_error_pipeline/output")
TWITTER_DIR = os.path.join(BASE, "mixed_error_pipeline_twitter/output")
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

# Adult Income config (used for llm + kireev)
TARGET_COL = "class"
CODEPENDENT_PAIRS = [("education", "education-num")]
SENSITIVE_COLS = ["race", "sex"]

# TwiBot-20 config
TWITTER_TARGET_COL = "label"
TWITTER_CODEPENDENT_PAIRS = []
TWITTER_SENSITIVE_COLS = []

RANDOM_STATE = 42

N_CLUSTERS = 15          # for KMeans, Hierarchical, GMM
SAMPLE_FRAC = 0.01       # 1% labeled
MIN_SAMPLES_PER_CLUSTER = 5   # minimum cells sampled from any cluster
KNN_K = 7

# ===========================================================================
# Tee-logger
# ===========================================================================

class TeeLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer = io.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.buffer.write(message)

    def flush(self):
        self.terminal.flush()
        self.buffer.flush()

    def get_log(self):
        return self.buffer.getvalue()


# ===========================================================================
# Data loaders
# ===========================================================================

def load_llm_dataset():
    dirty = pd.read_csv(os.path.join(LLM_DIR, "manipulated_records.csv"))
    masks_full = pd.read_csv(os.path.join(LLM_DIR, "masks.csv"))
    blind_mask = (masks_full != 0).astype(int)

    labels_dict = {}
    for row_idx in range(len(masks_full)):
        for col_name in masks_full.columns:
            val = masks_full.iloc[row_idx][col_name]
            if val != 0:
                labels_dict[(row_idx, col_name)] = 1 if val == 1 else 0

    labels = pd.Series(labels_dict, name="intent_label")
    labels.index = pd.MultiIndex.from_tuples(
        labels.index, names=["row_idx", "col_name"]
    )
    return dirty, blind_mask, labels


def load_kireev_dataset():
    dirty = pd.read_csv(os.path.join(KIREEV_DIR, "adult_phase2_final.csv"))
    mask_combined = pd.read_csv(os.path.join(KIREEV_DIR, "mask_combined.csv"))
    blind_mask = (mask_combined != 0).astype(int)

    labels_dict = {}
    for row_idx in range(len(mask_combined)):
        for col_name in mask_combined.columns:
            val = mask_combined.iloc[row_idx][col_name]
            if val != 0:
                labels_dict[(row_idx, col_name)] = 1 if val == 1 else 0

    labels = pd.Series(labels_dict, name="intent_label")
    labels.index = pd.MultiIndex.from_tuples(
        labels.index, names=["row_idx", "col_name"]
    )
    return dirty, blind_mask, labels


def load_twitter_dataset():
    dirty = pd.read_csv(os.path.join(TWITTER_DIR, "twibot20_phase2_final.csv"))
    mask_combined = pd.read_csv(os.path.join(TWITTER_DIR, "mask_combined.csv"))
    blind_mask = (mask_combined != 0).astype(int)

    labels_dict = {}
    for row_idx in range(len(mask_combined)):
        for col_name in mask_combined.columns:
            val = mask_combined.iloc[row_idx][col_name]
            if val != 0:
                labels_dict[(row_idx, col_name)] = 1 if val == 1 else 0

    labels = pd.Series(labels_dict, name="intent_label")
    labels.index = pd.MultiIndex.from_tuples(
        labels.index, names=["row_idx", "col_name"]
    )
    return dirty, blind_mask, labels


# ===========================================================================
# Clustering algorithms
# ===========================================================================

def cluster_kmeans(X_scaled, n_clusters=N_CLUSTERS):
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    runtime = time.time() - t0
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    info = {
        "algorithm": "kmeans",
        "n_clusters": n_clusters,
        "noise_count": 0,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  KMeans: {n_clusters} clusters, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


def cluster_dbscan(X_scaled, min_samples=5):
    t0 = time.time()
    # Auto-tune eps via 90th percentile of k-NN distances
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)
    k_dists = np.sort(dists[:, -1])
    eps = float(np.percentile(k_dists, 90))
    if eps <= 0:
        eps = 0.5
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    runtime = time.time() - t0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    sil = -1.0
    if n_clusters >= 2:
        valid = labels != -1
        if valid.sum() > n_clusters:
            sil = silhouette_score(
                X_scaled[valid], labels[valid],
                sample_size=min(5000, valid.sum())
            )
    info = {
        "algorithm": "dbscan",
        "eps": round(eps, 4),
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "noise_count": noise,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  DBSCAN: eps={eps:.3f}, {n_clusters} clusters, noise={noise}, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


def cluster_hierarchical_ward(X_scaled, n_clusters=N_CLUSTERS):
    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    runtime = time.time() - t0
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    info = {
        "algorithm": "hierarchical_ward",
        "n_clusters": n_clusters,
        "linkage": "ward",
        "noise_count": 0,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  Hierarchical-Ward: {n_clusters} clusters, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


def cluster_hierarchical_average(X_scaled, n_clusters=N_CLUSTERS):
    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    labels = hc.fit_predict(X_scaled)
    runtime = time.time() - t0
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    info = {
        "algorithm": "hierarchical_average",
        "n_clusters": n_clusters,
        "linkage": "average",
        "noise_count": 0,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  Hierarchical-Average: {n_clusters} clusters, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


def cluster_gmm(X_scaled, n_components=N_CLUSTERS):
    t0 = time.time()
    gmm = GaussianMixture(
        n_components=n_components, random_state=RANDOM_STATE, n_init=3
    )
    labels = gmm.fit_predict(X_scaled)
    runtime = time.time() - t0
    sil = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))
    info = {
        "algorithm": "gmm",
        "n_components": n_components,
        "bic": round(gmm.bic(X_scaled), 2),
        "aic": round(gmm.aic(X_scaled), 2),
        "converged": bool(gmm.converged_),
        "noise_count": 0,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  GMM: {n_components} components, converged={gmm.converged_}, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


def cluster_hdbscan(X_scaled, min_cluster_size=50, min_samples=5):
    t0 = time.time()
    if not HDBSCAN_AVAILABLE:
        print("  HDBSCAN not available — falling back to KMeans")
        return cluster_kmeans(X_scaled)
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_scaled)
    runtime = time.time() - t0
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    sil = -1.0
    if n_clusters >= 2:
        valid = labels != -1
        if valid.sum() > n_clusters:
            sil = silhouette_score(
                X_scaled[valid], labels[valid],
                sample_size=min(5000, valid.sum())
            )
    info = {
        "algorithm": "hdbscan",
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "noise_count": noise,
        "silhouette": round(sil, 4),
        "runtime": round(runtime, 2),
    }
    print(f"  HDBSCAN: {n_clusters} clusters, noise={noise}, sil={sil:.4f}, t={runtime:.2f}s")
    return labels, info


CLUSTERING_METHODS = {
    "kmeans":               cluster_kmeans,
    "dbscan":               cluster_dbscan,
    "hierarchical_ward":    cluster_hierarchical_ward,
    "hierarchical_average": cluster_hierarchical_average,
    "gmm":                  cluster_gmm,
    "hdbscan":              cluster_hdbscan,
}


# ===========================================================================
# Proportional sampling (Scenario B: 1% labeled)
# ===========================================================================

def proportional_sample(feat_df, cluster_labels, sample_frac=SAMPLE_FRAC):
    """
    Proportionally allocate ~sample_frac of all cells across clusters.
    Noise points (cluster == -1) from DBSCAN/HDBSCAN are excluded from
    the labeled pool but kept in the test set.

    Returns
    -------
    sampled_idx  : array of integer positions into feat_df (labeled set)
    remaining_idx: array of integer positions into feat_df (unlabeled / test set)
    """
    n_total = len(feat_df)
    target_n = max(int(round(n_total * sample_frac)), 10)

    valid_mask = cluster_labels != -1
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    if len(unique_clusters) == 0:
        # All noise — random sample
        sampled_pos = np.random.choice(n_total, target_n, replace=False)
        remaining_pos = np.setdiff1d(np.arange(n_total), sampled_pos)
        return sampled_pos, remaining_pos

    cluster_sizes = {c: int((cluster_labels == c).sum()) for c in unique_clusters}
    total_valid = sum(cluster_sizes.values())

    # Proportional allocation
    allocation = {}
    for c in unique_clusters:
        base = (cluster_sizes[c] / total_valid) * target_n
        allocation[c] = max(MIN_SAMPLES_PER_CLUSTER, int(round(base)))

    # Trim to exact target
    total_allocated = sum(allocation.values())
    if total_allocated != target_n:
        diff = total_allocated - target_n
        sorted_clusters = sorted(unique_clusters, key=lambda c: allocation[c], reverse=True)
        for i in range(abs(diff)):
            c = sorted_clusters[i % len(sorted_clusters)]
            if diff > 0:
                allocation[c] = max(MIN_SAMPLES_PER_CLUSTER, allocation[c] - 1)
            else:
                allocation[c] += 1

    # Sample from each cluster
    rng = np.random.default_rng(RANDOM_STATE)
    sampled_positions = []
    all_positions = np.arange(n_total)

    for c in unique_clusters:
        cluster_pos = all_positions[cluster_labels == c]
        n_draw = min(allocation[c], len(cluster_pos))
        chosen = rng.choice(cluster_pos, size=n_draw, replace=False)
        sampled_positions.extend(chosen.tolist())

    sampled_pos = np.array(sampled_positions)
    remaining_pos = np.setdiff1d(all_positions, sampled_pos)
    return sampled_pos, remaining_pos


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def _metrics(y_true, y_pred, method_name, **extra):
    """Return a dict of evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")
    row = {
        "method": method_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_intentional": round(f1, 4),
        "f1_weighted": round(f1_w, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else "nan",
    }
    row.update(extra)
    return row


def _print_metrics(row):
    print(
        f"    Acc={row['accuracy']:.4f}  Prec={row['precision']:.4f}  "
        f"Rec={row['recall']:.4f}  F1={row['f1_intentional']:.4f}  "
        f"F1w={row['f1_weighted']:.4f}  AUC={row['auc']}"
    )


# ===========================================================================
# Label methods
# ===========================================================================

def method_random_forest(X_train, y_train, X_test, y_test):
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    runtime = time.time() - t0
    row = _metrics(y_test, y_pred, "RandomForest", runtime=round(runtime, 2))
    _print_metrics(row)
    return row


def method_cluster_majority_vote(cluster_labels, sampled_pos, remaining_pos, y_all):
    """
    For each cluster, find the majority label among sampled cells.
    Assign that majority label to every cell in the cluster (test set).
    Noise points (-1) fall back to the overall majority.
    """
    t0 = time.time()
    # Compute cluster → majority label from sampled cells
    cluster_majority = {}
    for c in np.unique(cluster_labels):
        if c == -1:
            continue
        in_cluster_sampled = sampled_pos[cluster_labels[sampled_pos] == c]
        if len(in_cluster_sampled) == 0:
            cluster_majority[c] = int(np.bincount(y_all).argmax())
        else:
            votes = y_all[in_cluster_sampled]
            cluster_majority[c] = int(np.bincount(votes).argmax())

    overall_majority = int(np.bincount(y_all[sampled_pos]).argmax())

    y_pred = np.array([
        cluster_majority.get(cluster_labels[i], overall_majority)
        for i in remaining_pos
    ])
    y_test = y_all[remaining_pos]
    runtime = time.time() - t0
    row = _metrics(y_test, y_pred, "ClusterMajorityVote", runtime=round(runtime, 2))
    _print_metrics(row)
    return row


def method_knn(X_train, y_train, X_test, y_test, k=KNN_K):
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    runtime = time.time() - t0
    row = _metrics(y_test, y_pred, f"KNN_k{k}", runtime=round(runtime, 2))
    _print_metrics(row)
    return row


def method_label_propagation(X_train, y_train, X_test, y_test, X_all):
    """
    sklearn LabelPropagation:
    - Labeled samples keep their 0/1 label.
    - Unlabeled samples are marked -1 (sklearn convention).
    """
    t0 = time.time()
    n_labeled = len(X_train)
    y_partial = np.concatenate([y_train, np.full(len(X_test), -1, dtype=int)])

    lp = LabelPropagation(kernel="knn", n_neighbors=KNN_K, max_iter=1000)
    lp.fit(X_all, y_partial)
    y_pred = lp.predict(X_all[n_labeled:])
    runtime = time.time() - t0

    # Handle any remaining -1 predictions (shouldn't happen, but guard)
    fallback = int(np.bincount(y_train).argmax())
    y_pred = np.where(y_pred == -1, fallback, y_pred)

    row = _metrics(y_test, y_pred, "LabelPropagation", runtime=round(runtime, 2))
    _print_metrics(row)
    return row


def method_label_spreading(X_train, y_train, X_test, y_test, X_all):
    """
    sklearn LabelSpreading (alpha=0.2 → softer than LabelPropagation).
    """
    t0 = time.time()
    n_labeled = len(X_train)
    y_partial = np.concatenate([y_train, np.full(len(X_test), -1, dtype=int)])

    ls = LabelSpreading(kernel="knn", n_neighbors=KNN_K, alpha=0.2, max_iter=1000)
    ls.fit(X_all, y_partial)
    y_pred = ls.predict(X_all[n_labeled:])
    runtime = time.time() - t0

    fallback = int(np.bincount(y_train).argmax())
    y_pred = np.where(y_pred == -1, fallback, y_pred)

    row = _metrics(y_test, y_pred, "LabelSpreading", runtime=round(runtime, 2))
    _print_metrics(row)
    return row


# ===========================================================================
# Per-algorithm runner
# ===========================================================================

def run_algorithm(
    algo_name,
    cluster_fn,
    feat_df,
    X_scaled,
    y_all,
    ds_dir,
):
    """
    Run one clustering algorithm + all 5 label methods on a single dataset.

    Parameters
    ----------
    algo_name  : str — name of the algorithm
    cluster_fn : callable — returns (cluster_labels, info_dict)
    feat_df    : pd.DataFrame with MultiIndex (row_idx, col_name), shape (N, 13)
    X_scaled   : np.ndarray, scaled feature matrix
    y_all      : np.ndarray, integer labels (0/1) aligned with feat_df rows
    ds_dir     : pathlib.Path — dataset output directory

    Returns
    -------
    list of metric dicts (one per label method)
    """
    import pathlib
    algo_dir = pathlib.Path(ds_dir) / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*60}")
    print(f"  Clustering: {algo_name.upper()}")
    print(f"  {'─'*60}")

    # 1. Cluster
    cluster_labels, cluster_info = cluster_fn(X_scaled)
    n_total = len(feat_df)
    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    # 2. Save cluster assignments
    assign_df = feat_df.copy()
    assign_df["cluster"] = cluster_labels
    assign_df["true_label"] = y_all
    assign_df.to_csv(algo_dir / "cluster_assignments.csv")

    # 3. Proportional sample
    sampled_pos, remaining_pos = proportional_sample(feat_df, cluster_labels)
    n_labeled = len(sampled_pos)
    n_test = len(remaining_pos)
    pct_labeled = n_labeled / n_total * 100

    print(f"  Cells total : {n_total}")
    print(f"  Labeled     : {n_labeled} ({pct_labeled:.2f}%)")
    print(f"  Test (99%)  : {n_test}")
    print(f"  Clusters    : {n_clusters_found}")

    # 4. Save sampled cells
    sampled_df = feat_df.iloc[sampled_pos].copy()
    sampled_df["cluster"] = cluster_labels[sampled_pos]
    sampled_df["true_label"] = y_all[sampled_pos]
    sampled_df.to_csv(algo_dir / "sampled_cells.csv")

    # 5. Prepare train / test arrays
    X_train = X_scaled[sampled_pos]
    y_train = y_all[sampled_pos]
    X_test  = X_scaled[remaining_pos]
    y_test  = y_all[remaining_pos]

    # Stack for semi-supervised (labeled first, unlabeled second)
    X_all_stacked = np.vstack([X_train, X_test])

    # --- Method A: Random Forest ---
    print(f"\n  [A] Random Forest")
    rf_row = method_random_forest(X_train, y_train, X_test, y_test)
    rf_row.update({"n_labeled": n_labeled, "n_test": n_test,
                   "pct_labeled": round(pct_labeled, 2), **cluster_info})
    pd.DataFrame([rf_row]).to_csv(algo_dir / "metrics_rf.csv", index=False)

    # --- Method B: Cluster Majority Vote ---
    print(f"\n  [B] Cluster Majority Vote")
    mv_row = method_cluster_majority_vote(
        cluster_labels, sampled_pos, remaining_pos, y_all
    )
    mv_row.update({"n_labeled": n_labeled, "n_test": n_test,
                   "pct_labeled": round(pct_labeled, 2), **cluster_info})
    pd.DataFrame([mv_row]).to_csv(algo_dir / "metrics_majority_vote.csv", index=False)

    # --- Method C: KNN ---
    print(f"\n  [C] KNN (k={KNN_K})")
    knn_row = method_knn(X_train, y_train, X_test, y_test)
    knn_row.update({"n_labeled": n_labeled, "n_test": n_test,
                    "pct_labeled": round(pct_labeled, 2), **cluster_info})
    pd.DataFrame([knn_row]).to_csv(algo_dir / "metrics_knn.csv", index=False)

    # --- Method D: Label Propagation ---
    print(f"\n  [D] Label Propagation")
    lp_row = method_label_propagation(X_train, y_train, X_test, y_test, X_all_stacked)
    lp_row.update({"n_labeled": n_labeled, "n_test": n_test,
                   "pct_labeled": round(pct_labeled, 2), **cluster_info})
    pd.DataFrame([lp_row]).to_csv(algo_dir / "metrics_label_propagation.csv", index=False)

    # --- Method E: Label Spreading ---
    print(f"\n  [E] Label Spreading")
    ls_row = method_label_spreading(X_train, y_train, X_test, y_test, X_all_stacked)
    ls_row.update({"n_labeled": n_labeled, "n_test": n_test,
                   "pct_labeled": round(pct_labeled, 2), **cluster_info})
    pd.DataFrame([ls_row]).to_csv(algo_dir / "metrics_label_spreading.csv", index=False)

    # Save cluster info JSON
    with open(algo_dir / "cluster_info.json", "w") as fh:
        json.dump(cluster_info, fh, indent=2)

    all_method_rows = [rf_row, mv_row, knn_row, lp_row, ls_row]
    return all_method_rows


# ===========================================================================
# Comparison plots
# ===========================================================================

def save_comparison_plots(comparison_df, out_dir):
    """Bar plots of F1-weighted per clustering × method for each dataset."""
    import pathlib
    out_dir = pathlib.Path(out_dir)

    datasets = comparison_df["dataset"].unique()
    metrics_to_plot = ["f1_weighted", "accuracy", "f1_intentional"]

    for ds in datasets:
        ds_df = comparison_df[comparison_df["dataset"] == ds]

        for metric in metrics_to_plot:
            pivot = ds_df.pivot_table(
                index="algorithm", columns="method", values=metric, aggfunc="first"
            )
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 5))
            pivot.plot(kind="bar", ax=ax, width=0.7)
            ax.set_title(f"{ds} — {metric} by clustering × method")
            ax.set_xlabel("Clustering Algorithm")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.05)
            ax.legend(loc="lower right", fontsize=8)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            fname = out_dir / f"{ds}_{metric}.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            print(f"  Plot saved: {fname.name}")


# ===========================================================================
# Dataset runner
# ===========================================================================

def run_dataset(name, load_fn, run_dir, pipeline_cfg=None):
    """
    Full Scenario B pipeline for one dataset.

    Returns list of metric dicts (one per clustering × method).
    pipeline_cfg: dict with keys target_col, codependent_pairs, sensitive_cols
                  (defaults to Adult Income config if None)
    """
    if pipeline_cfg is None:
        pipeline_cfg = {
            "target_col": TARGET_COL,
            "codependent_pairs": CODEPENDENT_PAIRS,
            "sensitive_cols": SENSITIVE_COLS,
        }
    import pathlib
    ds_dir = pathlib.Path(run_dir) / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DATASET: {name.upper()}")
    print(f"{'='*70}")

    # ---- Load data ----
    print("\n[1] Loading data...")
    t0 = time.time()
    dirty, blind_mask, labels = load_fn()
    print(f"  Dirty shape : {dirty.shape}")
    print(f"  Error cells : {len(labels)}")
    print(f"  Intentional : {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    print(f"  Unintentional: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")

    # ---- Fit pipeline ----
    print("\n[2] Fitting AttributionPipeline (unsupervised)...")
    pipe = AttributionPipeline(
        target_col=pipeline_cfg["target_col"],
        codependent_pairs=pipeline_cfg["codependent_pairs"],
        sensitive_cols=pipeline_cfg["sensitive_cols"],
    )
    pipe.fit(dirty, blind_mask)
    print(f"  Pipeline fitted in {time.time()-t0:.2f}s")

    # ---- Compute features ----
    print("\n[3] Computing 12-feature matrix...")
    feat_df = pipe.compute_features(dirty, blind_mask)
    print(f"  Feature matrix shape: {feat_df.shape}")
    feat_df.to_csv(ds_dir / "feature_matrix.csv")

    # ---- Align labels ----
    common_idx = feat_df.index.intersection(labels.index)
    feat_df = feat_df.loc[common_idx]
    y_all = labels.loc[common_idx].values.astype(int)
    print(f"  After alignment: {len(feat_df)} cells")

    # ---- Scale ----
    print("\n[4] Scaling features...")
    X_raw = np.nan_to_num(feat_df.values, nan=-999.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ---- Run all clustering × label method combinations ----
    print("\n[5] Running 6 clustering algorithms × 5 label methods...")
    all_rows = []

    for algo_name, cluster_fn in CLUSTERING_METHODS.items():
        try:
            method_rows = run_algorithm(
                algo_name, cluster_fn, feat_df, X_scaled, y_all, ds_dir
            )
            for row in method_rows:
                row["dataset"] = name
                row["algorithm"] = algo_name
            all_rows.extend(method_rows)
        except Exception as exc:
            print(f"\n  ⚠ {algo_name} FAILED: {exc}")
            traceback.print_exc()

    # ---- Save per-dataset comparison table ----
    if all_rows:
        comp_df = pd.DataFrame(all_rows)
        comp_df.to_csv(ds_dir / "comparison_table.csv", index=False)

        # Summary: best method per clustering
        print(f"\n[6] Summary — best F1-weighted per clustering:")
        for algo in comp_df["algorithm"].unique():
            sub = comp_df[comp_df["algorithm"] == algo]
            best = sub.loc[sub["f1_weighted"].idxmax()]
            print(f"  {algo:<22s}  best={best['method']:<22s}  F1w={best['f1_weighted']:.4f}")

        # Plots
        save_comparison_plots(comp_df, ds_dir)

    return all_rows


# ===========================================================================
# Main
# ===========================================================================

def main():
    np.random.seed(RANDOM_STATE)

    # Create timestamped run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, f"run_scenario_b_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Start tee logger
    tee = TeeLogger()
    sys.stdout = tee

    print(f"Scenario B — Cluster → Sample 1% → Label → Propagate")
    print(f"Run ID   : {timestamp}")
    print(f"Run dir  : {run_dir}")
    print(f"Datasets : LLM adult income + Mixed SOTA (Adult) + Mixed SOTA (TwiBot-20)")
    print(f"Algorithms: {list(CLUSTERING_METHODS.keys())}")
    print(f"Methods  : RandomForest, ClusterMajorityVote, KNN, LabelPropagation, LabelSpreading")
    print(f"Sample % : {SAMPLE_FRAC*100:.1f}%")
    print(f"HDBSCAN  : {'available' if HDBSCAN_AVAILABLE else 'NOT available — will fallback to KMeans'}")

    ADULT_CFG = {
        "target_col": TARGET_COL,
        "codependent_pairs": CODEPENDENT_PAIRS,
        "sensitive_cols": SENSITIVE_COLS,
    }
    TWITTER_CFG = {
        "target_col": TWITTER_TARGET_COL,
        "codependent_pairs": TWITTER_CODEPENDENT_PAIRS,
        "sensitive_cols": TWITTER_SENSITIVE_COLS,
    }

    datasets = [
        ("llm",     load_llm_dataset,     ADULT_CFG),
        ("kireev",  load_kireev_dataset,  ADULT_CFG),
        ("twitter", load_twitter_dataset, TWITTER_CFG),
    ]

    all_rows = []
    summary_rows = []

    for ds_name, load_fn, ds_cfg in datasets:
        t_ds = time.time()
        try:
            rows = run_dataset(ds_name, load_fn, run_dir, pipeline_cfg=ds_cfg)
            all_rows.extend(rows)
            elapsed = time.time() - t_ds
            summary_rows.append({
                "dataset": ds_name,
                "status": "ok",
                "n_combinations": len(rows),
                "elapsed_s": round(elapsed, 1),
            })
        except Exception as exc:
            print(f"\n✗ Dataset '{ds_name}' FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({
                "dataset": ds_name,
                "status": f"error: {exc}",
                "n_combinations": 0,
                "elapsed_s": round(time.time() - t_ds, 1),
            })

    # ---- Global comparison table ----
    if all_rows:
        comp_df = pd.DataFrame(all_rows)

        # Ensure key columns are present
        for col in ["dataset", "algorithm", "method", "f1_weighted", "accuracy",
                    "f1_intentional", "precision", "recall", "auc", "runtime",
                    "n_labeled", "n_test", "pct_labeled"]:
            if col not in comp_df.columns:
                comp_df[col] = float("nan")

        # Reorder columns
        front_cols = ["dataset", "algorithm", "method", "f1_weighted",
                      "f1_intentional", "accuracy", "precision", "recall",
                      "auc", "runtime", "n_labeled", "n_test", "pct_labeled"]
        rest_cols = [c for c in comp_df.columns if c not in front_cols]
        comp_df = comp_df[front_cols + rest_cols]
        comp_df.to_csv(os.path.join(run_dir, "comparison_table.csv"), index=False)

        # Cross-dataset comparison plots
        save_comparison_plots(comp_df, run_dir)

        print(f"\n{'='*70}")
        print("GLOBAL BEST — F1-weighted per dataset × algorithm:")
        print(f"{'='*70}")
        for ds in comp_df["dataset"].unique():
            for algo in comp_df[comp_df["dataset"] == ds]["algorithm"].unique():
                sub = comp_df[(comp_df["dataset"] == ds) & (comp_df["algorithm"] == algo)]
                if sub.empty:
                    continue
                best = sub.loc[sub["f1_weighted"].idxmax()]
                print(
                    f"  {ds:<8s} {algo:<22s} → {best['method']:<22s} "
                    f"F1w={best['f1_weighted']:.4f}  Acc={best['accuracy']:.4f}"
                )

    # ---- Summary CSV ----
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    print(f"\nSummary saved to: {os.path.join(run_dir, 'summary.csv')}")

    # ---- Run log ----
    sys.stdout = tee.terminal
    log_path = os.path.join(run_dir, "run_log.txt")
    with open(log_path, "w") as fh:
        fh.write(tee.get_log())
    print(f"Run log  : {log_path}")
    print(f"Run dir  : {run_dir}")
    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
