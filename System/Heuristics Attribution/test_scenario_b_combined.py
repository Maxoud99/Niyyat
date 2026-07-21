#!/usr/bin/env python
"""
Scenario B — COMBINED FEATURES (Statistical + Heuristic)
=========================================================

Same Scenario B pipeline (Cluster → 1% labels → Train/Propagate → Evaluate)
but uses a COMBINED feature matrix of:

  • 13 heuristic features  (H1–H8, computed by AttributionPipeline)
  • N  statistical features (cell-level value statistics derived from
    original→dirty change, matching the old clustering system's signals
    but kept at cell-level, not aggregated to record-level)

Statistical features per erroneous cell:
  1.  change_magnitude        — |dirty_value − clean_value|
  2.  relative_change         — magnitude / (|clean_value| + 1)
  3.  change_direction        — sign(dirty − clean)  {−1, 0, +1}
  4.  original_magnitude      — |clean_value|  (numeric)
  5.  new_magnitude           — |dirty_value|  (numeric)
  6.  original_log            — log1p(|clean_value|)
  7.  new_log                 — log1p(|dirty_value|)
  8.  feature_name_encoded    — label-encoded column name
  9.  original_value_encoded  — label-encoded clean value (as string)
  10. new_value_encoded        — label-encoded dirty value (as string)

Total combined feature vector: 13 + 10 = 23 features per erroneous cell.

Datasets: LLM Adult · Mixed SOTA Adult · TwiBot-20
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, io, time, json, warnings, datetime, traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, silhouette_score,
)

warnings.filterwarnings("ignore")

try:
    import hdbscan as hdbscan_lib
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# ── path setup ──────────────────────────────────────────────────────────────
HEURISTIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
if HEURISTIC_ROOT not in sys.path:
    sys.path.insert(0, HEURISTIC_ROOT)

from attribution.heuristics.pipeline import AttributionPipeline

# ── paths ────────────────────────────────────────────────────────────────────
BASE        = "/home/mohamed/error_injector/llms_baseline"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Dataset directories
LLM_DIRTY   = os.path.join(BASE, "adult_income_dataset/tenth-trial/data/raw/run_v2_20260617_173016/manipulated_records.csv")
LLM_MASK    = os.path.join(BASE, "adult_income_dataset/tenth-trial/data/raw/run_v2_20260617_173016/masks.csv")
LLM_CLEAN   = os.path.join(BASE, "adult_income_dataset/tenth-trial/data/raw/correct_records.csv")

KIREEV_DIRTY = os.path.join(BASE, "mixed_error_pipeline/output/adult_phase2_final.csv")
KIREEV_MASK  = os.path.join(BASE, "mixed_error_pipeline/output/mask_combined.csv")
KIREEV_CLEAN = os.path.join(BASE, "mixed_error_pipeline/output/adult_clean.csv")

TWITTER_DIRTY = os.path.join(BASE, "mixed_error_pipeline_twitter/output/twibot20_phase2_final.csv")
TWITTER_MASK  = os.path.join(BASE, "mixed_error_pipeline_twitter/output/mask_combined.csv")
TWITTER_CLEAN = os.path.join(BASE, "mixed_error_pipeline_twitter/output/twibot20_clean.csv")

# ── config ───────────────────────────────────────────────────────────────────
ADULT_CFG = {
    "target_col":        "class",
    "codependent_pairs": [("education", "education-num")],
    "sensitive_cols":    ["race", "sex"],
}
TWITTER_CFG = {
    "target_col":        "label",
    "codependent_pairs": [],
    "sensitive_cols":    [],
}

RANDOM_STATE          = 42
N_CLUSTERS            = 15
SAMPLE_FRAC           = 0.01
MIN_SAMPLES_PER_CLUSTER = 5
KNN_K                 = 7

STAT_FEATURE_NAMES = [
    "stat_change_magnitude",
    "stat_relative_change",
    "stat_change_direction",
    "stat_original_magnitude",
    "stat_new_magnitude",
    "stat_original_log",
    "stat_new_log",
    "stat_feature_name_encoded",
    "stat_original_value_encoded",
    "stat_new_value_encoded",
]

# ── tee logger ───────────────────────────────────────────────────────────────
class TeeLogger:
    def __init__(self):
        self.terminal = sys.stdout
        self.buffer   = io.StringIO()
    def write(self, msg):
        self.terminal.write(msg)
        self.terminal.flush()
        self.buffer.write(msg)
        self.buffer.flush()
    def flush(self):
        self.terminal.flush()
        self.buffer.flush()
    def get_log(self):
        return self.buffer.getvalue()


# ============================================================================
# Statistical feature builder
# ============================================================================

def build_stat_features(dirty_df: pd.DataFrame,
                        clean_df: pd.DataFrame,
                        mask_df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised cell-level statistical feature builder.

    For the LLM dataset clean_df has 1/3 the rows of dirty_df — we
    expand it by repeating each row 3 times before doing column-wise ops.
    For Mixed SOTA and Twitter, shapes match 1:1.

    Returns
    -------
    pd.DataFrame with MultiIndex (row_idx, col_name) and columns
    STAT_FEATURE_NAMES.  Only erroneous cells (mask != 0) are included.
    """
    is_3x = (len(dirty_df) == 3 * len(clean_df))
    if is_3x:
        clean_aligned = clean_df.loc[clean_df.index.repeat(3)].reset_index(drop=True)
    else:
        clean_aligned = clean_df.reset_index(drop=True)

    dirty_r  = dirty_df.reset_index(drop=True)
    mask_r   = mask_df.reset_index(drop=True)

    all_cols = mask_r.columns.tolist()
    col_enc  = {c: i for i, c in enumerate(all_cols)}

    # Build long-format error-cell table via melt
    mask_long = mask_r.copy()
    mask_long.index.name = "row_idx"
    mask_long = mask_long.reset_index().melt(
        id_vars="row_idx", var_name="col_name", value_name="intent"
    )
    mask_long = mask_long[mask_long["intent"] != 0].reset_index(drop=True)

    # Look up original and new values
    def _get_vals(df, rows_idx, cols):
        """Vectorised lookup: for each (row, col) pair return the value."""
        # We use fancy indexing via to_numpy
        arr = df.to_numpy()
        col_map = {c: j for j, c in enumerate(df.columns)}
        col_indices = [col_map.get(c, 0) for c in cols]
        vals = arr[rows_idx, col_indices]
        return vals

    row_idxs = mask_long["row_idx"].values
    col_names = mask_long["col_name"].values

    orig_vals = _get_vals(clean_aligned if is_3x else clean_aligned,
                          row_idxs, col_names)
    new_vals  = _get_vals(dirty_r, row_idxs, col_names)

    # Numeric coercion
    def to_num(arr):
        result = np.zeros(len(arr), dtype=float)
        for i, v in enumerate(arr):
            try:
                result[i] = float(v) if pd.notna(v) else 0.0
            except (ValueError, TypeError):
                result[i] = 0.0
        return result

    orig_num = to_num(orig_vals)
    new_num  = to_num(new_vals)

    diff      = new_num - orig_num
    magnitude = np.abs(diff)
    # For non-numeric cells (magnitude stays 0 after coercion but values differ
    # as strings) set magnitude = 1
    for i, (ov, nv) in enumerate(zip(orig_vals, new_vals)):
        if str(ov) != str(nv) and magnitude[i] == 0.0:
            magnitude[i] = 1.0

    relative_change  = magnitude / (np.abs(orig_num) + 1.0)
    change_direction = np.sign(diff)
    orig_log         = np.log1p(np.abs(orig_num))
    new_log          = np.log1p(np.abs(new_num))
    feat_enc         = np.array([col_enc.get(c, 0) for c in col_names], dtype=float)

    # Encode original and new values per column
    per_col_le: dict[str, LabelEncoder] = {}
    for col in all_cols:
        if col not in dirty_r.columns:
            continue
        orig_s = clean_aligned[col].astype(str).tolist() if col in clean_aligned.columns else []
        new_s  = dirty_r[col].astype(str).tolist()
        le = LabelEncoder().fit(sorted(set(orig_s + new_s)))
        per_col_le[col] = le

    orig_enc_arr = np.full(len(mask_long), -1, dtype=float)
    new_enc_arr  = np.full(len(mask_long), -1, dtype=float)
    for i, (col, ov, nv) in enumerate(zip(col_names, orig_vals, new_vals)):
        le = per_col_le.get(col)
        if le is None:
            continue
        try:
            orig_enc_arr[i] = float(le.transform([str(ov)])[0])
        except ValueError:
            pass
        try:
            new_enc_arr[i] = float(le.transform([str(nv)])[0])
        except ValueError:
            pass

    stat_df = pd.DataFrame({
        "row_idx":  row_idxs,
        "col_name": col_names,
        "stat_change_magnitude":       magnitude,
        "stat_relative_change":        relative_change,
        "stat_change_direction":       change_direction,
        "stat_original_magnitude":     np.abs(orig_num),
        "stat_new_magnitude":          np.abs(new_num),
        "stat_original_log":           orig_log,
        "stat_new_log":                new_log,
        "stat_feature_name_encoded":   feat_enc,
        "stat_original_value_encoded": orig_enc_arr,
        "stat_new_value_encoded":      new_enc_arr,
    })
    stat_df = stat_df.set_index(["row_idx", "col_name"])
    stat_df.index = pd.MultiIndex.from_tuples(
        stat_df.index, names=["row_idx", "col_name"]
    )
    return stat_df


# ============================================================================
# Data loaders (return dirty, blind_mask, full_mask, labels, clean_df)
# ============================================================================

def _load(dirty_path, mask_path, clean_path):
    dirty      = pd.read_csv(dirty_path)
    mask_full  = pd.read_csv(mask_path)
    clean      = pd.read_csv(clean_path)
    blind_mask = (mask_full != 0).astype(int)

    labels_dict = {}
    for row_idx in range(len(mask_full)):
        for col_name in mask_full.columns:
            val = mask_full.iloc[row_idx][col_name]
            if val != 0:
                labels_dict[(row_idx, col_name)] = 1 if val == 1 else 0

    labels = pd.Series(labels_dict, name="intent_label")
    labels.index = pd.MultiIndex.from_tuples(
        labels.index, names=["row_idx", "col_name"]
    )
    return dirty, blind_mask, mask_full, labels, clean


def load_llm_dataset():
    return _load(LLM_DIRTY, LLM_MASK, LLM_CLEAN)

def load_kireev_dataset():
    return _load(KIREEV_DIRTY, KIREEV_MASK, KIREEV_CLEAN)

def load_twitter_dataset():
    return _load(TWITTER_DIRTY, TWITTER_MASK, TWITTER_CLEAN)


# ============================================================================
# Clustering
# ============================================================================

def cluster_kmeans(X, n_clusters=N_CLUSTERS):
    t0 = time.time()
    km     = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil    = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    info   = {"algorithm": "kmeans", "n_clusters": n_clusters,
               "noise_count": 0, "silhouette": round(sil, 4),
               "runtime": round(time.time()-t0, 2)}
    print(f"  KMeans: {n_clusters} clusters, sil={sil:.4f}, t={info['runtime']:.2f}s")
    return labels, info


def cluster_dbscan(X, min_samples=5):
    t0 = time.time()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min_samples).fit(X)
    dists, _ = nn.kneighbors(X)
    eps = float(np.percentile(np.sort(dists[:, -1]), 90))
    if eps <= 0: eps = 0.5
    labels  = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
    noise   = int((labels == -1).sum())
    valid   = labels != -1
    sil = silhouette_score(X[valid], labels[valid],
                            sample_size=min(5000, valid.sum())) \
          if n_cl >= 2 and valid.sum() > n_cl else -1.0
    info = {"algorithm": "dbscan", "eps": round(eps, 4),
            "min_samples": min_samples, "n_clusters": n_cl,
            "noise_count": noise, "silhouette": round(sil, 4),
            "runtime": round(time.time()-t0, 2)}
    print(f"  DBSCAN: eps={eps:.3f}, {n_cl} clusters, noise={noise}, sil={sil:.4f}, t={info['runtime']:.2f}s")
    return labels, info


def cluster_hierarchical(X, linkage="ward", n_clusters=N_CLUSTERS):
    t0     = time.time()
    labels = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit_predict(X)
    sil    = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    info   = {"algorithm": f"hierarchical_{linkage}", "n_clusters": n_clusters,
               "linkage": linkage, "noise_count": 0,
               "silhouette": round(sil, 4), "runtime": round(time.time()-t0, 2)}
    print(f"  Hierarchical-{linkage}: {n_clusters} clusters, sil={sil:.4f}, t={info['runtime']:.2f}s")
    return labels, info


def cluster_gmm(X, n_components=N_CLUSTERS):
    t0  = time.time()
    gmm = GaussianMixture(n_components=n_components, random_state=RANDOM_STATE, n_init=3)
    labels = gmm.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(5000, len(X)))
    info = {"algorithm": "gmm", "n_components": n_components,
             "bic": round(gmm.bic(X), 2), "aic": round(gmm.aic(X), 2),
             "converged": bool(gmm.converged_), "noise_count": 0,
             "silhouette": round(sil, 4), "runtime": round(time.time()-t0, 2)}
    print(f"  GMM: {n_components} components, converged={gmm.converged_}, sil={sil:.4f}, t={info['runtime']:.2f}s")
    return labels, info


def cluster_hdbscan(X, min_cluster_size=50, min_samples=5):
    t0 = time.time()
    if not HDBSCAN_AVAILABLE:
        print("  HDBSCAN not available — falling back to KMeans")
        return cluster_kmeans(X)
    clusterer = hdbscan_lib.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples, prediction_data=True)
    labels = clusterer.fit_predict(X)
    n_cl  = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    valid = labels != -1
    sil = silhouette_score(X[valid], labels[valid],
                            sample_size=min(5000, valid.sum())) \
          if n_cl >= 2 and valid.sum() > n_cl else -1.0
    info = {"algorithm": "hdbscan", "min_cluster_size": min_cluster_size,
             "min_samples": min_samples, "n_clusters": n_cl,
             "noise_count": noise, "silhouette": round(sil, 4),
             "runtime": round(time.time()-t0, 2)}
    print(f"  HDBSCAN: {n_cl} clusters, noise={noise}, sil={sil:.4f}, t={info['runtime']:.2f}s")
    return labels, info


CLUSTERING_METHODS = {
    "kmeans":               lambda X: cluster_kmeans(X),
    "dbscan":               lambda X: cluster_dbscan(X),
    "hierarchical_ward":    lambda X: cluster_hierarchical(X, "ward"),
    "hierarchical_average": lambda X: cluster_hierarchical(X, "average"),
    "gmm":                  lambda X: cluster_gmm(X),
    "hdbscan":              lambda X: cluster_hdbscan(X),
}


# ============================================================================
# Proportional sampling (identical to test_scenario_b.py)
# ============================================================================

def proportional_sample(n_total, cluster_labels):
    target_n = max(int(round(n_total * SAMPLE_FRAC)), 10)
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    if not unique_clusters:
        sampled  = np.random.choice(n_total, target_n, replace=False)
        return sampled, np.setdiff1d(np.arange(n_total), sampled)

    cluster_sizes = {c: int((cluster_labels == c).sum()) for c in unique_clusters}
    total_valid   = sum(cluster_sizes.values())

    allocation = {c: max(MIN_SAMPLES_PER_CLUSTER,
                         int(round(cluster_sizes[c] / total_valid * target_n)))
                  for c in unique_clusters}

    total_alloc = sum(allocation.values())
    diff = total_alloc - target_n
    sorted_c = sorted(unique_clusters, key=lambda c: allocation[c], reverse=True)
    for i in range(abs(diff)):
        c = sorted_c[i % len(sorted_c)]
        if diff > 0:
            allocation[c] = max(MIN_SAMPLES_PER_CLUSTER, allocation[c] - 1)
        else:
            allocation[c] += 1

    rng = np.random.default_rng(RANDOM_STATE)
    all_pos = np.arange(n_total)
    sampled_pos = []
    for c in unique_clusters:
        pos    = all_pos[cluster_labels == c]
        n_draw = min(allocation[c], len(pos))
        sampled_pos.extend(rng.choice(pos, size=n_draw, replace=False).tolist())

    sampled_pos   = np.array(sampled_pos)
    remaining_pos = np.setdiff1d(all_pos, sampled_pos)
    return sampled_pos, remaining_pos


# ============================================================================
# Metrics / label methods  (same as test_scenario_b.py)
# ============================================================================

def _metrics(y_true, y_pred, method_name, **extra):
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")
    row = {
        "method":         method_name,
        "accuracy":       round(accuracy_score(y_true, y_pred), 4),
        "precision":      round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":         round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_intentional": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "f1_weighted":    round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "auc":            round(auc, 4) if not np.isnan(auc) else "nan",
    }
    row.update(extra)
    return row


def _print_m(row):
    print(f"    Acc={row['accuracy']:.4f}  F1={row['f1_intentional']:.4f}  "
          f"F1w={row['f1_weighted']:.4f}  AUC={row['auc']}")


def method_rf(X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    row = _metrics(y_te, rf.predict(X_te), "RandomForest",
                   runtime=round(time.time()-t0, 2))
    _print_m(row)
    return row


def method_mv(cluster_labels, sampled_pos, remaining_pos, y_all):
    t0 = time.time()
    majority = {}
    for c in np.unique(cluster_labels):
        if c == -1: continue
        sp = sampled_pos[cluster_labels[sampled_pos] == c]
        majority[c] = int(np.bincount(y_all[sp]).argmax()) if len(sp) else \
                      int(np.bincount(y_all).argmax())
    overall = int(np.bincount(y_all[sampled_pos]).argmax())
    y_pred  = np.array([majority.get(cluster_labels[i], overall) for i in remaining_pos])
    row = _metrics(y_all[remaining_pos], y_pred, "ClusterMajorityVote",
                   runtime=round(time.time()-t0, 2))
    _print_m(row)
    return row


def method_knn(X_tr, y_tr, X_te, y_te, k=KNN_K):
    t0  = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    row = _metrics(y_te, knn.predict(X_te), f"KNN_k{k}",
                   runtime=round(time.time()-t0, 2))
    _print_m(row)
    return row


def method_lp(X_tr, y_tr, X_te, y_te, X_all):
    t0    = time.time()
    y_par = np.concatenate([y_tr, np.full(len(X_te), -1)])
    lp    = LabelPropagation(kernel="knn", n_neighbors=KNN_K, max_iter=1000)
    lp.fit(X_all, y_par)
    y_pred = lp.predict(X_all[len(X_tr):])
    fb     = int(np.bincount(y_tr).argmax())
    y_pred = np.where(y_pred == -1, fb, y_pred)
    row    = _metrics(y_te, y_pred, "LabelPropagation",
                      runtime=round(time.time()-t0, 2))
    _print_m(row)
    return row


def method_ls(X_tr, y_tr, X_te, y_te, X_all):
    t0    = time.time()
    y_par = np.concatenate([y_tr, np.full(len(X_te), -1)])
    ls    = LabelSpreading(kernel="knn", n_neighbors=KNN_K, alpha=0.2, max_iter=1000)
    ls.fit(X_all, y_par)
    y_pred = ls.predict(X_all[len(X_tr):])
    fb     = int(np.bincount(y_tr).argmax())
    y_pred = np.where(y_pred == -1, fb, y_pred)
    row    = _metrics(y_te, y_pred, "LabelSpreading",
                      runtime=round(time.time()-t0, 2))
    _print_m(row)
    return row


# ============================================================================
# Per-algorithm runner
# ============================================================================

def run_algorithm(algo_name, cluster_fn, feat_df, X_scaled, y_all, ds_dir):
    import pathlib
    algo_dir = pathlib.Path(ds_dir) / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'─'*60}")
    print(f"  Clustering: {algo_name.upper()}")
    print(f"  {'─'*60}")

    cluster_labels, cluster_info = cluster_fn(X_scaled)
    n_total = len(feat_df)

    # Save assignments
    asgn = feat_df.copy()
    asgn["cluster"]    = cluster_labels
    asgn["true_label"] = y_all
    asgn.to_csv(algo_dir / "cluster_assignments.csv")

    sampled_pos, remaining_pos = proportional_sample(n_total, cluster_labels)
    n_labeled  = len(sampled_pos)
    n_test     = len(remaining_pos)
    pct_labeled = n_labeled / n_total * 100

    print(f"  Cells total : {n_total}")
    print(f"  Labeled     : {n_labeled} ({pct_labeled:.2f}%)")
    print(f"  Test        : {n_test}")
    print(f"  Clusters    : {cluster_info.get('n_clusters', '?')}")

    # Save sampled cells
    sampled_df = feat_df.iloc[sampled_pos].copy()
    sampled_df["cluster"]    = cluster_labels[sampled_pos]
    sampled_df["true_label"] = y_all[sampled_pos]
    sampled_df.to_csv(algo_dir / "sampled_cells.csv")

    extra = {"n_labeled": n_labeled, "n_test": n_test,
             "pct_labeled": round(pct_labeled, 2), **cluster_info}

    X_tr  = X_scaled[sampled_pos];  y_tr = y_all[sampled_pos]
    X_te  = X_scaled[remaining_pos]; y_te = y_all[remaining_pos]
    X_all = np.vstack([X_tr, X_te])

    rows = []
    print(f"\n  [A] Random Forest")
    r = method_rf(X_tr, y_tr, X_te, y_te);  r.update(extra)
    pd.DataFrame([r]).to_csv(algo_dir / "metrics_rf.csv", index=False); rows.append(r)

    print(f"\n  [B] Cluster Majority Vote")
    r = method_mv(cluster_labels, sampled_pos, remaining_pos, y_all); r.update(extra)
    pd.DataFrame([r]).to_csv(algo_dir / "metrics_majority_vote.csv", index=False); rows.append(r)

    print(f"\n  [C] KNN (k={KNN_K})")
    r = method_knn(X_tr, y_tr, X_te, y_te);  r.update(extra)
    pd.DataFrame([r]).to_csv(algo_dir / "metrics_knn.csv", index=False); rows.append(r)

    print(f"\n  [D] Label Propagation")
    r = method_lp(X_tr, y_tr, X_te, y_te, X_all); r.update(extra)
    pd.DataFrame([r]).to_csv(algo_dir / "metrics_label_propagation.csv", index=False); rows.append(r)

    print(f"\n  [E] Label Spreading")
    r = method_ls(X_tr, y_tr, X_te, y_te, X_all); r.update(extra)
    pd.DataFrame([r]).to_csv(algo_dir / "metrics_label_spreading.csv", index=False); rows.append(r)

    with open(algo_dir / "cluster_info.json", "w") as fh:
        json.dump(cluster_info, fh, indent=2)

    return rows


# ============================================================================
# Dataset runner
# ============================================================================

def run_dataset(name, load_fn, pipeline_cfg, run_dir):
    import pathlib
    ds_dir = pathlib.Path(run_dir) / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DATASET: {name.upper()}")
    print(f"{'='*70}")

    # 1 — Load
    print("\n[1] Loading data...")
    t0 = time.time()
    dirty, blind_mask, mask_full, labels, clean_df = load_fn()
    print(f"  Dirty shape  : {dirty.shape}")
    print(f"  Clean shape  : {clean_df.shape}")
    print(f"  Error cells  : {len(labels)}")
    print(f"  Intentional  : {(labels == 1).sum()} ({(labels==1).mean()*100:.1f}%)")
    print(f"  Unintentional: {(labels == 0).sum()} ({(labels==0).mean()*100:.1f}%)")

    # 2 — Heuristic features
    print("\n[2] Computing heuristic features (AttributionPipeline)...")
    pipe = AttributionPipeline(
        target_col=pipeline_cfg["target_col"],
        codependent_pairs=pipeline_cfg["codependent_pairs"],
        sensitive_cols=pipeline_cfg["sensitive_cols"],
    )
    pipe.fit(dirty, blind_mask)
    heur_df = pipe.compute_features(dirty, blind_mask)
    print(f"  Heuristic feature matrix : {heur_df.shape}  "
          f"(columns: {heur_df.columns.tolist()})")

    # 3 — Statistical features (cell-level value statistics)
    print("\n[3] Computing statistical features (cell-level value statistics)...")
    stat_df = build_stat_features(dirty, clean_df, mask_full)
    print(f"  Statistical feature matrix: {stat_df.shape}  "
          f"(columns: {stat_df.columns.tolist()})")

    # 4 — Align and concatenate
    print("\n[4] Aligning and concatenating feature matrices...")
    common_idx = heur_df.index.intersection(stat_df.index).intersection(labels.index)
    heur_aligned = heur_df.loc[common_idx]
    stat_aligned = stat_df.loc[common_idx]
    y_all        = labels.loc[common_idx].values.astype(int)

    combined_df = pd.concat([heur_aligned, stat_aligned], axis=1)
    print(f"  Combined feature matrix  : {combined_df.shape}")
    print(f"  Feature names: {combined_df.columns.tolist()}")
    combined_df["true_label"] = y_all
    combined_df.to_csv(ds_dir / "feature_matrix_combined.csv")
    combined_df = combined_df.drop(columns=["true_label"])

    # 5 — Scale
    print("\n[5] Scaling features...")
    X_raw    = np.nan_to_num(combined_df.values, nan=-999.0)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print(f"  Total cells    : {len(combined_df)}")
    print(f"  Feature dim    : {X_scaled.shape[1]}")
    print(f"  Time so far    : {time.time()-t0:.2f}s")

    # 6 — Run all clustering × label method combinations
    print("\n[6] Running 6 clustering algorithms × 5 label methods...")
    all_rows = []

    for algo_name, cluster_fn in CLUSTERING_METHODS.items():
        try:
            method_rows = run_algorithm(
                algo_name, cluster_fn, combined_df, X_scaled, y_all, ds_dir
            )
            for row in method_rows:
                row["dataset"]   = name
                row["algorithm"] = algo_name
            all_rows.extend(method_rows)
        except Exception as exc:
            print(f"\n  ⚠ {algo_name} FAILED: {exc}")
            traceback.print_exc()

    # 7 — Save comparison table
    if all_rows:
        comp_df = pd.DataFrame(all_rows)
        comp_df.to_csv(ds_dir / "comparison_table.csv", index=False)

        print(f"\n[7] Summary — best F1-weighted per clustering:")
        for algo in comp_df["algorithm"].unique():
            sub  = comp_df[comp_df["algorithm"] == algo]
            best = sub.loc[sub["f1_weighted"].idxmax()]
            print(f"  {algo:<22s}  best={best['method']:<22s}  "
                  f"F1w={best['f1_weighted']:.4f}  F1_int={best['f1_intentional']:.4f}")

    return all_rows


# ============================================================================
# Main
# ============================================================================

def main():
    np.random.seed(RANDOM_STATE)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(OUTPUT_ROOT, f"run_scenario_b_combined_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    tee       = TeeLogger()
    sys.stdout = tee

    print(f"Scenario B — COMBINED (Statistical + Heuristic) Features")
    print(f"Run ID    : {timestamp}")
    print(f"Run dir   : {run_dir}")
    print(f"Features  : 13 heuristic + 10 statistical = 23 combined per cell")
    print(f"Datasets  : LLM Adult · Mixed SOTA Adult · TwiBot-20")
    print(f"Algorithms: {list(CLUSTERING_METHODS.keys())}")
    print(f"Methods   : RF · MajorityVote · KNN · LabelProp · LabelSpreading")
    print(f"Sample %  : {SAMPLE_FRAC*100:.1f}%")
    print(f"HDBSCAN   : {'available' if HDBSCAN_AVAILABLE else 'NOT available'}")

    DATASETS = [
        ("llm",     load_llm_dataset,     ADULT_CFG),
        ("kireev",  load_kireev_dataset,  ADULT_CFG),
        ("twitter", load_twitter_dataset, TWITTER_CFG),
    ]

    all_rows     = []
    summary_rows = []

    for ds_name, load_fn, cfg in DATASETS:
        t_ds = time.time()
        try:
            rows = run_dataset(ds_name, load_fn, cfg, run_dir)
            all_rows.extend(rows)
            summary_rows.append({
                "dataset": ds_name, "status": "ok",
                "n_combinations": len(rows),
                "elapsed_s": round(time.time()-t_ds, 1),
            })
        except Exception as exc:
            print(f"\n✗ Dataset '{ds_name}' FAILED: {exc}")
            traceback.print_exc()
            summary_rows.append({
                "dataset": ds_name, "status": f"error: {exc}",
                "n_combinations": 0,
                "elapsed_s": round(time.time()-t_ds, 1),
            })

    # Global comparison table
    if all_rows:
        comp_df = pd.DataFrame(all_rows)
        front   = ["dataset", "algorithm", "method", "f1_weighted",
                   "f1_intentional", "accuracy", "precision", "recall",
                   "auc", "runtime", "n_labeled", "n_test", "pct_labeled"]
        rest    = [c for c in comp_df.columns if c not in front]
        comp_df = comp_df[front + rest]
        comp_df.to_csv(os.path.join(run_dir, "comparison_table.csv"), index=False)

        print(f"\n{'='*70}")
        print("GLOBAL BEST — Combined features, F1-weighted per dataset × algorithm:")
        print(f"{'='*70}")
        for ds in comp_df["dataset"].unique():
            for algo in comp_df[comp_df["dataset"]==ds]["algorithm"].unique():
                sub  = comp_df[(comp_df["dataset"]==ds) & (comp_df["algorithm"]==algo)]
                best = sub.loc[sub["f1_weighted"].idxmax()]
                print(f"  {ds:<8s} {algo:<22s} → {best['method']:<22s} "
                      f"F1w={best['f1_weighted']:.4f}")

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(run_dir, "summary.csv"), index=False
    )

    sys.stdout = tee.terminal
    log_path = os.path.join(run_dir, "run_log.txt")
    with open(log_path, "w") as fh:
        fh.write(tee.get_log())
    print(f"\nRun log : {log_path}")
    print(f"Run dir : {run_dir}")
    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()
