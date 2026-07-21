#!/usr/bin/env python
"""
Heuristic Group Ablation — Scenario B (DBSCAN + RF only)
=========================================================

Tests the impact of removing each heuristic group:
  - Surface  = H1, H2, H3  (features: h1_plausible, h2_min_edit_distance,
                              h2_is_obfuscation, h3_distribution_score)
  - Context  = H4, H5      (features: h4_coherence_score, h5_error_count,
                              h5_codependent_flag)
  - Domain   = H6, H7, H8  (features: h6_column_importance,
                              h7_gain_direction, h7_comprehensibility,
                              h8_is_sensitive, h8_is_majority_value;
                              h7_mutability is excluded from the deployed
                              12-feature fingerprint, see pipeline.py)

Ablation configurations:
  1. Full (H1–H8)        — 12 features
  2. −Surface (H1–H3)    — remove 4 surface features → 8 features
  3. −Context (H4–H5)    — remove 3 context features → 9 features
  4. −Domain  (H6–H8)    — remove 5 domain features  → 7 features
  5. Surface only         — keep only H1–H3            → 4 features
  6. Context only         — keep only H4–H5            → 3 features
  7. Domain only          — keep only H6–H8            → 5 features

Uses DBSCAN+RF on all 3 datasets (same as the best B configuration on Adult).
"""

import matplotlib
matplotlib.use("Agg")

import sys, os, time, warnings, traceback
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score

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

from attribution.heuristics.pipeline import AttributionPipeline, FEATURE_COLUMNS

# ── paths ────────────────────────────────────────────────────────────────────
BASE = "/home/mohamed/error_injector/llms_baseline"
LLM_DIR = os.path.join(
    BASE, "adult_income_dataset/tenth-trial/data/raw/run_v2_20260617_173016"
)
KIREEV_DIR = os.path.join(BASE, "mixed_error_pipeline/output")
TWITTER_DIR = os.path.join(BASE, "mixed_error_pipeline_twitter/output")

# ── config ───────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
SAMPLE_FRAC = 0.01

# ── feature groups ───────────────────────────────────────────────────────────
SURFACE_FEATURES = [
    "h1_plausible",
    "h2_min_edit_distance",
    "h2_is_obfuscation",
    "h3_distribution_score",
]

CONTEXT_FEATURES = [
    "h4_coherence_score",
    "h5_error_count",
    "h5_codependent_flag",
]

DOMAIN_FEATURES = [
    "h6_column_importance",
    # "h7_mutability",  # excluded from the deployed 12-feature fingerprint
    # (see FEATURE_COLUMNS note in pipeline.py)
    "h7_gain_direction",
    "h7_comprehensibility",
    "h8_is_sensitive",
    "h8_is_majority_value",
]

ALL_FEATURES = SURFACE_FEATURES + CONTEXT_FEATURES + DOMAIN_FEATURES
assert set(ALL_FEATURES) == set(FEATURE_COLUMNS), \
    f"Feature mismatch!\n  Groups: {sorted(ALL_FEATURES)}\n  Pipeline: {sorted(FEATURE_COLUMNS)}"

# Ablation configs: (name, feature_list)
ABLATION_CONFIGS = [
    ("Full (H1-H8)",       ALL_FEATURES),
    ("-Surface (H1-H3)",   CONTEXT_FEATURES + DOMAIN_FEATURES),
    ("-Context (H4-H5)",   SURFACE_FEATURES + DOMAIN_FEATURES),
    ("-Domain (H6-H8)",    SURFACE_FEATURES + CONTEXT_FEATURES),
    ("Surface only",       SURFACE_FEATURES),
    ("Context only",       CONTEXT_FEATURES),
    ("Domain only",        DOMAIN_FEATURES),
]


# ===========================================================================
# Data loaders (same as test_scenario_b.py)
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
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["row_idx", "col_name"])
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
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["row_idx", "col_name"])
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
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["row_idx", "col_name"])
    return dirty, blind_mask, labels


# ===========================================================================
# DBSCAN clustering (same parameters as test_scenario_b.py)
# ===========================================================================

def cluster_dbscan(X_scaled, min_samples=5):
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    dists, _ = nn.kneighbors(X_scaled)
    k_dists = np.sort(dists[:, -1])
    eps = float(np.percentile(k_dists, 90))
    if eps <= 0:
        eps = 0.5
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = int((labels == -1).sum())
    return labels, n_clusters, noise


# ===========================================================================
# Proportional sampling (same as test_scenario_b.py)
# ===========================================================================

def proportional_sample(n_total, cluster_labels, sample_frac=SAMPLE_FRAC):
    """
    Cluster-proportional sampling with NO per-cluster minimum.
    Total seed = exactly round(n_total * sample_frac); clusters that round
    to 0 are skipped. A per-cluster floor (the previous MIN_SAMPLES_PER_CLUSTER
    behavior) silently inflated the budget on datasets where DBSCAN found
    many clusters — removed for consistency with the 1% budget claimed
    throughout the paper.
    """
    target_n = max(int(round(n_total * sample_frac)), 2)
    valid_mask = cluster_labels != -1
    unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]

    if len(unique_clusters) == 0:
        rng = np.random.default_rng(RANDOM_STATE)
        sampled_pos = rng.choice(n_total, target_n, replace=False)
        remaining_pos = np.setdiff1d(np.arange(n_total), sampled_pos)
        return sampled_pos, remaining_pos

    cluster_sizes = {c: int((cluster_labels == c).sum()) for c in unique_clusters}
    total_valid = sum(cluster_sizes.values())

    allocation = {}
    for c in unique_clusters:
        base = (cluster_sizes[c] / total_valid) * target_n
        allocation[c] = int(round(base))

    total_allocated = sum(allocation.values())
    diff = total_allocated - target_n
    for c in sorted(unique_clusters, key=lambda c: allocation[c], reverse=True):
        if diff == 0:
            break
        if diff > 0:
            if allocation[c] > 0:
                allocation[c] -= 1
                diff -= 1
        else:
            allocation[c] += 1
            diff += 1

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
# Run one ablation configuration
# ===========================================================================

def run_ablation_config(config_name, feature_subset, feat_df_full, y_all):
    """
    Run DBSCAN + RF with a subset of features.
    Returns dict with metrics.
    """
    # Select feature subset
    available = [f for f in feature_subset if f in feat_df_full.columns]
    if len(available) == 0:
        return {"config": config_name, "n_features": 0,
                "accuracy": 0.0, "f1_weighted": 0.0, "f1_intentional": 0.0,
                "note": "no features available"}

    X_raw = np.nan_to_num(feat_df_full[available].values, nan=-999.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Cluster with DBSCAN
    cluster_labels, n_clusters, n_noise = cluster_dbscan(X_scaled)

    # Sample
    sampled_pos, remaining_pos = proportional_sample(len(feat_df_full), cluster_labels)

    # Train RF
    X_train = X_scaled[sampled_pos]
    y_train = y_all[sampled_pos]
    X_test = X_scaled[remaining_pos]
    y_test = y_all[remaining_pos]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_int = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    return {
        "config": config_name,
        "n_features": len(available),
        "features_used": available,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_intentional": round(f1_int, 4),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_labeled": len(sampled_pos),
        "n_test": len(remaining_pos),
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    np.random.seed(RANDOM_STATE)

    ADULT_CFG = {
        "target_col": "class",
        "codependent_pairs": [("education", "education-num")],
        "sensitive_cols": ["race", "sex"],
    }
    TWITTER_CFG = {
        "target_col": "label",
        "codependent_pairs": [],
        "sensitive_cols": [],
    }

    datasets = [
        ("Adult-LLM",   load_llm_dataset,     ADULT_CFG),
        ("Adult-Mixed",  load_kireev_dataset,  ADULT_CFG),
        ("TwitterBot",   load_twitter_dataset, TWITTER_CFG),
    ]

    all_results = []

    for ds_name, load_fn, ds_cfg in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*70}")

        # Load data
        t0 = time.time()
        dirty, blind_mask, labels = load_fn()
        print(f"  Dirty shape: {dirty.shape}, Error cells: {len(labels)}")

        # Fit pipeline (unsupervised — same for all ablation configs)
        pipe = AttributionPipeline(
            target_col=ds_cfg["target_col"],
            codependent_pairs=ds_cfg["codependent_pairs"],
            sensitive_cols=ds_cfg["sensitive_cols"],
        )
        pipe.fit(dirty, blind_mask)

        # Compute full 12-feature matrix
        feat_df = pipe.compute_features(dirty, blind_mask)
        common_idx = feat_df.index.intersection(labels.index)
        feat_df = feat_df.loc[common_idx]
        y_all = labels.loc[common_idx].values.astype(int)
        print(f"  Feature matrix: {feat_df.shape}, aligned cells: {len(feat_df)}")
        print(f"  Pipeline fit + feature extraction: {time.time()-t0:.1f}s")

        # Run all ablation configs
        for config_name, feature_subset in ABLATION_CONFIGS:
            print(f"\n  --- {config_name} ({len([f for f in feature_subset if f in feat_df.columns])} features) ---")
            try:
                result = run_ablation_config(config_name, feature_subset, feat_df, y_all)
                result["dataset"] = ds_name
                all_results.append(result)
                print(f"      Acc={result['accuracy']:.4f}  F1w={result['f1_weighted']:.4f}  "
                      f"F1int={result['f1_intentional']:.4f}  "
                      f"clusters={result['n_clusters']}  labeled={result['n_labeled']}")
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}\n")

    results_df = pd.DataFrame(all_results)
    
    # Clean display
    for ds_name in ["Adult-LLM", "Adult-Mixed", "TwitterBot"]:
        ds_results = results_df[results_df["dataset"] == ds_name]
        if ds_results.empty:
            continue
        print(f"\n{ds_name}:")
        print(f"  {'Config':<25s} {'#Feat':>5s} {'Acc':>7s} {'F1_w':>7s} {'F1_int':>7s}")
        print(f"  {'-'*55}")
        for _, row in ds_results.iterrows():
            print(f"  {row['config']:<25s} {row['n_features']:>5d} "
                  f"{row['accuracy']:>7.4f} {row['f1_weighted']:>7.4f} {row['f1_intentional']:>7.4f}")

    # Save CSV
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results.csv")
    cols_to_save = ["dataset", "config", "n_features", "accuracy", "f1_weighted", 
                    "f1_intentional", "n_clusters", "n_noise", "n_labeled", "n_test"]
    results_df[cols_to_save].to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # LaTeX-ready table (for paper)
    print(f"\n\n{'='*70}")
    print("LATEX TABLE (F1_w, DBSCAN+RF):")
    print(f"{'='*70}\n")
    
    for ds_name in ["Adult-LLM", "Adult-Mixed", "TwitterBot"]:
        ds_results = results_df[results_df["dataset"] == ds_name]
        if ds_results.empty:
            continue
        for _, row in ds_results.iterrows():
            print(f"  {row['config']:<25s} & {row['n_features']:>2d} & {row['f1_weighted']:.3f} \\\\")
        print()


if __name__ == "__main__":
    main()
