#!/usr/bin/env python3
"""
Oracle vs. Imputed vs. Naive — Feature and Classification Comparison
=====================================================================

Three pipelines side by side:

  ORACLE   : direction/magnitude computed from (observed_value, correct_value)
             — the ideal upper bound, requires ground truth x*
  IMPUTED  : direction/magnitude computed from (observed_value, imputed_value)
             — our MICE pipeline, no ground truth needed
  NAIVE    : no reference point — only (observed_value, column stats)
             — baseline: can you detect intent without any imputation?

For each pipeline we compute the same diagnostic features, then run a 5-fold
stratified RF classifier and compare F1-macro, F1-intentional, F1-unintentional.

Additionally computes cell-level feature agreement between Oracle and Imputed
to understand *where* imputation succeeds and where it fails.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def _to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def build_oracle_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features using the ground-truth correct value (x*).
    Only cells where correct_value is not NaN are included.

    Features:
        oracle_direction   : sign(observed - correct)  — +1=inflated, -1=deflated
        oracle_magnitude   : |observed - correct|
        oracle_rel_change  : |observed - correct| / (|correct| + 1)
        oracle_cat_changed : 1 if observed != correct (categorical), else 0
        is_categorical
        intent_label
    """
    df = results_df[results_df["correct_value"].notna()].copy()

    rows = []
    for _, r in df.iterrows():
        obs = r["observed_value"]
        cor = r["correct_value"]
        is_cat = (r["col_type"] == "categorical")

        if is_cat:
            changed = int(str(obs).strip().lower() != str(cor).strip().lower())
            rows.append({
                "oracle_direction": changed,          # 1=different, 0=same
                "oracle_magnitude": changed,
                "oracle_rel_change": changed,
                "oracle_cat_changed": changed,
                "is_categorical": 1,
                "intent_label": int(r["intent_label"]),
                "column": r["column"],
                "row_idx": r["row_idx"],
            })
        else:
            obs_f = _to_numeric(pd.Series([obs])).iloc[0]
            cor_f = _to_numeric(pd.Series([cor])).iloc[0]
            if pd.isna(obs_f) or pd.isna(cor_f):
                continue
            direction = np.sign(obs_f - cor_f)
            magnitude = abs(obs_f - cor_f)
            rel_change = magnitude / (abs(cor_f) + 1)
            rows.append({
                "oracle_direction": direction,
                "oracle_magnitude": magnitude,
                "oracle_rel_change": rel_change,
                "oracle_cat_changed": 0,
                "is_categorical": 0,
                "intent_label": int(r["intent_label"]),
                "column": r["column"],
                "row_idx": r["row_idx"],
            })

    return pd.DataFrame(rows)


def build_imputed_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features using the MICE-imputed value (x̂*).
    Mirrors the DiagnosticFeatureExtractor logic.

    Features (parallel to oracle for direct comparison):
        imp_direction   : sign(observed - imputed)
        imp_magnitude   : |observed - imputed|
        imp_rel_change  : |observed - imputed| / (|imputed| + 1)
        imp_wrc         : |observed - imputed| / (|imputed| * (1 + sigma_tree))
        imp_cat_changed : 1 if observed != imputed (categorical)
        confidence      : 1 / (1 + sigma_tree)
        sigma_tree
        sigma_oob
        is_categorical
        intent_label
    """
    # Use only cells also in oracle set for fair comparison
    df = results_df[results_df["correct_value"].notna()].copy()

    rows = []
    for _, r in df.iterrows():
        obs = r["observed_value"]
        imp = r["imputed_value"]
        sigma = float(r["sigma_tree"])
        is_cat = (r["col_type"] == "categorical")

        if is_cat:
            changed = int(str(obs).strip().lower() != str(imp).strip().lower())
            rows.append({
                "imp_direction": changed,
                "imp_magnitude": changed,
                "imp_rel_change": changed,
                "imp_wrc": changed,
                "imp_cat_changed": changed,
                "confidence": float(r["confidence"]),
                "sigma_tree": sigma,
                "sigma_oob": float(r["sigma_oob"]),
                "is_categorical": 1,
                "intent_label": int(r["intent_label"]),
                "column": r["column"],
                "row_idx": r["row_idx"],
            })
        else:
            obs_f = _to_numeric(pd.Series([obs])).iloc[0]
            imp_f = _to_numeric(pd.Series([imp])).iloc[0]
            if pd.isna(obs_f) or pd.isna(imp_f):
                continue
            direction = np.sign(obs_f - imp_f)
            magnitude = abs(obs_f - imp_f)
            rel_change = magnitude / (abs(imp_f) + 1)
            denom = max(abs(imp_f), 1e-6) * (1.0 + sigma)
            wrc = magnitude / denom
            rows.append({
                "imp_direction": direction,
                "imp_magnitude": magnitude,
                "imp_rel_change": rel_change,
                "imp_wrc": wrc,
                "imp_cat_changed": 0,
                "confidence": float(r["confidence"]),
                "sigma_tree": sigma,
                "sigma_oob": float(r["sigma_oob"]),
                "is_categorical": 0,
                "intent_label": int(r["intent_label"]),
                "column": r["column"],
                "row_idx": r["row_idx"],
            })

    return pd.DataFrame(rows)


def build_naive_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features using ONLY the observed (corrupted) value + column stats.
    No reference point at all — purely based on how unusual the value looks.

    Features:
        obs_numeric     : numeric cast of observed value
        col_mean        : mean of column in clean rows (approximated from all rows here)
        col_std         : std of column
        col_median      : median of column
        z_score         : (observed - mean) / std
        is_categorical  : 1/0
        intent_label
    """
    df = results_df[results_df["correct_value"].notna()].copy()

    # Build column stats from imputed reference (clean-row approximation)
    col_stats = {}
    for col in df["column"].unique():
        subset = df[df["column"] == col]
        nums = _to_numeric(subset["observed_value"].astype(str)).dropna()
        if len(nums) > 1:
            col_stats[col] = {
                "mean": float(nums.mean()),
                "std": max(float(nums.std()), 1e-6),
                "median": float(nums.median()),
            }
        else:
            col_stats[col] = {"mean": 0.0, "std": 1.0, "median": 0.0}

    rows = []
    for _, r in df.iterrows():
        obs = r["observed_value"]
        col = r["column"]
        is_cat = (r["col_type"] == "categorical")

        obs_f = 0.0
        try:
            obs_f = float(obs)
        except (ValueError, TypeError):
            obs_f = 0.0

        stats = col_stats.get(col, {"mean": 0.0, "std": 1.0, "median": 0.0})
        z = (obs_f - stats["mean"]) / stats["std"]

        rows.append({
            "obs_numeric": obs_f,
            "col_mean": stats["mean"],
            "col_std": stats["std"],
            "col_median": stats["median"],
            "z_score": z,
            "is_categorical": int(is_cat),
            "intent_label": int(r["intent_label"]),
            "column": r["column"],
            "row_idx": r["row_idx"],
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray,
           n_folds: int = 5, seed: int = 42) -> dict:
    """5-fold stratified CV with a balanced RF."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc_list, f1m_list, f1i_list, f1u_list = [], [], [], []

    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        y_test = y[test_idx]

        acc_list.append(accuracy_score(y_test, y_pred))
        f1m_list.append(f1_score(y_test, y_pred, average="macro", labels=[-1, 1]))
        f1i_list.append(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1u_list.append(f1_score(y_test, y_pred, pos_label=-1, zero_division=0))

    def ms(lst):
        return {"mean": float(np.mean(lst)), "std": float(np.std(lst))}

    return {
        "accuracy": ms(acc_list),
        "f1_macro": ms(f1m_list),
        "f1_intentional": ms(f1i_list),
        "f1_unintentional": ms(f1u_list),
        "n_samples": len(y),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3.  CELL-LEVEL FEATURE AGREEMENT
# ──────────────────────────────────────────────────────────────────────────────

def compute_feature_agreement(oracle_df: pd.DataFrame,
                               imputed_df: pd.DataFrame) -> dict:
    """
    For each cell where both oracle and imputed features are computed,
    compare them directly.

    Returns per-column and overall agreement on:
        direction_match  : oracle_direction == imp_direction  (numerical only)
        magnitude_ratio  : imp_magnitude / oracle_magnitude   (how close)
    """
    # Merge on (row_idx, column)
    merged = oracle_df.merge(
        imputed_df[["row_idx", "column", "imp_direction", "imp_magnitude",
                    "imp_wrc", "imp_cat_changed", "confidence", "is_categorical"]],
        on=["row_idx", "column"],
        how="inner",
        suffixes=("", "_imp"),
    )
    # After merge, oracle's is_categorical is "is_categorical", imputed's is "is_categorical_imp"
    # Use oracle's version (they should agree)
    is_cat_col = "is_categorical"

    num = merged[merged[is_cat_col] == 0].copy()
    cat = merged[merged[is_cat_col] == 1].copy()

    results = {}

    # Numerical direction match
    if len(num) > 0:
        num["dir_match"] = (num["oracle_direction"] == num["imp_direction"]).astype(int)
        # Avoid /0
        safe_oracle_mag = num["oracle_magnitude"].replace(0, np.nan)
        num["mag_ratio"] = num["imp_magnitude"] / safe_oracle_mag

        results["numerical"] = {
            "n": len(num),
            "direction_match_rate": float(num["dir_match"].mean()),
            "median_magnitude_ratio": float(num["mag_ratio"].median()),
            "mean_magnitude_ratio": float(num["mag_ratio"].mean()),
            # Stratified by confidence
            "direction_match_high_conf": float(
                num[num["confidence"] >= 0.5]["dir_match"].mean()
                if (num["confidence"] >= 0.5).sum() > 0 else float("nan")
            ),
            "direction_match_low_conf": float(
                num[num["confidence"] < 0.1]["dir_match"].mean()
                if (num["confidence"] < 0.1).sum() > 0 else float("nan")
            ),
        }

        # Per-column direction match
        per_col = {}
        for col, grp in num.groupby("column"):
            per_col[col] = {
                "n": len(grp),
                "direction_match": round(float(grp["dir_match"].mean()), 4),
                "median_mag_ratio": round(float(
                    (grp["imp_magnitude"] / grp["oracle_magnitude"].replace(0, np.nan)).median()
                ), 4),
            }
        results["numerical"]["per_column"] = per_col

    # Categorical changed/not changed agreement
    if len(cat) > 0:
        cat["change_match"] = (
            (cat["oracle_cat_changed"] == cat["imp_cat_changed"]) |
            ((cat["oracle_cat_changed"] == 1) & (cat["imp_cat_changed"] == 1))
        ).astype(int)
        results["categorical"] = {
            "n": len(cat),
            "change_detection_match_rate": float(cat["change_match"].mean()),
        }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MAIN COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(results_path: str, output_path: str = None,
                   n_folds: int = 5, seed: int = 42) -> dict:
    """
    Full Oracle vs. Imputed vs. Naive comparison.

    Parameters
    ----------
    results_path : str
        Path to imputation_results.csv (must contain correct_value column).
    output_path : str, optional
        If given, writes a markdown report there.
    n_folds : int
        CV folds for classification.
    seed : int
        Random seed.

    Returns
    -------
    report : dict
        All comparison metrics.
    """
    print("=" * 70)
    print("ORACLE vs. IMPUTED vs. NAIVE — COMPARISON")
    print("=" * 70)

    results_df = pd.read_csv(results_path)
    has_gt = results_df["correct_value"].notna().sum()
    print(f"\nLoaded {len(results_df)} cells  ({has_gt} with ground truth)")

    if has_gt == 0:
        raise ValueError("No correct_value entries found — cannot run oracle comparison.")

    # ── Build features ─────────────────────────────────────────────────────────
    print("\nBuilding feature sets...")
    oracle_feats  = build_oracle_features(results_df)
    imputed_feats = build_imputed_features(results_df)
    naive_feats   = build_naive_features(results_df)

    # Align all three to the same cell set (inner join on row_idx+column)
    common_keys = (
        set(zip(oracle_feats["row_idx"], oracle_feats["column"])) &
        set(zip(imputed_feats["row_idx"], imputed_feats["column"])) &
        set(zip(naive_feats["row_idx"], naive_feats["column"]))
    )
    print(f"Common cells across all three pipelines: {len(common_keys)}")

    def filter_common(df):
        mask = [
            (r, c) in common_keys
            for r, c in zip(df["row_idx"], df["column"])
        ]
        return df[mask].reset_index(drop=True)

    oracle_feats  = filter_common(oracle_feats)
    imputed_feats = filter_common(imputed_feats)
    naive_feats   = filter_common(naive_feats)

    assert (oracle_feats["intent_label"].values == imputed_feats["intent_label"].values).all()
    y = oracle_feats["intent_label"].values.astype(int)

    # ── Classification ─────────────────────────────────────────────────────────
    print("\nRunning 5-fold CV classification...")

    oracle_X = oracle_feats[
        ["oracle_direction", "oracle_magnitude", "oracle_rel_change",
         "oracle_cat_changed", "is_categorical"]
    ].fillna(0).values.astype(float)

    imputed_X = imputed_feats[
        ["imp_direction", "imp_magnitude", "imp_rel_change", "imp_wrc",
         "imp_cat_changed", "confidence", "sigma_tree", "sigma_oob",
         "is_categorical"]
    ].fillna(0).values.astype(float)

    naive_X = naive_feats[
        ["obs_numeric", "col_mean", "col_std", "col_median",
         "z_score", "is_categorical"]
    ].fillna(0).values.astype(float)

    print("  [1/3] Oracle features...")
    oracle_cv  = run_cv(oracle_X, y, n_folds, seed)
    print("  [2/3] Imputed features (MICE)...")
    imputed_cv = run_cv(imputed_X, y, n_folds, seed)
    print("  [3/3] Naive features (no imputation)...")
    naive_cv   = run_cv(naive_X, y, n_folds, seed)

    # ── Feature agreement ─────────────────────────────────────────────────────
    print("\nComputing feature agreement (Oracle ↔ Imputed)...")
    agreement = compute_feature_agreement(oracle_feats, imputed_feats)

    # ── Cell-level direction comparison table ─────────────────────────────────
    num_only = oracle_feats[oracle_feats["is_categorical"] == 0].copy()
    imp_num  = imputed_feats[imputed_feats["is_categorical"] == 0].copy()
    merged_num = num_only.merge(
        imp_num[["row_idx", "column", "imp_direction", "imp_magnitude",
                 "imp_wrc", "confidence"]],
        on=["row_idx", "column"],
    )
    merged_num["dir_match"] = (
        merged_num["oracle_direction"] == merged_num["imp_direction"]
    )

    # ── Print results ──────────────────────────────────────────────────────────
    _print_results(oracle_cv, imputed_cv, naive_cv, agreement, merged_num)

    report = {
        "n_cells": len(y),
        "n_intentional": int((y == 1).sum()),
        "n_unintentional": int((y == -1).sum()),
        "oracle_cv": oracle_cv,
        "imputed_cv": imputed_cv,
        "naive_cv": naive_cv,
        "feature_agreement": agreement,
    }

    if output_path:
        _write_report(report, oracle_cv, imputed_cv, naive_cv,
                      agreement, merged_num, output_path)
        print(f"\n✓ Report saved to: {output_path}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PRINTING / REPORTING
# ──────────────────────────────────────────────────────────────────────────────

def _print_results(oracle_cv, imputed_cv, naive_cv, agreement, merged_num):
    print()
    print("=" * 70)
    print("CLASSIFICATION RESULTS (5-fold CV)")
    print("=" * 70)
    header = f"{'Pipeline':<22}  {'Acc':>6}  {'F1-macro':>8}  {'F1-intent':>9}  {'F1-uninten':>10}  {'n':>5}"
    print(header)
    print("-" * 70)

    for name, cv in [("Oracle (x*)", oracle_cv),
                     ("Imputed (MICE x̂*)", imputed_cv),
                     ("Naive (no ref)", naive_cv)]:
        acc = f"{cv['accuracy']['mean']:.4f}±{cv['accuracy']['std']:.3f}"
        f1m = f"{cv['f1_macro']['mean']:.4f}"
        f1i = f"{cv['f1_intentional']['mean']:.4f}"
        f1u = f"{cv['f1_unintentional']['mean']:.4f}"
        n   = cv["n_samples"]
        print(f"  {name:<20}  {acc:>12}  {f1m:>8}  {f1i:>9}  {f1u:>10}  {n:>5}")

    # Gap analysis
    oracle_f1  = oracle_cv["f1_macro"]["mean"]
    imputed_f1 = imputed_cv["f1_macro"]["mean"]
    naive_f1   = naive_cv["f1_macro"]["mean"]

    gap_oracle_imputed = oracle_f1 - imputed_f1
    gap_imputed_naive  = imputed_f1 - naive_f1
    pct_recovered      = gap_imputed_naive / max(oracle_f1 - naive_f1, 1e-6) * 100

    print()
    print(f"  Oracle  → Imputed gap : −{gap_oracle_imputed:.4f} F1  (imputation cost)")
    print(f"  Imputed → Naive gap   : +{gap_imputed_naive:.4f} F1  (imputation value)")
    print(f"  % of oracle gap recovered by imputation: {pct_recovered:.1f}%")

    print()
    print("=" * 70)
    print("FEATURE AGREEMENT (Oracle ↔ Imputed, numerical cells)")
    print("=" * 70)
    if "numerical" in agreement:
        ag = agreement["numerical"]
        print(f"  Direction match (all conf):   {ag['direction_match_rate']:.4f}")
        print(f"  Direction match (conf ≥ 0.5): {ag['direction_match_high_conf']:.4f}")
        print(f"  Direction match (conf < 0.1): {ag['direction_match_low_conf']:.4f}")
        print(f"  Median magnitude ratio:        {ag['median_magnitude_ratio']:.4f}  "
              f"(1.0 = perfect; >1 = imputed over-estimates)")
        print()
        print(f"  Per-column direction match rate:")
        for col, m in sorted(ag["per_column"].items(), key=lambda x: -x[1]["direction_match"]):
            bar = "█" * int(m["direction_match"] * 20)
            print(f"    {col:20s}  {m['direction_match']:.3f}  {bar}  n={m['n']}")
    if "categorical" in agreement:
        ag = agreement["categorical"]
        print(f"\n  Categorical change-detection match: {ag['change_detection_match_rate']:.4f}")

    print()
    print("=" * 70)
    print("DIRECTION CORRECTNESS BY CONFIDENCE TIER (numerical)")
    print("=" * 70)
    if len(merged_num) > 0:
        bins = [(0.0, 0.1, "0.0–0.1 (very low)"),
                (0.1, 0.3, "0.1–0.3 (low)"),
                (0.3, 0.6, "0.3–0.6 (medium)"),
                (0.6, 1.01,"0.6–1.0 (high)")]
        print(f"  {'Confidence tier':<22}  {'n':>5}  {'Dir match':>10}  {'Median mag ratio':>16}")
        print(f"  {'-'*60}")
        for lo, hi, label in bins:
            tier = merged_num[(merged_num["confidence"] >= lo) &
                              (merged_num["confidence"] < hi)]
            if len(tier) == 0:
                continue
            dm = tier["dir_match"].mean()
            safe_oracle = tier["oracle_magnitude"].replace(0, np.nan)
            mr = (tier["imp_magnitude"] / safe_oracle).median()
            print(f"  {label:<22}  {len(tier):>5}  {dm:>10.4f}  {mr:>16.4f}")


def _write_report(report, oracle_cv, imputed_cv, naive_cv,
                  agreement, merged_num, path: str):
    """Write a markdown report file."""
    lines = []
    A = lines.append

    A("# Oracle vs. Imputed vs. Naive — Comparison Report")
    A("")
    A(f"**Cells compared:** {report['n_cells']}  ")
    A(f"**Intentional:** {report['n_intentional']}  ")
    A(f"**Unintentional:** {report['n_unintentional']}  ")
    A("")
    A("---")
    A("")
    A("## 1. Classification Performance (5-fold CV)")
    A("")
    A("| Pipeline | Accuracy | F1-macro | F1-intent | F1-uninten |")
    A("|---|---|---|---|---|")
    for name, cv in [("Oracle (x\\*)", oracle_cv),
                     ("Imputed (MICE x̂\\*)", imputed_cv),
                     ("Naive (no reference)", naive_cv)]:
        acc = f"{cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.3f}"
        f1m = f"{cv['f1_macro']['mean']:.4f}"
        f1i = f"{cv['f1_intentional']['mean']:.4f}"
        f1u = f"{cv['f1_unintentional']['mean']:.4f}"
        A(f"| {name} | {acc} | {f1m} | {f1i} | {f1u} |")

    oracle_f1  = oracle_cv["f1_macro"]["mean"]
    imputed_f1 = imputed_cv["f1_macro"]["mean"]
    naive_f1   = naive_cv["f1_macro"]["mean"]
    gap_oi     = oracle_f1 - imputed_f1
    gap_in     = imputed_f1 - naive_f1
    pct        = gap_in / max(oracle_f1 - naive_f1, 1e-6) * 100

    A("")
    A(f"- **Oracle → Imputed gap:** −{gap_oi:.4f} F1 (cost of not having ground truth)")
    A(f"- **Imputed → Naive gap:** +{gap_in:.4f} F1 (value added by MICE imputation)")
    A(f"- **% of oracle gap recovered:** {pct:.1f}%")
    A("")
    A("---")
    A("")
    A("## 2. Feature Agreement (Oracle ↔ Imputed, Numerical Cells)")
    A("")

    if "numerical" in agreement:
        ag = agreement["numerical"]
        A(f"- Direction match (all confidence): **{ag['direction_match_rate']:.4f}**")
        A(f"- Direction match (conf ≥ 0.5):    **{ag['direction_match_high_conf']:.4f}**")
        A(f"- Direction match (conf < 0.1):    **{ag['direction_match_low_conf']:.4f}**")
        A(f"- Median magnitude ratio:           **{ag['median_magnitude_ratio']:.4f}** (1.0 = perfect)")
        A("")
        A("### Per-Column Direction Match")
        A("")
        A("| Column | n | Direction Match | Median Mag Ratio |")
        A("|---|---|---|---|")
        for col, m in sorted(ag["per_column"].items(),
                             key=lambda x: -x[1]["direction_match"]):
            A(f"| `{col}` | {m['n']} | {m['direction_match']:.4f} | {m['median_mag_ratio']:.4f} |")
        A("")

    if "categorical" in agreement:
        ag = agreement["categorical"]
        A(f"- Categorical change-detection match: **{ag['change_detection_match_rate']:.4f}**")
        A("")

    A("---")
    A("")
    A("## 3. Direction Correctness by Confidence Tier")
    A("")
    A("| Confidence tier | n | Direction match | Median mag ratio |")
    A("|---|---|---|---|")
    bins = [(0.0, 0.1, "0.0–0.1"), (0.1, 0.3, "0.1–0.3"),
            (0.3, 0.6, "0.3–0.6"), (0.6, 1.01, "0.6–1.0")]
    for lo, hi, label in bins:
        tier = merged_num[(merged_num["confidence"] >= lo) &
                          (merged_num["confidence"] < hi)]
        if len(tier) == 0:
            continue
        dm = tier["dir_match"].mean()
        safe_oracle = tier["oracle_magnitude"].replace(0, np.nan)
        mr = (tier["imp_magnitude"] / safe_oracle).median()
        A(f"| {label} | {len(tier)} | {dm:.4f} | {mr:.4f} |")

    A("")
    A("---")
    A("")
    A("## 4. Interpretation")
    A("")
    A(f"The **Oracle pipeline** represents the theoretical ceiling: if we knew the")
    A(f"correct value x\\* for every cell, what F1 could we achieve?")
    A(f"Answer: **F1-macro = {oracle_f1:.4f}**.")
    A("")
    A(f"The **Imputed pipeline** uses only the MICE estimate x̂\\*. It achieves")
    A(f"**F1-macro = {imputed_f1:.4f}** — a gap of only {gap_oi:.4f} below oracle.")
    A(f"It recovers **{pct:.1f}%** of the total gap between oracle and naive.")
    A("")
    A(f"The **Naive pipeline** (no reference point, no imputation) achieves")
    A(f"**F1-macro = {naive_f1:.4f}**. The {gap_in:.4f} F1 gap above naive is the")
    A(f"measurable value added by running MICE imputation.")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Oracle vs. Imputed vs. Naive intent attribution pipelines."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to imputation_results.csv (must have correct_value column).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the markdown report (optional).",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    run_comparison(
        results_path=args.results,
        output_path=args.output,
        n_folds=args.folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
