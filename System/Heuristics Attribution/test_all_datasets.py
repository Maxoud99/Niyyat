#!/usr/bin/env python
"""
Heuristics Attribution (H1-H8) -- generic runner across all 8 datasets.

Unlike test_adult_income.py / test_twitterbot.py (which hardcode paths for
2-3 datasets each), this reuses the canonical loaders in fraud_baseline so
every dataset -- including eBay, which has no oracle clean reference -- gets
evaluated the same way: fit the 8 heuristics directly on (dirty, mask), no
clean reference required (H7 derives its own reference internally from the
fully-clean rows within the dirty data, see h7_user_incentive.py), then
StratifiedKFold + RandomForest on the 12-feature matrix.

Single-class datasets (TwitterBot-LLM, TabFact: every flagged cell is
intentional by construction) cannot support a binary CV split -- these are
reported with feature distributions only, CV is skipped and noted.

Output: heuristics/output/run_YYYYMMDD_HHMMSS/<dataset_key>/...
"""

import sys
import os
import time
import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

HEURISTIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../error_detection_system/src
ROOT = os.path.dirname(os.path.dirname(HEURISTIC_ROOT))  # .../llms_baseline
FRAUD = os.path.join(ROOT, "fraud_baseline")
for _p in (HEURISTIC_ROOT, FRAUD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from attribution.heuristics.pipeline import AttributionPipeline  # noqa: E402
from datasets import LOADERS  # noqa: E402  (fraud_baseline canonical loaders)

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RANDOM_STATE = 42
N_SPLITS = 5

# Per-dataset semantic config (mirrors declarative/run_all_declarative.py's
# DATASET_CONFIGS pipeline_cfg, duplicated here per this codebase's existing
# convention of per-script dataset config rather than a shared module).
PIPELINE_CFG = {
    "adult_llm":        dict(target_col="class", codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"]),
    "adult_mixed":      dict(target_col="class", codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"]),
    "adult_tfm":        dict(target_col=None,    codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"]),
    "twitterbot_llm":   dict(target_col=None, codependent_pairs=[], sensitive_cols=[]),
    "twitterbot_mixed": dict(target_col=None, codependent_pairs=[], sensitive_cols=[]),
    "twitterbot_tfm":   dict(target_col=None, codependent_pairs=[], sensitive_cols=[]),
    "tabfact":          dict(target_col="claim_domain", codependent_pairs=[], sensitive_cols=[]),
    "ebay":             dict(target_col=None, codependent_pairs=[("spec_brand", "title"), ("category", "marketplace")], sensitive_cols=[]),
}


def compute_feature_distributions(feat_df, labels):
    aligned = labels.loc[feat_df.index]
    rows = []
    for col in feat_df.columns:
        int_vals = feat_df.loc[aligned == 1, col].dropna()
        unint_vals = feat_df.loc[aligned == 0, col].dropna()
        rows.append({
            "feature": col,
            "intentional_mean": int_vals.mean(),
            "unintentional_mean": unint_vals.mean(),
            "delta": int_vals.mean() - unint_vals.mean(),
        })
    return pd.DataFrame(rows)


def run_cv(feat_df, labels):
    aligned = labels.loc[feat_df.index]
    X = np.nan_to_num(feat_df.values.copy(), nan=-999.0)
    y = aligned.values.copy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_rows = []
    last_rf = None

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                     n_jobs=-1, class_weight="balanced")
        rf.fit(X[train_idx], y[train_idx])
        y_pred = rf.predict(X[test_idx])
        y_proba = rf.predict_proba(X[test_idx])[:, 1]
        fold_rows.append({
            "fold": fold_i + 1,
            "accuracy": accuracy_score(y[test_idx], y_pred),
            "precision": precision_score(y[test_idx], y_pred, zero_division=0),
            "recall": recall_score(y[test_idx], y_pred, zero_division=0),
            "f1": f1_score(y[test_idx], y_pred, zero_division=0),
            "auc": roc_auc_score(y[test_idx], y_proba),
        })
        all_y_true.extend(y[test_idx]); all_y_pred.extend(y_pred); all_y_proba.extend(y_proba)
        last_rf = rf

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1", "auc"],
        "mean": [fold_df[c].mean() for c in ("accuracy", "precision", "recall", "f1", "auc")],
        "std":  [fold_df[c].std()  for c in ("accuracy", "precision", "recall", "f1", "auc")],
    })
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(cm, index=["true_unintentional", "true_intentional"],
                          columns=["pred_unintentional", "pred_intentional"])
    report_str = classification_report(all_y_true, all_y_pred,
                                        target_names=["Unintentional", "Intentional"])
    imp_df = (pd.DataFrame({"feature": feat_df.columns, "importance": last_rf.feature_importances_})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    imp_df.index += 1
    return fold_df, summary_df, cm_df, report_str, imp_df


def run_dataset(key: str, run_dir: str) -> dict:
    print(f"\n{'='*70}\n  DATASET: {key}\n{'='*70}")
    ds_dir = os.path.join(run_dir, key)
    os.makedirs(ds_dir, exist_ok=True)

    ds = LOADERS[key]()
    dirty, mask = ds["dirty"], ds["mask"]
    blind_mask = (mask != 0).astype(int)

    labels_dict = {}
    for row_idx in range(len(mask)):
        row = mask.iloc[row_idx]
        for col_name in mask.columns:
            val = row[col_name]
            if val != 0:
                labels_dict[(row_idx, col_name)] = 1 if val == 1 else 0
    labels = pd.Series(labels_dict, name="intent_label")
    labels.index = pd.MultiIndex.from_tuples(labels.index, names=["row_idx", "col_name"])

    n_errors, n_int, n_unint = len(labels), int((labels == 1).sum()), int((labels == 0).sum())
    print(f"  {ds['name']}: dirty={dirty.shape} errors={n_errors} int={n_int} unint={n_unint}")

    cfg = PIPELINE_CFG.get(key, dict(target_col=None, codependent_pairs=[], sensitive_cols=[]))
    pipe = AttributionPipeline(**cfg)
    pipe.fit(dirty, blind_mask)
    feat_df = pipe.compute_features(dirty, blind_mask)
    feat_df.to_csv(os.path.join(ds_dir, "feature_matrix.csv"))

    dist_df = compute_feature_distributions(feat_df, labels)
    dist_df.to_csv(os.path.join(ds_dir, "feature_distributions.csv"), index=False)

    result = {"dataset": key, "name": ds["name"], "n_errors": n_errors,
              "n_int": n_int, "n_unint": n_unint, "note": ds.get("note", "")}

    if n_int == 0 or n_unint == 0:
        print(f"  [skip CV] single-class dataset (int={n_int}, unint={n_unint})")
        result.update(cv_skipped=True, f1=float("nan"), accuracy=float("nan"), auc=float("nan"))
        return result

    fold_df, summary_df, cm_df, report_str, imp_df = run_cv(feat_df, labels)
    fold_df.to_csv(os.path.join(ds_dir, "cv_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(ds_dir, "cv_summary.csv"), index=False)
    cm_df.to_csv(os.path.join(ds_dir, "confusion_matrix.csv"))
    imp_df.to_csv(os.path.join(ds_dir, "feature_importances.csv"))
    with open(os.path.join(ds_dir, "classification_report.txt"), "w") as fh:
        fh.write(report_str)

    f1_mean = summary_df.loc[summary_df.metric == "f1", "mean"].iloc[0]
    acc_mean = summary_df.loc[summary_df.metric == "accuracy", "mean"].iloc[0]
    auc_mean = summary_df.loc[summary_df.metric == "auc", "mean"].iloc[0]
    print(f"  [CV] acc={acc_mean:.4f} f1={f1_mean:.4f} auc={auc_mean:.4f}")
    result.update(cv_skipped=False, f1=f1_mean, accuracy=acc_mean, auc=auc_mean)
    return result


def main():
    run_dir = os.path.join(OUTPUT_ROOT, "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    results = []
    for key in LOADERS:
        t0 = time.time()
        try:
            r = run_dataset(key, run_dir)
            r["elapsed_s"] = round(time.time() - t0, 1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {"dataset": key, "name": key, "error": str(e)}
        results.append(r)

    summary = pd.DataFrame(results)
    summary.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved -> {run_dir}")


if __name__ == "__main__":
    main()
