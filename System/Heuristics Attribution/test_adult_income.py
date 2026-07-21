#!/usr/bin/env python
"""
Test the Attribution Pipeline (H1-H8) on BOTH Adult Income datasets.

Each run creates a timestamped output folder under:
  .../heuristics/output/run_YYYYMMDD_HHMMSS/

Output files per dataset (in a sub-folder):
  - feature_matrix.csv
  - feature_distributions.csv
  - cv_fold_metrics.csv
  - cv_summary.csv
  - confusion_matrix.csv
  - feature_importances.csv
  - classification_report.txt

Global output:
  - summary.csv
  - run_log.txt
"""

import sys
import os
import io
import time
import warnings
import datetime
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")

# -- Add the heuristics package to path --
HEURISTIC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
if HEURISTIC_ROOT not in sys.path:
    sys.path.insert(0, HEURISTIC_ROOT)

from attribution.heuristics.pipeline import AttributionPipeline

# -- Paths --
BASE = "/home/mohamed/error_injector/llms_baseline"
LLM_DIR = os.path.join(
    BASE, "adult_income_dataset/tenth-trial/data/raw/run_v2_20260617_173016"
)
KIREEV_DIR = os.path.join(BASE, "mixed_error_pipeline/output")
OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output"
)

# -- Config --
TARGET_COL = "class"
CODEPENDENT_PAIRS = [("education", "education-num")]
SENSITIVE_COLS = ["race", "sex"]
RANDOM_STATE = 42
N_SPLITS = 5


# ===========================================================================
# Tee-logger: prints AND captures to string
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


# ===========================================================================
# Evaluation helpers
# ===========================================================================

def compute_feature_distributions(feat_df, labels):
    aligned = labels.loc[feat_df.index]
    rows = []
    for col in feat_df.columns:
        int_vals = feat_df.loc[aligned == 1, col].dropna()
        unint_vals = feat_df.loc[aligned == 0, col].dropna()
        rows.append({
            "feature": col,
            "intentional_mean": int_vals.mean(),
            "intentional_std": int_vals.std(),
            "unintentional_mean": unint_vals.mean(),
            "unintentional_std": unint_vals.std(),
            "delta": int_vals.mean() - unint_vals.mean(),
        })
    return pd.DataFrame(rows)


def run_cv(feat_df, labels):
    aligned = labels.loc[feat_df.index]
    X = feat_df.values.copy()
    y = aligned.values.copy()
    X = np.nan_to_num(X, nan=-999.0)

    skf = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )

    all_y_true, all_y_pred, all_y_proba = [], [], []
    fold_rows = []
    last_rf = None

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced",
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]

        fold_rows.append({
            "fold": fold_i + 1,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        last_rf = rf

    fold_df = pd.DataFrame(fold_rows)

    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    summary_rows = {"metric": [], "mean": [], "std": []}
    for mc in metric_cols:
        summary_rows["metric"].append(mc)
        summary_rows["mean"].append(fold_df[mc].mean())
        summary_rows["std"].append(fold_df[mc].std())
    summary_df = pd.DataFrame(summary_rows)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["true_unintentional", "true_intentional"],
        columns=["pred_unintentional", "pred_intentional"],
    )

    report_str = classification_report(
        all_y_true, all_y_pred,
        target_names=["Unintentional", "Intentional"],
    )

    importances = last_rf.feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feat_df.columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.index = imp_df.index + 1
    imp_df.index.name = "rank"

    return fold_df, summary_df, cm_df, report_str, imp_df


# ===========================================================================
# Run one dataset end-to-end
# ===========================================================================

def run_dataset(name, load_fn, run_dir):
    slug = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )
    ds_dir = os.path.join(run_dir, slug)
    os.makedirs(ds_dir, exist_ok=True)

    sep = "=" * 70
    print("\n" + sep)
    print("  DATASET: " + name)
    print(sep)

    # -- Load --
    print("\nLoading " + name + "...")
    t0 = time.time()
    dirty, blind_mask, labels = load_fn()
    n_errors = int(blind_mask.values.sum())
    n_intent = int((labels == 1).sum())
    n_unintent = int((labels == 0).sum())
    print("  Dirty shape:      " + str(dirty.shape))
    print("  Erroneous cells:  " + str(n_errors))
    print("  Intentional:      " + str(n_intent))
    print("  Unintentional:    " + str(n_unintent))
    print("  Loaded in {:.1f}s".format(time.time() - t0))

    # -- Fit pipeline --
    print("\nFitting pipeline (unsupervised)...")
    t0 = time.time()
    pipe = AttributionPipeline(
        target_col=TARGET_COL,
        codependent_pairs=CODEPENDENT_PAIRS,
        sensitive_cols=SENSITIVE_COLS,
    )
    pipe.fit(dirty, blind_mask)
    fit_time = time.time() - t0
    print("  Fit completed in {:.1f}s".format(fit_time))

    # -- Compute features --
    print("\nComputing 12-feature matrix...")
    t0 = time.time()
    feat_df = pipe.compute_features(dirty, blind_mask)
    compute_time = time.time() - t0
    print("  Feature matrix shape: " + str(feat_df.shape))
    print("  Computed in {:.1f}s".format(compute_time))
    assert feat_df.shape[1] == 12, (
        "Expected 12 features, got " + str(feat_df.shape[1])
    )

    # -- Save feature matrix --
    feat_df.to_csv(os.path.join(ds_dir, "feature_matrix.csv"))
    print("  -> Saved feature_matrix.csv")

    # -- Feature distributions --
    dist_df = compute_feature_distributions(feat_df, labels)
    dist_df.to_csv(
        os.path.join(ds_dir, "feature_distributions.csv"), index=False
    )
    print("  -> Saved feature_distributions.csv")

    print("\n" + "-" * 70)
    print("Feature Distributions -- " + name)
    print("-" * 70)
    hdr = "{:<30} {:>12} {:>14} {:>10}".format(
        "Feature", "Intentional", "Unintentional", "Delta"
    )
    print("\n" + hdr)
    print("=" * 70)
    for _, r in dist_df.iterrows():
        delta = r["delta"]
        arrow = ""
        if abs(delta) > 0.05:
            arrow = " ^" if delta > 0 else " v"
        line = "{:<30} {:>12.4f} {:>14.4f} {:>+10.4f}{}".format(
            r["feature"],
            r["intentional_mean"],
            r["unintentional_mean"],
            delta,
            arrow,
        )
        print(line)

    # -- Cross-validation --
    print("\n" + "=" * 70)
    print("Cross-Validation Evaluation -- " + name)
    print("=" * 70)
    print("\nTotal erroneous cells: " + str(n_errors))
    print("Intentional (1):      {} ({:.1f}%)".format(
        n_intent, n_intent / n_errors * 100
    ))
    print("Unintentional (0):    {} ({:.1f}%)".format(
        n_unintent, n_unintent / n_errors * 100
    ))

    fold_df, summary_df, cm_df, report_str, imp_df = run_cv(feat_df, labels)

    # Save all CV outputs
    fold_df.to_csv(os.path.join(ds_dir, "cv_fold_metrics.csv"), index=False)
    summary_df.to_csv(os.path.join(ds_dir, "cv_summary.csv"), index=False)
    cm_df.to_csv(os.path.join(ds_dir, "confusion_matrix.csv"))
    imp_df.to_csv(os.path.join(ds_dir, "feature_importances.csv"))
    with open(os.path.join(ds_dir, "classification_report.txt"), "w") as fh:
        fh.write(report_str)

    # Print fold metrics
    print("\n{:<6} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "Fold", "Acc", "Prec", "Recall", "F1", "AUC"
    ))
    print("-" * 50)
    for _, r in fold_df.iterrows():
        print("{:<6} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            int(r["fold"]),
            r["accuracy"], r["precision"],
            r["recall"], r["f1"], r["auc"],
        ))

    mean_s = summary_df.set_index("metric")["mean"]
    std_s = summary_df.set_index("metric")["std"]
    print("-" * 50)
    print("{:<6} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "Mean",
        mean_s["accuracy"], mean_s["precision"],
        mean_s["recall"], mean_s["f1"], mean_s["auc"],
    ))
    print("{:<6} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        "Std",
        std_s["accuracy"], std_s["precision"],
        std_s["recall"], std_s["f1"], std_s["auc"],
    ))

    # Confusion matrix
    print("\nConfusion Matrix (aggregated):")
    print("  {:>20} Pred=Unintent  Pred=Intent".format(""))
    print("  {:>20}  {:>10}  {:>10}".format(
        "True=Unintent", cm_df.iloc[0, 0], cm_df.iloc[0, 1]
    ))
    print("  {:>20}  {:>10}  {:>10}".format(
        "True=Intent", cm_df.iloc[1, 0], cm_df.iloc[1, 1]
    ))

    print("\nClassification Report:")
    print(report_str)

    # Feature importances
    print("RF Feature Importances (last fold):")
    for rank, (_, r) in enumerate(imp_df.iterrows()):
        bar = "#" * int(r["importance"] * 50)
        print("  {:>2}. {:<30} {:.4f}  {}".format(
            rank + 1, r["feature"], r["importance"], bar
        ))

    print("\n  -> All dataset outputs saved to " + ds_dir + "/")

    return {
        "dataset": name,
        "rows": dirty.shape[0],
        "errors": n_errors,
        "intentional": n_intent,
        "unintentional": n_unintent,
        "accuracy_mean": mean_s["accuracy"],
        "accuracy_std": std_s["accuracy"],
        "precision_mean": mean_s["precision"],
        "precision_std": std_s["precision"],
        "recall_mean": mean_s["recall"],
        "recall_std": std_s["recall"],
        "f1_mean": mean_s["f1"],
        "f1_std": std_s["f1"],
        "auc_mean": mean_s["auc"],
        "auc_std": std_s["auc"],
        "fit_time_s": fit_time,
        "compute_time_s": compute_time,
    }


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_ROOT, "run_" + timestamp)
    os.makedirs(run_dir, exist_ok=True)

    tee = TeeLogger()
    sys.stdout = tee

    try:
        print("=" * 70)
        print("  ATTRIBUTION PIPELINE -- ADULT INCOME DATASET EVALUATION")
        print("  Run: " + timestamp)
        print("  Output: " + run_dir)
        print("=" * 70)

        all_summaries = []

        s = run_dataset("LLM (tenth-trial)", load_llm_dataset, run_dir)
        all_summaries.append(s)

        s = run_dataset(
            "Mixed SOTA (mixed_error_pipeline)", load_kireev_dataset, run_dir
        )
        all_summaries.append(s)

        # -- Global summary --
        print("\n" + "=" * 70)
        print("  SUMMARY COMPARISON")
        print("=" * 70)
        print("\n{:<30} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
            "Dataset", "Acc", "Prec", "Recall", "F1", "AUC"
        ))
        print("=" * 70)
        for s in all_summaries:
            print("{:<30} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                s["dataset"],
                s["accuracy_mean"], s["precision_mean"],
                s["recall_mean"], s["f1_mean"], s["auc_mean"],
            ))

        global_df = pd.DataFrame(all_summaries)
        global_df.to_csv(os.path.join(run_dir, "summary.csv"), index=False)
        print("\n  -> Global summary saved to " + run_dir + "/summary.csv")
        print("\nDone.")

    except Exception:
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    finally:
        sys.stdout = tee.terminal
        log_path = os.path.join(run_dir, "run_log.txt")
        with open(log_path, "w") as fh:
            fh.write(tee.get_log())
        print("\n  -> Full log saved to " + log_path)
        print("  -> All outputs in: " + run_dir + "/")
