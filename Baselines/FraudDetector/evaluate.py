"""
evaluate.py
Metrics for intent attribution: mirrors the metric set used in §5 of the paper.

Metrics computed:
  - F1_int      : F1 for the intentional class  (+1)
  - F1_unint    : F1 for the unintentional class (-1)
  - F1_weighted : weighted average F1
  - Accuracy    : overall cell-level accuracy
  - AUC         : ROC-AUC using delta_score as the soft score
  - Precision_int / Recall_int
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)


def compute_metrics(results: pd.DataFrame, dataset_name: str = "") -> dict:
    """
    results: DataFrame with columns y_true, y_pred, delta_score
             y_true / y_pred in {+1, -1}
    """
    y_true = results["y_true"].values
    y_pred = results["y_pred"].values
    scores = results["delta_score"].values

    # Handle single-class datasets (e.g., TwitterBot-LLM: all +1)
    classes = sorted(set(y_true))
    single_class = len(classes) == 1

    metrics = {
        "dataset"       : dataset_name,
        "n_cells"       : len(y_true),
        "n_int"         : int((y_true == 1).sum()),
        "n_unint"       : int((y_true == -1).sum()),
        "accuracy"      : round(accuracy_score(y_true, y_pred), 4),
        "f1_int"        : round(f1_score(y_true, y_pred, pos_label=1,
                                          average="binary", zero_division=0), 4),
        "f1_unint"      : round(f1_score(y_true, y_pred, pos_label=-1,
                                          average="binary", zero_division=0), 4),
        "f1_weighted"   : round(f1_score(y_true, y_pred,
                                          average="weighted", zero_division=0), 4),
        "f1_macro"      : round(f1_score(y_true, y_pred,
                                          average="macro", zero_division=0), 4),
        "precision_int" : round(precision_score(y_true, y_pred, pos_label=1,
                                                 zero_division=0), 4),
        "recall_int"    : round(recall_score(y_true, y_pred, pos_label=1,
                                              zero_division=0), 4),
    }

    # AUC: only meaningful if both classes present
    if not single_class:
        try:
            # Remap +1/-1 → 1/0 for sklearn
            y_bin = (y_true == 1).astype(int)
            metrics["auc"] = round(roc_auc_score(y_bin, scores), 4)
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")  # undefined for single class

    return metrics


def threshold_sweep(
    results   : pd.DataFrame,
    thresholds: np.ndarray = None,
    metric    : str = "f1_weighted",
) -> tuple:
    """
    Sweep delta_score threshold and return (best_threshold, best_metric, sweep_df).
    """
    if thresholds is None:
        thresholds = np.percentile(results["delta_score"],
                                   np.linspace(1, 99, 50))

    rows = []
    for t in thresholds:
        results_t = results.copy()
        results_t["y_pred"] = (results_t["delta_score"] > t).map({True: 1, False: -1})
        m = compute_metrics(results_t)
        rows.append({"threshold": t, **m})

    sweep_df     = pd.DataFrame(rows)
    best_idx     = sweep_df[metric].idxmax()
    best_thresh  = float(sweep_df.loc[best_idx, "threshold"])
    best_val     = float(sweep_df.loc[best_idx, metric])

    return best_thresh, best_val, sweep_df


def print_report(metrics: dict) -> None:
    print(f"\n{'─'*55}")
    print(f"  Dataset   : {metrics['dataset']}")
    print(f"  Cells     : {metrics['n_cells']}  "
          f"(int={metrics['n_int']}, unint={metrics['n_unint']})")
    print(f"{'─'*55}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  F1 INT      : {metrics['f1_int']:.4f}")
    print(f"  F1 UNINT    : {metrics['f1_unint']:.4f}")
    print(f"  F1 Weighted : {metrics['f1_weighted']:.4f}")
    print(f"  Prec INT    : {metrics['precision_int']:.4f}")
    print(f"  Recall INT  : {metrics['recall_int']:.4f}")
    print(f"  AUC         : {metrics['auc']}")
    print(f"{'─'*55}")
