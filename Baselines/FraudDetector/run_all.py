"""
run_all.py
End-to-end driver: fits ECOD on each dataset's clean distribution,
runs LOO cell attribution, sweeps threshold, saves results.

Usage:
    python run_all.py [--datasets adult_llm adult_mixed ...]
    python run_all.py           # runs all datasets
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from datasets import LOADERS
from ecod_loo import ECODDetector, run_loo_on_dataset
from evaluate  import compute_metrics, threshold_sweep, print_report

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def run_dataset(name: str, ds: dict, default_threshold: float = 0.0) -> dict:
    print(f"\n{'='*60}")
    print(f"  Dataset: {ds['name']}")
    print(f"  Note   : {ds['note']}")
    print(f"{'='*60}")

    t0 = time.time()

    # ── 1. Fit ECOD on clean data ──────────────────────────────────────────
    print(f"  Fitting ECOD on {len(ds['clean'])} clean rows ...")
    detector = ECODDetector(contamination=0.05)
    detector.fit(ds["clean"])

    # ── 2. Run LOO attribution ─────────────────────────────────────────────
    print(f"  Running LOO on {(ds['mask'] != 0).any(axis=1).sum()} dirty rows ...")
    results = run_loo_on_dataset(
        detector     = detector,
        clean        = ds["clean"],
        dirty        = ds["dirty"],
        mask         = ds["mask"],
        feature_cols = ds["feature_cols"],
        threshold    = default_threshold,
        verbose      = True,
    )

    if results.empty:
        print("  [WARN] No flagged cells found — skipping.")
        return {}

    # ── 3. Save raw results ────────────────────────────────────────────────
    raw_path = os.path.join(OUT_DIR, f"{name}_loo_raw.csv")
    results.to_csv(raw_path, index=False)
    print(f"  Raw results saved → {raw_path}")

    # ── 4. Threshold sweep ─────────────────────────────────────────────────
    best_thresh, best_f1w, sweep_df = threshold_sweep(results, metric="f1_weighted")
    sweep_path = os.path.join(OUT_DIR, f"{name}_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Best threshold: {best_thresh:.4f}  (F1-weighted={best_f1w:.4f})")

    # ── 5. Final metrics at best threshold ────────────────────────────────
    results["y_pred"] = (results["delta_score"] > best_thresh).map({True: 1, False: -1})
    metrics = compute_metrics(results, dataset_name=ds["name"])
    metrics["best_threshold"] = round(best_thresh, 4)
    metrics["elapsed_sec"]    = round(time.time() - t0, 1)

    print_report(metrics)

    # ── 6. Save metrics ────────────────────────────────────────────────────
    metrics_path = os.path.join(OUT_DIR, f"{name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    return metrics


def build_summary_table(all_metrics: dict) -> pd.DataFrame:
    """Compile one row per dataset into a summary DataFrame."""
    cols = ["dataset", "n_cells", "n_int", "n_unint",
            "f1_int", "f1_unint", "f1_weighted", "f1_macro", "accuracy", "auc",
            "precision_int", "recall_int", "best_threshold"]
    rows = []
    for _, m in all_metrics.items():
        if m:
            rows.append({c: m.get(c, float("nan")) for c in cols})
    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(LOADERS.keys()),
                        choices=list(LOADERS.keys()),
                        help="Which datasets to run (default: all).")
    args = parser.parse_args()

    all_metrics = {}

    for name in args.datasets:
        print(f"\n[LOADING] {name}")
        try:
            ds = LOADERS[name]()
        except Exception as e:
            print(f"  [FAIL] Could not load {name}: {e}")
            all_metrics[name] = {}
            continue

        try:
            m = run_dataset(name, ds)
            all_metrics[name] = m
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()
            all_metrics[name] = {}

    # ── Summary table ──────────────────────────────────────────────────────
    summary = build_summary_table(all_metrics)
    summary_path = os.path.join(OUT_DIR, "summary_all_datasets.csv")
    summary.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
