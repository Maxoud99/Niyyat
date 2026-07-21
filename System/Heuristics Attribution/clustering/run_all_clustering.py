#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_clustering.py
=========================
Run Scenario B (Heuristic, 12 features) and B+ (Reference-Augmented, 10
clean-based statistical features -- NOT combined with the Scenario-B
heuristics) for all 8 datasets WITH DBSCAN cluster-proportional seed
sampling.

This is the clustered counterpart of no_clustering/run_all_no_clustering.py
(same B/B+ feature extraction, same RF, same 1% budget; only the seed
selection differs), and completes the set alongside
declarative/run_all_declarative.py's C / BC / BplusC scenarios -- together
these five scenarios are all clustered with the SAME single DBSCAN
parameterisation (per project decision: one clustering algorithm, not a
multi-algorithm sweep). The B+heuristic combination (13+10 features) is a
separate scenario, Full, computed only when concatenated with C in
declarative/run_all_declarative.py's BplusC scenario.

B+ is skipped automatically for datasets with no oracle clean reference
(currently: eBay).

Output
------
  results/<dataset_key>/<scenario>/metrics.json
  results/summary_clustering.csv

Usage
-----
  python run_all_clustering.py
  python run_all_clustering.py --datasets adult_llm ebay
  python run_all_clustering.py --scenarios B
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).parent.resolve()
ATTR_DIR = THIS_DIR.parent
EDS_SRC = ATTR_DIR.parent
ROOT = EDS_SRC.parents[1]
FRAUD = ROOT / "fraud_baseline"
DECLARATIVE = ATTR_DIR / "declarative"
NO_CLUSTERING = ATTR_DIR / "no_clustering"

for _p in [str(EDS_SRC), str(FRAUD), str(NO_CLUSTERING)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(THIS_DIR))  # shadow any pipeline.py name collisions

from features import extract_b_features, extract_statistical_features, attach_labels  # noqa: E402
from pipeline import ClusteringPipeline, save_results  # noqa: E402
from datasets import LOADERS  # noqa: E402

RANDOM_STATE = 42
SAMPLE_RATE = 0.01

RESULTS_DIR = THIS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = DECLARATIVE / "configs"

# Identical per-dataset semantic config to declarative/run_all_declarative.py
# and no_clustering/run_all_no_clustering.py (duplicated per this codebase's
# existing convention -- not imported, to avoid a pipeline.py module name
# collision with declarative's own pipeline.py).
DATASET_CONFIGS: Dict[str, Dict] = {
    "adult_llm": dict(name="Adult-LLM", pipeline_cfg=dict(
        target_col="class", codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"])),
    "adult_mixed": dict(name="Adult-Mixed", pipeline_cfg=dict(
        target_col="class", codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"])),
    "adult_tfm": dict(name="Adult-TFM", pipeline_cfg=dict(
        target_col=None, codependent_pairs=[("education", "education-num")], sensitive_cols=["race", "sex"])),
    "twitterbot_llm": dict(name="TwitterBot-LLM V2", pipeline_cfg=dict(
        target_col=None, codependent_pairs=[], sensitive_cols=[])),
    "twitterbot_mixed": dict(name="TwitterBot-Mixed", pipeline_cfg=dict(
        target_col=None, codependent_pairs=[], sensitive_cols=[])),
    "twitterbot_tfm": dict(name="TwitterBot-TFM", pipeline_cfg=dict(
        target_col=None, codependent_pairs=[], sensitive_cols=[])),
    "tabfact": dict(name="TabFact", pipeline_cfg=dict(
        target_col="claim_domain", codependent_pairs=[], sensitive_cols=[])),
    "ebay": dict(name="eBay", pipeline_cfg=dict(
        target_col=None, codependent_pairs=[("spec_brand", "title"), ("category", "marketplace")], sensitive_cols=[])),
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())
ALL_SCENARIOS = ["B", "Bplus"]

# B+ requires an oracle clean reference; eBay's "clean" is a pseudo-clean
# approximation only (see fraud_baseline/datasets.py:load_ebay docstring).
NO_ORACLE_CLEAN = {"ebay"}


def run_dataset(ds_key: str, scenarios: List[str], verbose: bool) -> List[Dict]:
    cfg = DATASET_CONFIGS[ds_key]
    print(f"\n{'='*70}\nDataset: {cfg['name']} ({ds_key})  [DBSCAN CLUSTERING]\n{'='*70}")

    try:
        ds = LOADERS[ds_key]()
    except Exception as e:
        print(f"[FAIL] Could not load {ds_key}: {e}")
        return []

    dirty, gt_mask, clean = ds["dirty"], ds["mask"], ds["clean"]
    blind = (gt_mask != 0).astype(int)
    n_err = int((gt_mask != 0).sum().sum())
    n_int = int((gt_mask == 1).sum().sum())
    n_unint = int((gt_mask == -1).sum().sum())
    print(f"  {n_err} erroneous cells - INT={n_int}, UNINT={n_unint}")

    scenarios = [s for s in scenarios if not (s == "Bplus" and ds_key in NO_ORACLE_CLEAN)]
    if not scenarios:
        print("  [skip] No applicable scenarios (no oracle clean reference for B+)")
        return []

    b_feats = None
    if "B" in scenarios:
        print("\n  [B features] Extracting 12 structural heuristic features...")
        b_feats = extract_b_features(dirty, blind, cfg["pipeline_cfg"])

    # Bplus = Reference-Augmented = the 10 clean-based statistical features
    # alone. It must NOT include the 13 Scenario-B heuristic features --
    # that combination is a separate scenario (Full, B+ + C, computed in
    # declarative/run_all_declarative.py), not standalone Reference-Augmented.
    stat_feats = None
    if "Bplus" in scenarios:
        print("\n  [Reference-Augmented features] Extracting 10 clean-based statistical features...")
        stat_feats = extract_statistical_features(dirty, clean, gt_mask)

    scenario_feats = {"B": b_feats, "Bplus": stat_feats}

    rows = []
    out_base = RESULTS_DIR / ds_key
    for scenario in scenarios:
        feat_df = scenario_feats.get(scenario)
        if feat_df is None:
            print(f"\n  [skip] Scenario {scenario}: features unavailable")
            continue

        print(f"\n  -- Scenario {scenario} (DBSCAN clustering) --")
        t_start = time.time()
        labeled_df = attach_labels(feat_df, gt_mask)
        if len(labeled_df) == 0:
            print(f"  [skip] Scenario {scenario}: no labeled cells after join")
            continue

        try:
            pipe = ClusteringPipeline(sampling_rate=SAMPLE_RATE, random_state=RANDOM_STATE, verbose=verbose)
            result = pipe.run(labeled_df)
        except Exception as e:
            print(f"  [FAIL] Scenario {scenario}: {e}")
            import traceback
            traceback.print_exc()
            continue

        save_results(result, str(out_base), scenario)
        metrics = result.get("metrics") or {}
        elapsed = result.get("elapsed", time.time() - t_start)
        row = {
            "dataset": cfg["name"], "ds_key": ds_key, "scenario": scenario,
            "n_cells": n_err, "n_int": n_int, "n_unint": n_unint,
            "n_seed": result.get("n_seed", 0), "n_clusters": result.get("n_clusters", 0),
            "f1_weighted": round(metrics.get("f1_weighted", float("nan")), 4),
            "f1_int": round(metrics.get("f1_intentional", float("nan")), 4),
            "f1_unint": round(metrics.get("f1_unintentional", float("nan")), 4),
            "accuracy": round(metrics.get("accuracy", float("nan")), 4),
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        print(f"\n  [Result] {scenario}  F1-w={row['f1_weighted']:.4f}  "
              f"F1-INT={row['f1_int']:.4f}  F1-UNINT={row['f1_unint']:.4f}  ({elapsed:.1f}s)")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run B / B+ attribution WITH DBSCAN clustering for all datasets")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dataset_keys = args.datasets or ALL_DATASETS
    scenarios = args.scenarios or ALL_SCENARIOS
    verbose = not args.quiet

    for k in dataset_keys:
        if k not in DATASET_CONFIGS:
            print(f"[error] Unknown dataset: {k}. Choose from: {ALL_DATASETS}")
            sys.exit(1)

    print(f"Datasets  : {dataset_keys}\nScenarios : {scenarios}\nLabel budget: {SAMPLE_RATE*100:.0f}% (DBSCAN cluster-proportional)")

    all_rows = []
    t_global = time.time()
    for ds_key in dataset_keys:
        all_rows.extend(run_dataset(ds_key, scenarios, verbose))

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = RESULTS_DIR / "summary_clustering.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved -> {summary_path}")
        print(summary.to_string(index=False))
    print(f"\nTotal time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
