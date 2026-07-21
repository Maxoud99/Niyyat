#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_no_clustering.py
=========================
Run Scenario B, B+, C, B+C, (B+)+C for all 7 datasets — WITHOUT any
clustering step. The 1% label budget is drawn by plain uniform random
sampling instead of cluster-proportional sampling.

This is the no-clustering counterpart of:
  - supervised_baseline/run_scenario_b_sweep.py        (B, B+)
  - declarative/run_all_declarative.py                 (C, B+C, (B+)+C)

Everything else (feature engineering, RF hyperparameters, label budget,
held-out evaluation) is identical, so any metric delta is attributable
to removing the clustering step.

Scenarios
---------
  B        — 12 structural heuristic features
  Bplus    — 10 clean-based statistical features (Reference-Augmented,
             standalone — does NOT include the Scenario-B heuristics)
  C        — declarative constraint features only (standalone)
  BC       — 12 structural (B) + constraint features
  BplusC   — 23 structural+statistical (B + Bplus, i.e. Full's non-C half)
             + constraint features

Output
------
  results/<dataset_key>/<scenario>/metrics.json
  results/<dataset_key>/<scenario>/features.csv
  results/<dataset_key>/<scenario>/feature_importance.csv
  results/summary_no_clustering.csv

Usage
-----
  python run_all_no_clustering.py
  python run_all_no_clustering.py --datasets adult_llm adult_mixed
  python run_all_no_clustering.py --scenarios B Bplus C
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).parent.resolve()
ATTR_DIR     = THIS_DIR.parent                      # attribution/
EDS_SRC      = ATTR_DIR.parent                      # error_detection_system/src/
ROOT         = EDS_SRC.parents[1]                   # llms_baseline/
FRAUD        = ROOT / "fraud_baseline"
DECLARATIVE  = ATTR_DIR / "declarative"

for _p in [str(EDS_SRC), str(FRAUD), str(DECLARATIVE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# THIS_DIR inserted last so it shadows declarative/pipeline.py and
# declarative/__pycache__ name collisions (both dirs have a pipeline.py).
sys.path.insert(0, str(THIS_DIR))

from features import (                                # noqa: E402
    extract_b_features,
    extract_statistical_features,
    combine_b_and_stat,
    attach_labels,
)
from pipeline import NoClusteringPipeline, save_results  # noqa: E402
from datasets import LOADERS                          # noqa: E402
from extractor import extract_constraints              # noqa: E402
from evaluator import ConstraintEvaluator               # noqa: E402

RANDOM_STATE = 42
SAMPLE_RATE  = 0.01

# ── Output dirs ───────────────────────────────────────────────────────────────
RESULTS_DIR     = THIS_DIR / "results"
CONSTRAINTS_DIR = DECLARATIVE / "results" / "constraints"   # reuse cached constraints
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = DECLARATIVE / "configs"

# ── Dataset → config mapping (identical to declarative/run_all_declarative.py) ─
DATASET_CONFIGS: Dict[str, Dict] = {
    "adult_llm": {
        "name":        "Adult-LLM",
        "description": CONFIGS_DIR / "adult_income.txt",
        "constraints_json": CONSTRAINTS_DIR / "adult_income_constraints.json",
        "pipeline_cfg": dict(
            target_col="class",
            codependent_pairs=[("education", "education-num")],
            sensitive_cols=["race", "sex"],
        ),
    },
    "adult_mixed": {
        "name":        "Adult-Mixed",
        "description": CONFIGS_DIR / "adult_income.txt",
        "constraints_json": CONSTRAINTS_DIR / "adult_income_constraints.json",
        "pipeline_cfg": dict(
            target_col="class",
            codependent_pairs=[("education", "education-num")],
            sensitive_cols=["race", "sex"],
        ),
    },
    "adult_tfm": {
        "name":        "Adult-TFM",
        "description": CONFIGS_DIR / "adult_income.txt",
        "constraints_json": CONSTRAINTS_DIR / "adult_income_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,
            codependent_pairs=[("education", "education-num")],
            sensitive_cols=["race", "sex"],
        ),
    },
    "twitterbot_llm": {
        "name":        "TwitterBot-LLM V2",
        "description": CONFIGS_DIR / "twibot20.txt",
        "constraints_json": CONSTRAINTS_DIR / "twibot20_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,
            codependent_pairs=[],
            sensitive_cols=[],
        ),
    },
    "twitterbot_mixed": {
        "name":        "TwitterBot-Mixed",
        "description": CONFIGS_DIR / "twibot20.txt",
        "constraints_json": CONSTRAINTS_DIR / "twibot20_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,
            codependent_pairs=[],
            sensitive_cols=[],
        ),
    },
    "twitterbot_tfm": {
        "name":        "TwitterBot-TFM",
        "description": CONFIGS_DIR / "twibot20.txt",
        "constraints_json": CONSTRAINTS_DIR / "twibot20_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,
            codependent_pairs=[],
            sensitive_cols=[],
        ),
    },
    "tabfact": {
        "name":        "TabFact",
        "description": CONFIGS_DIR / "tabfact.txt",
        "constraints_json": CONSTRAINTS_DIR / "tabfact_constraints.json",
        "pipeline_cfg": dict(
            target_col="claim_domain",
            codependent_pairs=[],
            sensitive_cols=[],
        ),
    },
    "ebay": {
        "name":        "eBay",
        "description": CONFIGS_DIR / "ebay.txt",
        "constraints_json": CONSTRAINTS_DIR / "ebay_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,
            codependent_pairs=[("spec_brand", "title"), ("category", "marketplace")],
            sensitive_cols=[],
        ),
    },
}

ALL_DATASETS  = list(DATASET_CONFIGS.keys())
ALL_SCENARIOS = ["B", "Bplus", "C", "BC", "BplusC"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(ds_key: str, scenarios: List[str], force_reextract: bool, verbose: bool) -> List[Dict]:
    cfg = DATASET_CONFIGS[ds_key]
    print(f"\n{'='*70}")
    print(f"Dataset: {cfg['name']} ({ds_key})  [NO CLUSTERING]")
    print(f"{'='*70}")

    try:
        ds = LOADERS[ds_key]()
    except Exception as e:
        print(f"[FAIL] Could not load {ds_key}: {e}")
        return []

    dirty   = ds["dirty"]
    gt_mask = ds["mask"]                       # +1 / -1 / 0
    clean   = ds["clean"]
    blind   = (gt_mask != 0).astype(int)        # binary erroneous-cell mask

    n_err   = int((gt_mask != 0).sum().sum())
    n_int   = int((gt_mask == 1).sum().sum())
    n_unint = int((gt_mask == -1).sum().sum())
    print(f"  {n_err} erroneous cells — INT={n_int}, UNINT={n_unint}")

    # ── B features (needed for B, BC, and as an ingredient of BplusC/Full) ──
    b_feats = None
    if any(s in scenarios for s in ("B", "BC", "BplusC")):
        print("\n  [B features] Extracting 12 structural heuristic features...")
        b_feats = extract_b_features(dirty, blind, cfg["pipeline_cfg"])

    # ── Statistical (clean-based) features — Bplus = Reference-Augmented.
    # Bplus is the 10 clean-based statistical features ALONE; it must NOT
    # include the 13 Scenario-B heuristic features. That combination
    # (13+10 = 23 features) is only assembled for BplusC (= Full), never
    # exposed as standalone "Bplus". ──
    stat_feats = None
    if any(s in scenarios for s in ("Bplus", "BplusC")):
        print("\n  [Reference-Augmented features] Extracting 10 clean-based statistical features...")
        stat_feats = extract_statistical_features(dirty, clean, gt_mask)

    # ── B+heuristics combined features — needed only for BplusC (Full) ──
    bp_feats = None
    if "BplusC" in scenarios and b_feats is not None and stat_feats is not None:
        bp_feats = combine_b_and_stat(b_feats, stat_feats)

    # ── C features (needed for C, BC, BplusC) ──
    c_feats = None
    n_constraints = 0
    if any(s in scenarios for s in ("C", "BC", "BplusC")):
        desc_path  = cfg["description"]
        cache_path = str(cfg["constraints_json"])
        if not Path(str(desc_path)).exists():
            print(f"  [skip C] Description file not found: {desc_path}")
        else:
            try:
                constraints = extract_constraints(
                    description_path=str(desc_path),
                    output_json=cache_path,
                    force_reextract=force_reextract,
                )
                n_constraints = len(constraints)
                print(f"  {n_constraints} valid constraints")
                evaluator = ConstraintEvaluator(constraints)
                c_feats = evaluator.extract_features(dirty, blind)
            except Exception as e:
                print(f"  [FAIL] Constraint extraction/evaluation failed: {e}")

    # ── Assemble per-scenario feature matrices ──
    scenario_feats: Dict[str, Optional[pd.DataFrame]] = {}
    scenario_feats["B"]      = b_feats
    scenario_feats["Bplus"]  = stat_feats
    scenario_feats["C"]      = c_feats
    if b_feats is not None and c_feats is not None:
        scenario_feats["BC"] = b_feats.merge(c_feats, on=["row_idx", "column_name"], how="inner")
    else:
        scenario_feats["BC"] = None
    if bp_feats is not None and c_feats is not None:
        scenario_feats["BplusC"] = bp_feats.merge(c_feats, on=["row_idx", "column_name"], how="inner")
    else:
        scenario_feats["BplusC"] = None

    # ── Run each requested scenario ──
    rows = []
    out_base = RESULTS_DIR / ds_key

    for scenario in scenarios:
        feat_df = scenario_feats.get(scenario)
        if feat_df is None:
            print(f"\n  [skip] Scenario {scenario}: features unavailable")
            continue

        print(f"\n  ── Scenario {scenario} (no clustering) ──")
        t_start = time.time()

        labeled_df = attach_labels(feat_df, gt_mask)
        if len(labeled_df) == 0:
            print(f"  [skip] Scenario {scenario}: no labeled cells after join")
            continue

        try:
            pipe = NoClusteringPipeline(
                sampling_rate=SAMPLE_RATE,
                random_state=RANDOM_STATE,
                verbose=verbose,
            )
            result = pipe.run(labeled_df)
        except Exception as e:
            print(f"  [FAIL] Scenario {scenario}: {e}")
            import traceback; traceback.print_exc()
            continue

        save_results(result, str(out_base), scenario)

        metrics = result.get("metrics") or {}
        elapsed = result.get("elapsed", time.time() - t_start)
        n_seed  = result.get("n_seed", 0)

        row = {
            "dataset":       cfg["name"],
            "ds_key":        ds_key,
            "scenario":      scenario,
            "n_cells":       n_err,
            "n_int":         n_int,
            "n_unint":       n_unint,
            "n_constraints": n_constraints,
            "n_seed":        n_seed,
            "f1_weighted":   round(metrics.get("f1_weighted", float("nan")), 4),
            "f1_int":        round(metrics.get("f1_intentional", float("nan")), 4),
            "f1_unint":      round(metrics.get("f1_unintentional", float("nan")), 4),
            "accuracy":      round(metrics.get("accuracy", float("nan")), 4),
            "elapsed_s":     round(elapsed, 1),
        }
        rows.append(row)
        print(f"\n  [Result] {scenario}  n_seed={n_seed}  F1-w={row['f1_weighted']:.4f}  "
              f"F1-INT={row['f1_int']:.4f}  F1-UNINT={row['f1_unint']:.4f}  ({elapsed:.1f}s)")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run B, B+, C, B+C, (B+)+C attribution WITHOUT clustering for all datasets"
    )
    parser.add_argument("--datasets", nargs="*", default=None,
                         help=f"Dataset keys to run (default: all 7). Choices: {', '.join(ALL_DATASETS)}")
    parser.add_argument("--scenarios", nargs="*", default=None,
                         help="Scenarios to run (default: B Bplus C BC BplusC).")
    parser.add_argument("--force-reextract", action="store_true",
                         help="Re-call LLM even if constraint JSON cache exists.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dataset_keys = args.datasets or ALL_DATASETS
    scenarios    = args.scenarios or ALL_SCENARIOS
    verbose      = not args.quiet

    for k in dataset_keys:
        if k not in DATASET_CONFIGS:
            print(f"[error] Unknown dataset: {k}. Choose from: {ALL_DATASETS}")
            sys.exit(1)

    print("=" * 70)
    print("No-Clustering Ablation (B, B+, C, B+C, (B+)+C) — Full Experiment Run")
    print("=" * 70)
    print(f"Datasets  : {dataset_keys}")
    print(f"Scenarios : {scenarios}")
    print(f"Label budget: {SAMPLE_RATE*100:.0f}% (uniform random, no clustering)")

    all_rows = []
    t_global = time.time()

    for ds_key in dataset_keys:
        if ds_key not in LOADERS:
            print(f"[skip] {ds_key} not in LOADERS")
            continue
        rows = run_dataset(ds_key, scenarios, args.force_reextract, verbose)
        all_rows.extend(rows)

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = RESULTS_DIR / "summary_no_clustering.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        cols = ["dataset", "scenario", "n_cells", "n_seed", "n_constraints",
                "f1_weighted", "f1_int", "f1_unint", "accuracy"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print(f"\nSaved → {summary_path}")
    else:
        print("\n[warn] No results produced.")

    print(f"\nTotal time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
