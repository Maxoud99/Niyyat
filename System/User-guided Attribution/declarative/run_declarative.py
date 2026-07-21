#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Declarative Heuristics — CLI Runner
=====================================

Runs Scenario C (declarative constraints only) and optionally
Scenario B+C and Scenario (B+)+C.

Usage examples:
    # Scenario C only
    python run_declarative.py \
        --description configs/adult_income.txt \
        --dirty /path/to/dirty.csv \
        --mask  /path/to/mask.csv \
        --gt    /path/to/ground_truth_mask.csv \
        --out   results/adult_income

    # B+C: provide pre-computed B fingerprint features
    python run_declarative.py \
        --description configs/adult_income.txt \
        --dirty /path/to/dirty.csv \
        --mask  /path/to/mask.csv \
        --gt    /path/to/ground_truth_mask.csv \
        --external-features /path/to/B_features.csv \
        --scenario BC \
        --out results/adult_income

    # (B+)+C: provide pre-computed B+ features
    python run_declarative.py \
        --description configs/adult_income.txt \
        --dirty /path/to/dirty.csv \
        --mask  /path/to/mask.csv \
        --gt    /path/to/ground_truth_mask.csv \
        --external-features /path/to/Bplus_features.csv \
        --scenario BplusC \
        --out results/adult_income

    # Re-extract constraints (ignore cache)
    python run_declarative.py --description configs/adult_income.txt ... --force-reextract
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from extractor import extract_constraints
from pipeline import DeclarativePipeline, save_results


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Declarative (Family C) intent attribution pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--description", required=True,
                   help="Path to user's NL dataset description (.txt)")
    p.add_argument("--dirty", required=True,
                   help="Path to dirty dataset CSV")
    p.add_argument("--mask", required=True,
                   help="Path to binary error mask CSV (1 = erroneous cell)")

    # Optional
    p.add_argument("--gt", default=None,
                   help="Path to ground-truth intent mask CSV (-1/1). "
                        "Required for evaluation metrics.")
    p.add_argument("--external-features", default=None,
                   help="Path to pre-computed B or B+ features CSV "
                        "(required for --scenario BC or BplusC)")
    p.add_argument("--scenario", default="C",
                   choices=["C", "BC", "BplusC"],
                   help="Scenario to run: C (default), BC, or BplusC")
    p.add_argument("--out", default="results",
                   help="Output directory (default: results/)")
    p.add_argument("--constraints-json", default=None,
                   help="Path to save/load extracted constraints JSON. "
                        "Defaults to <out>/constraints.json")
    p.add_argument("--force-reextract", action="store_true",
                   help="Re-extract constraints even if cache exists")
    p.add_argument("--clustering", default="hdbscan",
                   choices=["hdbscan", "kmeans"])
    p.add_argument("--n-clusters", type=int, default=15,
                   help="Number of clusters for KMeans (ignored for HDBSCAN)")
    p.add_argument("--sampling-rate", type=float, default=0.01,
                   help="Fraction of cells to label per cluster (default: 0.01)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    verbose = not args.quiet

    # ── Resolve paths ──
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    constraints_json = args.constraints_json or str(out_dir / "constraints.json")

    # ── Extract constraints ──
    print(f"\n{'=' * 70}")
    print(f"DECLARATIVE ATTRIBUTION — SCENARIO {args.scenario}")
    print(f"{'=' * 70}")

    constraints = extract_constraints(
        description_path=args.description,
        output_json=constraints_json,
        force_reextract=args.force_reextract,
    )

    if not constraints:
        print("ERROR: No constraints extracted. Check description file and LLM output.")
        sys.exit(1)

    print(f"\n[main] {len(constraints)} constraints loaded:")
    for c in constraints:
        print(f"  {c['id']}: {c['description']}")

    # ── Load datasets ──
    print(f"\n[main] Loading datasets...")
    dirty_df = pd.read_csv(args.dirty, dtype=str, keep_default_na=False)
    mask_df  = pd.read_csv(args.mask,  dtype=str, keep_default_na=False)
    print(f"  dirty: {dirty_df.shape}  mask: {mask_df.shape}")

    gt_mask = None
    if args.gt:
        gt_mask = pd.read_csv(args.gt, dtype=str, keep_default_na=False)
        # Convert gt mask values to int
        for col in gt_mask.columns:
            gt_mask[col] = pd.to_numeric(gt_mask[col], errors="coerce").fillna(0).astype(int)
        print(f"  ground truth: {gt_mask.shape}")

    external_features = None
    if args.external_features:
        if args.scenario == "C":
            print("[main] Warning: --external-features provided but scenario is C. Ignoring.")
        else:
            external_features = pd.read_csv(args.external_features)
            print(f"  external features: {external_features.shape}")

    if args.scenario in ("BC", "BplusC") and external_features is None:
        print(f"ERROR: Scenario {args.scenario} requires --external-features.")
        sys.exit(1)

    # ── Run pipeline ──
    pipeline = DeclarativePipeline(
        constraints=constraints,
        sampling_rate=args.sampling_rate,
        clustering=args.clustering,
        n_clusters=args.n_clusters,
        random_state=args.seed,
        verbose=verbose,
    )

    result = pipeline.run(
        dirty_df=dirty_df,
        mask_df=mask_df,
        ground_truth_mask=gt_mask,
        external_features_df=external_features,
    )

    # ── Save results ──
    save_results(result, str(out_dir), scenario=args.scenario)

    # ── Print summary ──
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — Scenario {args.scenario}")
    print(f"{'=' * 70}")
    print(f"Description : {args.description}")
    print(f"Constraints : {len(constraints)}")
    print(f"Clustering  : {args.clustering}")
    print(f"Scenario    : {args.scenario}")
    metrics = result.get("metrics")
    if metrics:
        print(f"\nResults:")
        print(f"  Accuracy       : {metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  F1 Weighted    : {metrics.get('f1_weighted', 'N/A'):.4f}")
        print(f"  F1 Macro       : {metrics.get('f1_macro', 'N/A'):.4f}")
        print(f"  F1 Intentional : {metrics.get('f1_intentional', 'N/A'):.4f}")
        print(f"  F1 Uninten.    : {metrics.get('f1_unintentional', 'N/A'):.4f}")
    else:
        print("\nNo ground truth provided — metrics not computed.")
    print(f"\nOutputs saved to: {out_dir}/{args.scenario}/")


if __name__ == "__main__":
    main()
