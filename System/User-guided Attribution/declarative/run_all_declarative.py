#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_declarative.py
======================
Run Scenario C, B+C, and (B+)+C for all 7 datasets.

Scenarios
---------
  C        — declarative constraint features only (standalone)
  B+C      — 12 structural (B) + constraint features
  (B+)+C   — 23 structural+statistical (B+) + constraint features

All at 1% label budget, DBSCAN clustering (fixed across every scenario in
the paper so that comparisons isolate the feature set, not the clustering
choice), Random Forest.

Output
------
  results/<dataset_key>/<scenario>/metrics.json
  results/<dataset_key>/<scenario>/features.csv
  results/<dataset_key>/<scenario>/feature_importance.csv
  results/constraints/<name>_constraints.json    (one-time, cached)
  results/summary_declarative.csv                (all datasets × scenarios)

Usage
-----
  # All datasets, all scenarios
  python run_all_declarative.py

  # Specific datasets
  python run_all_declarative.py --datasets adult_llm adult_mixed

  # Force re-extract constraints from LLM
  python run_all_declarative.py --force-reextract

  # Specific scenarios only
  python run_all_declarative.py --scenarios C BC BplusC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
DECL_DIR = Path(__file__).parent.resolve()
EDS_SRC  = DECL_DIR.parents[1]       # error_detection_system/src/
ROOT     = DECL_DIR.parents[3]       # llms_baseline/
FRAUD    = ROOT / "fraud_baseline"

for _p in [str(DECL_DIR), str(EDS_SRC), str(FRAUD)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Declarative imports ───────────────────────────────────────────────────────
from extractor import extract_constraints             # noqa: E402
from evaluator import ConstraintEvaluator            # noqa: E402
from pipeline  import DeclarativePipeline, save_results  # noqa: E402

# ── Dataset loaders (fraud_baseline) ─────────────────────────────────────────
from datasets import LOADERS                         # noqa: E402

# ── B heuristics pipeline ─────────────────────────────────────────────────────
try:
    from attribution.heuristics.pipeline import AttributionPipeline
    HEURISTICS_OK = True
except Exception as _e:
    print(f"[warning] Heuristics pipeline unavailable: {_e}")
    print("  B+C and (B+)+C scenarios will be skipped.")
    HEURISTICS_OK = False

RANDOM_STATE = 42
SAMPLE_RATE  = 0.01

# ── Output dirs ───────────────────────────────────────────────────────────────
CONFIGS_DIR      = DECL_DIR / "configs"
RESULTS_DIR      = DECL_DIR / "results"
CONSTRAINTS_DIR  = RESULTS_DIR / "constraints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset → config mapping ──────────────────────────────────────────────────
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
            target_col=None,   # 'class' is excluded from dirty_df by the TFM loader
            codependent_pairs=[("education", "education-num")],
            sensitive_cols=["race", "sex"],
        ),
    },
    "twitterbot_llm": {
        "name":        "TwitterBot-LLM V2",
        "description": CONFIGS_DIR / "twibot20.txt",
        "constraints_json": CONSTRAINTS_DIR / "twibot20_constraints.json",
        "pipeline_cfg": dict(
            target_col=None,   # 'label' is excluded from dirty_df by the loader
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
            target_col=None,   # no classification target; real scraped listings
            codependent_pairs=[("spec_brand", "title"), ("category", "marketplace")],
            sensitive_cols=[],
        ),
    },
}

ALL_DATASETS  = list(DATASET_CONFIGS.keys())
ALL_SCENARIOS = ["C", "BC", "BplusC"]


# ─────────────────────────────────────────────────────────────────────────────
# B/B+ feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_b_features(dirty: pd.DataFrame,
                         blind: pd.DataFrame,
                         cfg: dict) -> Optional[pd.DataFrame]:
    """
    Extract 12 structural heuristic features (Scenario B) for each erroneous cell.
    Returns DataFrame with columns: row_idx, column_name, h1_..., h2_..., ...
    """
    if not HEURISTICS_OK:
        return None
    try:
        # Ensure blind has same columns as dirty (add zeros for any missing)
        blind_aligned = blind.reindex(columns=dirty.columns, fill_value=0)

        pipe = AttributionPipeline(
            target_col=cfg.get("target_col", None),
            codependent_pairs=cfg.get("codependent_pairs", []),
            sensitive_cols=cfg.get("sensitive_cols", []),
        )
        pipe.fit(dirty, blind_aligned)
        feat_df = pipe.compute_features(dirty, blind_aligned)
        # feat_df has MultiIndex (row_idx, col_name)
        feat_df = feat_df.reset_index()
        feat_df = feat_df.rename(columns={"col_name": "column_name"})
        print(f"  [B features] shape: {feat_df.shape}")
        return feat_df
    except Exception as e:
        print(f"  [warning] B feature extraction failed: {e}")
        return None


def _extract_statistical_features(dirty: pd.DataFrame,
                                   clean: pd.DataFrame,
                                   mask: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract 10 statistical features (the B+ addition) for each erroneous cell.
    Returns DataFrame with columns: row_idx, column_name, stat_*
    """
    try:
        clean = clean.reset_index(drop=True)
        dirty = dirty.reset_index(drop=True)
        mask  = mask.reset_index(drop=True)

        # For Adult-LLM: dirty is 3× the clean (LLM was run on 3 copies)
        if len(dirty) == 3 * len(clean):
            clean = clean.loc[clean.index.repeat(3)].reset_index(drop=True)

        # Align clean columns to dirty
        shared_cols = [c for c in dirty.columns if c in clean.columns]
        clean = clean[shared_cols].reset_index(drop=True)
        dirty = dirty[shared_cols].reset_index(drop=True)

        all_cols = [c for c in mask.columns if c in shared_cols]
        col_enc  = {c: i for i, c in enumerate(all_cols)}

        mask_r = mask[all_cols].reset_index(drop=True)

        # Long-form erroneous cells
        mask_long = mask_r.copy()
        mask_long.index.name = "row_idx"
        mask_long = mask_long.reset_index().melt(
            id_vars="row_idx", var_name="col_name", value_name="intent"
        )
        mask_long = mask_long[mask_long["intent"] != 0].reset_index(drop=True)

        if len(mask_long) == 0:
            return None

        row_idxs  = mask_long["row_idx"].values
        col_names = mask_long["col_name"].values

        def _get_vals(df, rows, cols):
            arr     = df.to_numpy()
            col_map = {c: j for j, c in enumerate(df.columns)}
            cidxs   = [col_map.get(c, 0) for c in cols]
            return arr[rows, cidxs]

        orig_vals = _get_vals(clean, row_idxs, col_names)
        new_vals  = _get_vals(dirty, row_idxs, col_names)

        def _to_num(arr):
            out = np.zeros(len(arr), dtype=float)
            for i, v in enumerate(arr):
                try:
                    out[i] = float(v) if pd.notna(v) else 0.0
                except (ValueError, TypeError):
                    out[i] = 0.0
            return out

        orig_num  = _to_num(orig_vals)
        new_num   = _to_num(new_vals)
        diff      = new_num - orig_num
        magnitude = np.abs(diff)
        for i, (ov, nv) in enumerate(zip(orig_vals, new_vals)):
            if str(ov) != str(nv) and magnitude[i] == 0.0:
                magnitude[i] = 1.0

        feat_enc = np.array([col_enc.get(c, 0) for c in col_names], dtype=float)

        per_col_le: Dict = {}
        for col in all_cols:
            if col not in dirty.columns:
                continue
            orig_s = clean[col].astype(str).tolist() if col in clean.columns else []
            new_s  = dirty[col].astype(str).tolist()
            per_col_le[col] = LabelEncoder().fit(sorted(set(orig_s + new_s)))

        orig_enc = np.full(len(mask_long), -1, dtype=float)
        new_enc  = np.full(len(mask_long), -1, dtype=float)
        for i, (col, ov, nv) in enumerate(zip(col_names, orig_vals, new_vals)):
            le = per_col_le.get(col)
            if le is None:
                continue
            try:
                orig_enc[i] = float(le.transform([str(ov)])[0])
            except ValueError:
                pass
            try:
                new_enc[i] = float(le.transform([str(nv)])[0])
            except ValueError:
                pass

        stat_df = pd.DataFrame({
            "row_idx":                     row_idxs,
            "column_name":                 col_names,
            "stat_change_magnitude":       magnitude,
            "stat_relative_change":        magnitude / (np.abs(orig_num) + 1.0),
            "stat_change_direction":       np.sign(diff),
            "stat_original_magnitude":     np.abs(orig_num),
            "stat_new_magnitude":          np.abs(new_num),
            "stat_original_log":           np.log1p(np.abs(orig_num)),
            "stat_new_log":                np.log1p(np.abs(new_num)),
            "stat_feature_name_encoded":   feat_enc,
            "stat_original_value_encoded": orig_enc,
            "stat_new_value_encoded":      new_enc,
        })
        print(f"  [Statistical features] shape: {stat_df.shape}")
        return stat_df

    except Exception as e:
        print(f"  [warning] Statistical feature extraction failed: {e}")
        return None


def _combine_b_and_stat(b_feats: pd.DataFrame,
                         stat_feats: pd.DataFrame) -> pd.DataFrame:
    """Merge B (13) and statistical (10) features on (row_idx, column_name) → 23 features."""
    merged = b_feats.merge(stat_feats, on=["row_idx", "column_name"], how="inner")
    print(f"  [B+ combined] shape: {merged.shape}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset runner
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset(
    ds_key: str,
    scenarios: List[str],
    force_reextract: bool,
    verbose: bool,
) -> List[Dict]:
    cfg = DATASET_CONFIGS[ds_key]
    print(f"\n{'='*70}")
    print(f"Dataset: {cfg['name']} ({ds_key})")
    print(f"{'='*70}")

    # ── Load dataset ──
    try:
        ds = LOADERS[ds_key]()
    except Exception as e:
        print(f"[FAIL] Could not load {ds_key}: {e}")
        return []

    dirty    = ds["dirty"]
    gt_mask  = ds["mask"]          # +1 / -1 / 0
    clean    = ds["clean"]
    blind    = (gt_mask != 0).astype(int)   # binary erroneous-cell mask

    n_err  = (gt_mask != 0).sum().sum()
    n_int  = (gt_mask == 1).sum().sum()
    n_unint = (gt_mask == -1).sum().sum()
    print(f"  {n_err} erroneous cells — INT={n_int}, UNINT={n_unint}")

    # ── Extract constraints (one-time per description file) ──
    desc_path  = cfg["description"]
    cache_path = str(cfg["constraints_json"])

    if not Path(str(desc_path)).exists():
        print(f"  [skip] Description file not found: {desc_path}")
        return []

    try:
        constraints = extract_constraints(
            description_path=str(desc_path),
            output_json=cache_path,
            force_reextract=force_reextract,
        )
    except Exception as e:
        print(f"  [FAIL] Constraint extraction failed: {e}")
        return []

    print(f"  {len(constraints)} valid constraints")

    # ── Pre-compute B and B+ features (shared across BC and BplusC) ──
    b_feats  = None
    bp_feats = None

    needs_b  = any(s in scenarios for s in ("BC", "BplusC"))
    needs_bp = "BplusC" in scenarios

    if needs_b and HEURISTICS_OK:
        print("\n  [B features] Extracting 12 structural heuristic features...")
        b_feats = _extract_b_features(dirty, blind, cfg["pipeline_cfg"])

    if needs_bp and b_feats is not None:
        print("\n  [B+ features] Extracting 10 statistical features...")
        stat_feats = _extract_statistical_features(dirty, clean, gt_mask)
        if stat_feats is not None and b_feats is not None:
            bp_feats = _combine_b_and_stat(b_feats, stat_feats)

    # ── Run each scenario ──
    rows = []
    out_base = RESULTS_DIR / ds_key

    for scenario in scenarios:
        print(f"\n  ── Scenario {scenario} ──")
        t_start = time.time()

        external = None
        if scenario == "BC":
            if b_feats is None:
                print("  [skip] B features unavailable — skipping B+C")
                continue
            external = b_feats.copy()
        elif scenario == "BplusC":
            if bp_feats is None:
                print("  [skip] B+ features unavailable — skipping (B+)+C")
                continue
            external = bp_feats.copy()

        try:
            pipe = DeclarativePipeline(
                constraints=constraints,
                sampling_rate=SAMPLE_RATE,
                clustering="dbscan",
                random_state=RANDOM_STATE,
                verbose=verbose,
            )
            result = pipe.run(
                dirty_df=dirty,
                mask_df=blind,
                ground_truth_mask=gt_mask,
                external_features_df=external,
            )
        except Exception as e:
            print(f"  [FAIL] Scenario {scenario}: {e}")
            import traceback; traceback.print_exc()
            continue

        save_results(result, str(out_base), scenario)

        metrics = result.get("metrics") or {}
        elapsed = result.get("elapsed", time.time() - t_start)

        row = {
            "dataset":   cfg["name"],
            "ds_key":    ds_key,
            "scenario":  scenario,
            "n_cells":   n_err,
            "n_int":     n_int,
            "n_unint":   n_unint,
            "n_constraints": len(constraints),
            "f1_weighted":   round(metrics.get("f1_weighted", float("nan")), 4),
            "f1_int":        round(metrics.get("f1_intentional", float("nan")), 4),
            "f1_unint":      round(metrics.get("f1_unintentional", float("nan")), 4),
            "accuracy":      round(metrics.get("accuracy", float("nan")), 4),
            "elapsed_s":     round(elapsed, 1),
        }
        rows.append(row)
        print(f"\n  [Result] {scenario}  F1-w={row['f1_weighted']:.4f}  "
              f"F1-INT={row['f1_int']:.4f}  F1-UNINT={row['f1_unint']:.4f}  "
              f"({elapsed:.1f}s)")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run C, B+C, (B+)+C declarative attribution for all datasets"
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Dataset keys to run (default: all 7). "
             f"Choices: {', '.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Scenarios to run (default: C BC BplusC).",
    )
    parser.add_argument(
        "--force-reextract", action="store_true",
        help="Re-call LLM even if constraint JSON cache exists.",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed pipeline output.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose pipeline output.",
    )
    args = parser.parse_args()

    dataset_keys = args.datasets or ALL_DATASETS
    scenarios    = args.scenarios or ALL_SCENARIOS
    verbose      = not args.quiet

    # Validate
    for k in dataset_keys:
        if k not in DATASET_CONFIGS:
            print(f"[error] Unknown dataset: {k}. Choose from: {ALL_DATASETS}")
            sys.exit(1)

    print("=" * 70)
    print("Declarative Heuristics (Family C) — Full Experiment Run")
    print("=" * 70)
    print(f"Datasets  : {dataset_keys}")
    print(f"Scenarios : {scenarios}")
    print(f"Label budget: {SAMPLE_RATE*100:.0f}%")
    print(f"Force re-extract: {args.force_reextract}")

    all_rows = []
    t_global = time.time()

    for ds_key in dataset_keys:
        if ds_key not in LOADERS:
            print(f"[skip] {ds_key} not in LOADERS")
            continue
        rows = run_dataset(
            ds_key=ds_key,
            scenarios=scenarios,
            force_reextract=args.force_reextract,
            verbose=verbose,
        )
        all_rows.extend(rows)

    # ── Summary ──────────────────────────────────────────────────────────────
    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = RESULTS_DIR / "summary_declarative.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        cols = ["dataset", "scenario", "n_cells", "n_constraints",
                "f1_weighted", "f1_int", "f1_unint", "accuracy"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print(f"\nSaved → {summary_path}")
    else:
        print("\n[warn] No results produced.")

    print(f"\nTotal time: {time.time() - t_global:.1f}s")


if __name__ == "__main__":
    main()
