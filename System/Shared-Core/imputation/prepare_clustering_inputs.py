#!/usr/bin/env python3
"""
prepare_clustering_inputs.py
============================

Converts the v3 labeled dataset (1:1 mask/dirty/correct) into the format
expected by compare_clustering_algorithms.py (3 dirty variants per clean record),
and produces TWO versions of the dirty data:

  ORACLE  : dirty = combined_dataset_no_id_v3.csv  (real corrupted values, correct = true GT)
  IMPUTED : dirty = combined_dataset_no_id_v3.csv  (same), correct = MICE imputed values
            i.e. wherever a cell was corrupted, the "correct" value is the MICE estimate

Both share the same masks.csv.  The script outputs to:
  outputs/clustering_comparison_v3/oracle/
  outputs/clustering_comparison_v3/imputed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path("/home/mohamed/error_injector/llms_baseline/error_detection_system")
DATASET_DIR = BASE / "datasets/adult_income_v3"
IMPUTATION_RESULTS = BASE / "outputs/imputation/adult_income_v3/imputation_results.csv"
OUT_DIR = BASE / "outputs/clustering_comparison_v3"

MASKS_PATH    = DATASET_DIR / "ground_truth_masks_v3.csv"
CORRECT_PATH  = DATASET_DIR / "aligned_correct_v3.csv"
DIRTY_PATH    = DATASET_DIR / "combined_dataset_no_id_v3.csv"


def triple_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat every row 3 times so the clustering script's idx//3 logic maps
    correctly: dirty_row[i] -> correct_row[i//3] == correct_row[i]."""
    return pd.DataFrame(
        np.repeat(df.values, 3, axis=0),
        columns=df.columns
    ).reset_index(drop=True)


def build_imputed_correct(correct_df: pd.DataFrame,
                           imputation_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build an 'imputed correct' dataframe:
      - Start from aligned_correct_v3 (the true correct values)
      - For every cell in imputation_results, REPLACE the correct value
        with the MICE imputed value.
    This simulates: "what if we used MICE imputation instead of ground truth?"
    """
    imputed_correct = correct_df.copy()

    n_replaced = 0
    n_missing  = 0

    for _, row in imputation_results.iterrows():
        row_idx = int(row["row_idx"])
        col     = row["column"]
        imp_val = row["imputed_value"]

        if col not in imputed_correct.columns:
            continue
        if pd.isna(imp_val):
            n_missing += 1
            continue

        # Replace correct value with imputed estimate
        imputed_correct.at[row_idx, col] = imp_val
        n_replaced += 1

    print(f"  Replaced {n_replaced} cells with MICE imputed values")
    print(f"  Skipped  {n_missing} cells (imputed_value was NaN → kept true correct)")
    return imputed_correct


def main():
    print("=" * 65)
    print("PREPARING CLUSTERING INPUTS — Oracle & Imputed")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    masks   = pd.read_csv(MASKS_PATH)
    correct = pd.read_csv(CORRECT_PATH)
    dirty   = pd.read_csv(DIRTY_PATH)
    imp_res = pd.read_csv(IMPUTATION_RESULTS)

    print(f"\nLoaded:")
    print(f"  masks   : {masks.shape}")
    print(f"  correct : {correct.shape}")
    print(f"  dirty   : {dirty.shape}")
    print(f"  imp_res : {imp_res.shape}")

    # ── Build imputed-correct ──────────────────────────────────────────────────
    print("\nBuilding IMPUTED correct (replacing GT with MICE estimates)...")
    imputed_correct = build_imputed_correct(correct, imp_res)

    # Quick sanity check
    diffs = 0
    for col in correct.columns:
        try:
            diff = (correct[col].astype(str) != imputed_correct[col].astype(str)).sum()
            diffs += diff
        except Exception:
            pass
    print(f"  Total cells that differ (oracle vs imputed correct): {diffs}")

    # ── Triple rows (1:1 → 3:1 ratio required by clustering script) ────────────
    print("\nTripling rows to match clustering script's idx//3 expectation...")
    masks_3x          = triple_rows(masks)
    correct_3x        = triple_rows(correct)
    imputed_correct_3x = triple_rows(imputed_correct)
    dirty_3x          = triple_rows(dirty)

    print(f"  masks_3x          : {masks_3x.shape}")
    print(f"  correct_3x        : {correct_3x.shape}")
    print(f"  imputed_correct_3x: {imputed_correct_3x.shape}")
    print(f"  dirty_3x          : {dirty_3x.shape}")

    # ── Save oracle inputs ─────────────────────────────────────────────────────
    oracle_dir = OUT_DIR / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)

    masks_3x.to_csv(oracle_dir / "masks.csv",           index=False)
    correct_3x.to_csv(oracle_dir / "correct_records.csv", index=False)
    dirty_3x.to_csv(oracle_dir / "dirty_records.csv",   index=False)

    print(f"\n✓ Oracle inputs saved to: {oracle_dir}")

    # ── Save imputed inputs ────────────────────────────────────────────────────
    imputed_dir = OUT_DIR / "imputed"
    imputed_dir.mkdir(parents=True, exist_ok=True)

    masks_3x.to_csv(imputed_dir / "masks.csv",           index=False)
    imputed_correct_3x.to_csv(imputed_dir / "correct_records.csv", index=False)
    dirty_3x.to_csv(imputed_dir / "dirty_records.csv",   index=False)

    print(f"✓ Imputed inputs saved to: {imputed_dir}")

    # ── Print the two run commands ─────────────────────────────────────────────
    script = BASE / "src/attribution/clustering-organized/scripts/compare_clustering_algorithms.py"
    print()
    print("=" * 65)
    print("READY — Run these two commands:")
    print("=" * 65)
    print()
    print("# ORACLE (uses true ground truth):")
    print(f"python {script} \\")
    print(f"  --mask-path    {oracle_dir/'masks.csv'} \\")
    print(f"  --clean-data-path {oracle_dir/'correct_records.csv'} \\")
    print(f"  --dirty-data-path {oracle_dir/'dirty_records.csv'}")
    print()
    print("# IMPUTED (uses MICE estimates as ground truth):")
    print(f"python {script} \\")
    print(f"  --mask-path    {imputed_dir/'masks.csv'} \\")
    print(f"  --clean-data-path {imputed_dir/'correct_records.csv'} \\")
    print(f"  --dirty-data-path {imputed_dir/'dirty_records.csv'}")


if __name__ == "__main__":
    main()
