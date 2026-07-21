#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ground Truth Label Generator
-------------------------------------------------------------------------------
Creates labeled dataset from original data + masks.

Input:
  - combined_dataset_no_id.csv: Original records (N rows)
  - combined_mask_no_id.csv: Binary masks (N rows, 1=changed, 0=unchanged)

Output:
  - labeled_dataset.csv: Original data + 'is_erroneous' label column
    - is_erroneous=1 if ANY feature in mask row has value 1
    - is_erroneous=0 if ALL features in mask row have value 0

Logic:
  Row i is erroneous if: sum(mask_row_i) > 0
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_ground_truth_labels(dataset_path: str, mask_path: str, output_path: str):
    """
    Create labeled dataset with ground truth based on masks.
    
    Parameters
    ----------
    dataset_path : str
        Path to original dataset CSV (without labels)
    mask_path : str
        Path to mask CSV (binary, 1=changed, 0=unchanged)
    output_path : str
        Path to save labeled dataset
        
    Returns
    -------
    dict
        Statistics about the labeling process
    """
    print("="*80)
    print("GROUND TRUTH LABEL GENERATOR")
    print("="*80)
    
    # Load data
    print(f"\n1. Loading dataset: {dataset_path}")
    df_data = pd.read_csv(dataset_path)
    print(f"   - Shape: {df_data.shape}")
    print(f"   - Columns: {df_data.columns.tolist()}")
    
    print(f"\n2. Loading masks: {mask_path}")
    df_mask = pd.read_csv(mask_path)
    print(f"   - Shape: {df_mask.shape}")
    print(f"   - Columns: {df_mask.columns.tolist()}")
    
    # Verify alignment
    assert len(df_data) == len(df_mask), \
        f"Row count mismatch! Dataset: {len(df_data)}, Masks: {len(df_mask)}"
    assert list(df_data.columns) == list(df_mask.columns), \
        f"Column mismatch! Dataset columns != Mask columns"
    
    print(f"\n3. Generating ground truth labels...")
    
    # Create binary label: 1 if ANY mask value is 1, else 0
    # Sum across columns (axis=1) to get total changes per row
    mask_sums = df_mask.sum(axis=1)
    is_erroneous = (mask_sums > 0).astype(int)
    
    # Add label column to dataset
    df_labeled = df_data.copy()
    df_labeled['is_erroneous'] = is_erroneous
    
    # Statistics
    num_total = len(df_labeled)
    num_erroneous = is_erroneous.sum()
    num_clean = num_total - num_erroneous
    pct_erroneous = 100 * num_erroneous / num_total
    
    print(f"\n4. Label Statistics:")
    print(f"   - Total records: {num_total:,}")
    print(f"   - Erroneous records (label=1): {num_erroneous:,} ({pct_erroneous:.2f}%)")
    print(f"   - Clean records (label=0): {num_clean:,} ({100-pct_erroneous:.2f}%)")
    
    # Per-record change statistics
    print(f"\n5. Changes per Erroneous Record:")
    if num_erroneous > 0:
        changes_per_record = mask_sums[mask_sums > 0]
        print(f"   - Min changes: {changes_per_record.min()}")
        print(f"   - Max changes: {changes_per_record.max()}")
        print(f"   - Mean changes: {changes_per_record.mean():.2f}")
        print(f"   - Median changes: {changes_per_record.median():.0f}")
    
    # Save labeled dataset
    print(f"\n6. Saving labeled dataset: {output_path}")
    df_labeled.to_csv(output_path, index=False)
    print(f"   - Output shape: {df_labeled.shape}")
    print(f"   - Output columns: {df_labeled.columns.tolist()}")
    
    # Create summary statistics
    stats = {
        'num_total': int(num_total),
        'num_erroneous': int(num_erroneous),
        'num_clean': int(num_clean),
        'pct_erroneous': float(pct_erroneous),
        'pct_clean': float(100 - pct_erroneous),
        'min_changes': int(changes_per_record.min()) if num_erroneous > 0 else 0,
        'max_changes': int(changes_per_record.max()) if num_erroneous > 0 else 0,
        'mean_changes': float(changes_per_record.mean()) if num_erroneous > 0 else 0.0,
        'median_changes': float(changes_per_record.median()) if num_erroneous > 0 else 0.0
    }
    
    print("\n" + "="*80)
    print("✅ Ground truth labels created successfully!")
    print("="*80)
    
    return stats


def main():
    """Main execution."""
    # Paths (from error_detection_system/src/preprocessing/ -> llms_baseline/)
    BASE_DIR = Path(__file__).parent.parent.parent.parent  # Go up to llms_baseline/
    DATASET_PATH = BASE_DIR / 'klim-kireev' / 'datasets' / 'twitter-bot' / 'combined_dataset_no_id.csv'
    MASK_PATH = BASE_DIR / 'klim-kireev' / 'datasets' / 'twitter-bot' / 'combined_mask_no_id.csv'
    OUTPUT_PATH = BASE_DIR / 'error_detection_system' / 'datasets' / 'twitter_bot' / 'labeled_dataset.csv'
    
    # Create output directory if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate labels
    stats = create_ground_truth_labels(
        dataset_path=str(DATASET_PATH),
        mask_path=str(MASK_PATH),
        output_path=str(OUTPUT_PATH)
    )
    
    # Save statistics
    stats_path = OUTPUT_PATH.parent / 'labeling_statistics.json'
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Statistics saved to: {stats_path}")


if __name__ == "__main__":
    main()
