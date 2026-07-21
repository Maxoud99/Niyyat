"""
TWITTERBOT CLUSTERING COMPARISON
=================================
Runs the same clustering + label propagation pipeline as compare_clustering_algorithms.py
but adapted for 1:1 row format data (TwiBot-20).

The original pipeline expects 3 variants per original record (idx // 3).
TwiBot-20 uses a simple dirty/clean/mask format where each row IS one record.

Files used:
  dirty:  mixed_error_pipeline_twitter/output/twibot20_phase2_final.csv
  mask:   mixed_error_pipeline_twitter/output/mask_combined.csv
  clean:  mixed_error_pipeline_twitter/output/twibot20_clean.csv
"""

import sys
import os
from pathlib import Path

# ── path setup ──────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd
import numpy as np
from tqdm import tqdm
from compare_clustering_algorithms import ClusteringComparison

# ── data paths ───────────────────────────────────────────────────────────────
BASE = Path('/home/mohamed/error_injector/llms_baseline')
TWITTER_DIR = BASE / 'mixed_error_pipeline_twitter' / 'output'

DIRTY_PATH = TWITTER_DIR / 'twibot20_phase2_final.csv'
MASK_PATH  = TWITTER_DIR / 'mask_combined.csv'
CLEAN_PATH = TWITTER_DIR / 'twibot20_clean.csv'

# ── subclass overriding load_data ─────────────────────────────────────────────
class TwitterClusteringComparison(ClusteringComparison):
    """
    Drop-in replacement for ClusteringComparison that handles 1:1 row data
    instead of the 3-variants-per-record format used by the LLM adult dataset.
    """

    def load_data(self):
        """
        Load TwiBot-20 data in 1:1 format.
        Each row in the mask/dirty/clean CSVs is a SINGLE record
        (no idx//3 lookup needed).
        """
        self._log('main', "\n" + "=" * 70)
        self._log('main', "LOADING TWITTERBOT DATA (1:1 format)")
        self._log('main', "=" * 70)

        # ── load files ───────────────────────────────────────────────────────
        print(f"Loading Twitter data files …")
        print(f"  Mask:  {self.mask_path}")
        print(f"  Clean: {self.clean_data_path}")
        print(f"  Dirty: {self.dirty_data_path}")

        masks_df      = pd.read_csv(self.mask_path)
        clean_df      = pd.read_csv(self.clean_data_path)
        dirty_df      = pd.read_csv(self.dirty_data_path)

        print(f"✓ Loaded masks:   {masks_df.shape}")
        print(f"✓ Loaded clean:   {clean_df.shape}")
        print(f"✓ Loaded dirty:   {dirty_df.shape}")

        assert len(masks_df) == len(clean_df) == len(dirty_df), \
            "Shape mismatch between mask / clean / dirty files!"

        # ── determine feature columns (intersection of all three) ────────────
        feature_cols = [c for c in masks_df.columns
                        if c in clean_df.columns and c in dirty_df.columns]
        print(f"Feature columns to process: {feature_cols}")

        # ── build per-cell change rows ────────────────────────────────────────
        all_changes = []
        print(f"\nProcessing {len(masks_df)} records (1:1 format)…")

        for idx in tqdm(range(len(masks_df)), desc="Processing records",
                        unit="record", leave=True):

            for feature in feature_cols:
                intent_label = masks_df.iloc[idx][feature]

                # skip clean cells
                if intent_label == 0:
                    continue

                original_value = clean_df.iloc[idx][feature]
                new_value      = dirty_df.iloc[idx][feature]

                try:
                    orig_num = float(original_value) if pd.notna(original_value) else 0.0
                    new_num  = float(new_value)      if pd.notna(new_value)      else 0.0
                    magnitude = abs(new_num - orig_num)
                except (ValueError, TypeError):
                    magnitude = 1.0

                all_changes.append({
                    'variant_record_id': idx,      # one record = one "variant"
                    'original_record_id': idx,     # same; no 3x expansion
                    'variant_idx': 0,
                    'feature_name': feature,
                    'original_value': original_value,
                    'new_value': new_value,
                    'change_magnitude': magnitude,
                    'intent_label': intent_label,
                })

        self.df = pd.DataFrame(all_changes)

        print(f"\n✓ Created dataset with {len(self.df)} feature changes")
        print(f"  Intentional   (1) : {(self.df['intent_label'] ==  1).sum()}")
        print(f"  Unintentional (-1): {(self.df['intent_label'] == -1).sum()}")
        print(f"  Unique records    : {self.df['variant_record_id'].nunique()}")

        # ── encode categoricals ───────────────────────────────────────────────
        for col in ['feature_name', 'original_value', 'new_value']:
            self.df[f'{col}_encoded'] = pd.Categorical(
                self.df[col].astype(str)).codes

        def safe_numeric(series):
            return pd.to_numeric(series, errors='coerce').fillna(0)

        orig_num = safe_numeric(self.df['original_value'])
        new_num  = safe_numeric(self.df['new_value'])

        self.df['relative_change']   = self.df['change_magnitude'] / (orig_num.abs() + 1)
        self.df['change_direction']  = np.sign(new_num - orig_num)
        self.df['original_log']      = np.log1p(orig_num.abs())
        self.df['new_log']           = np.log1p(new_num.abs())
        self.df['original_magnitude']= orig_num.abs()
        self.df['new_magnitude']     = new_num.abs()

        for feat in self.df['feature_name'].unique():
            self.df[f'feat_{feat}'] = (self.df['feature_name'] == feat).astype(int)

        self.total_variants = self.df['variant_record_id'].nunique()
        self.total_samples  = len(self.df)

        if self.target_samples is None:
            self.target_samples = max(1, int(self.total_variants * 0.01))
            print(f"\n⚙️  Auto-calculated target_samples: {self.target_samples} "
                  f"(1% of {self.total_variants} records)")

        print(f"\nTotal dataset:")
        print(f"  Total records  : {self.total_variants}")
        print(f"  Total changes  : {self.total_samples}")
        print(f"  Target samples : {self.target_samples} "
              f"({self.target_samples/self.total_variants*100:.2f}% of records)")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    from datetime import datetime
    import io

    print(f"\n{'='*70}")
    print(f"TWITTERBOT CLUSTERING COMPARISON")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    comparison = TwitterClusteringComparison(
        target_samples=None,       # auto = 1% of records
        random_state=42,
        mask_path=str(MASK_PATH),
        clean_data_path=str(CLEAN_PATH),
        dirty_data_path=str(DIRTY_PATH),
        enable_detailed_logging=True,
    )

    # tee stdout to a log file inside the run dir
    log_filename = comparison.logs_dir / f"twitter_clustering_{comparison.timestamp}.log"
    log_file     = open(log_filename, 'w')

    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeLogger(sys.stdout, log_file)

    try:
        results = comparison.run_comparison()

        print(f"\n{'='*70}")
        print("✓ TWITTERBOT CLUSTERING COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📁 Outputs saved to: {comparison.run_dir}")
        print(f"   ├── results/  — CSV result files")
        print(f"   ├── plots/    — PNG visualizations")
        print(f"   └── logs/     — execution log")
        print(f"\nFiles used:")
        print(f"  Dirty: {DIRTY_PATH}")
        print(f"  Mask:  {MASK_PATH}")
        print(f"  Clean: {CLEAN_PATH}")
        print(f"\n{'='*70}")
        print(f"EXECUTION COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

    finally:
        sys.stdout = original_stdout
        log_file.close()


if __name__ == '__main__':
    main()
