#!/usr/bin/env python3
"""
build_validation_sample.py
============================
Reduces the full 284,807-row Credit Card Fraud table to a tractable
validation sample for the attribution pipelines, mirroring Chile Customs'
build_validation_sample.py: keep every row that carries at least one
labelled cell in full, plus a random fill of additional legitimate,
unflagged rows so the heuristics that need a representative "background"
population (H4 row-coherence, H6 column-importance, both train per-column
models on the full dirty frame) see a realistic clean distribution without
paying the cost of all 284,807 rows.

29,428 / 284,807 rows carry >=1 labelled (mask != 0) cell. We keep all of
them plus a random_state=42 draw of 15,572 additional Class==0, fully-clean
rows, for a 45,000-row sample -- the same order of magnitude as Chile
Customs' 49,689-row v1 sample and eBay's 10,336 listings.

This does NOT change which cells are labelled or how (Class is still the
direct label, behavioural-only restriction still applies) -- it only
restricts which ROWS participate in the released working dataset, exactly
as Chile Customs' "stratified sample of declarations" restricts which
declarations participate rather than running on the full 460,571-
declaration population.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "raw_cache", "creditcard_raw.csv")
MASK_PATH = os.path.join(HERE, "data", "masks.csv")

TARGET_FILL = 15_572  # -> 45,000-row sample total
RANDOM_STATE = 42


def main():
    raw = pd.read_csv(RAW_PATH)
    raw["Class"] = raw["Class"].astype(str).str.strip("'").astype(int)
    mask = pd.read_csv(MASK_PATH)

    labelled_rows = mask.index[(mask != 0).any(axis=1)]
    print(f"Rows with >=1 labelled cell: {len(labelled_rows):,}")

    unlabelled_legit = raw.index[(raw["Class"] == 0) & ~raw.index.isin(labelled_rows)]
    fill = pd.Index(unlabelled_legit).to_series().sample(
        n=min(TARGET_FILL, len(unlabelled_legit)), random_state=RANDOM_STATE
    ).index

    keep_idx = labelled_rows.union(fill).sort_values()
    print(f"Random fill (unflagged legit): {len(fill):,}")
    print(f"Total validation sample: {len(keep_idx):,} rows")

    raw_sample = raw.loc[keep_idx].reset_index(drop=True)
    mask_sample = mask.loc[keep_idx].reset_index(drop=True)

    raw_out = os.path.join(HERE, "data", "creditcard_validation_sample.csv")
    mask_out = os.path.join(HERE, "data", "masks_sample.csv")
    raw_sample.to_csv(raw_out, index=False)
    mask_sample.to_csv(mask_out, index=False)
    print(f"Wrote {raw_out}  shape={raw_sample.shape}")
    print(f"Wrote {mask_out}  shape={mask_sample.shape}")

    n_int = int((mask_sample.values == 1).sum())
    n_unint = int((mask_sample.values == -1).sum())
    print(f"Labelled cells preserved: INTENTIONAL={n_int:,}  UNINTENTIONAL={n_unint:,}  "
          f"(full-population had 3,503 / 61,931)")


if __name__ == "__main__":
    main()
