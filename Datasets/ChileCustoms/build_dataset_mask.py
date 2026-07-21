#!/usr/bin/env python3
"""
build_dataset_mask.py
======================
Builds the cell-level intent mask for Chile Customs, in the same
0/1/-1 format as NIYYAT/Datasets/<dataset>/masks.csv (eBay, Adult, etc.):

    0  -> clean (no error) OR an error excluded from intent labelling
    1  -> intentional (behavioural-anomaly cell on a declaration that
          customs genuinely selected for physical inspection, NU_REGR)
   -1  -> unintentional (behavioural-anomaly cell on a declaration that
          was not inspected)

This is the ground truth approved after investigation: NU_REGR is used
DIRECTLY as the label (never as a feature -- excluded from every
dirty/clean column), restricted to behavioural-type cells only.

Why restrict to behavioural cells (not every flagged cell in an
inspected row)
---------------------------------------------------------------------
An earlier version of this dataset's mask used an importer-behavioural-
tier proxy instead of NU_REGR directly. That was replaced by this direct
construction. The remaining design choice -- behavioural cells only,
not every flagged cell -- was checked empirically: among flagged cells on
inspected declarations, 83.2% are structural (truncated fields, broken
arithmetic, typos) vs. 86.4% on non-inspected declarations -- almost the
same mixture. Labelling 100% of an inspected row's structural cells as
"intentional" would inject labels we have independent, mechanistic reason
to believe are wrong (e.g. a CIF arithmetic check that is enforced
server-side cannot be a deliberate human choice). Behavioural cells (price
/ outlier / rare-value anomalies) are the only category with a plausible
link to what customs risk engines are documented to act on, so structural
cells are excluded from the labelled set entirely (mask=0, same as clean)
rather than forced into either class.

No oracle clean reference exists for this dataset (real customs filings,
not synthetic injection) -- `clean` here is a pseudo-clean approximation
(per-(HS4-chapter, origin-country) median/mode, see build_pseudo_clean.py),
exactly analogous to eBay's pseudo_clean.csv. Reference-Augmented (B+) is
therefore not applicable, same as eBay.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATA_CSV = BASE / "data" / "chile_din_validation_sample_small.csv"
DETAIL_CSV = BASE / "results_validation" / "chile_din_all_errors_detailed.csv"

OUT_LOCAL = BASE / "data" / "masks.csv"
OUT_NIYYAT = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/ChileCustoms/masks.csv")
OUT_BLIND = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/ChileCustoms/masks_blind.csv")

EXCLUDE_FROM_FEATURES = ["data_month", "NUMENCRIPTADO", "NUMITEM", "NUM_UNICO_IMPORTADOR", "NU_REGR"]

BEHAVIOURAL_TYPES = {
    "crossfield_unit_price_extreme_high_for_hs4", "crossfield_unit_price_extreme_low_for_hs4",
    "statistical_outlier_iqr", "statistical_outlier_zscore",
    "ml_outlier_isolation_forest", "autotest_rare_categorical_value",
}

INTENTIONAL = 1
UNINTENTIONAL = -1
CLEAN = 0


def main():
    print("=" * 70)
    print("Chile Customs -- Dataset Mask Builder (direct NU_REGR label)")
    print("=" * 70)

    print("\n[1] Loading data...")
    raw = pd.read_csv(DATA_CSV, dtype=str, low_memory=False).reset_index(drop=True)
    raw.index.name = "row_idx"
    feature_cols = [c for c in raw.columns if c not in EXCLUDE_FROM_FEATURES]
    print(f"    {len(raw):,} rows x {len(feature_cols)} feature columns")

    inspected_decl = set(
        raw.drop_duplicates("NUMENCRIPTADO")
           .loc[lambda d: d["NU_REGR"].notna() & (d["NU_REGR"] != "0"), "NUMENCRIPTADO"]
    )
    print(f"    {len(inspected_decl):,} genuinely inspected declarations (NU_REGR)")

    print("\n[2] Restricting to behavioural-type flagged cells...")
    det = pd.read_csv(DETAIL_CSV, usecols=["row_idx", "column", "error_type"])
    det = det[det["error_type"].isin(BEHAVIOURAL_TYPES) & det["column"].isin(feature_cols)]
    det = det.drop_duplicates(["row_idx", "column"])
    det = det.merge(raw[["NUMENCRIPTADO"]], left_on="row_idx", right_index=True, how="left")
    det["cell_value"] = np.where(det["NUMENCRIPTADO"].isin(inspected_decl), INTENTIONAL, UNINTENTIONAL)
    print(f"    behavioural cells: INTENTIONAL={int((det['cell_value']==INTENTIONAL).sum()):,}  "
          f"UNINTENTIONAL={int((det['cell_value']==UNINTENTIONAL).sum()):,}")

    print("\n[3] Building full row x column mask matrix...")
    mask = pd.DataFrame(CLEAN, index=raw.index, columns=feature_cols)
    col_loc = {c: i for i, c in enumerate(feature_cols)}
    for row_idx, col, val in det[["row_idx", "column", "cell_value"]].itertuples(index=False):
        mask.iat[row_idx, col_loc[col]] = val

    total_cells = mask.shape[0] * mask.shape[1]
    n_int = int((mask.values == INTENTIONAL).sum())
    n_unint = int((mask.values == UNINTENTIONAL).sum())
    n_clean = int((mask.values == CLEAN).sum())
    print(f"\n    Mask shape: {mask.shape[0]:,} x {mask.shape[1]} = {total_cells:,} cells")
    print(f"      CLEAN/excluded (0) : {n_clean:,}  ({100*n_clean/total_cells:.2f}%)")
    print(f"      UNINTENTIONAL (-1) : {n_unint:,}  ({100*n_unint/total_cells:.2f}%)")
    print(f"      INTENTIONAL (1)    : {n_int:,}  ({100*n_int/total_cells:.2f}%)")

    print("\n[4] Saving outputs...")
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_LOCAL, index=False)
    print(f"    {OUT_LOCAL}")

    OUT_NIYYAT.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_NIYYAT, index=False)
    print(f"    {OUT_NIYYAT}")

    mask_blind = mask.copy()
    mask_blind[mask_blind != CLEAN] = 1
    mask_blind.to_csv(OUT_BLIND, index=False)
    print(f"    {OUT_BLIND}  (detection-only)")

    return mask


if __name__ == "__main__":
    main()
