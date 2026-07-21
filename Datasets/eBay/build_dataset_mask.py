"""
Full Dataset Mask Builder
==========================
Builds a cell-level intentionality mask matching the shape of
data/ebay_all_listings.csv (10,336 rows x 26 columns), in the same format as
NIYYAT/Datasets/<dataset>/masks.csv:

    0  -> clean (no error / no flagged anomaly)
    1  -> intentional error
   -1  -> unintentional error

Why this differs from intentionality_labels.csv
-------------------------------------------------
intentionality_labeler.py deliberately restricts intentionality labelling to
Type A (genuine data errors) and excludes Type B (behavioural anomalies,
e.g. a seller with 0.0% feedback, a rare category, a brand missing from the
title) on principled grounds: a Type B value is not wrong, so asking whether
it is an "intentional error" is a category error.

That framing is correct for error-detection research, but it is the wrong
scope for a ground-truth MASK whose purpose is to flag manipulation. Almost
all of the actual evidence of deliberate seller manipulation in this corpus
lives in Type B (bait pricing, hidden brand behind a known EAN, rare values
used to dodge filters) -- restricting the mask to Type A produces only 9
intentional cells out of 7,402, which undercounts manipulation by roughly
two orders of magnitude relative to what manual review of even a small
sample finds.

Resolution used here
---------------------
Type A (genuine errors): keep the already-validated rules from
    intentionality_labeler.py (HARD_UNINTENTIONAL, TYPO_PREFIX, SOFT_RULES).
    Two disputed rules were checked against the underlying data and held up:
      - invalid_return_days_not_standard_option: value=1.0 in 170/171 cases,
        spread across LEGIT (155) and SUSPICIOUS (15) with no concentration
        on bad sellers -> systemic artifact, not manipulation.
      - fd_violation_duplicate_item_id: spot-checked example showed the two
        rows differ in dataset_source/search_query -- the same listing
        scraped twice in different category-targeted crawl passes, a
        crawler artifact, not seller action.

Type B (behavioural anomalies): a Type B flag is not an error on a LEGIT
    seller -- the value is genuinely valid and the cell is marked CLEAN (0).
    On a SUSPICIOUS or COUNTERFEIT seller, the same anomalous-but-valid value
    is evidence of deliberate behaviour (the seller's bad-actor status was
    already established independently via signals S1-S9) and the cell is
    marked INTENTIONAL (1). No Type B flag is ever marked UNINTENTIONAL,
    because by definition the value itself is not wrong.

Multiple flags per cell: a single (row, column) cell can be flagged by more
    than one detector. Priority order when aggregating: INTENTIONAL (1) >
    UNINTENTIONAL (-1) > CLEAN (0) -- if any signal on that cell indicates
    deliberate manipulation, that takes precedence.
"""

from pathlib import Path

import pandas as pd

import intentionality_labeler as il

BASE        = Path(__file__).parent.parent
DATA_CSV    = BASE / "data" / "ebay_all_listings.csv"
ERRORS_CSV  = BASE / "results" / "ebay_all_errors_detailed.csv"
LABELS_CSV  = BASE / "data" / "ebay_all_listings_labelled.csv"

OUT_LOCAL   = Path(__file__).parent / "data" / "masks.csv"
OUT_NIYYAT  = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/eBay/masks.csv")
OUT_BLIND   = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/eBay/masks_blind.csv")

INTENTIONAL   = 1
UNINTENTIONAL = -1
CLEAN         = 0


def main():
    print("=" * 70)
    print("Full Dataset Mask Builder")
    print("=" * 70)

    print("\n[1] Loading data …")
    base     = pd.read_csv(DATA_CSV, low_memory=False)
    errors   = pd.read_csv(ERRORS_CSV, low_memory=False)
    listings = pd.read_csv(LABELS_CSV, low_memory=False)
    print(f"    Base dataset : {base.shape[0]:,} rows x {base.shape[1]} columns")
    print(f"    Raw flags    : {len(errors):,}")

    seller_lookup = (
        listings.drop_duplicates(subset="item_id", keep="first")
                .set_index("item_id")[["auto_label", "signals_fired", "signal_count"]]
                .to_dict("index")
    )

    def _get(iid, key, default):
        return seller_lookup.get(iid, {}).get(key, default)

    errors["auto_label"] = errors["item_id"].map(lambda x: _get(x, "auto_label", "LEGIT"))

    print("\n[2] Classifying Type A (genuine errors) vs Type B (behavioural anomalies) …")

    def _is_type_a(etype: str) -> bool:
        return etype in il.TYPE_A_ERRORS or etype.startswith(il.TYPO_PREFIX)

    mask_a = errors["error_type"].map(_is_type_a)
    type_a = errors[mask_a].copy()
    type_b = errors[~mask_a].copy()
    print(f"    Type A: {len(type_a):,}   Type B: {len(type_b):,}")

    print("\n[3] Assigning per-flag intentionality …")

    # Type A: reuse the validated rules from intentionality_labeler.py
    results = [
        il.assign_label(r.error_type, r.auto_label)
        for r in type_a[["error_type", "auto_label"]].itertuples(index=False)
    ]
    type_a["cell_value"] = [r[0] for r in results]
    # AMBIGUOUS (+2) should not occur post-promotion; guard just in case.
    n_residual_ambiguous = (type_a["cell_value"] == il.AMBIGUOUS).sum()
    if n_residual_ambiguous:
        print(f"    WARNING: {n_residual_ambiguous} Type A flags still AMBIGUOUS — "
              f"treating as UNINTENTIONAL for the mask (conservative default).")
        type_a.loc[type_a["cell_value"] == il.AMBIGUOUS, "cell_value"] = UNINTENTIONAL

    # Type B: LEGIT -> clean (not an error at all), SUSPICIOUS/COUNTERFEIT -> intentional
    type_b["cell_value"] = type_b["auto_label"].map(
        lambda tier: INTENTIONAL if tier in ("SUSPICIOUS", "COUNTERFEIT") else CLEAN
    )
    # Drop LEGIT Type B rows entirely — they contribute nothing to the mask (clean = no flag)
    type_b = type_b[type_b["cell_value"] == INTENTIONAL]

    combined = pd.concat([
        type_a[["row_idx", "column", "cell_value"]],
        type_b[["row_idx", "column", "cell_value"]],
    ], ignore_index=True)
    print(f"    Combined flag records contributing to mask: {len(combined):,}")

    print("\n[4] Aggregating multiple flags per cell (priority: INTENTIONAL > UNINTENTIONAL) …")
    # Priority aggregation: take the max cell_value per (row_idx, column).
    # INTENTIONAL=1 > UNINTENTIONAL=-1 is wrong under plain max() since -1 < 0;
    # use explicit priority order instead.
    PRIORITY = {INTENTIONAL: 2, UNINTENTIONAL: 1}
    combined["priority"] = combined["cell_value"].map(PRIORITY)
    combined = combined.sort_values("priority", ascending=False)
    deduped = combined.drop_duplicates(subset=["row_idx", "column"], keep="first")
    print(f"    Unique flagged cells: {len(deduped):,}")
    print(f"      INTENTIONAL cells   : {(deduped['cell_value']==INTENTIONAL).sum():,}")
    print(f"      UNINTENTIONAL cells : {(deduped['cell_value']==UNINTENTIONAL).sum():,}")

    print("\n[5] Building full row x column mask matrix …")
    mask = pd.DataFrame(CLEAN, index=base.index, columns=base.columns)
    for row_idx, col, val in deduped[["row_idx", "column", "cell_value"]].itertuples(index=False):
        if col in mask.columns and 0 <= row_idx < len(mask):
            mask.iat[row_idx, mask.columns.get_loc(col)] = val

    total_cells = mask.shape[0] * mask.shape[1]
    n_intentional   = (mask.values == INTENTIONAL).sum()
    n_unintentional = (mask.values == UNINTENTIONAL).sum()
    n_clean         = (mask.values == CLEAN).sum()
    print(f"\n    Mask shape: {mask.shape[0]:,} x {mask.shape[1]} = {total_cells:,} cells")
    print(f"      CLEAN (0)          : {n_clean:,}  ({100*n_clean/total_cells:.1f}%)")
    print(f"      UNINTENTIONAL (-1) : {n_unintentional:,}  ({100*n_unintentional/total_cells:.1f}%)")
    print(f"      INTENTIONAL (1)    : {n_intentional:,}  ({100*n_intentional/total_cells:.1f}%)")

    print(f"\n[6] Per-column breakdown …")
    col_summary = pd.DataFrame({
        "intentional":   (mask == INTENTIONAL).sum(),
        "unintentional": (mask == UNINTENTIONAL).sum(),
    })
    col_summary = col_summary[(col_summary["intentional"] > 0) | (col_summary["unintentional"] > 0)]
    col_summary = col_summary.sort_values("intentional", ascending=False)
    print(col_summary.to_string())

    print(f"\n[7] Saving outputs …")
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_LOCAL, index=False)
    print(f"    {OUT_LOCAL}")

    OUT_NIYYAT.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_NIYYAT, index=False)
    print(f"    {OUT_NIYYAT}")

    mask_blind = mask.copy()
    mask_blind[mask_blind != CLEAN] = 1
    mask_blind.to_csv(OUT_BLIND, index=False)
    print(f"    {OUT_BLIND}  (detection-only: any error -> 1, no intentionality distinction)")

    return mask


if __name__ == "__main__":
    main()
