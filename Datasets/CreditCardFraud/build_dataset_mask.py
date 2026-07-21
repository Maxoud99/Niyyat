#!/usr/bin/env python3
"""
build_dataset_mask.py
======================
Builds the cell-level intent mask for Credit Card Fraud (ULB), in the same
0/1/-1 format as NIYYAT/Datasets/<dataset>/masks.csv (eBay, Chile Customs):

    0  -> clean (no error) OR an error excluded from intent labelling
          (structural: exact-duplicate rows -- a data-export artefact,
          not a value a human chose)
    1  -> intentional (behavioural-anomaly cell on a transaction CONFIRMED
          fraudulent, Class == 1)
   -1  -> unintentional (behavioural-anomaly cell on a transaction NOT
          flagged as fraud, Class == 0)

Why this is a stronger ground truth than Chile Customs
---------------------------------------------------------------------
Chile Customs' NU_REGR label is "genuinely selected for physical
inspection" -- a real enforcement action, but a noisy population-level
signal: inspection-selection does not confirm that the specific flagged
cell is why the declaration was chosen (Intent_Paper, sec:chile-customs).
Class here is the opposite: a CONFIRMED fraud determination already made
by the card issuer/bank on this exact transaction (the canonical ULB/
Kaggle release; see Dal Pozzolo et al. 2015 and the dataset card). There
is no "suspicious-but-unconfirmed" tier to confuse with intent -- a row's
Class is either a confirmed-fraudulent transaction or it isn't.

We still restrict intent labelling to BEHAVIOURAL cells (statistical /
multivariate value anomalies: z-score, IQR, Isolation-Forest), excluding
STRUCTURAL cells (exact-duplicate rows), for the same reason as Chile
Customs and eBay: a structural artefact (the row appears twice in the
export) has no plausible link to what a fraudster or a legitimate
cardholder did -- it is a data-pipeline property, not a value choice.
Behavioural cells -- a transaction whose feature vector is statistically
extreme relative to the population -- are the only category with a
plausible link to fraud (fraud transactions are, definitionally, the
reason this kind of anomaly detector exists for this domain).

No oracle clean reference exists for the cell-level counterfactual (we
do not know what a fraudulent transaction's value "would have been" had
it been legitimate), so `clean` is a pseudo-clean approximation: the
per-column median computed over the CONFIRMED-LEGITIMATE (Class==0)
population, broadcast to every row (see build_pseudo_clean.py). This is
the same approximation status as eBay/Chile's pseudo-clean, but grounded
in a population that is itself oracle-correct (genuinely non-fraudulent
transactions), not a constructed behavioural-tier proxy.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW_PATH = BASE / "raw_cache" / "creditcard_raw.csv"
DETAIL_CSV = BASE / "results" / "creditcard_all_errors_detailed.csv"

OUT_LOCAL = BASE / "data" / "masks.csv"
OUT_BLIND_LOCAL = BASE / "data" / "masks_blind.csv"
OUT_NIYYAT = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/CreditCardFraud/masks.csv")
OUT_BLIND = Path("/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/CreditCardFraud/masks_blind.csv")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

INTENTIONAL = 1
UNINTENTIONAL = -1
CLEAN = 0


def main():
    print("=" * 70)
    print("Credit Card Fraud -- Dataset Mask Builder (direct Class label)")
    print("=" * 70)

    print("\n[1] Loading data...")
    raw = pd.read_csv(RAW_PATH)
    raw["Class"] = raw["Class"].astype(str).str.strip("'").astype(int)
    print(f"    {len(raw):,} rows x {len(FEATURE_COLS)} feature columns "
          f"({(raw['Class']==1).sum()} confirmed fraud, {(raw['Class']==0).sum()} legitimate)")

    print("\n[2] Restricting to behavioural-type flagged cells...")
    det = pd.read_csv(DETAIL_CSV, usecols=["row_idx", "column", "category"])
    det = det[(det["category"] == "behavioural") & det["column"].isin(FEATURE_COLS)]
    det = det.drop_duplicates(["row_idx", "column"])
    det = det.merge(raw[["Class"]], left_on="row_idx", right_index=True, how="left")
    det["cell_value"] = det["Class"].map({1: INTENTIONAL, 0: UNINTENTIONAL})
    print(f"    behavioural cells: INTENTIONAL={int((det['cell_value']==INTENTIONAL).sum()):,}  "
          f"UNINTENTIONAL={int((det['cell_value']==UNINTENTIONAL).sum()):,}")

    print("\n[3] Building full row x column mask matrix...")
    mask = pd.DataFrame(CLEAN, index=raw.index, columns=FEATURE_COLS, dtype="int8")
    col_loc = {c: i for i, c in enumerate(FEATURE_COLS)}
    for row_idx, col, val in det[["row_idx", "column", "cell_value"]].itertuples(index=False):
        mask.iat[row_idx, col_loc[col]] = val

    total_cells = mask.shape[0] * mask.shape[1]
    n_int = int((mask.values == INTENTIONAL).sum())
    n_unint = int((mask.values == UNINTENTIONAL).sum())
    n_clean = int((mask.values == CLEAN).sum())
    print(f"\n    Mask shape: {mask.shape[0]:,} x {mask.shape[1]} = {total_cells:,} cells")
    print(f"      CLEAN/excluded (0) : {n_clean:,}  ({100*n_clean/total_cells:.3f}%)")
    print(f"      UNINTENTIONAL (-1) : {n_unint:,}  ({100*n_unint/total_cells:.3f}%)")
    print(f"      INTENTIONAL (1)    : {n_int:,}  ({100*n_int/total_cells:.3f}%)")
    print(f"      intentional share of labelled cells: {100*n_int/(n_int+n_unint):.2f}%")

    print("\n[4] Saving outputs...")
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_LOCAL, index=False)
    print(f"    {OUT_LOCAL}")

    OUT_NIYYAT.parent.mkdir(parents=True, exist_ok=True)
    mask.to_csv(OUT_NIYYAT, index=False)
    print(f"    {OUT_NIYYAT}")

    mask_blind = mask.copy()
    mask_blind[mask_blind != CLEAN] = 1
    mask_blind.to_csv(OUT_BLIND_LOCAL, index=False)
    mask_blind.to_csv(OUT_BLIND, index=False)
    print(f"    {OUT_BLIND_LOCAL}  /  {OUT_BLIND}  (detection-only)")

    return mask


if __name__ == "__main__":
    main()
