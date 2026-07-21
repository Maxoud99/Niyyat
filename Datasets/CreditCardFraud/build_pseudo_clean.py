#!/usr/bin/env python3
"""
Pseudo-clean reference for credit_card_fraud, mirroring eBay/Chile Customs'
build_pseudo_clean.py: a per-column reference value to revert a flagged
cell to and rescore (needed for ECOD-LOO and the B+/fingerprint statistical
features).

Unlike eBay/Chile Customs, this dataset *does* have an oracle-correct
"clean" population: Class==0 (confirmed-legitimate transactions). There is
no categorical grouping key in the anonymised V1-V28 PCA space (no
business-meaningful (category, region) analogue), so we fall back directly
to the global per-column median computed on the legitimate population --
the coarsest tier eBay/Chile Customs use only as a last resort, but here
the source population itself is genuinely correct (not a behavioural-tier
proxy), so the approximation is in the row-personalisation, not in
which rows it's drawn from.
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE, "raw_cache", "creditcard_raw.csv")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def build_pseudo_clean(df: pd.DataFrame) -> pd.DataFrame:
    legit = df[df["Class"] == 0]
    medians = legit[FEATURE_COLS].median()
    out = pd.DataFrame([medians.values] * len(df), columns=FEATURE_COLS, index=df.index)
    return out


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    raw["Class"] = raw["Class"].astype(str).str.strip("'").astype(int)

    pseudo_clean = build_pseudo_clean(raw)

    out_local = os.path.join(BASE, "data", "pseudo_clean.csv")
    out_niyyat = "/home/mohamed/error_injector/llms_baseline/NIYYAT/Datasets/CreditCardFraud/pseudo_clean.csv"
    os.makedirs(os.path.dirname(out_local), exist_ok=True)
    os.makedirs(os.path.dirname(out_niyyat), exist_ok=True)
    pseudo_clean.to_csv(out_local, index=False)
    pseudo_clean.to_csv(out_niyyat, index=False)
    print(f"Wrote {out_local}  shape={pseudo_clean.shape}")
    print(f"Wrote {out_niyyat}  shape={pseudo_clean.shape}")
