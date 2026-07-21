#!/usr/bin/env python3
"""
Pseudo-clean reference for chile_customs, mirroring eBay's
wdc_product_analysis/intentionality/build_pseudo_clean.py exactly: per-column
median (numeric) / mode (categorical) within the same (HS4 tariff chapter,
origin country) group, falling back to HS4-only, then country-only, then
global, when a group is too small. Needed for ECOD-LOO (requires a "clean"
counterfactual to revert a cell to and rescore).
"""
import os
import pandas as pd

BASE = "/home/mohamed/error_injector/llms_baseline/chile_customs"
SAMPLE_PATH = os.path.join(BASE, "data", "chile_din_validation_sample_small.csv")

NUMERIC_COLS = ["FOB", "FLETE", "SEGURO", "CIF", "CIF_ITEM", "PRE_UNIT",
                "TOT_PESO", "CANT_MERC", "ADVAL_ALA", "TOT_BULTOS"]
EXCLUDE = ["data_month", "NUMENCRIPTADO", "NUMITEM", "NUM_UNICO_IMPORTADOR", "NU_REGR"]


def numc(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode(dropna=True).iloc[0] if len(s) else float("nan")


def build_pseudo_clean(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    keys = pd.DataFrame({
        "hs4": df["ARANC_NAC"].astype(str).str[:4],
        "origin": df["PA_ORIG"],
    })
    out = df.copy()
    for col in feature_cols:
        if col in ("ARANC_NAC", "PA_ORIG"):
            continue  # grouping keys have no meaningful counterfactual w.r.t. themselves
        is_num = col in NUMERIC_COLS
        agg = (lambda s: s.median()) if is_num else _mode_or_nan

        global_ref = agg(df[col])
        origin_ref = df.groupby(keys["origin"])[col].transform(agg)
        hs4_ref = df.groupby(keys["hs4"])[col].transform(agg)
        group_ref = df.groupby([keys["hs4"], keys["origin"]])[col].transform(agg)

        filled = group_ref.combine_first(hs4_ref).combine_first(origin_ref).fillna(global_ref)
        out[col] = filled.values
    return out[feature_cols]


if __name__ == "__main__":
    raw = pd.read_csv(SAMPLE_PATH, dtype=str, low_memory=False).reset_index(drop=True)
    dirty = raw.drop(columns=EXCLUDE).copy()
    for c in NUMERIC_COLS:
        dirty[c] = numc(dirty[c])
    feature_cols = list(dirty.columns)

    pseudo_clean = build_pseudo_clean(dirty, feature_cols)
    out_path = os.path.join(BASE, "data", "chile_din_pseudo_clean.csv")
    pseudo_clean.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={pseudo_clean.shape}")
