"""
datasets.py
Canonical loader for all six intent-attribution datasets.

Returns a unified dict per dataset:
  {
    "name"      : str,
    "clean"     : pd.DataFrame,           # clean rows (feature cols only)
    "dirty"     : pd.DataFrame,           # dirty rows (same shape)
    "mask"      : pd.DataFrame,           # +1=intentional, -1=unintentional, 0=clean
    "feature_cols": list[str],
    "note"      : str,
  }

All paths are relative to BASE = this file's parent (fraud_baseline/).
"""

import os
import pandas as pd
import numpy as np

BASE   = os.path.dirname(os.path.abspath(__file__))
# This copy lives at llms_baseline/NIYYAT/Baselines/FraudDetector/ -- three
# levels below llms_baseline/, unlike the canonical fraud_baseline/datasets.py
# this file mirrors (one level below). Walk up three parents, not one.
ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))  # llms_baseline/


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _p(*parts):
    return os.path.join(ROOT, *parts)


def _combined_mask(mask_int: pd.DataFrame,
                   mask_unint: pd.DataFrame) -> pd.DataFrame:
    """Merge two binary (0/1) masks into +1 / -1 / 0."""
    m = pd.DataFrame(0, index=mask_int.index, columns=mask_int.columns)
    m[mask_int  == 1] =  1
    m[mask_unint == 1] = -1
    return m


# ─────────────────────────────────────────────────────────────────────────────
# (A) Adult-LLM
# ─────────────────────────────────────────────────────────────────────────────

def load_adult_llm() -> dict:
    clean_path = _p("mixed_error_pipeline", "output", "adult_clean.csv")
    # Most recent Gemini cell-level generation run (2026-06-17), superseding
    # the earlier run_v2_20260213_141240 used previously.
    dirty_path = _p("adult_income_dataset", "tenth-trial", "data", "raw",
                    "run_v2_20260617_173016", "manipulated_records.csv")
    mask_path  = _p("adult_income_dataset", "tenth-trial", "data", "raw",
                    "run_v2_20260617_173016", "masks.csv")

    dirty = pd.read_csv(dirty_path)
    mask  = pd.read_csv(mask_path)
    clean_full = pd.read_csv(clean_path)

    # Mask encodes: -1=unintentional, 0=clean, positive int=intentional subtype
    # Normalise to +1/−1/0
    mask_norm = mask.copy()
    mask_norm[mask > 0]  =  1
    mask_norm[mask < 0]  = -1

    # Feature cols: all except any id columns
    feature_cols = [c for c in dirty.columns]

    # Clean reference: sample 48,842 rows from the clean pool
    # (LLM was run on a subset; use the full clean as ECOD training distribution)
    clean = clean_full[feature_cols].reset_index(drop=True)

    return dict(
        name         = "Adult-LLM",
        clean        = clean,
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask_norm[feature_cols].reset_index(drop=True),
        feature_cols = feature_cols,
        note         = ("19,539 rows retained after LLM filtering. "
                        "Mask: positive=intentional subtype, -1=unintentional, 0=clean. "
                        "ECOD trained on full clean Adult distribution."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (B) Adult-Mixed
# ─────────────────────────────────────────────────────────────────────────────

def load_adult_mixed() -> dict:
    clean_path  = _p("mixed_error_pipeline", "output", "adult_clean.csv")
    dirty_path  = _p("mixed_error_pipeline", "output", "adult_phase2_final.csv")
    mask_i_path = _p("mixed_error_pipeline", "output", "mask_intentional.csv")
    mask_u_path = _p("mixed_error_pipeline", "output", "mask_unintentional.csv")

    clean   = pd.read_csv(clean_path)
    dirty   = pd.read_csv(dirty_path)
    mask_i  = pd.read_csv(mask_i_path)
    mask_u  = pd.read_csv(mask_u_path)

    feature_cols = list(clean.columns)
    mask = _combined_mask(mask_i[feature_cols], mask_u[feature_cols])

    return dict(
        name         = "Adult-Mixed",
        clean        = clean[feature_cols].reset_index(drop=True),
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask.reset_index(drop=True),
        feature_cols = feature_cols,
        note         = "tab_err unintentional + greedy adversarial intentional.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# (C) Adult-TFM
# ─────────────────────────────────────────────────────────────────────────────

def load_adult_tfm() -> dict:
    clean_path = _p("mixed_error_pipeline", "output", "adult_clean.csv")
    dirty_path = _p("tfm_error_injection", "output", "adult", "tabpfn", "dirty.csv")
    mask_path  = _p("tfm_error_injection", "output", "adult", "tabpfn", "mask.csv")

    clean = pd.read_csv(clean_path)
    dirty = pd.read_csv(dirty_path)
    mask  = pd.read_csv(mask_path)  # already -1/0/+1

    # Align columns (TFM mask may have slightly different order)
    feature_cols = [c for c in clean.columns if c in mask.columns]
    clean = clean[feature_cols].reset_index(drop=True)
    dirty = dirty[feature_cols].reset_index(drop=True)
    mask  = mask[feature_cols].reset_index(drop=True)

    return dict(
        name         = "Adult-TFM",
        clean        = clean,
        dirty        = dirty,
        mask         = mask,
        feature_cols = feature_cols,
        note         = "TabPFN v2.6 distribution-consistent injection. "
                       "Mask: +1=intentional, -1=unintentional, 0=clean.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# (D) TwitterBot-LLM
# ─────────────────────────────────────────────────────────────────────────────

def load_twitterbot_llm() -> dict:
    gemini_dir  = _p("klim-kireev", "datasets", "twitter-bot",
                     "gemini-run_v2_20260617_112647")
    clean_path  = _p("mixed_error_pipeline_twitter", "output", "twibot20_clean.csv")
    dirty_path  = os.path.join(gemini_dir, "manipulated_records.csv")
    mask_path   = os.path.join(gemini_dir, "masks.csv")

    clean_full  = pd.read_csv(clean_path)
    dirty       = pd.read_csv(dirty_path)
    mask_raw    = pd.read_csv(mask_path)  # 0=clean, 1=intentional

    feature_cols = [c for c in dirty.columns
                    if c in clean_full.columns
                    and c not in ("user_id", "label")]

    # V2 mask: 1=intentional, -1=unintentional, 0=clean
    mask = mask_raw[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    clean = clean_full[feature_cols].reset_index(drop=True)

    return dict(
        name         = "TwitterBot-LLM",
        clean        = clean,
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask.reset_index(drop=True),
        feature_cols = feature_cols,
        note         = ("100 bot accounts manipulated by Gemini to evade detection. "
                        "Single-class: all flagged cells are intentional. "
                        "F1-unint undefined; report precision/recall for INT only."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (E) TwitterBot-Mixed
# ─────────────────────────────────────────────────────────────────────────────

def load_twitterbot_mixed() -> dict:
    base = _p("mixed_error_pipeline_twitter", "output")
    clean   = pd.read_csv(os.path.join(base, "twibot20_clean.csv"))
    dirty   = pd.read_csv(os.path.join(base, "twibot20_phase2_final.csv"))
    mask_i  = pd.read_csv(os.path.join(base, "mask_intentional.csv"))
    mask_u  = pd.read_csv(os.path.join(base, "mask_unintentional.csv"))

    feature_cols = [c for c in clean.columns
                    if c in mask_i.columns and c not in ("user_id", "label")]

    mask = _combined_mask(mask_i[feature_cols], mask_u[feature_cols])

    return dict(
        name         = "TwitterBot-Mixed",
        clean        = clean[feature_cols].reset_index(drop=True),
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask.reset_index(drop=True),
        feature_cols = feature_cols,
        note         = "tab_err outlier injection + greedy adversarial attack.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# (F) TwitterBot-TFM
# ─────────────────────────────────────────────────────────────────────────────

def load_twitterbot_tfm() -> dict:
    clean_path = _p("mixed_error_pipeline_twitter", "output", "twibot20_clean.csv")
    dirty_path = _p("tfm_error_injection", "output", "twitterbot", "tabpfn", "dirty.csv")
    mask_path  = _p("tfm_error_injection", "output", "twitterbot", "tabpfn", "mask.csv")

    clean = pd.read_csv(clean_path)
    dirty = pd.read_csv(dirty_path)
    mask  = pd.read_csv(mask_path)

    feature_cols = [c for c in clean.columns
                    if c in mask.columns and c not in ("user_id", "label")]

    return dict(
        name         = "TwitterBot-TFM",
        clean        = clean[feature_cols].reset_index(drop=True),
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask[feature_cols].reset_index(drop=True),
        feature_cols = feature_cols,
        note         = "TabPFN distribution-consistent injection on TwiBot-20.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# (G) TabFact
# ─────────────────────────────────────────────────────────────────────────────

def load_tabfact() -> dict:
    """
    TabFact has no clean/dirty split and no cell-level mask.
    is_factual=0 → refuted row (all errors are intentional by construction).
    is_factual=1 → entailed (clean).

    We treat entailed rows as 'clean' training distribution and
    refuted rows as 'dirty'. ECOD score is used as a row-level
    intentionality proxy only.
    """
    path = _p("tabfact", "datasets", "final", "combined_dataset_TRF.csv")
    df   = pd.read_csv(path)

    feature_cols = [c for c in df.columns if c not in ("claim_text", "is_factual")]

    # Encode categorical columns as category codes
    df_enc = df[feature_cols].copy()
    for col in df_enc.select_dtypes(include="object").columns:
        df_enc[col] = pd.Categorical(df_enc[col]).codes

    clean_rows   = df_enc[df["is_factual"] == 1].reset_index(drop=True)
    dirty_rows   = df_enc[df["is_factual"] == 0].reset_index(drop=True)

    # Row-level mask: all refuted cells are +1 (intentional) — no cell distinction
    mask = pd.DataFrame(1, index=dirty_rows.index, columns=feature_cols)

    return dict(
        name         = "TabFact",
        clean        = clean_rows,
        dirty        = dirty_rows,
        mask         = mask,
        feature_cols = feature_cols,
        note         = ("No cell-level mask. All errors in refuted rows are "
                        "intentional by construction. Row-level anomaly score only. "
                        "Report precision for INT at threshold sweep."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (H) eBay
# ─────────────────────────────────────────────────────────────────────────────

def load_ebay() -> dict:
    """
    eBay has no oracle clean twin (it's real scraped data, not a synthetic
    corruption pipeline). "clean" here is a *pseudo*-clean reference: the
    most typical value per column within the same (category, marketplace)
    group, falling back to category- or marketplace-only, then global, when
    a group is too small (see intentionality/build_pseudo_clean.py).

    Leave-One-Out attribution against this pseudo-clean answers "does this
    cell look anomalous relative to what's typical for this kind of
    listing", not "was this cell's true original value restored" -- treat
    results as an approximation, not an oracle bound (cf. adult_llm etc.).

    mask.csv: 0=clean, 1=intentional, -1=unintentional
    (see wdc_product_analysis/intentionality/build_dataset_mask.py).
    """
    dirty_path  = _p("wdc_product_analysis", "data", "ebay_all_listings.csv")
    clean_path  = _p("wdc_product_analysis", "intentionality", "data", "pseudo_clean.csv")
    mask_path   = _p("wdc_product_analysis", "intentionality", "data", "masks.csv")

    dirty    = pd.read_csv(dirty_path)
    clean    = pd.read_csv(clean_path)
    mask_raw = pd.read_csv(mask_path)

    feature_cols = [c for c in dirty.columns if c not in ("item_id", "dataset_source")]

    # returns_accepted / has_detail load as real bool dtype (unlike the
    # 0/1-int booleans elsewhere in the project) -- normalise to int so
    # downstream numeric ops (np.percentile etc.) don't choke on bool dtype.
    for df in (dirty, clean):
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

    mask = mask_raw[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    return dict(
        name         = "eBay",
        clean        = clean[feature_cols].reset_index(drop=True),
        dirty        = dirty[feature_cols].reset_index(drop=True),
        mask         = mask.reset_index(drop=True),
        feature_cols = feature_cols,
        note         = ("10,336 real eBay listings (4 marketplaces). 'clean' is a "
                        "pseudo-clean per-(category,marketplace) reference, not an "
                        "oracle -- see docstring. Mask from the Type A/B + tiered "
                        "intentionality framework, not synthetic injection."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# (I) Chile Customs
# ─────────────────────────────────────────────────────────────────────────────

def _load_chile_customs(suffix: str = "", note: str = "") -> dict:
    """
    Shared implementation for load_chile_customs (v1, original 9-detector /
    42,456-row construction, cited in Intent_Paper/chapters/4_Datasets.tex --
    files left untouched) and load_chile_customs_v2 (extended 12-detector /
    NU_REGR-not-null-filtered / ~10k-row construction, files suffixed "2").

    Chile Customs has no oracle clean twin (real import declarations, not a
    synthetic corruption pipeline). "clean" here is a pseudo-clean reference:
    per-(HS4-chapter, origin-country) median/mode, falling back to HS4-only,
    then origin-only, then global (see build_pseudo_clean.py) -- same
    approximation status as eBay's pseudo_clean.

    Unlike every other dataset, the intent mask here is built directly from
    a REAL external enforcement signal (NU_REGR: genuine selection for
    customs physical inspection) rather than a synthetic injection pipeline
    or a constructed proxy. NU_REGR is excluded from every feature/clean/
    dirty column -- it is the label, never an input. Restricted to
    behavioural-type cells only (price/outlier/rare-value anomalies); see
    build_dataset_mask.py for why structural cells are excluded rather than
    labelled.

    mask.csv: 0=clean or excluded, 1=intentional (inspected), -1=unintentional
    (not inspected) -- see build_dataset_mask.py.
    """
    dirty_path = _p("chile_customs", "data", f"chile_din_validation_sample_small{suffix}.csv")
    clean_path = _p("chile_customs", "data", f"chile_din_pseudo_clean{suffix}.csv")
    mask_path  = _p("chile_customs", "data", f"masks{suffix}.csv")

    dirty_raw = pd.read_csv(dirty_path, dtype=str, low_memory=False)
    clean     = pd.read_csv(clean_path)
    mask_raw  = pd.read_csv(mask_path)

    feature_cols = list(mask_raw.columns)

    numeric_cols = ["FOB", "FLETE", "SEGURO", "CIF", "CIF_ITEM", "PRE_UNIT",
                    "TOT_PESO", "CANT_MERC", "ADVAL_ALA", "TOT_BULTOS",
                    "VALEXFAB",
                    "CANT_BUL1", "CANT_BUL2", "CANT_BUL3", "CANT_BUL4",
                    "CANT_BUL5", "CANT_BUL6", "CANT_BUL7", "CANT_BUL8"]
    numeric_cols = [c for c in numeric_cols if c in feature_cols]
    dirty = dirty_raw[feature_cols].copy()
    for c in numeric_cols:
        dirty[c] = pd.to_numeric(dirty[c].str.replace(",", ".", regex=False), errors="coerce")

    mask = mask_raw[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Precomputed per-HS4-chapter median unit price, exposed as an extra
    # row_dict column (not a feature_col / not in mask) so User-Guided's C1/C2
    # rules can reference a tariff-chapter peer price without requiring the
    # single-row constraint sandbox to compute a cross-row aggregate itself.
    hs4 = dirty["ARANC_NAC"].astype(str).str[:4]
    dirty["_HS4_MEDIAN_PRICE"] = hs4.map(dirty.groupby(hs4)["PRE_UNIT"].median())

    return dict(
        name         = "Chile Customs" + (" v2" if suffix else ""),
        clean        = clean[feature_cols].reset_index(drop=True),
        dirty        = dirty.reset_index(drop=True),
        mask         = mask.reset_index(drop=True),
        feature_cols = feature_cols,
        note         = note,
    )


def load_chile_customs() -> dict:
    return _load_chile_customs(
        suffix="",
        note=("42,456 item-rows from a stratified sample of real Chilean import "
              "declarations (all 381 genuinely-inspected declarations + 9,000 "
              "random non-inspected, out of 460,571 total). 'clean' is a "
              "pseudo-clean per-(HS4-chapter,origin-country) reference, not an "
              "oracle. Mask built directly from real NU_REGR inspection outcome "
              "(behavioural cells only), 9-detector ensemble -- see docstring."),
    )


def load_chile_customs_v2() -> dict:
    """
    Extended-detector variant (v2) for direct before/after comparison against
    load_chile_customs (v1, untouched, matches the published paper draft).
    Differences from v1:
      - 3 additional cross-field detectors (ARANC_ALA-vs-ARANC_NAC HS6-level
        tariff mismatch, VALEXFAB-vs-FOB ex-factory valuation, CANT_BUL1-8-
        vs-TOT_BULTOS package-count consistency) -- see
        chile_din_error_detector.py Detectors 12, 14, 16.
      - Sample population: drawn from declarations where NU_REGR is NOT
        truly null (the "0" sentinel + genuinely-inspected), rather than
        v1's all-inspected + 9,000-random-non-inspected construction.
      - Final size target ~10,000 item rows instead of v1's 42,456.
    See chile_customs/build_validation_sample.py for the construction.
    """
    return _load_chile_customs(
        suffix="2",
        note=("v2 (extended-detector) variant: ~10,000 item-rows sampled from "
              "declarations where NU_REGR is not truly null (sentinel '0' + "
              "genuinely-inspected), all genuinely-inspected declarations kept "
              "in full. 12-detector ensemble (9 original + tariff/VALEXFAB/"
              "package-count cross-field checks). Mask built directly from "
              "real NU_REGR inspection outcome (behavioural cells only) -- "
              "see build_validation_sample.py / load_chile_customs docstring."),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

LOADERS = {
    "adult_llm"        : load_adult_llm,
    "adult_mixed"      : load_adult_mixed,
    "adult_tfm"        : load_adult_tfm,
    "twitterbot_llm"   : load_twitterbot_llm,
    "twitterbot_mixed" : load_twitterbot_mixed,
    "twitterbot_tfm"   : load_twitterbot_tfm,
    "tabfact"          : load_tabfact,
    "ebay"             : load_ebay,
    "chile_customs"    : load_chile_customs,
    "chile_customs_v2" : load_chile_customs_v2,
}


def load_all() -> dict:
    return {name: loader() for name, loader in LOADERS.items()}


if __name__ == "__main__":
    for name, loader in LOADERS.items():
        try:
            ds = loader()
            n_err = (ds["mask"] != 0).sum().sum()
            n_int = (ds["mask"] == 1).sum().sum()
            n_uni = (ds["mask"] == -1).sum().sum()
            print(f"[OK] {ds['name']:25s}  "
                  f"dirty={len(ds['dirty']):6d} rows  "
                  f"errors={n_err:6d}  int={n_int:5d}  unint={n_uni:5d}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
