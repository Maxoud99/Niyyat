"""
Pseudo-Clean Reference Builder
================================
eBay has no oracle "clean" twin per cell (unlike Adult/TwitterBot/TabFact,
which are synthetic corruption pipelines that retain the true original
value). Several methods in the project structurally require a clean
counterfactual per cell:

  - FraudDetector's Leave-One-Out (revert cell to "clean", rescore)
  - Heuristic H7 (user-incentive / gain-direction: was the change beneficial?)
  - Reference-Augmented LLM prompts (clean row + dirty row + mask)

This script builds a *pseudo*-clean reference: for each column, the most
typical value observed for listings in the same (category, marketplace)
group -- falling back to category-only, then marketplace-only, then the
global value when a group is too small to be informative (164/357
category x marketplace groups here have fewer than 3 listings).

This is an approximation, not an oracle. It answers "what does a normal
listing like this one usually look like", not "what was this exact listing
before it was changed". Treat results derived from it accordingly -- see
NIYYAT/README.md for how each consumer is expected to use it.

Output: data/pseudo_clean.csv -- same shape/columns/index as
data/../data/ebay_all_listings.csv.
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)  # wdc_product_analysis/

NUMERIC_COLS = [
    "price", "seller_feedback_pct", "seller_feedback_score",
    "item_location_city", "n_images", "description_length",
    "return_period_days",
]
GROUP_COLS = ["category", "marketplace"]


def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return float("nan")
    return s.mode(dropna=True).iloc[0]


def build_pseudo_clean(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("item_id", "dataset_source")]
    keys = df[GROUP_COLS].copy()  # stable copy of the grouping keys, untouched

    out = df.copy()
    for col in feature_cols:
        if col in GROUP_COLS:
            continue  # a grouping key has no meaningful counterfactual w.r.t. itself

        is_num = col in NUMERIC_COLS
        agg = (lambda s: s.median()) if is_num else _mode_or_nan

        global_ref = agg(df[col])
        marketplace_ref = df.groupby(keys["marketplace"])[col].transform(agg)
        category_ref = df.groupby(keys["category"])[col].transform(agg)
        group_ref = df.groupby([keys["category"], keys["marketplace"]])[col].transform(agg)

        filled = group_ref.combine_first(category_ref) \
                           .combine_first(marketplace_ref) \
                           .fillna(global_ref)
        out[col] = filled.values

    return out[df.columns]


if __name__ == "__main__":
    dirty = pd.read_csv(os.path.join(DATA, "data", "ebay_all_listings.csv"))
    pseudo_clean = build_pseudo_clean(dirty)
    out_path = os.path.join(HERE, "data", "pseudo_clean.csv")
    pseudo_clean.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={pseudo_clean.shape}")

    # Sanity check: how often does pseudo-clean actually differ from dirty
    # on flagged cells (it should, most of the time, for the method to be
    # informative)?
    mask = pd.read_csv(os.path.join(HERE, "data", "masks.csv"))
    feature_cols = [c for c in dirty.columns if c not in ("item_id", "dataset_source")]
    flagged = (mask[feature_cols] != 0)
    n_flagged = int(flagged.values.sum())
    n_diff_on_flagged = int(((dirty[feature_cols].astype(str) !=
                              pseudo_clean[feature_cols].astype(str)) & flagged).values.sum())
    print(f"Flagged cells: {n_flagged}; pseudo-clean differs from dirty on "
          f"{n_diff_on_flagged} ({100*n_diff_on_flagged/max(n_flagged,1):.1f}%) of them")
