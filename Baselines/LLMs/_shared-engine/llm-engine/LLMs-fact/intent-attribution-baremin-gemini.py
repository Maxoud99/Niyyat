#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabFact Intent Attribution — Gemini, BARE-MINIMUM variant.
===========================================================

Plain prompt: role + strict JSON rules, no rubric, no few-shot examples.

Inputs:
  • dirty dataset CSV         (combined_dataset_TRF_no_claim.csv)
  • unified error mask CSV    (error_mask_updated.csv)
  • gemini explanations JSON  (explanations.json → provides correct values)

Output directory:
  <this-folder>/outputs/bareminimum-gemini/
"""

import os
import json
from typing import Dict, List

from _gemini_base import (
    ATTR_COLS,
    run_pipeline,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)

VARIANT_NAME = "bareminimum"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "bareminimum-gemini")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────
ROLE_BLOCK = """ROLE
You are a precise data-forensics assistant analysing factual claims about
people, teams, events, places, and statistics.

For each record you will be given:
  • CORRECT profile — the ground-truth values (from verified sources)
  • DIRTY   profile — the same record as it appears in our dataset
  • MASK    — which cells differ (1 = error cell, 0 = unchanged)

For EACH error cell (mask = 1) you must decide:
  1  → INTENTIONAL   (deliberate manipulation / misinformation /
                      meaning-altering swap such as wrong entity, wrong
                      location, wrong claim domain, or a value that
                      substantially changes the factual claim)
 -1  → UNINTENTIONAL (typo, capitalization, spacing, abbreviation,
                      ±1–2 numeric drift, or other trivial drift
                      that preserves the factual claim)

Evaluate EACH error cell INDEPENDENTLY.
"""


def make_chunk_prompt(
    clean_rows: List[Dict[str, str]],
    dirty_rows: List[Dict[str, str]],
    mask_rows:  List[Dict[str, str]],
    global_ids: List[int],
) -> str:
    sys_rules = """Rules:
- For EACH error cell (mask = 1) return exactly one JSON entry.
- Respond with ONLY one JSON object (no prose, no markdown fences).

Required JSON format:
{
  "feature_decisions": [
    {"row_id": <id>, "column": "<name>", "decision": 1 or -1,
     "reason": "<short>"}
  ]
}
"""
    records = []
    for i, gid in enumerate(global_ids):
        changed = [c for c in ATTR_COLS if mask_rows[i][c] == "1"]
        if not changed:
            continue
        records.append({
            "row_id":         gid,
            "correct":        clean_rows[i],
            "dirty":          dirty_rows[i],
            "error_columns":  changed,
        })
    bundle = {"columns": ATTR_COLS, "records": records}
    return (
        f"{ROLE_BLOCK}\n\n{sys_rules}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON now."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirty", default=DIRTY_CSV)
    p.add_argument("--mask",  default=MASK_CSV)
    p.add_argument("--expl",  default=EXPLANATIONS_JSON)
    p.add_argument("--orig",  default=ORIG_DIRTY_CSV)
    p.add_argument("--out",   default=OUT_DIR)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    run_pipeline(
        make_chunk_prompt=make_chunk_prompt,
        out_dir=args.out,
        variant_name=VARIANT_NAME,
        dirty_path=args.dirty,
        mask_path=args.mask,
        expl_path=args.expl,
        orig_path=args.orig,
        max_records=args.max_records,
    )
