#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabFact Intent Attribution — Gemini, INFO variant.
===================================================

Same pipeline as the bare-minimum script but the prompt now includes:
  • a RUBRIC with concrete INTENTIONAL vs UNINTENTIONAL indicators
    tailored to factual-claim datasets (entity, type, location, value,
    domain)
  • explicit per-cell DIFFS listed inline in the prompt

Output directory:
  <this-folder>/outputs/info-gemini/
"""

import os
import json
from typing import Dict, List

from _gemini_base import (
    ATTR_COLS,
    run_pipeline,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)

VARIANT_NAME = "info"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "info-gemini")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────
ROLE_BLOCK = """ROLE
You are a precise data-forensics assistant analysing factual claims about
people, teams, events, places, and statistics (TabFact-style records).

Per cell you must output:
  1  → INTENTIONAL   (deliberate factual manipulation)
 -1  → UNINTENTIONAL (typo, drift, trivial formatting)

Evaluate EACH error cell INDEPENDENTLY.
"""

RUBRIC = """Rubric for TabFact-style factual-claim errors:

INTENTIONAL (1) — meaning-altering manipulation:
  • subject_entity swapped to a DIFFERENT entity
    (e.g. "miami" → "washington",  "brown" → "cleveland browns"
    when the referent actually changes, not just a typo).
  • subject_type changed to a semantically DIFFERENT category
    (e.g. "Athlete" → "Team",  "Party" → "Public Figure").
  • subject_location swapped to a DIFFERENT place
    (e.g. "england" → "usa", "toronto" → "new york").
  • value changed to a DIFFERENT quantity that alters the fact
    (e.g. 92 → 214, a comparator flip ">X" ↔ "<X",
     or a large numeric jump).
  • claim_domain changed to a DIFFERENT domain
    (e.g. "Sports" → "Politics").

UNINTENTIONAL (-1) — trivial drift that preserves the fact:
  • Typos / misspellings         ("south melboure" → "south melbourne")
  • Case or spacing only         ("USA" → "usa", "Sports " → "Sports")
  • Abbreviation / expansion     ("the united state" → "usa")
  • Near-duplicate alias         ("Demographics" → "Demographic")
  • Tiny numeric drift (±1–2)    (214 → 213,  119 → 118)

When uncertain, prefer -1 only if the edit is clearly small/benign;
otherwise choose 1.
"""


def _build_diffs(clean: Dict[str, str], dirty: Dict[str, str],
                 mask_row: Dict[str, str]) -> List[str]:
    diffs = []
    for c in ATTR_COLS:
        if mask_row[c] == "1":
            diffs.append(f"  - {c}: {dirty[c]!r} → {clean[c]!r}")
    return diffs


def make_chunk_prompt(
    clean_rows: List[Dict[str, str]],
    dirty_rows: List[Dict[str, str]],
    mask_rows:  List[Dict[str, str]],
    global_ids: List[int],
) -> str:
    sys_rules = """Rules:
- Use the DIFFS section (dirty → correct) together with the rubric.
- For EACH error cell (mask = 1) return exactly one JSON entry.
- Respond with ONLY one JSON object (no prose, no markdown fences).

Required JSON format:
{
  "feature_decisions": [
    {"row_id": <id>, "column": "<name>", "decision": 1 or -1,
     "reason": "<≤15 words>"}
  ]
}
"""
    records = []
    for i, gid in enumerate(global_ids):
        changed = [c for c in ATTR_COLS if mask_rows[i][c] == "1"]
        if not changed:
            continue
        records.append({
            "row_id":        gid,
            "correct":       clean_rows[i],
            "dirty":         dirty_rows[i],
            "error_columns": changed,
            "diffs":         _build_diffs(clean_rows[i], dirty_rows[i], mask_rows[i]),
        })
    bundle = {"columns": ATTR_COLS, "records": records}
    return (
        f"{ROLE_BLOCK}\n{RUBRIC}\n{sys_rules}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON now."
    )


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
        max_output_tokens=6144,
    )
