#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabFact Intent Attribution — Gemini, FEW-SHOTS variant.
========================================================

Same pipeline as the ``info`` variant but the prompt also contains 3
in-context few-shot examples that demonstrate both INTENTIONAL (1) and
UNINTENTIONAL (-1) decisions on realistic TabFact-style error cells.

Output directory:
  <this-folder>/outputs/few-shots-gemini/
"""

import os
import json
from typing import Dict, List

from _gemini_base import (
    ATTR_COLS,
    run_pipeline,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)

VARIANT_NAME = "few-shots"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "few-shots-gemini")


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
  • subject_type changed to a semantically DIFFERENT category
  • subject_location swapped to a DIFFERENT place
  • value changed to a DIFFERENT quantity that alters the fact
    (large jump, comparator flip)
  • claim_domain changed to a DIFFERENT domain

UNINTENTIONAL (-1) — trivial drift that preserves the fact:
  • Typos / misspellings, case or spacing only, abbreviations,
    near-duplicate aliases, tiny numeric drift (±1–2).
"""

USE_FEW_SHOT = True

FEW_SHOT_BLOCK = """
EXAMPLE 1 — entity/location swap (intentional) vs tiny value drift (unintentional)

INPUT (abridged):
  row_id=0  dirty={subject_entity:"ian poulter", subject_location:"usa", value:"214"}
            correct={subject_entity:"ian poulter", subject_location:"england", value:"213"}
            error_columns=["subject_location","value"]

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 0, "column": "subject_location", "decision": 1,
     "reason": "location swapped to a different country"},
    {"row_id": 0, "column": "value", "decision": -1,
     "reason": "single-unit numeric drift (214→213)"}
  ]
}

EXAMPLE 2 — alias expansion (unintentional) vs domain swap (intentional)

INPUT (abridged):
  row_id=10 dirty={subject_entity:"brown", claim_domain:"Politics"}
            correct={subject_entity:"cleveland browns", claim_domain:"Sports"}
            error_columns=["subject_entity","claim_domain"]

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 10, "column": "subject_entity", "decision": -1,
     "reason": "alias expansion, same referent"},
    {"row_id": 10, "column": "claim_domain", "decision": 1,
     "reason": "domain flipped between Sports and Politics"}
  ]
}

EXAMPLE 3 — typo (unintentional), type mismatch (intentional), comparator flip (intentional)

INPUT (abridged):
  row_id=25 dirty={subject_location:"south melboure", subject_type:"Athlete", value:">132"}
            correct={subject_location:"south melbourne", subject_type:"Team", value:"<132"}
            error_columns=["subject_location","subject_type","value"]

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 25, "column": "subject_location", "decision": -1,
     "reason": "spelling typo (melboure→melbourne)"},
    {"row_id": 25, "column": "subject_type", "decision": 1,
     "reason": "Athlete vs Team is a semantic swap"},
    {"row_id": 25, "column": "value", "decision": 1,
     "reason": "comparator flipped (>132 vs <132)"}
  ]
}
""".strip()


def _build_diffs(clean: Dict[str, str], dirty: Dict[str, str],
                 mask_row: Dict[str, str]) -> List[str]:
    return [f"  - {c}: {dirty[c]!r} → {clean[c]!r}"
            for c in ATTR_COLS if mask_row[c] == "1"]


def make_chunk_prompt(
    clean_rows: List[Dict[str, str]],
    dirty_rows: List[Dict[str, str]],
    mask_rows:  List[Dict[str, str]],
    global_ids: List[int],
) -> str:
    sys_rules = """Rules:
- Use the DIFFS section (dirty → correct) together with the rubric
  and the in-context examples above.
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
    fewshot = (FEW_SHOT_BLOCK + "\n\n") if USE_FEW_SHOT else ""
    return (
        f"{ROLE_BLOCK}\n{RUBRIC}\n\n{fewshot}{sys_rules}\n\n"
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
        max_output_tokens=8192,
    )
