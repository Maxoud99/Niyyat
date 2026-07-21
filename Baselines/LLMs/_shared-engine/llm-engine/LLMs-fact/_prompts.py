#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared prompt builders for TabFact Intent Attribution.
=======================================================

Three variants, each of which returns the user-facing prompt content
(role + rubric/examples + rules + input JSON).

The *model wrapper* (Gemini call, Mixtral [INST], Llama-3 template,
Qwen ChatML, DeepSeek-R1 chat markers) is applied separately by
``_local_llm_base.wrap_for_model``.  Gemini passes the content through
unchanged.

Builders
--------
    make_prompt_baremin(clean_rows, dirty_rows, mask_rows, global_ids)
    make_prompt_info   (clean_rows, dirty_rows, mask_rows, global_ids)
    make_prompt_fewshot(clean_rows, dirty_rows, mask_rows, global_ids)
"""

from __future__ import annotations
import json
from typing import Dict, List

from _gemini_base import ATTR_COLS


# ─────────────────────────────────────────────────────────────────────────────
# BARE-MINIMUM
# ─────────────────────────────────────────────────────────────────────────────
_BAREMIN_ROLE = """ROLE
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

_BAREMIN_RULES = """Rules:
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


def _build_records(clean_rows, dirty_rows, mask_rows, global_ids, with_diffs: bool):
    records = []
    for i, gid in enumerate(global_ids):
        changed = [c for c in ATTR_COLS if mask_rows[i][c] == "1"]
        if not changed:
            continue
        entry = {
            "row_id":        gid,
            "correct":       clean_rows[i],
            "dirty":         dirty_rows[i],
            "error_columns": changed,
        }
        if with_diffs:
            entry["diffs"] = [
                f"  - {c}: {dirty_rows[i][c]!r} → {clean_rows[i][c]!r}"
                for c in changed
            ]
        records.append(entry)
    return records


def make_prompt_baremin(clean_rows, dirty_rows, mask_rows, global_ids) -> str:
    records = _build_records(clean_rows, dirty_rows, mask_rows, global_ids, with_diffs=False)
    bundle = {"columns": ATTR_COLS, "records": records}
    return (
        f"{_BAREMIN_ROLE}\n\n{_BAREMIN_RULES}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON now."
    )


# ─────────────────────────────────────────────────────────────────────────────
# INFO (role + rubric + diffs)
# ─────────────────────────────────────────────────────────────────────────────
_INFO_ROLE = """ROLE
You are a precise data-forensics assistant analysing factual claims about
people, teams, events, places, and statistics (TabFact-style records).

Per cell you must output:
  1  → INTENTIONAL   (deliberate factual manipulation)
 -1  → UNINTENTIONAL (typo, drift, trivial formatting)

Evaluate EACH error cell INDEPENDENTLY.
"""

_INFO_RUBRIC = """Rubric for TabFact-style factual-claim errors:

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

_INFO_RULES = """Rules:
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


def make_prompt_info(clean_rows, dirty_rows, mask_rows, global_ids) -> str:
    records = _build_records(clean_rows, dirty_rows, mask_rows, global_ids, with_diffs=True)
    bundle = {"columns": ATTR_COLS, "records": records}
    return (
        f"{_INFO_ROLE}\n{_INFO_RUBRIC}\n{_INFO_RULES}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON now."
    )


# ─────────────────────────────────────────────────────────────────────────────
# FEW-SHOTS (info + 3 in-context examples)
# ─────────────────────────────────────────────────────────────────────────────
_FEWSHOT_BLOCK = """
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

_FEWSHOT_RULES = """Rules:
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


def make_prompt_fewshot(clean_rows, dirty_rows, mask_rows, global_ids) -> str:
    records = _build_records(clean_rows, dirty_rows, mask_rows, global_ids, with_diffs=True)
    bundle = {"columns": ATTR_COLS, "records": records}
    return (
        f"{_INFO_ROLE}\n{_INFO_RUBRIC}\n\n{_FEWSHOT_BLOCK}\n\n{_FEWSHOT_RULES}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY the JSON now."
    )


# Convenience registry
PROMPT_BUILDERS = {
    "bareminimum": make_prompt_baremin,
    "info":        make_prompt_info,
    "few-shots":   make_prompt_fewshot,
}
