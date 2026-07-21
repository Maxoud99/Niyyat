#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Declarative Intent-Signal Extractor
=====================================

Reads a user-written natural language description of ANY dataset and uses
an LLM to extract *intent-indicative rules*: declarative, domain-grounded
patterns that point toward INTENTIONAL or UNINTENTIONAL cell modification.

This is deliberately NOT a denial-constraint / data-validity extractor.
Generic "is this value valid?" rules are an error-DETECTION signal — a
violation just means *something* is wrong, with no information about *why*
it changed. Intent attribution needs the opposite: each rule here is a
domain claim of the form "if a flagged cell shows pattern P, that is
evidence the change was {deliberate | accidental}, because P matches
{a gain-targeted / fairness-masking / obfuscation motive | no plausible
motive at all}." A rule may incidentally also describe a validity
violation (e.g. a cross-column mismatch), but it is extracted and scored
for its *motive content*, not its validity content.

The extraction is dataset-agnostic: the same prompt and code work for any
text (README, paper description, user paragraph) as long as the user also
states what an intentional change would look like for that domain (which
columns a rational actor would target and in which direction). No column
names or dataset-specific logic is hard-coded here.

Usage:
    from extractor import extract_constraints
    rules = extract_constraints("configs/adult_income.txt", output_json="adult_income_constraints.json")
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional

import google.generativeai as genai

try:
    import sys as _sys, os as _os
    _cfg = _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "config.py")
    _ns: dict = {}
    with open(_os.path.abspath(_cfg)) as _f:
        exec(_f.read(), _ns)
    GEMINI_API_KEY: str = _ns.get("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = ""

# Cost policy (2026-06-26): gemini-2.5-pro is never called from this codebase
# again after its usage ran up a ~550 EUR bill. Always try flash-lite first,
# fall back to flash on failure, then give up.
MODEL_NAME_LITE = "gemini-2.5-flash-lite"
MODEL_NAME_FLASH = "gemini-2.5-flash"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt — dataset-agnostic, works on any NL text
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are a fraud / data-manipulation analyst. You are NOT doing error
detection — every cell you will ever apply these rules to has ALREADY been
flagged as erroneous by a separate detector. Your only question is WHY it
changed: was the change made DELIBERATELY by a rational actor pursuing a
concrete goal ("intentional"), or is it an accidental artefact like a typo,
OCR error, or random noise ("unintentional")?

Read the natural-language dataset description below and extract a list of
INTENT-INDICATIVE RULES. Each rule is a domain claim of the form: "if a
flagged cell matches pattern P, that by itself is evidence the change was
{{intentional | unintentional}}." Ground every rule in one of these four
motive classes (state which one in the description):

  1. gain-targeted    — the new value moves a column in the direction that
                         plausibly helps a rational actor reach a favourable
                         outcome (e.g. a higher predicted income, evading a
                         classifier, looking more like a legitimate human
                         account). Look for explicit incentive language in
                         the description (target variable, what a "good"
                         outcome looks like, which columns drive it).
  2. fairness-masking  — the new value replaces a sensitive/protected
                         attribute (or a field correlated with it) with the
                         historically majority/privileged category, to
                         dodge discriminatory treatment.
  3. obfuscation       — the new value is a placeholder or null-surrogate
                         (e.g. "Unknown", "N/A", "?", empty string) chosen
                         to hide the true value, REGARDLESS of whether the
                         schema happens to also call that placeholder
                         "valid". Presence of a deliberate-looking
                         placeholder on a flagged cell is itself intent
                         evidence, independent of validity.
  4. no-plausible-motive (NOISE) — the new value is invalid/nonsensical in
                         a way that serves none of the above motives (e.g.
                         a negative age, an out-of-range count, a garbled
                         string that is not a recognizable placeholder).
                         A rational actor avoids implausible values because
                         they invite scrutiny; values that fail this hard
                         are therefore evidence AGAINST deliberate intent.

For each rule produce:
  - id            : unique identifier (C1, C2, ...)
  - description   : one sentence stating the pattern AND which motive class
                     it belongs to, e.g. "relationship=Husband with a
                     non-Male sex value suggests the sex field was
                     deliberately changed to mask gender (fairness-masking)."
  - columns       : list of column name(s) this rule attributes evidence to
                    when they are the flagged cell (use EXACT column names
                    as they appear in the text)
  - expression    : a Python boolean expression that evaluates to True
                    exactly when the PATTERN IS PRESENT in the row (not
                    "constraint satisfied" — the opposite convention from a
                    validity checker). Reference column values as
                    row['column_name']; every value arrives as a string.
                    Use str(), int(), float() for conversion, wrapped safely:
                      good:  (lambda v: int(float(v)))(row['age'])
                      avoid: int(row['age'])   # crashes on "17.0"
                    The expression may ONLY use the current (dirty) row —
                    no pre-modification / clean value is ever available.
  - intent_signal : +1 if the pattern's presence is evidence of INTENTIONAL
                    change (motive classes 1-3), or -1 if it is evidence of
                    UNINTENTIONAL change (motive class 4, no plausible
                    motive). Never 0.
  - motive        : one of "gain_targeted", "fairness_masking",
                     "obfuscation", "noise".

Return ONLY a JSON object in this exact format, no prose, no markdown:
{{
  "constraints": [
    {{
      "id": "C1",
      "description": "relationship=Husband with non-Male sex suggests the sex field was deliberately altered to mask gender",
      "columns": ["relationship", "sex"],
      "expression": "str(row['relationship']) == 'Husband' and str(row['sex']) != 'Male'",
      "intent_signal": 1,
      "motive": "fairness_masking"
    }},
    {{
      "id": "C2",
      "description": "An age outside [17, 90] is physically nonsensical and serves no manipulation goal, indicating accidental corruption",
      "columns": ["age"],
      "expression": "not (17 <= (lambda v: int(float(v)))(row['age']) <= 90)",
      "intent_signal": -1,
      "motive": "noise"
    }}
  ]
}}

Rules:
  - Extract rules for EVERY motive class that the description gives you
    enough information to ground (gain direction, sensitive attributes,
    placeholders, and impossible/out-of-range values). Do not force a
    motive class the text does not support — it is fine for a dataset to
    have zero rules in a class.
  - Do NOT invent a gain direction, a privileged category, or a placeholder
    convention that is not stated or clearly implied by the text.
  - Use EXACT column names as written in the text.
  - For cross-column rules list ALL involved columns.
  - Expressions must be a SINGLE Python EXPRESSION evaluable with eval() —
    NOT a statement. Do NOT use try/except, if/else statements, assignments,
    or multiple lines; eval() can only run one expression, not a try block.
    A bare conditional value can be written as a ternary expression
    (`a if cond else b`) if you truly need branching, but for a boolean
    pattern rule you almost never do.
  - Do NOT guard conversions with try/except. The harness already wraps
    every expression's evaluation in its own try/except and treats any
    runtime error (e.g. int() on a non-numeric string) as "rule not
    applicable" automatically. Just write the direct boolean expression,
    e.g. `(lambda v: int(float(v)))(row['age']) < 0`, and trust that a
    non-numeric value will safely fall through as not-applicable rather
    than crash anything.
  - Prefer many narrow, single-motive rules over one broad rule that mixes
    motives — the per-rule sign must be unambiguous.

Dataset description:
---
{text}
---

Return ONLY the JSON object.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Gemini call
# ─────────────────────────────────────────────────────────────────────────────

def _call_gemini_tier(prompt: str, model_name: str) -> Optional[str]:
    """Returns response text on success (possibly "" if content-filtered --
    not retried further), or None if the call raised (signals: try the next
    tier)."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name)
    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 32768,
                "candidate_count": 1,
            },
            safety_settings=safety,
            request_options={"timeout": 300},
        )
        if not resp.candidates:
            return ""
        cand = resp.candidates[0]
        if cand.finish_reason.value not in (1, 2):
            return ""
        if not cand.content or not getattr(cand.content, "parts", None):
            return ""
        return cand.content.parts[0].text or ""
    except Exception as e:
        print(f"[extractor] Gemini call failed ({model_name}): {e}")
        return None


def _call_gemini(prompt: str) -> str:
    # Cost policy: flash-lite first, then flash, then give up. Never pro.
    result = _call_gemini_tier(prompt, MODEL_NAME_LITE)
    if result is not None:
        return result
    result = _call_gemini_tier(prompt, MODEL_NAME_FLASH)
    if result is not None:
        return result
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing — robust to markdown fences
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.strip()
    for fence in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(fence, "")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Try to extract the outermost JSON object
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation: smoke-test each expression with a dummy row
# ─────────────────────────────────────────────────────────────────────────────

def _validate_expression(expr: str, columns: List[str]) -> bool:
    """
    Try to evaluate the expression with a dummy row (all empty strings).
    Returns True if it runs without a syntax error (runtime errors are ok —
    they will be caught per-row during evaluation).
    """
    dummy_row = {col: "" for col in columns}
    try:
        eval(expr, {"__builtins__": {}}, {
            "row": dummy_row,
            "int": int, "float": float, "str": str,
            "len": len, "abs": abs, "round": round,
            "min": min, "max": max, "lambda": None,
        })
    except SyntaxError:
        return False
    except Exception:
        pass  # runtime errors (type error on empty string) are expected
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_constraints(
    description_path: str,
    output_json: Optional[str] = None,
    force_reextract: bool = False,
) -> List[Dict]:
    """
    Extract integrity constraints from a user-written NL description file.

    Parameters
    ----------
    description_path : str
        Path to the plain-text file containing the dataset description.
    output_json : str, optional
        If provided, save the extracted constraints here (for reproducibility).
        On subsequent calls, load from this file instead of calling the LLM again
        (unless force_reextract=True).
    force_reextract : bool
        If True, always call the LLM even if output_json already exists.

    Returns
    -------
    List[Dict]
        Each dict has keys: id, description, columns, expression,
        intent_signal (+1/-1), motive.
    """
    # Load cached result if available
    if output_json and Path(output_json).exists() and not force_reextract:
        print(f"[extractor] Loading cached constraints from {output_json}")
        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("constraints", [])

    # Read user description
    desc_path = Path(description_path)
    if not desc_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")
    text = desc_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Description file is empty: {description_path}")

    print(f"[extractor] Extracting constraints from: {description_path}")
    print(f"[extractor] Description length: {len(text)} chars")

    prompt = _PROMPT_TEMPLATE.format(text=text)

    t0 = time.perf_counter()
    raw = _call_gemini(prompt)
    elapsed = time.perf_counter() - t0
    print(f"[extractor] LLM call: {elapsed:.1f}s")

    if not raw:
        raise RuntimeError("LLM returned empty response. Check API key and model availability.")

    parsed = _parse_json(raw)
    if parsed is None or "constraints" not in parsed:
        print(f"[extractor] Raw LLM output:\n{raw[:500]}")
        raise RuntimeError("Failed to parse constraint JSON from LLM response.")

    constraints = parsed["constraints"]
    print(f"[extractor] Extracted {len(constraints)} constraints")

    # Validate and report
    valid, invalid = [], []
    for c in constraints:
        cid    = c.get("id", "?")
        cols   = c.get("columns", [])
        expr   = c.get("expression", "")
        signal = c.get("intent_signal")
        ok_expr   = _validate_expression(expr, cols)
        ok_signal = signal in (1, -1, 1.0, -1.0)
        if ok_expr and ok_signal:
            c["intent_signal"] = int(signal)
            c.setdefault("motive", "unspecified")
            valid.append(c)
            sign = "+1 intentional" if c["intent_signal"] == 1 else "-1 unintentional"
            print(f"  [OK]   {cid} ({sign}, {c['motive']}): {c.get('description','')[:60]}")
        elif not ok_expr:
            invalid.append(c)
            print(f"  [SKIP] {cid} — syntax error in expression: {expr[:60]}")
        else:
            invalid.append(c)
            print(f"  [SKIP] {cid} — missing/invalid intent_signal ({signal!r}), must be +1 or -1")

    if invalid:
        print(f"[extractor] Warning: {len(invalid)} constraints dropped (syntax error or invalid intent_signal).")

    if not valid:
        raise RuntimeError("No valid constraints could be extracted.")

    # Re-index IDs to be consecutive
    for i, c in enumerate(valid):
        c["id"] = f"C{i+1}"

    result = {"constraints": valid, "source": str(desc_path), "model": MODEL_NAME}

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[extractor] Saved to {output_json}")

    return valid


def load_constraints(json_path: str) -> List[Dict]:
    """Load a previously extracted constraint spec from JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("constraints", [])
