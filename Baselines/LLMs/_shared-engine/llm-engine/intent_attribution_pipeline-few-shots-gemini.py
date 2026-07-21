#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Google Gemini — Few-shot, Robust JSON
------------------------------------------------------------------------------------
Gemini API with few-shot examples:
- Info-enhanced with DIFFS and rubric
- Few-shot demos included in prompt (USE_FEW_SHOT=True)
- Accept multiple JSON schemas (compact map, decisions list, decisions array)
- Robust parsing with markdown fence handling
- Pure fallback: if no parse, set all three to -1 (no heuristics)
"""

from __future__ import annotations
import json
import sys
import os
import time
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# Import API key from config
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import GEMINI_API_KEY

# =========================
# ======== CONFIG =========
# =========================
# Using Gemini 2.5 Pro - Flash model's context window is too small for this prompt
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/manipulated_records.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/masks-blind.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-gemini-pro/intent_labels.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-gemini-pro/intent_explanations.csv"

# Adult schema
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","class"
]

# ======= Few-shot config (ON by default) =======
USE_FEW_SHOT = True

FEW_SHOT_BLOCK = """
EXAMPLE 1 — Utility-seeking vs. benign slips (abridged)
Context: Clean row vs. 3 manipulated rows. DIFFS show only changed cells.

local_id 0 DIFFS:
- education-num: 9 → 13
- education: HS-grad → Bachelors
- capital-gain: 0 → 12000
- hours-per-week: 40 → 52
Rationale: multiple strong utility signals → INTENTIONAL (1).

local_id 1 DIFFS:
- age: 36 → 35 (±1)
- workclass: "Private" → "private" (case-only)
Rationale: tiny numeric drift + case change → UNINTENTIONAL (-1).

local_id 2 DIFFS:
- capital-loss: 0 → 1 (tiny)
Rationale: trivial change → UNINTENTIONAL (-1).

Gold decisions:
{"decisions":[
  {"row_id":100,"local_id":0,"decision":1,"reason":"large utility-seeking upgrades"},
  {"row_id":101,"local_id":1,"decision":-1,"reason":"small numeric drift & case-only"},
  {"row_id":102,"local_id":2,"decision":-1,"reason":"tiny change; no advantage"}
]}

EXAMPLE 2 — Fairness/attribute masking & privacy (abridged)
local_id 0 DIFFS:
- sex: Female → Male
- race: Black → White
- hours-per-week: 35 → 46
Rationale: edits likely intended to escape bias / seek advantage + larger hours → INTENTIONAL (1).

local_id 1 DIFFS:
- native-country: "United-States" → "?"
- occupation: "Tech-support" → "Tech support" (spacing)
Rationale: disguised missing value (privacy) is intentional; spacing change is incidental. Net effect → INTENTIONAL (1).

local_id 2 DIFFS:
- age: 28 → 29 (+1)
Rationale: tiny numeric slips → UNINTENTIONAL (-1).

Gold decisions:
{"decisions":[
  {"row_id":200,"local_id":0,"decision":1,"reason":"protected-attribute edits + workload increase"},
  {"row_id":201,"local_id":1,"decision":1,"reason":"DMV/privacy masking dominates"},
  {"row_id":202,"local_id":2,"decision":-1,"reason":"minor numeric slips"}
]}
""".strip()

# =========================
# ====== UTILITIES ========
# =========================
def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}

def mask_row_to_dict(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: "1" if str(mask_df.at[idx, c]).strip() in ("1","1.0") else "0" for c in ADULT_COLS}

def compute_diffs(clean_row: Dict[str,str], manip_row: Dict[str,str], mask_row: Dict[str,str]) -> List[Dict[str,str]]:
    """Compute diffs for fields that were modified (mask=1)."""
    diffs = []
    for c in ADULT_COLS:
        if c == "class":
            continue
        if mask_row.get(c) == "1":
            diffs.append({"column": c, "from": clean_row.get(c,""), "to": manip_row.get(c,"")})
    return diffs

# =========================
# ===== GEMINI CALL =======
# =========================
def call_gemini(model, prompt: str):
    """Call Gemini API and return response text."""
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0,  # Deterministic
                'max_output_tokens': 2048,
                'candidate_count': 1,
            },
            safety_settings=safety_settings,
            request_options={'timeout': 180}
        )
        
        # Check if response has valid content
        if not response.candidates:
            return ""
        
        candidate = response.candidates[0]
        
        # finish_reason: 1 = STOP (normal), 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION, 5 = OTHER
        # Accept STOP (1) or MAX_TOKENS (2) as valid
        if candidate.finish_reason.value not in (1, 2):
            return ""
        
        # Check if content has parts (actual text)
        if not candidate.content or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            return ""
        
        # Extract text from parts
        try:
            text = candidate.content.parts[0].text
            return text if text else ""
        except (AttributeError, IndexError):
            return ""
            
    except Exception as e:
        # Silent failure, return empty string
        return None  # signals caller: fall back to next model tier


def call_gemini_with_fallback(model_lite, model_flash, prompt: str) -> str:
    """Cost policy (2026-06-26): flash-lite first, then flash, then give up.
    Never escalates to gemini-2.5-pro."""
    result = call_gemini(model_lite, prompt)
    if result is not None:
        return result
    result = call_gemini(model_flash, prompt)
    if result is not None:
        return result
    return ""

# =========================
# ======= PARSING =========
# =========================
def parse_llm_json(text: str) -> Optional[dict]:
    """Robustly extract first top-level JSON (object/array) from model text."""
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()

    # Remove common wrappers/fences/markers
    for junk in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(junk, "")
    s = s.strip()

    # If whole output is quoted JSON, unescape once
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = json.loads(s)
        except Exception:
            pass
        s = (s or "").strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Extract first balanced JSON fragment ({...} or [...])
    def extract_top_level_fragment(txt: str) -> Optional[str]:
        stack = []
        start = None
        for i, ch in enumerate(txt):
            if ch in "{[":
                if not stack:
                    start = i
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and ch == stack[-1]:
                    stack.pop()
                    if not stack and start is not None:
                        return txt[start:i+1]
        return None

    frag = extract_top_level_fragment(s)
    if not frag:
        return None
    try:
        return json.loads(frag)
    except Exception:
        # last resort (single quotes etc.)
        try:
            import ast
            obj = ast.literal_eval(frag)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            return None
    return None

def decisions_from_obj(obj: dict, global_ids: List[int]) -> Optional[List[Tuple[int, Optional[int], str]]]:
    """
    Accepts:
      A) {"decisions":[ {"local_id":..,"row_id":..,"decision":..,"reason":..}, ... ]}
      B) {"decisions":[ 1, -1, -1 ]}                           # by local_id order 0,1,2
      C) {"decisions": {"0": 1, "1": -1, "2": -1}}             # map by local_id
      D) {"0": 1, "1": -1, "2": -1}                           # top-level map (no "decisions" key)

    Returns [(rid, decision|None, reason), ...] for lids 0..2, or None if unusable.
    """
    if not isinstance(obj, dict):
        return None

    decs = obj.get("decisions", None)
    
    # If no "decisions" key, check if it's a top-level map (format D)
    if decs is None:
        # Try top-level map: {"0": 1, "1": -1, "2": -1}
        if all(str(k) in obj for k in (0, 1, 2)) or all(k in obj for k in ("0", "1", "2")):
            out3: List[Tuple[int, Optional[int], str]] = []
            have_any = False
            for lid in (0, 1, 2):
                v = obj.get(str(lid), obj.get(lid))
                if v is None:
                    out3.append((global_ids[lid], None, "top_level_map_missing"))
                    continue
                try:
                    dec = 1 if int(v) >= 1 else -1
                    have_any = True
                    out3.append((global_ids[lid], dec, "top_level_map"))
                except Exception:
                    out3.append((global_ids[lid], None, "top_level_map_bad_value"))
            return out3 if have_any else None
        return None

    # If model put the list or map INSIDE a string, decode it once.
    if isinstance(decs, str):
        decs_str = decs.strip()
        try:
            decoded = json.loads(decs_str)
            decs = decoded
        except Exception:
            pass

    # Case A: list of dicts
    if isinstance(decs, list) and (len(decs) == 0 or isinstance(decs[0], dict)):
        out: List[Tuple[int, Optional[int], str]] = []
        by_lid: Dict[int, dict] = {}
        for it in decs:
            try:
                lid = int(it.get("local_id"))
                by_lid[lid] = it
            except Exception:
                continue
        for lid in (0, 1, 2):
            rid = global_ids[lid]
            it = by_lid.get(lid)
            if it is None:
                out.append((rid, None, "json_missing_entry"))
                continue
            try:
                # Try both "decision" and "intentional" keys
                dec_val = it.get("decision", it.get("intentional", -1))
                dec = 1 if int(dec_val) >= 1 else -1
            except Exception:
                dec = None
            # Get reason with fallback
            reason = str(it.get("reason", "")).strip()
            if not reason:
                if "intentional" in it:
                    reason = f"intentional_key_value_{it.get('intentional')}"
                elif "decision" in it:
                    reason = f"decision_key_value_{it.get('decision')}"
                else:
                    reason = "reason_not_provided"
            out.append((rid, dec, reason))
        return out

    # Case B: list of 3 scalars
    if isinstance(decs, list) and len(decs) >= 3 and all(not isinstance(x, dict) for x in decs[:3]):
        out2: List[Tuple[int, Optional[int], str]] = []
        for lid in (0, 1, 2):
            v = decs[lid]
            try:
                dec = 1 if int(v) >= 1 else -1
                out2.append((global_ids[lid], dec, "array_schema"))
            except Exception:
                out2.append((global_ids[lid], None, "array_schema_bad_value"))
        return out2

    # Case C: dict map inside "decisions"
    if isinstance(decs, dict):
        out3: List[Tuple[int, Optional[int], str]] = []
        have_any = False
        for lid in (0, 1, 2):
            v = decs.get(str(lid), decs.get(lid))
            if v is None:
                out3.append((global_ids[lid], None, "map_schema_missing"))
                continue
            try:
                dec = 1 if int(v) >= 1 else -1
                have_any = True
                out3.append((global_ids[lid], dec, "map_schema"))
            except Exception:
                out3.append((global_ids[lid], None, "map_schema_bad_value"))
        return out3 if have_any else None

    return None

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "You are a precise data-forensics assistant. Decide for each manipulated RECORD "
    "whether changes are INTENTIONAL (1) or UNINTENTIONAL (-1). Deterministic behavior."
)

RUBRIC = (
    "Rubric:\n"
    "- INTENTIONAL (1): large utility-seeking edits (education-num +≥2 or categorical upgrade), "
    "big jump in capital-gain from 0, hours-per-week +≥10, coordinated multiple edits, fairness gaming, privacy DMV.\n"
    "- UNINTENTIONAL (-1): tiny numeric drift (±1–2), case-only/formatting, single trivial change.\n"
    "- If uncertain: pick -1 only when edits are small/benign; otherwise 1."
)

def make_group_prompt(clean_row: Dict[str, str],
                      group: List[Dict[str, str]],
                      group_masks: List[Dict[str, str]],
                      group_global_ids: List[int]) -> str:
    """
    Few-shot enhanced prompt with DIFFS and rubric (same as few-shots-qwen).
    """
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the clean row + three manipulated rows with their masks and DIFFS.\n"
        "- Return ONLY ONE JSON. Choose exactly ONE format:\n"
        '  A) {"0":1|-1,"1":1|-1,"2":1|-1}\n'
        '  B) {"decisions":[{"row_id":<global>,"local_id":0,"decision":1|-1,"reason":"<≤10w>"},'
        '{"row_id":<global>,"local_id":1,"decision":1|-1,"reason":"<≤10w>"},'
        '{"row_id":<global>,"local_id":2,"decision":1|-1,"reason":"<≤10w>"}]}\n'
        "- No extra text outside the JSON."
    )

    manipulated_bundle = []
    for k in range(len(group)):
        manipulated_bundle.append({
            "local_id": k,
            "row_id": group_global_ids[k],
            "row": group[k],
            "mask": group_masks[k],
            "diffs": compute_diffs(clean_row, group[k], group_masks[k]),
        })

    bundle = {
        "columns": ADULT_COLS,
        "clean": clean_row,
        "manipulated": manipulated_bundle
    }

    fewshot = (FEW_SHOT_BLOCK + "\n\n") if USE_FEW_SHOT else ""
    
    prompt = (
        f"{ROLE_BLOCK}\n\n{RUBRIC}\n\n{sys_rules}\n\n"
        f"{fewshot}"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return ONLY one JSON now."
    )
    return prompt

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str):
    t0 = time.perf_counter()

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME_LITE)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)

    # Ensure output directory exists
    out_dir = os.path.dirname(OUT_LABELS_CSV) if OUT_LABELS_CSV else "."
    os.makedirs(out_dir, exist_ok=True)
    stats_json_path = os.path.join(out_dir, "run_stats.json")
    per_group_csv_path = os.path.join(out_dir, "per_group_times.csv")

    # Load CSVs as strings
    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)
    masks  = pd.read_csv(masks_path,  dtype=str, keep_default_na=False)

    # Ensure columns & order
    for df, name in [(manip,"manipulated_records"), (correct,"correct_records"), (masks,"masks")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        df[:] = df[ADULT_COLS]

    # Sanity: manipulated must be 3x clean
    n_clean = len(correct)
    n_manip = len(manip)
    if n_manip != 3 * n_clean:
        raise ValueError(f"Expected manipulated rows == 3 * clean rows; got {n_manip} vs 3*{n_clean}={3*n_clean}")

    # Prepare outputs
    out_mask_df = masks.copy(deep=True)  # generated mask (1 -> decision; 0 stays 0)
    explanations_rows: List[Dict[str, str]] = []

    # Timing & status trackers
    group_times: List[float] = []
    llm_times:   List[float] = []
    sources_hist = {"llm":0, "fallback":0}
    llm_status_hist = {"ok": 0, "api_error": 0, "parse_error": 0,
                       "ok_decisions_list": 0, "ok_simple_map": 0, "ok_array": 0, "ok_top_level_map": 0}

    # Iterate groups: clean j with manipulated 3j..3j+2
    for j in tqdm(range(n_clean), desc="Groups (clean + 3 manipulated)", ncols=100):
        g_t0 = time.perf_counter()

        clean_row = row_to_dict(correct, j)
        manip_rows, mask_rows, global_ids = [], [], []
        for k in range(3):
            mi = 3*j + k
            manip_rows.append(row_to_dict(manip, mi))
            mask_rows.append(mask_row_to_dict(masks, mi))
            global_ids.append(mi)

        # Build prompt
        prompt_text = make_group_prompt(clean_row, manip_rows, mask_rows, global_ids)

        obj = None
        llm_text = ""
        l0 = time.perf_counter()
        try:
            llm_text = call_gemini_with_fallback(model, model_flash, prompt_text)
            obj = parse_llm_json(llm_text)
            llm_status_hist["ok"] += 1
        except Exception:
            llm_status_hist["api_error"] += 1
            obj = None
            llm_text = ""
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)

        # ----- accept multiple schemas -----
        parsed: Optional[List[Tuple[int, Optional[int], str]]] = None

        if isinstance(obj, dict):
            parsed = decisions_from_obj(obj, global_ids)
            if parsed is not None:
                # Count which branch hit, best-effort by "reason" tag
                tag = parsed[0][2] if parsed and len(parsed) >= 1 else ""
                if tag.startswith("top_level_map"):
                    llm_status_hist["ok_top_level_map"] += 1
                elif tag.startswith("map_schema"):
                    llm_status_hist["ok_simple_map"] += 1
                elif tag.startswith("array_schema"):
                    llm_status_hist["ok_array"] += 1
                else:
                    llm_status_hist["ok_decisions_list"] += 1

        # decisions per rid
        decisions: Dict[int, Tuple[int, str, str]] = {}
        if parsed is None:
            # Final fallback: mark all three rows -1
            for rid in global_ids:
                decisions[rid] = (-1, "fallback_pure_minus1", "fallback")
            sources_hist["fallback"] += 1
        else:
            any_from_llm = False
            by_rid = {rid: (dec, rsn) for (rid, dec, rsn) in parsed if dec is not None}
            for rid in global_ids:
                if rid in by_rid:
                    dec, rsn = by_rid[rid]
                    decisions[rid] = (dec, rsn, "llm")
                    any_from_llm = True
                else:
                    decisions[rid] = (-1, "missing_llm_entry->-1", "fallback")
            sources_hist["llm" if any_from_llm else "fallback"] += 1

        # Fill generated mask + explanations
        for k, rid in enumerate(global_ids):
            decision, reason, src = decisions[rid]
            for c in ADULT_COLS:
                if c == "class":
                    out_mask_df.at[rid, c] = "0"
                else:
                    out_mask_df.at[rid, c] = (
                        str(decision) if str(masks.at[rid, c]).strip() in ("1","1.0") else "0"
                    )
            # per-cell explanations (record-level reason)
            for c in ADULT_COLS:
                if str(masks.at[rid, c]).strip() in ("1","1.0"):
                    explanations_rows.append({
                        "row_id": rid,
                        "column": c,
                        "manipulated_value": manip_rows[k][c],
                        "original_value": clean_row[c],
                        "mask": 1,
                        "diagnosis": str(decision),
                        "rule": "record_intent_decision",
                        "details": json.dumps({"reason": reason, "source": src}, ensure_ascii=False),
                        "mapping_status": "matched",
                        "mapping_score": ""
                    })

        g_t1 = time.perf_counter()
        group_times.append(g_t1 - g_t0)

    # Save outputs
    out_mask_df.to_csv(OUT_LABELS_CSV, index=False)

    expl_df = pd.DataFrame(explanations_rows, columns=[
        "row_id","column","manipulated_value","original_value","mask",
        "diagnosis","rule","details","mapping_status","mapping_score"
    ])
    expl_df.to_csv(OUT_EXPL_CSV, index=False)

    # Summary
    flat_vals = out_mask_df.replace({"1.0":"1","-1.0":"-1"}).values.flatten().tolist()
    counts = pd.Series(flat_vals).value_counts().to_dict()
    print("\n=== Generated Mask Counts (all cells) ===")
    for k in ["1","-1","0"]:
        print(f"{k:>3}: {counts.get(k,0)}")
    print(f"ALL: {len(flat_vals)}")

    print("\n=== Instrumentation ===")
    print("LLM status histogram:", json.dumps(llm_status_hist, indent=2))
    print("Decision sources histogram:", json.dumps(sources_hist, indent=2))

    # Runtime stats
    t1 = time.perf_counter()
    stats_payload = {
        "model": MODEL_NAME,
        "clean_rows": n_clean,
        "manip_rows": n_manip,
        "groups": n_clean,
        "total_time_sec": t1 - t0,
        "avg_group_time_sec": float(pd.Series(group_times).mean()) if group_times else None,
        "median_group_time_sec": float(pd.Series(group_times).median()) if group_times else None,
        "total_llm_time_sec": float(pd.Series(llm_times).sum()) if llm_times else None,
        "avg_llm_time_sec": float(pd.Series(llm_times).mean()) if llm_times else None,
        "llm_status_hist": llm_status_hist,
        "decision_sources_hist": sources_hist,
        "outputs": {
            "generated_mask_csv": OUT_LABELS_CSV,
            "explanations_csv": OUT_EXPL_CSV,
            "per_group_times_csv": per_group_csv_path
        }
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    per_group_df = pd.DataFrame({
        "group_id": list(range(n_clean)),
        "group_time_sec": group_times,
        "llm_time_sec": llm_times
    })
    per_group_df.to_csv(per_group_csv_path, index=False)

    print(f"\nSaved runtime stats to:\n  - {stats_json_path}\n  - {per_group_csv_path}")
    print(f"Outputs:\n  - {OUT_LABELS_CSV}\n  - {OUT_EXPL_CSV}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else MANIPULATED_CSV
    c = sys.argv[2] if len(sys.argv) > 2 else CORRECT_CSV
    k = sys.argv[3] if len(sys.argv) > 3 else MASKS_CSV
    run_pipeline(m, c, k)
