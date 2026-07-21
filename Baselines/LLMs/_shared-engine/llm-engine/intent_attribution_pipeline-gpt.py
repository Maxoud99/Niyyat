#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Local LLM (GPT-2)
------------------------------------------------------
- Assumes: for clean row j, manipulated rows are 3*j, 3*j+1, 3*j+2.
- LLM decides per manipulated RECORD: 1 (intentional) or -1 (unintentional).
- Output "generated mask": same shape as masks.csv; every 1 becomes the record decision; 0 stays 0; 'class' is always 0.
- tqdm progress bar + runtime stats saved next to outputs.
- Hardened: strict JSON via GBNF grammar with auto-fallback if unsupported; return_full_text disabled; explicit success/fallback counters.

Requirements: pandas, requests, tqdm
Server: Text Generation Inference (TGI) on http://127.0.0.1:6600
"""

from __future__ import annotations
import json
import sys
import os
import time
import re, json
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# ======== CONFIG =========
# =========================
SERVER_URL = "http://127.0.0.1:6600/generate"  # Mistral-7B-Instruct endpoint
MAX_NEW_TOKENS = 100                           # enough for JSON response
DO_SAMPLE = False                              # greedy, deterministic
REQUEST_TIMEOUT_SEC = 120

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/sixth_trial/run_20251030_130850/manipulated_records-test.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/sixth_trial/correct_records-test.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/sixth_trial/run_20251030_130850/masks-blind-test.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/sixth_trial/local-llms/first-trial-gpt/intent_labels-test.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/sixth_trial/local-llms/first-trial-gpt/intent_explanations-test.csv"

# Adult schema
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","class"
]

# JSON grammar to force strict shape (3 entries, decisions in {-1,1})
GBNF_DECISIONS = r'''
root        ::= "{" ws '"decisions"' ws ":" ws "[" ws decision ws "," ws decision ws "," ws decision ws "]" ws "}"
decision    ::= "{" ws '"row_id"' ws ":" ws int ws "," ws '"local_id"' ws ":" ws local ws "," ws '"decision"' ws ":" ws sign ws "," ws '"reason"' ws ":" ws string ws "}"
int         ::= "-"? [0-9]+
local       ::= "0" | "1" | "2"
sign        ::= "-" "1" | "1"
string      ::= "\"" ( [^"\\] | "\\" ["\\/bfnrt] )* "\""
ws          ::= [ \t\n\r]*
'''

# Retries
USE_GRAMMAR_BY_DEFAULT = True
LLM_MAX_RETRIES = 1  # one retry without grammar if grammar fails/unsupported

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

def call_tgi(prompt: str, grammar: Optional[str] = None) -> str:
    """Greedy call to TGI. If grammar is provided, include it; otherwise omit.
       Returns the 'generated_text' string or raises on HTTP error."""
    params = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "return_full_text": False,  # IMPORTANT: only completion
    }
    if grammar:
        params["grammar"] = grammar

    payload = {"inputs": prompt, "parameters": params}
    r = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # Unexpected schema: stringify as last resort (caller will parse/decide)
    return json.dumps(data)

def parse_llm_json(text: str) -> Optional[dict]:
    """Extract the first JSON object in text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def parse_either_schema(llm_text: str, global_ids):
    """Return [(row_id, decision, reason), ...] for local_ids 0..2 or None on failure."""
    # 1) normalize & strip code fences / leading junk
    s = llm_text.strip().lstrip("\ufeff")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()

    # 2) isolate the outermost JSON object if there's extra text
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]

    # 3) parse
    try:
        obj = json.loads(s)
    except Exception:
        return None

    # 4) schema A: {"decisions":[{"row_id":...,"local_id":...,"decision":...,"reason":...}, ...]}
    if isinstance(obj, dict) and isinstance(obj.get("decisions"), list):
        out, seen = [], set()
        for it in obj["decisions"]:
            try:
                lid = it.get("local_id", None)
                rid = it.get("row_id", global_ids[lid] if lid in (0,1,2) else None)
                dec = 1 if int(it.get("decision")) >= 1 else -1
                rsn = (str(it.get("reason","")).strip() or "no_reason")
                rid = int(rid)
                out.append((rid, dec, rsn)); seen.add(rid)
            except Exception:
                continue
        for lid in (0,1,2):
            rid = global_ids[lid]
            if rid not in seen:
                out.append((rid, -1, "json_missing_entry"))
        return out

    # 5) schema B: simple map {"0": 1, "1": -1, "2": -1} (or with int keys)
    if isinstance(obj, dict):
        vals = []
        for lid in (0,1,2):
            v = obj.get(str(lid), obj.get(lid))
            if v is None:
                return None
            try:
                dec = 1 if int(v) >= 1 else -1
            except Exception:
                return None
            vals.append((global_ids[lid], dec, "no_reason"))
        return vals

    return None

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = "Task: Classify each record change as 1 (intentional) or -1 (unintentional)."

def make_group_prompt(clean_row: Dict[str, str],
                      group: List[Dict[str, str]],
                      group_masks: List[Dict[str, str]],
                      group_global_ids: List[int]) -> str:
    """
    Completion-style prompt for GPT-2 (base model, not instruction-tuned)
    """
    # Only include fields where mask=1 to reduce token count
    def filter_by_mask(row_dict, mask_dict):
        return {k: v for k, v in row_dict.items() if mask_dict.get(k) == "1"}
    
    # Create few-shot style examples that GPT-2 can complete
    prompt_parts = ["Data manipulation classification task.\n"]
    
    for k in range(len(group)):
        changed = filter_by_mask(group[k], group_masks[k])
        if changed:
            prompt_parts.append(f"Record {k} changes: {', '.join(changed.keys())}")
        else:
            prompt_parts.append(f"Record {k} changes: none")
    
    prompt_parts.append("\nClassification (1=intentional, -1=unintentional):")
    prompt_parts.append('\n{"0":')
    
    return "\n".join(prompt_parts)

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str):
    t0 = time.perf_counter()

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
    llm_success_groups = 0
    fallback_groups = 0
    # Expanded histogram keys (grammar keys kept for compatibility)
    llm_status_hist = {"ok_with_grammar": 0, "ok_no_grammar": 0, "http_error": 0, "parse_error": 0,
                       "ok_decisions_list": 0, "ok_simple_map": 0}

    # -------- helpers (local to this function) ----------
    def _from_simple_triplet_map_text(llm_text: str, global_ids: List[int]):
        """
        Accepts outputs like '\n\n{"0": 1, "1": -1, "2": -1}' (or with int keys).
        Returns list[(row_id, decision, reason)] or None.
        """
        s = llm_text.strip().lstrip("\ufeff")
        # slice outermost JSON object if extra text exists
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None

        out = []
        for lid in (0, 1, 2):
            v = obj.get(str(lid), obj.get(lid))
            if v is None:
                return None
            try:
                dec = 1 if int(v) >= 1 else -1
            except Exception:
                return None
            out.append((global_ids[lid], dec, "no_reason"))
        return out

    def _from_decisions_list_obj(obj: dict, global_ids: List[int]):
        """
        Accepts {"decisions":[{"row_id":..,"local_id":..,"decision":..,"reason":..}, ...]}.
        Returns list[(row_id, decision, reason)] or None.
        """
        if not (isinstance(obj, dict) and isinstance(obj.get("decisions"), list)):
            return None
        out, seen = [], set()
        for item in obj["decisions"]:
            try:
                lid = item.get("local_id", None)
                rid = item.get("row_id", global_ids[lid] if lid in (0,1,2) else None)
                dec = 1 if int(item.get("decision")) >= 1 else -1
                reason = (str(item.get("reason","")).strip() or "no_reason")
                rid = int(rid)
                out.append((rid, dec, reason))
                seen.add(rid)
            except Exception:
                continue
        # ensure all three entries exist
        for lid in (0,1,2):
            rid = global_ids[lid]
            if rid not in seen:
                out.append((rid, -1, "json_missing_entry"))
        return out
    # ----------------------------------------------------

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

        prompt = make_group_prompt(clean_row, manip_rows, mask_rows, global_ids)

        # One call (no grammar). Parse both schemas.
        obj = None
        l0 = time.perf_counter()
        try:
            llm_text = call_tgi(prompt)  # returns generated_text string
            obj = parse_llm_json(llm_text)
        except requests.HTTPError:
            llm_status_hist["http_error"] += 1
            obj = None
            llm_text = ""
        except Exception:
            obj = None
            llm_text = ""
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)

        decisions: Dict[int, tuple[int, str]] = {}

        # ----- accept either schema -----
        parsed = None
        # 1) structured list
        if isinstance(obj, dict):
            parsed = _from_decisions_list_obj(obj, global_ids)
            if parsed is not None:
                llm_success_groups += 1
                llm_status_hist["ok_decisions_list"] += 1
        # 2) simple map, using the raw text (handles leading \n\n)
        if parsed is None and llm_text:
            parsed = _from_simple_triplet_map_text(llm_text, global_ids)
            if parsed is not None:
                llm_success_groups += 1
                llm_status_hist["ok_simple_map"] += 1
        # --------------------------------

        if parsed is None:
            # Final fallback: mark all three rows -1 with reason
            fallback_groups += 1
            for rid in global_ids:
                decisions[rid] = (-1, "llm_fallback")
        else:
            for rid, dec, reason in parsed:
                decisions[rid] = (dec, reason)

        # Fill generated mask + explanations
        for k, rid in enumerate(global_ids):
            decision, reason = decisions[rid]
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
                        "details": json.dumps({"reason": reason}, ensure_ascii=False),
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
    flat_vals = out_mask_df.values.flatten().tolist()
    counts = pd.Series(flat_vals).value_counts().to_dict()
    print("\n=== Generated Mask Counts ===")
    for k in ["1","-1","0"]:
        print(f"{k:>3}: {counts.get(k,0)}")
    print(f"ALL: {len(flat_vals)}")

    # Runtime stats
    t1 = time.perf_counter()
    stats_payload = {
        "clean_rows": n_clean,
        "manip_rows": n_manip,
        "groups": n_clean,
        "total_time_sec": t1 - t0,
        "avg_group_time_sec": float(pd.Series(group_times).mean()) if group_times else None,
        "median_group_time_sec": float(pd.Series(group_times).median()) if group_times else None,
        "total_llm_time_sec": float(pd.Series(llm_times).sum()) if llm_times else None,
        "avg_llm_time_sec": float(pd.Series(llm_times).mean()) if llm_times else None,
        "llm_success_groups": llm_success_groups,
        "fallback_groups": fallback_groups,
        "llm_status_hist": llm_status_hist,
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
