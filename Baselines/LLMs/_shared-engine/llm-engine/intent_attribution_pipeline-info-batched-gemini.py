#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Google Gemini — BATCHED Version
------------------------------------------------------------------------------------
Key differences from single-request version:
- Processes MULTIPLE groups (e.g., 10-50) in a SINGLE API call
- Dramatically reduces API overhead and total runtime
- Expects JSON response with decisions for ALL groups in the batch
- Robust parsing with per-group fallback i                "manip_rows": group_data["manip_rows"],
                "mask_rows": group_data["mask_rows"],
                "global_ids": group_data["global_ids"]
            })
            batch_group_ids.append(group_data["global_ids"])
        
        # Build batch prompt
        prompt_text = make_batch_prompt(batch_data)parsing fails
- Same info-enhanced features: DIFFS, rubric, etc.

Batch size is configurable via BATCH_SIZE (default: 20 groups per request).
For 6,513 groups with batch_size=20: ~326 API calls instead of 6,513!
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
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

# Batching config
BATCH_SIZE = 10  # Optimal: 10 groups per batch (proven reliable with info-enhanced prompts)
MAX_BATCHES_FOR_TESTING = None  # Set to None to process all batches, or a number to test with fewer batches
MAX_INPUT_TOKENS = 950000  # Conservative limit (95% of 1M to leave room for overhead)
TOKENS_PER_GROUP = 1500  # Higher estimate for info-enhanced (includes diffs + rubric)

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/manipulated_records.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/masks-blind.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/batched-trial-gemini-pro-info/intent_labels.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/batched-trial-gemini-pro-info/intent_explanations.csv"

# Adult schema
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","class"
]

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
                'max_output_tokens': 8192,  # Increased for info-enhanced responses (larger due to diffs/rubric)
                'candidate_count': 1,
            },
            safety_settings=safety_settings,
            request_options={'timeout': 300}  # 5 min timeout for batches
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
            
    except Exception:
        return None  # signals caller: fall back to next model tier


def call_gemini_with_fallback(model_lite, model_flash, *args, **kwargs) -> str:
    """Cost policy (2026-06-26): flash-lite first, then flash, then give up.
    Never escalates to gemini-2.5-pro."""
    result = call_gemini(model_lite, *args, **kwargs)
    if result is not None:
        return result
    result = call_gemini(model_flash, *args, **kwargs)
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

def parse_batch_decisions(obj: dict, batch_group_ids: List[List[int]]) -> Optional[Dict[int, Tuple[int, str]]]:
    """
    Parse batched response. Expected formats:
    
    Format A (list of groups, each with decisions list):
    {
      "groups": [
        {
          "group_id": 0,
          "decisions": [
            {"local_id": 0, "row_id": X, "decision": 1, "reason": "..."},
            {"local_id": 1, "row_id": Y, "decision": -1, "reason": "..."},
            {"local_id": 2, "row_id": Z, "decision": -1, "reason": "..."}
          ]
        },
        {"group_id": 1, "decisions": [...]}
      ]
    }
    
    Format B (flat list of all decisions with group_id):
    {
      "decisions": [
        {"group_id": 0, "local_id": 0, "row_id": X, "decision": 1, "reason": "..."},
        {"group_id": 0, "local_id": 1, "row_id": Y, "decision": -1, "reason": "..."},
        ...
      ]
    }
    
    Returns: {row_id: (decision, reason)} for all rows, or None if unusable
    """
    if not isinstance(obj, dict):
        return None
    
    decisions: Dict[int, Tuple[int, str]] = {}
    
    # Format A: groups list
    if "groups" in obj and isinstance(obj["groups"], list):
        for group_obj in obj["groups"]:
            if not isinstance(group_obj, dict):
                continue
            gid = group_obj.get("group_id")
            if gid is None or gid < 0 or gid >= len(batch_group_ids):
                continue
            
            group_global_ids = batch_group_ids[gid]
            group_decs = group_obj.get("decisions", [])
            
            if not isinstance(group_decs, list):
                continue
            
            # Map by local_id
            by_lid: Dict[int, dict] = {}
            for item in group_decs:
                if isinstance(item, dict):
                    try:
                        lid = int(item.get("local_id", -1))
                        if 0 <= lid < 3:
                            by_lid[lid] = item
                    except:
                        pass
            
            # Extract decisions for this group
            for lid in (0, 1, 2):
                rid = group_global_ids[lid]
                item = by_lid.get(lid)
                if item:
                    try:
                        dec_val = item.get("decision", item.get("intentional", -1))
                        dec = 1 if int(dec_val) >= 1 else -1
                        reason = str(item.get("reason", "")).strip() or "batch_parsed"
                        decisions[rid] = (dec, reason)
                    except:
                        decisions[rid] = (-1, "batch_parse_error")
                else:
                    decisions[rid] = (-1, "batch_missing_local_id")
        
        return decisions if decisions else None
    
    # Format B: flat decisions list with group_id
    if "decisions" in obj and isinstance(obj["decisions"], list):
        for item in obj["decisions"]:
            if not isinstance(item, dict):
                continue
            
            try:
                gid = int(item.get("group_id", -1))
                lid = int(item.get("local_id", -1))
                
                if gid < 0 or gid >= len(batch_group_ids):
                    continue
                if lid < 0 or lid >= 3:
                    continue
                
                rid = batch_group_ids[gid][lid]
                dec_val = item.get("decision", item.get("intentional", -1))
                dec = 1 if int(dec_val) >= 1 else -1
                reason = str(item.get("reason", "")).strip() or "batch_flat_parsed"
                decisions[rid] = (dec, reason)
            except:
                pass
        
        return decisions if decisions else None
    
    return None

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "You are a precise data-forensics assistant. Decide for EACH manipulated RECORD in EACH GROUP "
    "whether changes are INTENTIONAL (1) or UNINTENTIONAL (-1). Deterministic behavior."
)

RUBRIC = (
    "Rubric:\n"
    "- INTENTIONAL (1): large utility-seeking edits (education-num +≥2 or categorical upgrade), "
    "big jump in capital-gain from 0, hours-per-week +≥10, coordinated multiple edits, fairness gaming, privacy DMV.\n"
    "- UNINTENTIONAL (-1): tiny numeric drift (±1–2), case-only/formatting, single trivial change.\n"
    "- If uncertain: pick -1 only when edits are small/benign; otherwise 1."
)

def make_batch_prompt(batch_data: List[Dict]) -> str:
    """
    Create a batched prompt for multiple groups.
    
    batch_data: List of dicts, each containing:
      - group_id: int (0-indexed within this batch)
      - clean_row: Dict[str, str]
      - manip_rows: List[Dict[str, str]] (3 rows)
      - mask_rows: List[Dict[str, str]] (3 masks)
      - global_ids: List[int] (3 row IDs)
    """
    sys_rules = (
        "Rules:\n"
        f"- You will receive {len(batch_data)} GROUPS, each with 1 clean row + 3 manipulated rows.\n"
        "- For EACH group, decide 1 (intentional) or -1 (unintentional) for EACH of the 3 manipulated records.\n"
        "- Return ONLY ONE JSON with format:\n"
        '{\n'
        '  "groups": [\n'
        '    {\n'
        '      "group_id": 0,\n'
        '      "decisions": [\n'
        '        {"local_id": 0, "row_id": <global>, "decision": 1|-1, "reason": "<≤10w>"},\n'
        '        {"local_id": 1, "row_id": <global>, "decision": 1|-1, "reason": "<≤10w>"},\n'
        '        {"local_id": 2, "row_id": <global>, "decision": 1|-1, "reason": "<≤10w>"}\n'
        '      ]\n'
        '    },\n'
        '    {"group_id": 1, "decisions": [...]},\n'
        '    ...\n'
        '  ]\n'
        '}\n'
        "- No extra text outside the JSON."
    )
    
    groups_bundle = []
    for batch_item in batch_data:
        manipulated_bundle = []
        for k in range(3):
            manipulated_bundle.append({
                "local_id": k,
                "row_id": batch_item["global_ids"][k],
                "row": batch_item["manip_rows"][k],
                "mask": batch_item["mask_rows"][k],
                "diffs": compute_diffs(
                    batch_item["clean_row"],
                    batch_item["manip_rows"][k],
                    batch_item["mask_rows"][k]
                ),
            })
        
        groups_bundle.append({
            "group_id": batch_item["group_id"],
            "columns": ADULT_COLS,
            "clean": batch_item["clean_row"],
            "manipulated": manipulated_bundle
        })
    
    prompt = (
        f"{ROLE_BLOCK}\n\n{RUBRIC}\n\n{sys_rules}\n\n"
        f"INPUT ({len(batch_data)} groups):\n{json.dumps({'groups': groups_bundle}, ensure_ascii=False)}\n\n"
        "Return the JSON object now."
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
    per_batch_csv_path = os.path.join(out_dir, "per_batch_times.csv")

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

    # =========================
    # PRE-LOAD ALL GROUPS INTO MEMORY
    # =========================
    print(f"\n=== Loading ALL {n_clean:,} groups into memory ===")
    all_groups_data = []
    for j in range(n_clean):
        clean_row = row_to_dict(correct, j)
        manip_rows, mask_rows, global_ids = [], [], []
        for k in range(3):
            mi = 3*j + k
            manip_rows.append(row_to_dict(manip, mi))
            mask_rows.append(mask_row_to_dict(masks, mi))
            global_ids.append(mi)
        all_groups_data.append({
            "clean_row": clean_row,
            "manip_rows": manip_rows,
            "mask_rows": mask_rows,
            "global_ids": global_ids
        })
    print(f"✓ Loaded {len(all_groups_data):,} groups into memory")

    # Prepare outputs
    out_mask_df = masks.copy(deep=True)
    explanations_rows: List[Dict[str, str]] = []
    processed_rows = set()  # Track which rows we actually processed

    # Timing & status trackers
    batch_times: List[float] = []
    llm_times:   List[float] = []
    sources_hist = {"llm":0, "fallback":0}
    llm_status_hist = {"ok": 0, "api_error": 0, "parse_error": 0, "batch_parsed": 0}
    
    batch_size = BATCH_SIZE  # Use local variable
    total_batches = (n_clean + batch_size - 1) // batch_size
    print(f"\n=== Batched Processing (Info-enhanced, ALL DATA IN MEMORY) ===")
    print(f"Total groups: {n_clean:,}")
    print(f"Batch size: {batch_size}")
    print(f"Total batches: {total_batches}")
    print(f"API calls needed: {total_batches} (vs {n_clean:,} for non-batched)")
    print(f"Expected tokens per batch: ~{batch_size * TOKENS_PER_GROUP:,}")
    print(f"Speedup: ~{n_clean / total_batches:.1f}x\n")
    
    if MAX_BATCHES_FOR_TESTING is not None:
        print(f"⚠️  TESTING MODE: Processing only {MAX_BATCHES_FOR_TESTING} batches (set MAX_BATCHES_FOR_TESTING=None for full run)\n")

    # Process in batches
    for batch_idx in tqdm(range(0, n_clean, batch_size), desc="Batches", ncols=100):
        # Early exit for testing
        current_batch_num = batch_idx // batch_size
        if MAX_BATCHES_FOR_TESTING is not None and current_batch_num >= MAX_BATCHES_FOR_TESTING:
            print(f"\n✓ Completed {MAX_BATCHES_FOR_TESTING} test batches. Stopping early.")
            break
            
        batch_t0 = time.perf_counter()
        
        batch_end = min(batch_idx + batch_size, n_clean)
        batch_data = []
        batch_group_ids: List[List[int]] = []  # For parsing: batch_group_ids[group_id_in_batch] = [rid0, rid1, rid2]
        
        # Use pre-loaded data from memory
        for j in range(batch_idx, batch_end):
            group_data = all_groups_data[j]
            batch_data.append({
                "group_id": j - batch_idx,  # 0-indexed within this batch
                "clean_row": group_data["clean_row"],
                "manip_rows": group_data["manip_rows"],
                "mask_rows": group_data["mask_rows"],
                "global_ids": group_data["global_ids"]
            })
            batch_group_ids.append(group_data["global_ids"])
        
        # Build batch prompt
        prompt_text = make_batch_prompt(batch_data)
        
        # Call LLM
        obj = None
        llm_text = ""
        llm_t0 = time.perf_counter()
        try:
            llm_text = call_gemini_with_fallback(model, model_flash, prompt_text)
            obj = parse_llm_json(llm_text)
            llm_status_hist["ok"] += 1
        except Exception:
            llm_status_hist["api_error"] += 1
            obj = None
            llm_text = ""
        llm_t1 = time.perf_counter()
        llm_times.append(llm_t1 - llm_t0)
        
        # Parse batch decisions
        batch_decisions = None
        if isinstance(obj, dict):
            batch_decisions = parse_batch_decisions(obj, batch_group_ids)
            if batch_decisions is not None:
                llm_status_hist["batch_parsed"] += 1
        
        # Fill decisions (with per-row fallback if batch parse failed)
        decisions: Dict[int, Tuple[int, str, str]] = {}
        if batch_decisions is None:
            # Batch parse failed - fallback all rows in this batch
            for global_ids in batch_group_ids:
                for rid in global_ids:
                    decisions[rid] = (-1, "batch_fallback_parse_failed", "fallback")
            sources_hist["fallback"] += 1
        else:
            # Batch parsed successfully
            any_from_llm = False
            for global_ids in batch_group_ids:
                for rid in global_ids:
                    if rid in batch_decisions:
                        dec, rsn = batch_decisions[rid]
                        decisions[rid] = (dec, rsn, "llm")
                        any_from_llm = True
                    else:
                        decisions[rid] = (-1, "batch_missing_rid", "fallback")
            sources_hist["llm" if any_from_llm else "fallback"] += 1
        
        # Fill generated mask + explanations for this batch
        for batch_item in batch_data:
            for k, rid in enumerate(batch_item["global_ids"]):
                processed_rows.add(rid)  # Track this row as processed
                decision, reason, src = decisions.get(rid, (-1, "batch_missing", "fallback"))
                
                # Update mask
                for c in ADULT_COLS:
                    if c == "class":
                        out_mask_df.at[rid, c] = "0"
                    else:
                        out_mask_df.at[rid, c] = (
                            str(decision) if str(masks.at[rid, c]).strip() in ("1","1.0") else "0"
                        )
                
                # Add explanations
                for c in ADULT_COLS:
                    if str(masks.at[rid, c]).strip() in ("1","1.0"):
                        explanations_rows.append({
                            "row_id": rid,
                            "column": c,
                            "manipulated_value": batch_item["manip_rows"][k][c],
                            "original_value": batch_item["clean_row"][c],
                            "mask": 1,
                            "diagnosis": str(decision),
                            "rule": "record_intent_decision_batched",
                            "details": json.dumps({"reason": reason, "source": src, "batch_idx": batch_idx}, ensure_ascii=False),
                            "mapping_status": "matched",
                            "mapping_score": ""
                        })
        
        batch_t1 = time.perf_counter()
        batch_times.append(batch_t1 - batch_t0)

    # Save outputs
    out_mask_df.to_csv(OUT_LABELS_CSV, index=False)

    expl_df = pd.DataFrame(explanations_rows, columns=[
        "row_id","column","manipulated_value","original_value","mask",
        "diagnosis","rule","details","mapping_status","mapping_score"
    ])
    expl_df.to_csv(OUT_EXPL_CSV, index=False)

    # Summary
    # Statistics for all cells (includes unprocessed rows with original mask values)
    flat_vals = out_mask_df.replace({"1.0":"1","-1.0":"-1"}).values.flatten().tolist()
    counts = pd.Series(flat_vals).value_counts().to_dict()
    
    # Statistics for processed cells only
    processed_mask_df = out_mask_df.loc[list(processed_rows)]
    processed_flat_vals = processed_mask_df.replace({"1.0":"1","-1.0":"-1"}).values.flatten().tolist()
    processed_counts = pd.Series(processed_flat_vals).value_counts().to_dict()
    
    print("\n=== Generated Mask Counts (PROCESSED cells only) ===")
    for k in ["1","-1","0"]:
        print(f"{k:>3}: {processed_counts.get(k,0)}")
    print(f"ALL: {len(processed_flat_vals)}")
    print(f"Processed rows: {len(processed_rows):,} / {len(out_mask_df):,}")
    
    if MAX_BATCHES_FOR_TESTING is not None:
        print(f"\n=== All Cells (including unprocessed) ===")
        for k in ["1","-1","0"]:
            print(f"{k:>3}: {counts.get(k,0)}")
        print(f"ALL: {len(flat_vals)}")
        print("Note: Unprocessed rows retain original mask values (0 or 1)")

    print("\n=== Instrumentation ===")
    print("LLM status histogram:", json.dumps(llm_status_hist, indent=2))
    print("Decision sources histogram:", json.dumps(sources_hist, indent=2))

    # Runtime stats
    t1 = time.perf_counter()
    stats_payload = {
        "model": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "clean_rows": n_clean,
        "manip_rows": n_manip,
        "groups": n_clean,
        "batches": total_batches,
        "total_time_sec": t1 - t0,
        "avg_batch_time_sec": float(pd.Series(batch_times).mean()) if batch_times else None,
        "median_batch_time_sec": float(pd.Series(batch_times).median()) if batch_times else None,
        "total_llm_time_sec": float(pd.Series(llm_times).sum()) if llm_times else None,
        "avg_llm_time_sec": float(pd.Series(llm_times).mean()) if llm_times else None,
        "llm_status_hist": llm_status_hist,
        "decision_sources_hist": sources_hist,
        "outputs": {
            "generated_mask_csv": OUT_LABELS_CSV,
            "explanations_csv": OUT_EXPL_CSV,
            "per_batch_times_csv": per_batch_csv_path
        }
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    # Only include completed batches in per-batch stats
    num_completed_batches = len(batch_times)
    per_batch_df = pd.DataFrame({
        "batch_idx": list(range(0, num_completed_batches * batch_size, batch_size)),
        "batch_time_sec": batch_times,
        "llm_time_sec": llm_times
    })
    per_batch_df.to_csv(per_batch_csv_path, index=False)

    print(f"\nSaved runtime stats to:\n  - {stats_json_path}\n  - {per_batch_csv_path}")
    print(f"Outputs:\n  - {OUT_LABELS_CSV}\n  - {OUT_EXPL_CSV}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else MANIPULATED_CSV
    c = sys.argv[2] if len(sys.argv) > 2 else CORRECT_CSV
    k = sys.argv[3] if len(sys.argv) > 3 else MASKS_CSV
    run_pipeline(m, c, k)
