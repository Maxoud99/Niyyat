#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution for Twitter Bot Detection via Mixtral — Few-shot
------------------------------------------------------------------------------------
Mixtral via TGI with few-shot examples for Twitter bot detection:
- Info-enhanced with rubric for better decision guidance
- Few-shot demos included in prompt (USE_FEW_SHOT=True)
- Feature-level analysis with specific examples
- Mixtral [INST] format
- Pure fallback: if no parse, set all changed features to -1

Model: Mixtral (via TGI on port 6000)
"""

from __future__ import annotations
import json
import sys
import os
import time
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# ======== CONFIG =========
# =========================
SERVER_URL = "http://127.0.0.1:6000/generate"  # Mixtral TGI endpoint (port 6000)
MAX_NEW_TOKENS = 450                           # Reduced to fit within 4000 token limit with few-shot examples (3501 input + 450 = 3951 < 4000)
DO_SAMPLE = False                              # Greedy, deterministic
REQUEST_TIMEOUT_SEC = 240                      # 4 minutes for large responses

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_dataset.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_dataset_clean.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_mask.csv"

# Outputs
OUT_DIR = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/few-shots-mixtral"
OUT_LABELS_CSV = os.path.join(OUT_DIR, "intent_labels.csv")
OUT_EXPL_CSV   = os.path.join(OUT_DIR, "intent_explanations.csv")

# Twitter bot schema (19 columns - removed derived features)
TWITTER_BOT_COLS = [
    "user_id", "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "verified", "protected",
    "default_profile", "default_profile_image", "geo_enabled",
    "profile_use_background_image", "has_created_date",
    "description_length", "screen_name_length", "name_length",
    "has_description", "has_location", "label"
]

# Processing configuration
CHUNK_SIZE = 3  # Reduced from 5 to ensure all chunks fit within token limit with few-shot examples
MAX_RECORDS = None  # Set to None for full dataset
DEBUG_MODE = False  # Save LLM responses to debug file (disable for full run)

# ======= Few-shot config (ON by default) =======
USE_FEW_SHOT = True

FEW_SHOT_BLOCK = """
EXAMPLE 1 — Bot-to-Human Evasion: Follower Manipulation & Profile Completion (abridged)
Context: Clean bot profile vs. changed profile. Changed features show modifications.

Row ID: 1001
Changed Features:
- followers_count: 45 → 15000
  Rationale: Massive follower increase to appear popular and human-like → INTENTIONAL (1)
  
- friends_count: 32 → 12500  
  Rationale: Large increase to appear socially active → INTENTIONAL (1)
  
- has_description: 0 → 1
  Rationale: Adding description to complete profile and appear authentic → INTENTIONAL (1)
  
- description_length: 0 → 85
  Rationale: Profile completion strategy (correlated with has_description) → INTENTIONAL (1)

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 1001, "column": "followers_count", "decision": 1, "reason": "massive follower jump for popularity"},
    {"row_id": 1001, "column": "friends_count", "decision": 1, "reason": "large increase for social appearance"},
    {"row_id": 1001, "column": "has_description", "decision": 1, "reason": "profile completion strategy"},
    {"row_id": 1001, "column": "description_length", "decision": 1, "reason": "coordinated with description addition"}
  ]
}

EXAMPLE 2 — Mixed Changes: Evasion Attempts vs. Natural Variations (abridged)
Context: Multiple bot profiles with various changes showing different intent levels.

Row ID: 2001
Changed Features:
- verified: 0 → 1
  Rationale: Verification gaming to appear legitimate → INTENTIONAL (1)
  
- default_profile: 1 → 0
  Rationale: Customizing profile to appear authentic → INTENTIONAL (1)
  
- statuses_count: 120 → 135
  Rationale: Small activity increase, natural variation → UNINTENTIONAL (-1)

Row ID: 2002  
Changed Features:
- followers_count: 500 → 550
  Rationale: Minor follower growth, normal activity → UNINTENTIONAL (-1)
  
- description_length: 45 → 48
  Rationale: Small text edit, natural → UNINTENTIONAL (-1)

Row ID: 2003
Changed Features:
- friends_count: 150 → 50000
  Rationale: Extreme jump suggesting bot evasion → INTENTIONAL (1)
  
- favourites_count: 25 → 5000
  Rationale: Large engagement boost to appear active → INTENTIONAL (1)
  
- geo_enabled: 0 → 1
  Rationale: Strategic feature activation for authenticity → INTENTIONAL (1)

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 2001, "column": "verified", "decision": 1, "reason": "verification gaming"},
    {"row_id": 2001, "column": "default_profile", "decision": 1, "reason": "customization for authenticity"},
    {"row_id": 2001, "column": "statuses_count", "decision": -1, "reason": "small activity variation"},
    {"row_id": 2002, "column": "followers_count", "decision": -1, "reason": "minor natural growth"},
    {"row_id": 2002, "column": "description_length", "decision": -1, "reason": "small text edit"},
    {"row_id": 2003, "column": "friends_count", "decision": 1, "reason": "extreme jump for evasion"},
    {"row_id": 2003, "column": "favourites_count", "decision": 1, "reason": "engagement gaming"},
    {"row_id": 2003, "column": "geo_enabled", "decision": 1, "reason": "strategic feature activation"}
  ]
}

EXAMPLE 3 — Coordinated Multi-Feature Evasion (abridged)
Context: Bot profile with multiple coordinated changes to evade detection.

Row ID: 3001
Changed Features:
- followers_count: 80 → 25000
  Rationale: Large follower manipulation → INTENTIONAL (1)
  
- protected: 1 → 0
  Rationale: Making account public to appear open → INTENTIONAL (1)
  
- has_location: 0 → 1
  Rationale: Profile completion for legitimacy → INTENTIONAL (1)
  
- profile_use_background_image: 0 → 1
  Rationale: Visual customization strategy → INTENTIONAL (1)
  
- name_length: 5 → 8
  Rationale: Minor name edit, likely part of broader evasion → INTENTIONAL (1)

Gold feature decisions:
{
  "feature_decisions": [
    {"row_id": 3001, "column": "followers_count", "decision": 1, "reason": "follower manipulation"},
    {"row_id": 3001, "column": "protected", "decision": 1, "reason": "strategic visibility change"},
    {"row_id": 3001, "column": "has_location", "decision": 1, "reason": "profile completion"},
    {"row_id": 3001, "column": "profile_use_background_image", "decision": 1, "reason": "visual customization"},
    {"row_id": 3001, "column": "name_length", "decision": 1, "reason": "coordinated with other changes"}
  ]
}
""".strip()

# =========================
# ====== UTILITIES ========
# =========================
def to_str(x) -> str:
    """Convert value to string, handling NaN/None."""
    if pd.isna(x):
        return ""
    return str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    """Convert a DataFrame row to a dictionary."""
    return {c: to_str(df.at[idx, c]) for c in TWITTER_BOT_COLS}

def mask_row_to_dict(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    """Convert a mask row to dictionary with 0/1 values."""
    return {c: "1" if str(mask_df.at[idx, c]).strip() in ("1", "1.0") else "0" 
            for c in TWITTER_BOT_COLS}

# =========================
# ===== TGI CALL ==========
# =========================
def call_tgi(inputs: str) -> str:
    """Greedy call to TGI; returns generated_text."""
    params = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "return_full_text": False,  # IMPORTANT: only completion
    }
    payload = {"inputs": inputs, "parameters": params}
    
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
        
        # DEBUG: Log error details before raising
        if r.status_code != 200 and DEBUG_MODE:
            print(f"[DEBUG] TGI Status Code: {r.status_code}")
            print(f"[DEBUG] TGI Error Response: {r.text[:1000]}")
        
        r.raise_for_status()
        data = r.json()
        
        # DEBUG: Log the raw response
        if DEBUG_MODE:
            print(f"[DEBUG] TGI Response type: {type(data)}")
            print(f"[DEBUG] TGI Response keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
            print(f"[DEBUG] TGI Response: {json.dumps(data, indent=2)[:500]}")
        
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        
        # Unexpected schema: stringify as last resort
        return json.dumps(data)
    except Exception as e:
        # Log error in debug mode
        if DEBUG_MODE:
            print(f"[DEBUG] TGI Error: {str(e)}")
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
    for junk in ("</think>", "<|eot_id|>", "<|im_end|>", "```json", "```", "~~~json", "~~~"):
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

def decisions_from_obj(obj: dict, global_ids: List[int]) -> Optional[Dict[Tuple[int, str], Tuple[int, str]]]:
    """
    Parse feature-level decisions from LLM response.
    
    Expected format:
    {
      "feature_decisions": [
        {
          "row_id": <id>,
          "column": "<col_name>",
          "decision": 1 or -1,
          "reason": "<explanation>"
        },
        ...
      ]
    }
    
    Returns Dict[(row_id, column), (decision, reason)] or None if unusable.
    """
    if not isinstance(obj, dict):
        return None

    # Try to get "feature_decisions" key
    feat_decs = obj.get("feature_decisions", None)
    
    if feat_decs is None:
        return None

    # Must be a list of feature decisions
    if not isinstance(feat_decs, list):
        return None
    
    # Parse each feature decision
    decisions: Dict[Tuple[int, str], Tuple[int, str]] = {}
    
    for item in feat_decs:
        if not isinstance(item, dict):
            continue
        
        try:
            row_id = int(item.get("row_id"))
            column = str(item.get("column", ""))
            decision_val = int(item.get("decision", -1))
            reason = str(item.get("reason", "")).strip() or "reason_not_provided"
            
            # Normalize decision to 1 or -1
            decision = 1 if decision_val >= 1 else -1
            
            # Store by (row_id, column) key
            decisions[(row_id, column)] = (decision, reason)
            
        except (ValueError, TypeError, KeyError):
            continue
    
    return decisions if decisions else None

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = """ROLE
You are a precise data-forensics assistant specializing in Twitter bot detection evasion analysis.

Your task is to analyze changes made to individual features in Twitter bot profiles and determine whether each change was:
- INTENTIONAL (1): Deliberately made to make a bot profile appear more human-like or evade bot detection
- UNINTENTIONAL (-1): Accidental changes, or normal profile variations

IMPORTANT: You must evaluate EACH CHANGED FEATURE INDEPENDENTLY, not the entire record.

"""

RUBRIC = """Rubric for Twitter Bot Evasion Detection:

INTENTIONAL (1) - Bot-to-Human Evasion Indicators:
- **Follower manipulation**: Large increases in followers_count or friends_count to appear more social
- **Engagement gaming**: Significant jumps in favourites_count or statuses_count
- **Verification gaming**: Changing verified status
- **Profile completeness**: Adding description, adding location
- **Visual authenticity**: Changing default_profile or default_profile_image to appear customized
- **Strategic feature activation**: Enabling geo_enabled, profile_use_background_image to appear authentic

UNINTENTIONAL (-1) - Benign/Natural Changes:
- **Small numeric drift**: Count changes (natural activity fluctuation)
- **Natural growth**: Gradual increases in statuses_count, favourites_count consistent with normal use
- **Formatting/encoding**: No semantic change, just formatting differences
- **Listed count variations**: Small changes in listed_count - natural list membership changes
"""

def make_chunk_prompt(clean_rows: List[Dict[str, str]],
                     changed_rows: List[Dict[str, str]],
                     mask_rows: List[Dict[str, str]],
                     global_ids: List[int]) -> str:
    """
    Create few-shot enhanced prompt for a chunk of records.
    Includes rubric and few-shot examples for feature-level analysis.
    Uses Mixtral [INST] format.
    """
    sys_rules = """Rules:
- Analyze each pair of (clean, changed) bot profiles
- For EACH CHANGED FEATURE (where mask=1), provide a separate decision
- Decide 1 (intentional bot evasion) or -1 (unintentional/normal change) PER FEATURE
- Each feature should have its own specific explanation (≤15 words)
- Be deterministic - same input should yield same output
- Return ONLY a JSON object, no additional text

Required JSON format:
{
  "feature_decisions": [
    {
      "row_id": <global_id>,
      "column": "<feature_name>",
      "changed_value": "<new_value>",
      "original_value": "<old_value>",
      "decision": 1 or -1,
      "reason": "<specific explanation for THIS feature change, ≤15 words>"
    },
    ...
  ]
}

CRITICAL: Each changed feature needs its own entry with a specific reason explaining why THAT PARTICULAR feature change is intentional or unintentional.
"""
    
    # Build the data bundle with changed features explicitly listed
    records_with_changes = []
    for i in range(len(clean_rows)):
        changed_features = []
        for col in TWITTER_BOT_COLS:
            if col != "user_id" and mask_rows[i][col] == "1":
                changed_features.append({
                    "column": col,
                    "original_value": clean_rows[i][col],
                    "changed_value": changed_rows[i][col]
                })
        
        if changed_features:  # Only include records with changes
            records_with_changes.append({
                "index": i,
                "row_id": global_ids[i],
                "clean_profile": clean_rows[i],
                "changed_profile": changed_rows[i],
                "changed_features": changed_features
            })
    
    bundle = {
        "columns": TWITTER_BOT_COLS,
        "records": records_with_changes
    }
    
    fewshot = (FEW_SHOT_BLOCK + "\n\n") if USE_FEW_SHOT else ""
    
    user_content = (
        f"{ROLE_BLOCK}\n{RUBRIC}\n{sys_rules}\n\n{fewshot}"
        f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Analyze each changed feature using the rubric and examples, then return your feature-level decisions in JSON format now."
    )
    
    # Wrap in Mixtral [INST] tags
    prompt = f"[INST] {user_content} [/INST]"
    
    return prompt

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str, max_records: Optional[int] = None):
    """Main pipeline to process Twitter bot intent attribution with few-shot learning."""
    t0 = time.perf_counter()

    # Ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    stats_json_path = os.path.join(OUT_DIR, "run_stats.json")
    per_chunk_csv_path = os.path.join(OUT_DIR, "per_chunk_times.csv")

    # Load CSVs as strings
    print(f"Loading data from:")
    print(f"  - Manipulated: {manip_path}")
    print(f"  - Clean: {correct_path}")
    print(f"  - Masks: {masks_path}")
    
    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)
    masks = pd.read_csv(masks_path, dtype=str, keep_default_na=False)

    # Verify columns
    for df, name in [(manip, "manipulated"), (correct, "clean"), (masks, "masks")]:
        missing = [c for c in TWITTER_BOT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    # Ensure same order
    manip = manip[TWITTER_BOT_COLS].copy()
    correct = correct[TWITTER_BOT_COLS].copy()
    masks = masks[TWITTER_BOT_COLS].copy()

    # Verify 1-to-1 mapping
    if len(manip) != len(correct) or len(manip) != len(masks):
        raise ValueError(f"Row count mismatch: manip={len(manip)}, correct={len(correct)}, masks={len(masks)}")

    n_total = len(manip)
    if max_records is not None:
        n_total = min(n_total, max_records)
        manip = manip.iloc[:n_total].copy()
        correct = correct.iloc[:n_total].copy()
        masks = masks.iloc[:n_total].copy()

    print(f"Processing {n_total} records in chunks of {CHUNK_SIZE}")

    # Prepare outputs
    out_mask_df = masks.copy(deep=True)
    explanations_rows: List[Dict[str, str]] = []

    # Timing & status trackers
    chunk_times: List[float] = []
    llm_times: List[float] = []
    llm_success_chunks = 0
    llm_fail_chunks = 0
    parse_errors_total = 0
    features_evaluated = 0

    # Process in chunks
    n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    with tqdm(total=n_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for chunk_idx in range(n_chunks):
            chunk_t0 = time.perf_counter()
            
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, n_total)
            chunk_size = end_idx - start_idx
            
            # Get chunk data
            clean_rows = [row_to_dict(correct, i) for i in range(start_idx, end_idx)]
            changed_rows = [row_to_dict(manip, i) for i in range(start_idx, end_idx)]
            mask_rows = [mask_row_to_dict(masks, i) for i in range(start_idx, end_idx)]
            global_ids = list(range(start_idx, end_idx))
            
            # Count features to evaluate in this chunk
            chunk_features = sum(
                sum(1 for col in TWITTER_BOT_COLS if col != "user_id" and mask_rows[i][col] == "1")
                for i in range(chunk_size)
            )
            
            # Build prompt
            prompt = make_chunk_prompt(clean_rows, changed_rows, mask_rows, global_ids)
            
            # Debug: Save prompt
            if DEBUG_MODE:
                debug_prompt_file = os.path.join(OUT_DIR, f"debug_chunk_{chunk_idx}_prompt.txt")
                with open(debug_prompt_file, "w", encoding="utf-8") as f:
                    f.write(f"=== CHUNK {chunk_idx} PROMPT ===\n\n")
                    f.write(prompt)
                    f.write("\n\n=== END PROMPT ===\n")
            
            # Call LLM
            llm_t0 = time.perf_counter()
            llm_response = call_tgi(prompt)
            llm_elapsed = time.perf_counter() - llm_t0
            llm_times.append(llm_elapsed)
            
            # Debug: Save LLM response
            if DEBUG_MODE:
                debug_file = os.path.join(OUT_DIR, f"debug_chunk_{chunk_idx}_response.txt")
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(f"=== CHUNK {chunk_idx} LLM RESPONSE ===\n\n")
                    f.write(llm_response)
                    f.write("\n\n=== END RESPONSE ===\n")
            
            # Parse response
            obj = parse_llm_json(llm_response)
            decisions = decisions_from_obj(obj, global_ids) if obj else None
            
            if decisions is None:
                # LLM failure - mark all changed features as -1 (conservative)
                llm_fail_chunks += 1
                parse_errors_total += chunk_features
                
                for i in range(chunk_size):
                    rid = global_ids[i]
                    for col in TWITTER_BOT_COLS:
                        if col != "user_id" and mask_rows[i][col] == "1":
                            # Mark as -1 in output mask
                            out_mask_df.at[rid, col] = "-1"
                            
                            # Add to explanations (match Gemini format)
                            explanations_rows.append({
                                "row_id": str(rid),
                                "column": col,
                                "changed_value": changed_rows[i][col],
                                "original_value": clean_rows[i][col],
                                "mask": "1",
                                "diagnosis": "0",  # Parse error
                                "rule": "intent_attribution",
                                "details": '{"reason": "llm_json_parse_failure"}',
                                "mapping_status": "matched",
                                "mapping_score": ""
                            })
            else:
                # LLM success
                llm_success_chunks += 1
                
                # Process each changed feature
                for i in range(chunk_size):
                    rid = global_ids[i]
                    for col in TWITTER_BOT_COLS:
                        if col != "user_id" and mask_rows[i][col] == "1":
                            features_evaluated += 1
                            
                            # Check if we have a decision for this feature
                            key = (rid, col)
                            if key in decisions:
                                decision, reason = decisions[key]
                                # Update mask
                                out_mask_df.at[rid, col] = str(decision)
                                
                                # Add to explanations (match Gemini format)
                                explanations_rows.append({
                                    "row_id": str(rid),
                                    "column": col,
                                    "changed_value": changed_rows[i][col],
                                    "original_value": clean_rows[i][col],
                                    "mask": "1",
                                    "diagnosis": str(decision),
                                    "rule": "intent_attribution",
                                    "details": json.dumps({"reason": reason}),
                                    "mapping_status": "matched",
                                    "mapping_score": ""
                                })
                            else:
                                # Feature was changed but LLM didn't provide decision
                                parse_errors_total += 1
                                out_mask_df.at[rid, col] = "-1"
                                
                                # Add to explanations (match Gemini format)
                                explanations_rows.append({
                                    "row_id": str(rid),
                                    "column": col,
                                    "changed_value": changed_rows[i][col],
                                    "original_value": clean_rows[i][col],
                                    "mask": "1",
                                    "diagnosis": "0",
                                    "rule": "intent_attribution",
                                    "details": '{"reason": "llm_missing_feature_decision"}',
                                    "mapping_status": "matched",
                                    "mapping_score": ""
                                })
            
            chunk_elapsed = time.perf_counter() - chunk_t0
            chunk_times.append(chunk_elapsed)
            
            pbar.update(1)
            pbar.set_postfix({
                "success": llm_success_chunks,
                "fail": llm_fail_chunks,
                "parse_err": parse_errors_total,
                "features": features_evaluated
            })

    # Save outputs
    print("\nSaving outputs...")
    out_mask_df.to_csv(OUT_LABELS_CSV, index=False)
    print(f"  ✓ Saved intent labels: {OUT_LABELS_CSV}")
    
    expl_df = pd.DataFrame(explanations_rows)
    expl_df.to_csv(OUT_EXPL_CSV, index=False)
    print(f"  ✓ Saved explanations: {OUT_EXPL_CSV}")

    # Save timing stats
    total_elapsed = time.perf_counter() - t0
    avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
    avg_llm_time = sum(llm_times) / len(llm_times) if llm_times else 0

    stats = {
        "model": "Mixtral",
        "variant": "few-shots",
        "endpoint": SERVER_URL,
        "total_records": n_total,
        "chunk_size": CHUNK_SIZE,
        "total_chunks": n_chunks,
        "llm_success_chunks": llm_success_chunks,
        "llm_fail_chunks": llm_fail_chunks,
        "features_evaluated": features_evaluated,
        "parse_errors_total": parse_errors_total,
        "parse_error_rate": parse_errors_total / features_evaluated if features_evaluated > 0 else 0,
        "total_time_seconds": total_elapsed,
        "avg_chunk_time_seconds": avg_chunk_time,
        "avg_llm_time_seconds": avg_llm_time,
        "throughput_records_per_second": n_total / total_elapsed if total_elapsed > 0 else 0,
    }

    with open(stats_json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved run stats: {stats_json_path}")

    # Save per-chunk timing
    chunk_timing_df = pd.DataFrame({
        "chunk_idx": list(range(len(chunk_times))),
        "chunk_time_sec": chunk_times,
        "llm_time_sec": llm_times
    })
    chunk_timing_df.to_csv(per_chunk_csv_path, index=False)
    print(f"  ✓ Saved per-chunk times: {per_chunk_csv_path}")

    # Print summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Model:              Mixtral (few-shots)")
    print(f"Records processed:  {n_total:,}")
    print(f"Features evaluated: {features_evaluated:,}")
    print(f"Chunks processed:   {n_chunks}")
    print(f"LLM success:        {llm_success_chunks}/{n_chunks} ({llm_success_chunks/n_chunks*100:.1f}%)")
    print(f"Parse errors:       {parse_errors_total:,} ({stats['parse_error_rate']*100:.2f}%)")
    print(f"Total time:         {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print(f"Avg chunk time:     {avg_chunk_time:.2f}s")
    print(f"Avg LLM time:       {avg_llm_time:.2f}s")
    print(f"Throughput:         {stats['throughput_records_per_second']:.2f} records/sec")
    print("="*80)

if __name__ == "__main__":
    run_pipeline(MANIPULATED_CSV, CORRECT_CSV, MASKS_CSV, max_records=MAX_RECORDS)
