#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution for Twitter Bot Detection via Google Gemini
------------------------------------------------------------------------------
Info-enhanced version with rubric for better decision-making guidance.
Adapted from adult income dataset info pipeline for Twitter bot data.

Key features:
- Process Twitter bot profiles: clean vs changed (potentially manipulated)
- 1-to-1 mapping: each clean bot has one corresponding changed version
- Feed records in chunks to the LLM
- FEATURE-LEVEL analysis: each changed feature gets independent decision
- Determine if changes are INTENTIONAL bot-to-human evasion (1) or UNINTENTIONAL (-1)
- Enhanced prompt with detailed RUBRIC for better guidance
- Accept multiple JSON response formats from Gemini
- Configurable MAX_RECORDS for testing or full runs
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

# =========================
# ======== CONFIG =========
# =========================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_dataset.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_dataset_clean.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_mask.csv"

# Outputs
OUT_DIR = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/info-gemini"
OUT_LABELS_CSV = os.path.join(OUT_DIR, "intent_labels.csv")
OUT_EXPL_CSV   = os.path.join(OUT_DIR, "intent_explanations.csv")

# Twitter bot schema (22 columns)
TWITTER_BOT_COLS = [
    "user_id", "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "verified", "protected",
    "default_profile", "default_profile_image", "geo_enabled",
    "profile_use_background_image", "has_created_date",
    "description_length", "screen_name_length", "name_length",
    "has_description", "has_location", "label"
]
# Note: Removed derived features (followers_friends_ratio, tweets_per_friend, followers_per_tweet)
# as they are computed from other features and should not be modified directly

# Processing configuration
CHUNK_SIZE = 10  # Process 10 records at a time
MAX_RECORDS = None  # Set to a number (e.g., 100) for testing, None for full dataset

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
# ===== GEMINI CALL =======
# =========================
def call_gemini(model, prompt: str):
    """Call Gemini API and return response text."""
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
                'max_output_tokens': 4096,  # Large output buffer
                'candidate_count': 1,
            },
            safety_settings=safety_settings,
            request_options={'timeout': 180}  # 3 minute timeout
        )
        
        # Check if response has valid content
        if not response.candidates:
            return ""
        
        candidate = response.candidates[0]
        
        # finish_reason: 1 = STOP (normal), 2 = MAX_TOKENS, 3 = SAFETY, 4 = RECITATION, 5 = OTHER
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
- **Verification gaming**: Changing verified 
- **Profile completeness**: Adding description, adding location
- **Visual authenticity**: Changing default_profile or default_profile_imageto appear customized
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
    Create info-enhanced prompt for a chunk of records.
    Includes rubric for better decision-making guidance.
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
    
    prompt = (
        f"{ROLE_BLOCK}\n{RUBRIC}\n{sys_rules}\n\n"
        f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Analyze each changed feature using the rubric and return your feature-level decisions in JSON format now."
    )
    return prompt

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str, max_records: Optional[int] = None):
    """Main pipeline to process Twitter bot intent attribution."""
    t0 = time.perf_counter()

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME_LITE)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)

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

    # Ensure columns & order
    for df, name in [(manip, "manipulated"), (correct, "clean"), (masks, "masks")]:
        missing = [c for c in TWITTER_BOT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        df[:] = df[TWITTER_BOT_COLS]

    # Verify same number of records
    n_records = len(correct)
    if len(manip) != n_records or len(masks) != n_records:
        raise ValueError(f"Mismatch in record counts: clean={n_records}, manip={len(manip)}, masks={len(masks)}")
    
    # Apply max_records limit if specified
    if max_records is not None and max_records < n_records:
        print(f"\n⚠️  Limiting to first {max_records} records (out of {n_records} total)")
        manip = manip.iloc[:max_records]
        correct = correct.iloc[:max_records]
        masks = masks.iloc[:max_records]
        n_records = max_records
    
    print(f"\nProcessing {n_records} records in chunks of {CHUNK_SIZE}")

    # Prepare outputs
    out_mask_df = masks.copy(deep=True)  # Will store decisions (1/-1) where mask=1
    explanations_rows: List[Dict[str, str]] = []

    # Timing & status trackers
    chunk_times: List[float] = []
    llm_times: List[float] = []
    llm_success_chunks = 0
    fallback_chunks = 0
    llm_status_hist = {
        "ok_feature_decisions": 0,
        "api_error": 0,
        "parse_error": 0
    }

    # Process in chunks
    n_chunks = (n_records + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks", ncols=100):
        chunk_t0 = time.perf_counter()
        
        # Get chunk boundaries
        start_idx = chunk_idx * CHUNK_SIZE
        end_idx = min(start_idx + CHUNK_SIZE, n_records)
        chunk_size = end_idx - start_idx
        
        # Prepare chunk data
        clean_rows = [row_to_dict(correct, i) for i in range(start_idx, end_idx)]
        changed_rows = [row_to_dict(manip, i) for i in range(start_idx, end_idx)]
        mask_rows = [mask_row_to_dict(masks, i) for i in range(start_idx, end_idx)]
        global_ids = list(range(start_idx, end_idx))
        
        # Build prompt
        prompt_text = make_chunk_prompt(clean_rows, changed_rows, mask_rows, global_ids)
        
        # Call LLM
        obj = None
        llm_text = ""
        l0 = time.perf_counter()
        try:
            llm_text = call_gemini_with_fallback(model, model_flash, prompt_text)
            obj = parse_llm_json(llm_text)
        except Exception:
            llm_status_hist["api_error"] += 1
            obj = None
            llm_text = ""
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)
        
        # Parse decisions
        feature_decisions: Dict[Tuple[int, str], Tuple[int, str]] = {}
        parsed = None
        
        if isinstance(obj, dict):
            parsed = decisions_from_obj(obj, global_ids)
            if parsed is not None:
                llm_success_chunks += 1
                llm_status_hist["ok_feature_decisions"] += 1
                feature_decisions = parsed
        
        # Fallback if parsing failed
        if parsed is None:
            fallback_chunks += 1
            llm_status_hist["parse_error"] += 1
            # Default all changed features to -1 (unintentional)
            for rid in global_ids:
                for col in TWITTER_BOT_COLS:
                    local_idx = rid - start_idx
                    if col not in ("user_id", "label") and mask_rows[local_idx][col] == "1":
                        feature_decisions[(rid, col)] = (-1, "llm_fallback")
        
        # Fill output mask and explanations
        for local_idx, rid in enumerate(global_ids):
            # Update output mask: where original mask=1, put the decision (1 or -1)
            for col in TWITTER_BOT_COLS:
                if str(masks.at[rid, col]).strip() in ("1", "1.0"):
                    # Get feature-specific decision
                    decision, reason = feature_decisions.get((rid, col), (-1, "guard_fallback"))
                    out_mask_df.at[rid, col] = str(decision)
                    
                    # Create explanation for this specific feature
                    explanations_rows.append({
                        "row_id": rid,
                        "column": col,
                        "changed_value": changed_rows[local_idx][col],
                        "original_value": clean_rows[local_idx][col],
                        "mask": 1,
                        "diagnosis": str(decision),
                        "rule": "intent_attribution_info",
                        "details": json.dumps({"reason": reason, "source": "llm" if parsed else "fallback"}, ensure_ascii=False),
                        "mapping_status": "matched",
                        "mapping_score": ""
                    })
                else:
                    out_mask_df.at[rid, col] = "0"
        
        chunk_t1 = time.perf_counter()
        chunk_times.append(chunk_t1 - chunk_t0)
    
    # Save outputs
    print(f"\nSaving results...")
    out_mask_df.to_csv(OUT_LABELS_CSV, index=False)
    print(f"  - Intent labels: {OUT_LABELS_CSV}")
    
    expl_df = pd.DataFrame(explanations_rows, columns=[
        "row_id", "column", "changed_value", "original_value", "mask",
        "diagnosis", "rule", "details", "mapping_status", "mapping_score"
    ])
    expl_df.to_csv(OUT_EXPL_CSV, index=False)
    print(f"  - Intent explanations: {OUT_EXPL_CSV}")
    
    # Summary statistics
    flat_vals = out_mask_df.replace({"1.0": "1", "-1.0": "-1"}).values.flatten().tolist()
    counts = pd.Series(flat_vals).value_counts().to_dict()
    
    print("\n" + "="*60)
    print("GENERATED MASK COUNTS")
    print("="*60)
    for k in ["1", "-1", "0"]:
        print(f"{k:>3}: {counts.get(k, 0):>8,}")
    print(f"{'ALL':>3}: {len(flat_vals):>8,}")
    
    print("\n" + "="*60)
    print("LLM STATUS HISTOGRAM")
    print("="*60)
    for status, count in llm_status_hist.items():
        print(f"{status:>25}: {count:>6}")
    
    # Runtime stats
    t1 = time.perf_counter()
    stats_payload = {
        "model": MODEL_NAME,
        "dataset": "twitter_bot",
        "version": "info_enhanced_with_rubric",
        "total_records": n_records,
        "chunk_size": CHUNK_SIZE,
        "num_chunks": n_chunks,
        "max_records_limit": max_records,
        "total_time_sec": t1 - t0,
        "avg_chunk_time_sec": float(pd.Series(chunk_times).mean()) if chunk_times else None,
        "median_chunk_time_sec": float(pd.Series(chunk_times).median()) if chunk_times else None,
        "total_llm_time_sec": float(pd.Series(llm_times).sum()) if llm_times else None,
        "avg_llm_time_sec": float(pd.Series(llm_times).mean()) if llm_times else None,
        "llm_success_chunks": llm_success_chunks,
        "fallback_chunks": fallback_chunks,
        "llm_success_rate": f"{100*llm_success_chunks/n_chunks:.1f}%" if n_chunks > 0 else "N/A",
        "llm_status_hist": llm_status_hist,
        "decision_counts": {
            "intentional (1)": counts.get("1", 0),
            "unintentional (-1)": counts.get("-1", 0),
            "unchanged (0)": counts.get("0", 0)
        },
        "outputs": {
            "intent_labels_csv": OUT_LABELS_CSV,
            "intent_explanations_csv": OUT_EXPL_CSV,
            "per_chunk_times_csv": per_chunk_csv_path
        }
    }
    
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)
    print(f"  - Run stats: {stats_json_path}")
    
    # Save per-chunk timing
    per_chunk_df = pd.DataFrame({
        "chunk_id": list(range(n_chunks)),
        "chunk_time_sec": chunk_times,
        "llm_time_sec": llm_times
    })
    per_chunk_df.to_csv(per_chunk_csv_path, index=False)
    print(f"  - Per-chunk times: {per_chunk_csv_path}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print(f"Total time: {t1-t0:.2f}s")
    print(f"LLM success rate: {llm_success_chunks}/{n_chunks} chunks ({100*llm_success_chunks/n_chunks:.1f}%)")
    

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Twitter Bot Intent Attribution Pipeline (Info-Enhanced)")
    parser.add_argument("--manipulated", default=MANIPULATED_CSV, help="Path to manipulated CSV")
    parser.add_argument("--clean", default=CORRECT_CSV, help="Path to clean CSV")
    parser.add_argument("--masks", default=MASKS_CSV, help="Path to masks CSV")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS, 
                       help="Maximum records to process (default: all)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("TWITTER BOT INTENT ATTRIBUTION PIPELINE")
    print("Info-Enhanced with Rubric (Gemini)")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE} records")
    print(f"Max records: {args.max_records if args.max_records else 'ALL'}")
    print(f"Output directory: {OUT_DIR}")
    print("="*60)
    
    run_pipeline(args.manipulated, args.clean, args.masks, args.max_records)
