#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: LLM-Based Cell-Level Attribution - Twitter Bot Dataset
----------------------------------------------------------------
Evaluate Gemini's cell-level error attribution on Twitter bot dataset erroneous records.

Approach:
- Sample 10 fixed clean reference records at the start
- Process erroneous records in chunks (default: 5 per chunk)
- For each chunk, pass 5 erroneous + 10 clean records to Gemini in a single API call
- Save results: cell_suspicions.csv, predicted_masks.csv, explanations.json, metrics.json

Usage:
    python src/run_stage2_llm_twitter.py --chunk-size 5
    python src/run_stage2_llm_twitter.py --chunk-size 5 --num-records 50  # Test on first 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

# Import API key from config
# From: llms_baseline/error_detection_system/src/detectors/stage2/
# To:   / (root)
# Need to go up 5 levels: stage2 -> detectors -> src -> error_detection_system -> llms_baseline -> root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Set it in config.py or as environment variable.")

# =========================
# ======== CONFIG =========
# =========================
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

# Get absolute paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets/twitter_bot-llm/combined_dataset/combined_data_n4000.csv")
MASKS_PATH = os.path.join(PROJECT_ROOT, "datasets/twitter_bot-llm/combined_dataset/combined_mask_n4000.csv")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs/stage2/LLMs/twitterBot-dataset")

SUSPICION_THRESHOLD = 0.5  # Convert suspicion scores to binary masks (lowered from 0.5 due to subtle errors)

# =========================
# ====== UTILITIES ========
# =========================
def to_str(x) -> str:
    """Convert value to string, handling NaN."""
    if pd.isna(x):
        return ""
    return str(x)

def row_to_dict(df: pd.DataFrame, idx: int, columns: List[str]) -> Dict[str, str]:
    """Convert a DataFrame row to a dictionary."""
    return {c: to_str(df.at[idx, c]) for c in columns}

# =========================
# ===== GEMINI CALL =======
# =========================
def call_gemini(model, prompt: str, timeout: int = 120) -> "str | None":
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
                'temperature': 0,
                'max_output_tokens': 16384,  # gemini-2.5-pro is more verbose, needs larger limit
                'candidate_count': 1,
            },
            safety_settings=safety_settings,
            request_options={'timeout': timeout}
        )
        
        if not response.candidates:
            return ""
        
        candidate = response.candidates[0]
        
        # Accept STOP (1) or MAX_TOKENS (2) as valid
        if candidate.finish_reason.value not in (1, 2):
            return ""
        
        if not candidate.content or not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            return ""
        
        try:
            text = candidate.content.parts[0].text
            return text if text else ""
        except (AttributeError, IndexError):
            return ""
            
    except Exception as e:
        print(f"API Error: {e}")
        return None  # signals caller: fall back to next model tier


def call_gemini_with_fallback(model_lite, model_flash, prompt: str, timeout: int = 120) -> str:
    """Cost policy (2026-06-26): flash-lite first, then flash, then give up.
    Never escalates to gemini-2.5-pro."""
    result = call_gemini(model_lite, prompt, timeout=timeout)
    if result is not None:
        return result
    result = call_gemini(model_flash, prompt, timeout=timeout)
    if result is not None:
        return result
    return ""

# =========================
# ======= PARSING =========
# =========================
def parse_llm_json(text: str) -> dict:
    """Robustly extract first top-level JSON from model text."""
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()

    # Remove common wrappers
    for junk in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(junk, "")
    s = s.strip()

    # If whole output is quoted JSON, unescape
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

    # Extract first balanced JSON fragment
    def extract_fragment(txt: str) -> str:
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

    frag = extract_fragment(s)
    if not frag:
        return None
    try:
        return json.loads(frag)
    except Exception:
        try:
            import ast
            obj = ast.literal_eval(frag)
            if isinstance(obj, (dict, list)):
                return obj
        except Exception:
            return None
    return None

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "ROLE\n"
    "You are a data quality auditor analyzing Twitter bot dataset for detecting erroneous cells. "
    "IMPORTANT: The erroneous records you receive ARE CONFIRMED to contain errors - your job is to find ALL of them.\n\n"
    "You will be given:\n"
    "  - Clean reference records (known correct examples)\n"
    "  - Erroneous records (CONFIRMED to have one or more errors per record)\n"
    "  - Feature descriptions with valid ranges and constraints\n\n"
    "Your task: Thoroughly examine EVERY cell in EVERY erroneous record to identify ALL suspicious values. "
    "Focus and think critically - each erroneous record contains at least one error that needs to be found."
)

def make_batch_prompt(
    clean_refs: List[Dict[str, str]],
    erroneous_rows: List[Dict[str, str]],
    erroneous_indices: List[int],
    columns: List[str],
    dataset_info: Dict[str, str]
) -> str:
    """
    Build prompt for Gemini to identify erroneous cells in a batch of records.
    
    Parameters
    ----------
    clean_refs : List[Dict[str, str]]
        n clean reference records
    erroneous_rows : List[Dict[str, str]]
        Batch of erroneous records to analyze
    erroneous_indices : List[int]
        Original dataset indices of erroneous records
    columns : List[str]
        Feature names
    dataset_info : Dict[str, str]
        Dataset context
    
    Returns
    -------
    str
        Formatted prompt for Gemini
    """
    sys_rules = (
        "ANALYSIS INSTRUCTIONS:\n\n"
        "1. MINDSET: These records are CONFIRMED ERRONEOUS - at least one error exists in each. Be conservative and precise.\n\n"
        "2. EXAMINATION PROCESS:\n"
        "   a) Compare each cell to clean reference records\n"
        "   b) Check if value violates stated constraints (e.g., exceeds Twitter limits)\n"
        "   c) Verify consistency with related fields (e.g., has_description must match description_length pattern)\n"
        "   d) Notice unusual patterns: extreme ratios, impossible combinations, atypical values\n"
        "   e) Only flag cells you are confident are erroneous - precision over recall\n\n"
        "3. SUSPICION SCORING (0.0 to 1.0):\n"
        "   • 0.0-0.2: Perfectly normal - matches clean reference patterns exactly\n"
        "   • 0.3-0.4: Slightly unusual - at edge of normal range, minor inconsistency\n"
        "   • 0.5-0.6: Moderately suspicious - uncommon value or unclear relationship\n"
        "   • 0.7-0.8: Highly suspicious - rare pattern, likely manipulated\n"
        "   • 0.9-1.0: Almost certainly erroneous - violates hard constraint or impossible value\n\n"
        "4. KEY CHECKS:\n"
        "   • Hard limits: name_length ≤50, screen_name_length ≤15, description_length ≤160\n"
        "   • Logical consistency: If has_description=0 then description_length should be 0\n"
        "   • Relational coherence: verified=1 typically with high followers, extreme friend/follower ratios\n"
        "   • Bot manipulation markers: Very high friends_count, low engagement despite high counts\n\n"
        "5. OUTPUT FORMAT (strict JSON, no extra text):\n"
        '   {"records": [\n'
        '     {"record_id": 0, "cells": [\n'
        '       {"feature": "followers_count", "value": "15000", "suspicion": 0.1, "reason": "normal"},\n'
        '       {"feature": "name_length", "value": "250", "suspicion": 1.0, "reason": "exceeds Twitter limit of 50"},\n'
        "       ...\n"
        "     ]},\n"
        "     ...\n"
        "   ]}\n\n"
        "6. MANDATORY: Include ALL features for ALL records with their suspicion scores (even if 0.0).\n"
        "   Provide reasoning for any cell with suspicion ≥ 0.5.\n\n"
        "7. BE CONSERVATIVE: Only assign high suspicion scores (≥ 0.5) when you have strong evidence.\n"
        "   False positives are worse than missed errors. When in doubt, use lower scores."
    )
    
    context = "\nDATASET CONTEXT:\n" + json.dumps(dataset_info, indent=2) + "\n"
    
    erroneous_with_ids = [
        {
            "record_id": i,
            "dataset_index": erroneous_indices[i],
            "data": erroneous_rows[i]
        }
        for i in range(len(erroneous_rows))
    ]
    
    bundle = {
        "columns": columns,
        "clean_references": clean_refs,
        "erroneous_records": erroneous_with_ids
    }
    
    prompt = (
        f"{ROLE_BLOCK}\n\n{sys_rules}\n{context}\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Return the JSON object now (strict format, all features and records included)."
    )
    return prompt

# =========================
# ==== MAIN EVALUATION ====
# =========================
def evaluate_llm_attribution(
    dataset_path: str,
    masks_path: str,
    output_dir: str,
    chunk_size: int = 5,
    num_records: int = None,
    n_clean_refs: int = 10,
    random_seed: int = 42
):
    """
    Evaluate Gemini's cell-level attribution on all erroneous records.
    
    Parameters
    ----------
    dataset_path : str
        Path to labeled dataset
    masks_path : str
        Path to ground truth cell masks
    output_dir : str
        Directory to save results
    chunk_size : int
        Number of erroneous records per API call
    num_records : int
        Number of erroneous records to process (None = all)
    n_clean_refs : int
        Number of clean reference records
    random_seed : int
        Random seed for sampling
    """
    print("="*80)
    print("STAGE 2: LLM-BASED CELL ATTRIBUTION - FULL EVALUATION")
    print("="*80)
    print(f"Model: {MODEL_NAME_LITE} (fallback: {MODEL_NAME_FLASH})")
    print(f"Chunk size: {chunk_size} erroneous records per API call")
    print(f"Clean references: {n_clean_refs}")
    print(f"Suspicion threshold: {SUSPICION_THRESHOLD}")
    print()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME_LITE)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(dataset_path)
    masks_gt = pd.read_csv(masks_path)
    
    # Split into clean and erroneous
    clean_records = df[df['is_erroneous'] == 0].drop(columns=['is_erroneous']).reset_index(drop=True)
    erroneous_records = df[df['is_erroneous'] == 1].drop(columns=['is_erroneous']).reset_index(drop=True)
    erroneous_original_indices = df[df['is_erroneous'] == 1].index.tolist()
    
    columns = list(clean_records.columns)
    
    print(f"Clean records: {len(clean_records)}")
    print(f"Erroneous records: {len(erroneous_records)}")
    
    # Limit number of records if specified
    if num_records:
        erroneous_records = erroneous_records.head(num_records)
        erroneous_original_indices = erroneous_original_indices[:num_records]
        print(f"Processing first {num_records} erroneous records")
    
    n_erroneous = len(erroneous_records)
    n_features = len(columns)
    
    # Sample fixed clean references
    np.random.seed(random_seed)
    if len(clean_records) >= n_clean_refs:
        clean_sample_indices = np.random.choice(len(clean_records), size=n_clean_refs, replace=False)
    else:
        clean_sample_indices = np.random.choice(len(clean_records), size=n_clean_refs, replace=True)
    
    clean_refs = [row_to_dict(clean_records, idx, columns) for idx in clean_sample_indices]
    print(f"Sampled {len(clean_refs)} clean references (fixed for all queries)")
    
    # Dataset context - UNSUPERVISED: Only domain knowledge, no statistics
    dataset_info = {
        "name": "Twitter Bot Dataset",
        "description": "Twitter account features - bots may manipulate features to appear legitimate",
        "feature_descriptions": {
            "followers_count": "numeric, follower count (can be artificially inflated)",
            "friends_count": "numeric, number following (bots often follow many accounts)",
            "favourites_count": "numeric, number of likes given",
            "statuses_count": "numeric, total tweets posted",
            "listed_count": "numeric, appearances in public lists",
            "verified": "binary (0/1), Twitter verified badge",
            "protected": "binary (0/1), private account status",
            "default_profile": "binary (0/1), using default profile (1=default, 0=customized)",
            "default_profile_image": "binary (0/1), using default avatar (1=default, 0=custom)",
            "geo_enabled": "binary (0/1), location tagging enabled",
            "profile_use_background_image": "binary (0/1), custom background present",
            "has_created_date": "binary (0/1), creation timestamp exists (should always be 1)",
            "description_length": "numeric, bio length (0-160 chars per Twitter policy)",
            "screen_name_length": "numeric, @username length (1-15 chars per Twitter policy)",
            "name_length": "numeric, display name length (1-50 chars per Twitter policy)",
            "has_description": "binary (0/1), bio exists (1=has bio, 0=empty bio)",
            "has_location": "binary (0/1), location field filled"
        },
        "constraints": {
            "hard_limits": "name_length≤50, screen_name_length≤15, description_length≤160",
            "consistency_rules": "has_description=1 ↔ description_length>0, has_created_date should always be 1"
        }
    }
    
    # Initialize outputs
    cell_suspicions = pd.DataFrame(
        np.zeros((n_erroneous, n_features)),
        columns=columns
    )
    predicted_masks = pd.DataFrame(
        np.zeros((n_erroneous, n_features), dtype=int),
        columns=columns
    )
    all_explanations = []
    
    # Timing and statistics
    api_times = []
    successful_chunks = 0
    failed_chunks = 0
    
    # Process in chunks
    n_chunks = (n_erroneous + chunk_size - 1) // chunk_size
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {n_erroneous} RECORDS IN {n_chunks} CHUNKS")
    print(f"{'='*80}\n")
    
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_erroneous)
        
        # Get chunk data
        chunk_records = []
        chunk_orig_indices = []
        for i in range(start_idx, end_idx):
            chunk_records.append(row_to_dict(erroneous_records, i, columns))
            chunk_orig_indices.append(erroneous_original_indices[i])
        
        # Build prompt
        prompt = make_batch_prompt(
            clean_refs=clean_refs,
            erroneous_rows=chunk_records,
            erroneous_indices=chunk_orig_indices,
            columns=columns,
            dataset_info=dataset_info
        )
        
        # Call Gemini
        t0 = time.perf_counter()
        try:
            llm_text = call_gemini_with_fallback(model, model_flash, prompt, timeout=120)
            t1 = time.perf_counter()
            api_times.append(t1 - t0)
            
            # Parse response
            obj = parse_llm_json(llm_text)
            
            # Debug: Save raw response if parsing fails
            if not obj or not isinstance(obj, dict) or "records" not in obj:
                debug_file = os.path.join(output_dir, f"debug_failed_chunk_{chunk_idx}.txt")
                with open(debug_file, 'w') as f:
                    f.write(f"=== FAILED PARSING FOR CHUNK {chunk_idx} ===\n\n")
                    f.write(f"Parsed object: {obj}\n\n")
                    f.write(f"Raw LLM response:\n{llm_text}\n")
                print(f"⚠️  Parsing failed for chunk {chunk_idx}. Debug saved to: {debug_file}")
            
            if obj and isinstance(obj, dict) and "records" in obj:
                records_list = obj["records"]
                
                # Process each record in the response
                for record_data in records_list:
                    if not isinstance(record_data, dict):
                        continue
                    
                    record_id = record_data.get("record_id")
                    if record_id is None:
                        continue
                    
                    # Map to actual index in our arrays
                    actual_idx = start_idx + record_id
                    if actual_idx >= n_erroneous:
                        continue
                    
                    cells_list = record_data.get("cells", [])
                    record_explanations = {}
                    
                    for cell_info in cells_list:
                        if not isinstance(cell_info, dict):
                            continue
                        
                        feature = cell_info.get("feature", "")
                        suspicion = cell_info.get("suspicion", 0.0)
                        reason = cell_info.get("reason", "")
                        
                        if feature in columns:
                            # Store suspicion score
                            cell_suspicions.iloc[actual_idx, columns.index(feature)] = float(suspicion)
                            
                            # Convert to binary mask
                            predicted_masks.iloc[actual_idx, columns.index(feature)] = 1 if suspicion >= SUSPICION_THRESHOLD else 0
                            
                            # Store explanation if suspicion > 0.3
                            if suspicion > 0.3 and reason:
                                record_explanations[feature] = {
                                    "suspicion": suspicion,
                                    "reason": reason,
                                    "value": cell_info.get("value", "")
                                }
                    
                    all_explanations.append({
                        "record_index": actual_idx,
                        "dataset_index": chunk_orig_indices[record_id],
                        "cells": record_explanations
                    })
                
                successful_chunks += 1
            else:
                # Failed to parse - fallback to uniform suspicion
                for i in range(start_idx, end_idx):
                    cell_suspicions.iloc[i, :] = 0.5
                    # Apply threshold consistently (only flag if fallback suspicion >= threshold)
                    predicted_masks.iloc[i, :] = 1 if 0.5 >= SUSPICION_THRESHOLD else 0
                    all_explanations.append({
                        "record_index": i,
                        "dataset_index": erroneous_original_indices[i - start_idx],
                        "cells": {},
                        "error": "parse_failed"
                    })
                failed_chunks += 1
                
        except Exception as e:
            # API error - fallback
            for i in range(start_idx, end_idx):
                cell_suspicions.iloc[i, :] = 0.5
                # Apply threshold consistently (only flag if fallback suspicion >= threshold)
                predicted_masks.iloc[i, :] = 1 if 0.5 >= SUSPICION_THRESHOLD else 0
                all_explanations.append({
                    "record_index": i,
                    "dataset_index": erroneous_original_indices[i - start_idx],
                    "cells": {},
                    "error": str(e)
                })
            failed_chunks += 1
            api_times.append(120)
        
        # Save intermediate results every 10 chunks
        if (chunk_idx + 1) % 10 == 0:
            cell_suspicions.to_csv(os.path.join(output_dir, "cell_suspicions_partial.csv"), index=False)
            predicted_masks.to_csv(os.path.join(output_dir, "predicted_masks_partial.csv"), index=False)
    
    print(f"\n{'='*80}")
    print(f"API CALLS COMPLETE")
    print(f"{'='*80}")
    print(f"Successful chunks: {successful_chunks}/{n_chunks}")
    print(f"Failed chunks: {failed_chunks}/{n_chunks}")
    if api_times:
        print(f"Avg API time per chunk: {np.mean(api_times):.2f}s")
        print(f"Total API time: {np.sum(api_times)/60:.1f} minutes")
    
    # Calculate metrics against ground truth
    print(f"\n{'='*80}")
    print(f"CALCULATING METRICS")
    print(f"{'='*80}")
    
    # Get ground truth masks for erroneous records
    gt_masks = masks_gt.iloc[erroneous_original_indices].reset_index(drop=True)
    
    # Flatten for overall metrics
    y_true = gt_masks.values.flatten()
    y_pred = predicted_masks.values.flatten()
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(y_true)
    
    metrics = {
        "model": f"{MODEL_NAME_LITE}+fallback:{MODEL_NAME_FLASH}",
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_records": n_erroneous,
            "total_cells": n_erroneous * n_features,
            "true_error_cells": int(y_true.sum()),
            "predicted_error_cells": int(y_pred.sum())
        },
        "performance": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        },
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        },
        "api_stats": {
            "total_chunks": n_chunks,
            "successful_chunks": successful_chunks,
            "failed_chunks": failed_chunks,
            "avg_time_per_chunk": float(np.mean(api_times)) if api_times else 0,
            "total_time_seconds": float(np.sum(api_times)) if api_times else 0
        },
        "config": {
            "chunk_size": chunk_size,
            "n_clean_refs": n_clean_refs,
            "suspicion_threshold": SUSPICION_THRESHOLD,
            "random_seed": random_seed
        }
    }
    
    print(f"\nAccuracy:  {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1-Score:  {f1:.1%}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TN: {tn:,}")
    
    # Save results
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS TO {output_dir}")
    print(f"{'='*80}")
    
    cell_suspicions.to_csv(os.path.join(output_dir, "cell_suspicions.csv"), index=False)
    print(f"✓ Saved cell_suspicions.csv")
    
    predicted_masks.to_csv(os.path.join(output_dir, "predicted_masks.csv"), index=False)
    print(f"✓ Saved predicted_masks.csv")
    
    with open(os.path.join(output_dir, "explanations.json"), "w") as f:
        json.dump(all_explanations, f, indent=2)
    print(f"✓ Saved explanations.json")
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics.json")
    
    # Save the prompt template for reproducibility
    sample_prompt = make_batch_prompt(
        clean_refs=[{c: "sample" for c in columns}],
        erroneous_rows=[{c: "sample" for c in columns}],
        erroneous_indices=[0],
        columns=columns,
        dataset_info=dataset_info
    )
    with open(os.path.join(output_dir, "prompt_template.txt"), "w") as f:
        f.write(sample_prompt)
    print(f"✓ Saved prompt_template.txt")
    
    # Clean up partial files
    for fname in ["cell_suspicions_partial.csv", "predicted_masks_partial.csv"]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")

# =========================
# ========= CLI ===========
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: LLM-Based Cell-Level Attribution - Full Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Number of erroneous records per API call (default: 5)"
    )
    
    parser.add_argument(
        "--num-records",
        type=int,
        default=None,
        help="Number of erroneous records to process (default: all 999)"
    )
    
    parser.add_argument(
        "--n-clean-refs",
        type=int,
        default=50,
        help="Number of clean reference records (default: 10)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f"Base output directory (default: {OUTPUT_BASE_DIR})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling clean references (default: 42)"
    )
    
    args = parser.parse_args()
    
    evaluate_llm_attribution(
        dataset_path=DATASET_PATH,
        masks_path=MASKS_PATH,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        num_records=args.num_records,
        n_clean_refs=args.n_clean_refs,
        random_seed=args.seed
    )

if __name__ == "__main__":
    main()
