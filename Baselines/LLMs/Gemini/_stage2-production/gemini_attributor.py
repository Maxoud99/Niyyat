#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-Based Cell-Level Attribution using Google Gemini
-----------------------------------------------------
Given a record known to be erroneous, ask Gemini to identify which specific
cells are likely erroneous and provide reasoning.

Approach:
1. Sample 10 clean records as reference
2. For each erroneous record, pass it with the clean references to Gemini
3. Ask Gemini to identify suspicious cells and explain why
4. Parse the response to get cell-level suspicion scores

Based on the intent attribution pipeline syntax from:
/home/mohamed/error_injector/llms_baseline/adult_income_dataset/intent-attribution/intent_attribution_pipeline-gemini.py
"""

from __future__ import annotations
import json
import os
import sys
import time
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm
import google.generativeai as genai

# Import API key from config
# From: llms_baseline/error_detection_system/src/detectors/stage2/
# To:   / (root) - need to go up 5 levels
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
MODEL_NAME = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro

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
def call_gemini(model, prompt: str, timeout: int = 60) -> Optional[str]:
    """
    Call Gemini API and return response text.
    Uses safety settings to avoid content policy blocks.

    Returns "" for a legitimate empty/filtered response (not retried, since
    the same content hits the same filter on any model), or None if the
    call itself raised (signals the caller to fall back to the next model
    tier -- see call_gemini_with_fallback).
    """
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
            request_options={'timeout': timeout}
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
        # Signals the caller to fall back to the next model tier.
        return None


def call_gemini_with_fallback(model_lite, model_flash, prompt: str, timeout: int = 60) -> str:
    """Cost policy (2026-06-26): try flash-lite first, fall back to flash on
    failure, then give up. Never escalates to gemini-2.5-pro."""
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

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "ROLE\n"
    "You are a precise data quality assistant. Your task is to identify which specific CELLS "
    "in an erroneous record are likely to contain errors, and explain why each cell is suspicious.\n"
    "You will be given:\n"
    "  - 10 CLEAN reference records (known to be correct)\n"
    "  - 1 ERRONEOUS record (known to contain errors)\n"
    "  - Feature names and dataset information\n\n"
    "Your job: Identify the most suspicious cells in the erroneous record by comparing it "
    "to the clean records and domain knowledge."
)

def make_cell_attribution_prompt(
    clean_refs: List[Dict[str, str]],
    erroneous_row: Dict[str, str],
    columns: List[str],
    dataset_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Build prompt for Gemini to identify erroneous cells in a single record.
    
    Parameters
    ----------
    clean_refs : List[Dict[str, str]]
        10 clean reference records
    erroneous_row : Dict[str, str]
        The erroneous record to analyze
    columns : List[str]
        Feature names
    dataset_info : Optional[Dict[str, str]]
        Additional dataset context (e.g., feature descriptions, value ranges)
    
    Returns
    -------
    str
        Formatted prompt for Gemini
    """
    sys_rules = (
        "RULES:\n"
        "1. Analyze the erroneous record cell-by-cell\n"
        "2. Compare each cell value to the clean reference records\n"
        "3. Consider: statistical outliers, impossible values, inconsistencies, rare combinations\n"
        "4. Assign suspicion score 0.0-1.0 for EACH cell (0.0=clean, 1.0=definitely erroneous)\n"
        "5. Provide brief reasoning for cells with suspicion > 0.3\n"
        "6. Return STRICT JSON format (no extra text):\n"
        '   {"cells": [\n'
        '      {"feature": "age", "value": "999", "suspicion": 0.95, "reason": "impossible age value"},\n'
        '      {"feature": "workclass", "value": "Private", "suspicion": 0.1, "reason": "normal value"},\n'
        "      ...\n"
        "   ]}\n"
        "7. Include ALL features in the response, even if suspicion is 0.0"
    )
    
    # Build dataset context
    context = ""
    if dataset_info:
        context = "\nDATASET CONTEXT:\n" + json.dumps(dataset_info, indent=2) + "\n"
    
    bundle = {
        "columns": columns,
        "clean_references": clean_refs,
        "erroneous_record": erroneous_row
    }
    
    prompt = (
        f"{ROLE_BLOCK}\n\n{sys_rules}\n{context}\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Return the JSON object now (strict format, all features included)."
    )
    return prompt

# =========================
# ==== MAIN ATTRIBUTOR ====
# =========================
class GeminiCellAttributor:
    """
    LLM-based cell-level error attribution using Google Gemini.
    
    For each erroneous record:
    1. Sample 10 clean records as reference
    2. Ask Gemini to identify suspicious cells
    3. Parse response to get cell-level suspicion scores
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = MODEL_NAME,
        n_clean_refs: int = 10,
        random_seed: int = 42,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        api_key : Optional[str]
            Gemini API key (defaults to env variable)
        model_name : str
            Gemini model to use
        n_clean_refs : int
            Number of clean reference records to provide (default: 10)
        random_seed : int
            Random seed for sampling clean records
        verbose : bool
            Print progress information
        """
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name
        self.n_clean_refs = n_clean_refs
        self.random_seed = random_seed
        self.verbose = verbose

        # Configure Gemini. Cost policy (2026-06-26): always flash-lite
        # first, fall back to flash on failure, never gemini-2.5-pro --
        # see call_gemini_with_fallback.
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash-lite")
        self.model_flash = genai.GenerativeModel("gemini-2.5-flash")
        
        # Storage
        self.X_clean = None
        self.columns = None
        self.dataset_info = None
    
    def fit(
        self,
        X_clean: pd.DataFrame,
        dataset_info: Optional[Dict[str, str]] = None
    ):
        """
        Store clean reference data for comparison.
        
        Parameters
        ----------
        X_clean : pd.DataFrame
            Clean records to use as reference
        dataset_info : Optional[Dict[str, str]]
            Dataset context (feature descriptions, value ranges, etc.)
        """
        self.X_clean = X_clean.copy()
        self.columns = list(X_clean.columns)
        self.dataset_info = dataset_info
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GEMINI CELL ATTRIBUTOR - FITTED")
            print(f"{'='*60}")
            print(f"Model: {self.model_name}")
            print(f"Clean references available: {len(X_clean)}")
            print(f"Features: {len(self.columns)}")
            print(f"Will sample {self.n_clean_refs} references per query")
    
    def attribute(
        self,
        X_erroneous: pd.DataFrame,
        batch_size: int = 1,
        timeout_per_record: int = 60
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Attribute errors at cell level for erroneous records.
        
        Parameters
        ----------
        X_erroneous : pd.DataFrame
            Erroneous records to analyze
        batch_size : int
            Number of records to process at once (currently only supports 1)
        timeout_per_record : int
            Timeout for each Gemini API call (seconds)
        
        Returns
        -------
        cell_suspicions : pd.DataFrame
            Suspicion scores (0.0-1.0) for each cell [n_records × n_features]
        explanations : List[Dict]
            Per-record explanations with cell-level reasoning
        """
        if self.X_clean is None:
            raise ValueError("Must call fit() before attribute()")
        
        if batch_size != 1:
            raise NotImplementedError("Currently only batch_size=1 is supported")
        
        n_erroneous = len(X_erroneous)
        n_features = len(self.columns)
        
        # Initialize output
        cell_suspicions = pd.DataFrame(
            np.zeros((n_erroneous, n_features)),
            columns=self.columns,
            index=X_erroneous.index
        )
        explanations = []
        
        # Timing stats
        api_times = []
        successful_calls = 0
        failed_calls = 0
        
        # Sample clean references once (reused for all queries for consistency)
        np.random.seed(self.random_seed)
        if len(self.X_clean) >= self.n_clean_refs:
            clean_sample_indices = np.random.choice(
                len(self.X_clean),
                size=self.n_clean_refs,
                replace=False
            )
        else:
            # If not enough clean records, use all and sample with replacement
            clean_sample_indices = np.random.choice(
                len(self.X_clean),
                size=self.n_clean_refs,
                replace=True
            )
        
        clean_refs = [
            row_to_dict(self.X_clean, idx, self.columns)
            for idx in clean_sample_indices
        ]
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PROCESSING {n_erroneous} ERRONEOUS RECORDS")
            print(f"{'='*60}")
        
        # Process each erroneous record
        for i in tqdm(range(n_erroneous), desc="Attributing cells", disable=not self.verbose):
            record_idx = X_erroneous.index[i]
            erroneous_row = row_to_dict(X_erroneous, record_idx, self.columns)
            
            # Build prompt
            prompt = make_cell_attribution_prompt(
                clean_refs=clean_refs,
                erroneous_row=erroneous_row,
                columns=self.columns,
                dataset_info=self.dataset_info
            )
            
            # Call Gemini
            t0 = time.perf_counter()
            try:
                llm_text = call_gemini_with_fallback(self.model, self.model_flash, prompt, timeout=timeout_per_record)
                t1 = time.perf_counter()
                api_times.append(t1 - t0)
                
                # Parse response
                obj = parse_llm_json(llm_text)
                
                if obj and isinstance(obj, dict) and "cells" in obj:
                    cells_list = obj["cells"]
                    
                    # Extract suspicion scores
                    record_explanations = {}
                    for cell_info in cells_list:
                        if not isinstance(cell_info, dict):
                            continue
                        
                        feature = cell_info.get("feature", "")
                        suspicion = cell_info.get("suspicion", 0.0)
                        reason = cell_info.get("reason", "")
                        
                        if feature in self.columns:
                            # Store suspicion score
                            cell_suspicions.at[record_idx, feature] = float(suspicion)
                            
                            # Store explanation if suspicion > 0.3
                            if suspicion > 0.3 and reason:
                                record_explanations[feature] = {
                                    "suspicion": suspicion,
                                    "reason": reason,
                                    "value": cell_info.get("value", "")
                                }
                    
                    explanations.append({
                        "record_index": record_idx,
                        "cells": record_explanations,
                        "raw_response": llm_text[:500]  # First 500 chars for debugging
                    })
                    
                    successful_calls += 1
                else:
                    # Failed to parse - fallback to uniform suspicion
                    cell_suspicions.iloc[i, :] = 0.5
                    explanations.append({
                        "record_index": record_idx,
                        "cells": {},
                        "error": "parse_failed",
                        "raw_response": llm_text[:500]
                    })
                    failed_calls += 1
                    
            except Exception as e:
                # API error - fallback to uniform suspicion
                cell_suspicions.iloc[i, :] = 0.5
                explanations.append({
                    "record_index": record_idx,
                    "cells": {},
                    "error": str(e),
                    "raw_response": ""
                })
                failed_calls += 1
                api_times.append(timeout_per_record)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ATTRIBUTION COMPLETE")
            print(f"{'='*60}")
            print(f"Successful API calls: {successful_calls}/{n_erroneous}")
            print(f"Failed calls: {failed_calls}/{n_erroneous}")
            if api_times:
                print(f"Avg API time: {np.mean(api_times):.2f}s")
                print(f"Total time: {np.sum(api_times):.2f}s")
        
        return cell_suspicions, explanations


# =========================
# ======== TESTING ========
# =========================
if __name__ == "__main__":
    """Quick test of the Gemini attributor."""
    
    # Example with Adult Income dataset
    print("Testing Gemini Cell Attributor...")
    
    # Load sample data
    dataset_path = "/home/mohamed/error_injector/llms_baseline/error_detection_system/datasets/adult_income/labeled_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Skipping test.")
    else:
        df = pd.read_csv(dataset_path)
        
        print(f"Loaded dataset: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Split into clean and erroneous (label column is 'is_erroneous')
        clean_records = df[df['is_erroneous'] == 0].drop(columns=['is_erroneous']).reset_index(drop=True)
        erroneous_records = df[df['is_erroneous'] == 1].drop(columns=['is_erroneous']).reset_index(drop=True)
        
        print(f"Clean records: {len(clean_records)}, Erroneous records: {len(erroneous_records)}")
        
        # Take small sample for testing
        clean_sample = clean_records.head(50)
        erroneous_sample = erroneous_records.head(3)  # Test with 3 records
        
        print(f"Testing with {len(clean_sample)} clean refs and {len(erroneous_sample)} erroneous records")
        
        # Create attributor
        attributor = GeminiCellAttributor(
            n_clean_refs=10,
            verbose=True
        )
        
        # Dataset info (optional context)
        dataset_info = {
            "name": "Adult Income Dataset",
            "description": "Census data predicting income >50K",
            "age": "numeric, typically 17-90",
            "education-num": "numeric, years of education (1-16)",
            "hours-per-week": "numeric, typically 1-99",
            "capital-gain": "numeric, typically 0-99999",
            "capital-loss": "numeric, typically 0-4356"
        }
        
        # Fit on clean records
        attributor.fit(clean_sample, dataset_info=dataset_info)
        
        # Attribute errors in erroneous records
        suspicions, explanations = attributor.attribute(erroneous_sample)
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print("\nCell Suspicion Scores:")
        print(suspicions)
        
        print("\nSuspicion DataFrame (detailed):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        for idx in suspicions.index:
            non_zero = suspicions.loc[idx][suspicions.loc[idx] > 0]
            if len(non_zero) > 0:
                print(f"\nRecord {idx} (non-zero suspicions):")
                print(non_zero)
        
        print("\nExplanations:")
        for expl in explanations:
            print(f"\nRecord {expl['record_index']}:")
            if 'error' in expl:
                print(f"  ERROR: {expl['error']}")
                print(f"  Raw response (first 200 chars): {expl.get('raw_response', '')[:200]}")
            else:
                for feature, info in expl.get('cells', {}).items():
                    print(f"  {feature}: {info['suspicion']:.2f} - {info['reason']}")
