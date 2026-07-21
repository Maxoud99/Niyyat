#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Attribution — Three Gemini Variants — Mixed SOTA Twitter Bot Dataset
============================================================================

Runs three prompt strategies back-to-back on the Mixed SOTA TwiBot-20 dataset
(mixed error pipeline output, 1:1 dirty-to-clean ratio, feature-level analysis):

  Variant 1 — zero_shot : plain JSON bundle, no rubric, no examples
  Variant 2 — info      : adds per-feature DIFFS + decision rubric
  Variant 3 — few_shot  : info + three in-context examples

Key design:
  - Feature-level decisions: each changed feature in a record gets its own 1/-1
  - Blind mask: model sees 0/1 (was this feature changed?) — NOT the true intent
  - Ground-truth mask (mask_combined.csv): 1=intentional, -1=unintentional, 0=clean
  - Records are chunked (10 at a time) and sent together for efficiency
  - Metrics (accuracy, F1, etc.) computed at feature-level after each variant

Dataset
-------
  Dirty  : mixed_error_pipeline_twitter/output/twibot20_phase2_final.csv
  Clean  : mixed_error_pipeline_twitter/output/twibot20_clean.csv
  Mask   : mixed_error_pipeline_twitter/output/mask_combined.csv

Usage
-----
  python run_gemini_twitterbot_mixed_sota.py --n_records 100   # quick test
  python run_gemini_twitterbot_mixed_sota.py                   # full run
  python run_gemini_twitterbot_mixed_sota.py --variants zero_shot,info

Arguments
---------
  --dirty       Path to dirty CSV
  --clean       Path to clean CSV
  --mask        Path to combined mask CSV (1=intentional, -1=unintentional, 0=clean)
  --n_records   How many records to process (default: ALL)
  --out_dir     Output directory (default: auto-generated timestamped folder)
  --variants    Comma-separated: zero_shot,info,few_shot  (default: all)
  --model       Gemini model name  (default: gemini-2.0-flash)
  --chunk_size  Records per API call  (default: 10)
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ── API key ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

import google.generativeai as genai

# ── Defaults ──────────────────────────────────────────────────────────────────
_BASE = "/home/mohamed/error_injector/llms_baseline/mixed_error_pipeline_twitter/output"

DEFAULT_DIRTY = os.path.join(_BASE, "twibot20_phase2_final.csv")
DEFAULT_CLEAN = os.path.join(_BASE, "twibot20_clean.csv")
DEFAULT_MASK  = os.path.join(_BASE, "mask_combined.csv")
DEFAULT_MODEL = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro

# Twitter bot feature schema (19 columns, matches the pipeline output)
TWITTER_BOT_COLS = [
    "user_id", "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "verified", "protected",
    "default_profile", "default_profile_image", "geo_enabled",
    "profile_use_background_image", "has_created_date",
    "description_length", "screen_name_length", "name_length",
    "has_description", "has_location", "label",
]

# Columns excluded from intent analysis (identifier + target)
_SKIP_COLS = {"user_id", "label"}

DEFAULT_CHUNK_SIZE = 10
REQUEST_TIMEOUT    = 180

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in TWITTER_BOT_COLS}

def blind_mask_row(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    """Returns '1' if cell was changed (|mask|!=0), '0' otherwise — no intent info."""
    return {
        c: ("1" if str(mask_df.at[idx, c]).strip() not in ("0", "0.0", "") else "0")
        for c in TWITTER_BOT_COLS
    }

def compute_diffs(
    clean_row: Dict[str, str],
    dirty_row: Dict[str, str],
    blind: Dict[str, str],
) -> List[Dict]:
    return [
        {"column": c, "from": clean_row.get(c, ""), "to": dirty_row.get(c, "")}
        for c in TWITTER_BOT_COLS
        if c not in _SKIP_COLS and blind.get(c) == "1"
    ]

def changed_features_list(
    clean_row: Dict[str, str],
    dirty_row: Dict[str, str],
    blind: Dict[str, str],
) -> List[Dict]:
    return [
        {"column": c, "original_value": clean_row[c], "changed_value": dirty_row[c]}
        for c in TWITTER_BOT_COLS
        if c not in _SKIP_COLS and blind.get(c) == "1"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API call
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def call_gemini(model, prompt: str, timeout: int = REQUEST_TIMEOUT) -> str:
    """Call Gemini and return response text, or '' on failure."""
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 4096,
                "candidate_count": 1,
            },
            safety_settings=SAFETY_SETTINGS,
            request_options={"timeout": timeout},
        )
        if not response.candidates:
            return ""
        cand = response.candidates[0]
        if int(cand.finish_reason) not in (1, 2):
            return ""
        if not cand.content or not hasattr(cand.content, "parts") or not cand.content.parts:
            return ""
        texts = []
        for part in cand.content.parts:
            try:
                t = part.text
                if t:
                    texts.append(t)
            except Exception:
                pass
        return "\n".join(texts)
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()
    for fence in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(fence, "")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # extract first balanced fragment
    stack, start = [], None
    for i, ch in enumerate(s):
        if ch in "{[":
            if not stack:
                start = i
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack and start is not None:
                    frag = s[start: i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        try:
                            obj = ast.literal_eval(frag)
                            return obj if isinstance(obj, (dict, list)) else None
                        except Exception:
                            return None
    return None


def decisions_from_obj(
    obj: dict, global_ids: List[int]
) -> Optional[Dict[Tuple[int, str], Tuple[int, str]]]:
    """Parse feature_decisions list from LLM response.
    Returns dict keyed by (row_id, column) → (decision, reason), or None if unusable."""
    if not isinstance(obj, dict):
        return None
    feat_decs = obj.get("feature_decisions")
    if not isinstance(feat_decs, list) or not feat_decs:
        return None
    decisions: Dict[Tuple[int, str], Tuple[int, str]] = {}
    for item in feat_decs:
        if not isinstance(item, dict):
            continue
        try:
            row_id   = int(item.get("row_id"))
            column   = str(item.get("column", "")).strip()
            dec_val  = int(item.get("decision", -1))
            reason   = str(item.get("reason", "")).strip() or "no_reason"
            decision = 1 if dec_val >= 1 else -1
            if column:
                decisions[(row_id, column)] = (decision, reason)
        except (ValueError, TypeError, KeyError):
            continue
    return decisions if decisions else None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt blocks (shared across variants)
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_BLOCK = """\
ROLE
You are a precise data-forensics assistant specializing in Twitter bot detection evasion analysis.
Your task is to analyze changes made to individual features in Twitter bot profiles and determine whether each change was:
- INTENTIONAL (1): Deliberately made to make a bot profile appear more human-like or evade bot detection
- UNINTENTIONAL (-1): Accidental changes, or normal profile variations

IMPORTANT: You must evaluate EACH CHANGED FEATURE INDEPENDENTLY, not the entire record."""

_RUBRIC = """\
Rubric for Twitter Bot Evasion Detection:

INTENTIONAL (1) - Bot-to-Human Evasion Indicators:
- **Follower manipulation**: Large increases in followers_count or friends_count to appear more social
- **Engagement gaming**: Significant jumps in favourites_count or statuses_count
- **Verification gaming**: Changing verified flag to appear legitimate
- **Profile completeness**: Adding description (has_description=1), adding location (has_location=1)
- **Visual authenticity**: Changing default_profile or default_profile_image to appear customized
- **Strategic feature activation**: Enabling geo_enabled, profile_use_background_image to appear authentic
- **Coordinated multi-feature edits**: Multiple related fields changed together toward a more human profile

UNINTENTIONAL (-1) - Benign/Natural Changes:
- **Small numeric drift**: Minor count changes consistent with natural activity fluctuation (±5%)
- **Natural growth**: Gradual increases in statuses_count, favourites_count from normal use
- **Formatting/encoding**: No semantic change, just formatting differences
- **Listed count variations**: Small changes in listed_count from natural list membership"""

_OUTPUT_RULES = """\
Rules:
- Analyze each pair of (clean, changed) bot profiles
- For EACH CHANGED FEATURE (where mask=1), provide a separate decision
- Decide 1 (intentional bot evasion) or -1 (unintentional/normal change) PER FEATURE
- Each feature should have its own specific explanation (≤15 words)
- Be deterministic — same input must yield same output
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

CRITICAL: Each changed feature needs its own entry with a specific reason explaining
why THAT PARTICULAR feature change is intentional or unintentional."""

_FEW_SHOT_BLOCK = """\
EXAMPLE 1 — Bot-to-Human Evasion: Follower Manipulation & Profile Completion
Row ID: 1001
Changed Features:
- followers_count: 45 → 15000  (massive follower jump to appear popular)
- friends_count: 32 → 12500    (large increase to appear socially active)
- has_description: 0 → 1       (adding description to complete profile)
- description_length: 0 → 85  (profile completion, correlated with has_description)
Gold decisions:
{
  "feature_decisions": [
    {"row_id": 1001, "column": "followers_count", "decision": 1, "reason": "massive follower jump for popularity"},
    {"row_id": 1001, "column": "friends_count", "decision": 1, "reason": "large increase for social appearance"},
    {"row_id": 1001, "column": "has_description", "decision": 1, "reason": "profile completion strategy"},
    {"row_id": 1001, "column": "description_length", "decision": 1, "reason": "coordinated with description addition"}
  ]
}

EXAMPLE 2 — Mixed Changes: Evasion vs Natural Variation
Row ID: 2001
Changed Features:
- verified: 0 → 1              (gaining verification flag)
- default_profile: 1 → 0       (customizing profile appearance)
- statuses_count: 1210 → 1215  (tiny increase)

Row ID: 2002
Changed Features:
- followers_count: 820 → 845   (small natural growth)
- description_length: 95 → 97  (minor text edit)

Row ID: 2003
Changed Features:
- friends_count: 150 → 50000   (extreme jump)
- favourites_count: 25 → 5000  (large engagement boost)
- geo_enabled: 0 → 1           (strategic feature activation)
Gold decisions:
{
  "feature_decisions": [
    {"row_id": 2001, "column": "verified", "decision": 1, "reason": "verification gaming for legitimacy"},
    {"row_id": 2001, "column": "default_profile", "decision": 1, "reason": "customization for authenticity"},
    {"row_id": 2001, "column": "statuses_count", "decision": -1, "reason": "tiny activity variation"},
    {"row_id": 2002, "column": "followers_count", "decision": -1, "reason": "minor natural growth"},
    {"row_id": 2002, "column": "description_length", "decision": -1, "reason": "small text edit"},
    {"row_id": 2003, "column": "friends_count", "decision": 1, "reason": "extreme jump indicating bot evasion"},
    {"row_id": 2003, "column": "favourites_count", "decision": 1, "reason": "engagement gaming"},
    {"row_id": 2003, "column": "geo_enabled", "decision": 1, "reason": "strategic feature activation"}
  ]
}

EXAMPLE 3 — Coordinated Multi-Feature Evasion
Row ID: 3001
Changed Features:
- default_profile: 1 → 0        (customized appearance)
- default_profile_image: 1 → 0  (added custom image)
- has_location: 0 → 1           (added location for authenticity)
- profile_use_background_image: 0 → 1  (strategic background activation)
Gold decisions:
{
  "feature_decisions": [
    {"row_id": 3001, "column": "default_profile", "decision": 1, "reason": "profile customization for authenticity"},
    {"row_id": 3001, "column": "default_profile_image", "decision": 1, "reason": "image customization to appear human"},
    {"row_id": 3001, "column": "has_location", "decision": 1, "reason": "location added for human-like authenticity"},
    {"row_id": 3001, "column": "profile_use_background_image", "decision": 1, "reason": "coordinated background activation"}
  ]
}"""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders — one per variant
# ─────────────────────────────────────────────────────────────────────────────

def _build_bundle(
    clean_rows: List[Dict], dirty_rows: List[Dict],
    blind_rows: List[Dict], global_ids: List[int],
    with_diffs: bool = False,
) -> dict:
    records = []
    for i in range(len(clean_rows)):
        cf = changed_features_list(clean_rows[i], dirty_rows[i], blind_rows[i])
        if not cf:
            continue
        entry = {
            "row_id": global_ids[i],
            "clean_profile": clean_rows[i],
            "changed_profile": dirty_rows[i],
            "changed_features": cf,
        }
        if with_diffs:
            entry["diffs"] = compute_diffs(clean_rows[i], dirty_rows[i], blind_rows[i])
        records.append(entry)
    return {"columns": TWITTER_BOT_COLS, "records": records}


def make_prompt_zero_shot(
    clean_rows, dirty_rows, blind_rows, global_ids
) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids)
    return (
        f"{_ROLE_BLOCK}\n\n"
        f"{_OUTPUT_RULES}\n\n"
        f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Analyze each changed feature and return your feature-level decisions in JSON format now."
    )


def make_prompt_info(
    clean_rows, dirty_rows, blind_rows, global_ids
) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids, with_diffs=True)
    return (
        f"{_ROLE_BLOCK}\n\n"
        f"{_RUBRIC}\n\n"
        f"{_OUTPUT_RULES}\n\n"
        f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Analyze each changed feature using the rubric and return your feature-level decisions in JSON format now."
    )


def make_prompt_few_shot(
    clean_rows, dirty_rows, blind_rows, global_ids
) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids, with_diffs=True)
    return (
        f"{_ROLE_BLOCK}\n\n"
        f"{_RUBRIC}\n\n"
        f"{_OUTPUT_RULES}\n\n"
        f"{_FEW_SHOT_BLOCK}\n\n"
        f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
        "Analyze each changed feature using the rubric and examples, then return your feature-level decisions in JSON format now."
    )


VARIANTS = {
    "zero_shot": make_prompt_zero_shot,
    "info":      make_prompt_info,
    "few_shot":  make_prompt_few_shot,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (feature-level)
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == -1)
    n  = len(y_true)
    acc    = (tp + tn) / n if n else 0
    prec_i = tp / (tp + fp) if (tp + fp) else 0
    rec_i  = tp / (tp + fn) if (tp + fn) else 0
    f1_i   = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) else 0
    prec_u = tn / (tn + fn) if (tn + fn) else 0
    rec_u  = tn / (tn + fp) if (tn + fp) else 0
    f1_u   = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) else 0
    f1_w   = (f1_i * (tp + fn) + f1_u * (tn + fp)) / n if n else 0
    return {
        "n_features":         n,
        "accuracy":           round(acc,    4),
        "precision_int":      round(prec_i, 4),
        "recall_int":         round(rec_i,  4),
        "f1_intentional":     round(f1_i,   4),
        "precision_unint":    round(prec_u, 4),
        "recall_unint":       round(rec_u,  4),
        "f1_unintentional":   round(f1_u,   4),
        "f1_weighted":        round(f1_w,   4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_counts": dict(Counter(y_pred)),
        "true_counts":  dict(Counter(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant_name: str,
    prompt_fn,
    model,
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    mask_df:  pd.DataFrame,
    n_records: int,
    chunk_size: int,
    out_dir: str,
) -> Dict:
    v_dir = os.path.join(out_dir, variant_name)
    os.makedirs(v_dir, exist_ok=True)

    # ── Build output intent-labeled mask (copy of mask, values replaced with 1/-1) ──
    out_mask_df = mask_df.copy(deep=True)

    pred_rows:       List[Dict]  = []
    y_true:          List[int]   = []
    y_pred:          List[int]   = []
    chunk_times:     List[float] = []
    llm_times:       List[float] = []
    llm_success      = 0
    fallback_chunks  = 0
    status_hist      = {"ok": 0, "api_error": 0, "parse_error": 0, "fallback": 0}

    print(f"\n{'='*65}")
    print(f"  VARIANT: {variant_name.upper()}  ({n_records} records, chunk={chunk_size})")
    print(f"{'='*65}")

    n_chunks = (n_records + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc=variant_name, ncols=90):
        ct0 = time.perf_counter()

        start_i = chunk_idx * chunk_size
        end_i   = min(start_i + chunk_size, n_records)

        clean_rows  = [row_to_dict(clean_df,  i) for i in range(start_i, end_i)]
        dirty_rows  = [row_to_dict(dirty_df,  i) for i in range(start_i, end_i)]
        blind_rows  = [blind_mask_row(mask_df, i) for i in range(start_i, end_i)]
        global_ids  = list(range(start_i, end_i))

        prompt_text = prompt_fn(clean_rows, dirty_rows, blind_rows, global_ids)

        obj      = None
        llm_text = ""
        lt0 = time.perf_counter()
        try:
            llm_text = call_gemini_with_fallback(model, model_flash, prompt_text)
            obj = parse_llm_json(llm_text)
            status_hist["ok"] += 1
        except Exception:
            status_hist["api_error"] += 1
        lt1 = time.perf_counter()
        llm_times.append(lt1 - lt0)

        parsed = decisions_from_obj(obj, global_ids) if isinstance(obj, dict) else None

        if parsed is not None:
            llm_success += 1
            feature_decisions = parsed
        else:
            fallback_chunks += 1
            status_hist["fallback"] += 1
            feature_decisions = {}
            # default all changed features to -1
            for local_idx, rid in enumerate(global_ids):
                for col in TWITTER_BOT_COLS:
                    if col not in _SKIP_COLS and blind_rows[local_idx].get(col) == "1":
                        feature_decisions[(rid, col)] = (-1, "llm_fallback")

        # ── Fill output mask + collect feature-level predictions ──
        for local_idx, rid in enumerate(global_ids):
            for col in TWITTER_BOT_COLS:
                raw_val = mask_df.at[rid, col]
                try:
                    true_raw = int(float(str(raw_val).strip()))
                except (ValueError, TypeError):
                    true_raw = 0

                if true_raw != 0:
                    # This feature was changed — get true intent
                    true_intent = 1 if true_raw > 0 else -1

                    # Get LLM's feature-level decision
                    decision, reason = feature_decisions.get((rid, col), (-1, "guard_fallback"))

                    out_mask_df.at[rid, col] = str(decision)
                    y_true.append(true_intent)
                    y_pred.append(decision)

                    pred_rows.append({
                        "row_id":       rid,
                        "column":       col,
                        "true_intent":  true_intent,
                        "pred_intent":  decision,
                        "reason":       reason,
                        "clean_val":    clean_rows[local_idx].get(col, ""),
                        "dirty_val":    dirty_rows[local_idx].get(col, ""),
                    })
                else:
                    out_mask_df.at[rid, col] = 0

        chunk_times.append(time.perf_counter() - ct0)

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df   = pd.DataFrame(pred_rows)
    pred_path = os.path.join(v_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    out_mask_path = os.path.join(v_dir, "intent_labels.csv")
    out_mask_df.to_csv(out_mask_path, index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)
    metrics["variant"]             = variant_name
    metrics["llm_success_chunks"]  = llm_success
    metrics["fallback_chunks"]     = fallback_chunks
    metrics["status_hist"]         = status_hist
    metrics["avg_llm_time_s"]      = round(sum(llm_times) / len(llm_times), 3) if llm_times else 0
    metrics["total_llm_time_s"]    = round(sum(llm_times), 2)

    metrics_path = os.path.join(v_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n  Results — {variant_name}:")
    print(f"    Features evaluated : {metrics['n_features']}")
    print(f"    Accuracy           : {metrics['accuracy']:.4f}")
    print(f"    F1 intentional     : {metrics['f1_intentional']:.4f}")
    print(f"    F1 unintentional   : {metrics['f1_unintentional']:.4f}")
    print(f"    F1 weighted        : {metrics['f1_weighted']:.4f}")
    print(f"    Fallback chunks    : {fallback_chunks}/{n_chunks}")
    print(f"    Avg LLM time       : {metrics['avg_llm_time_s']:.2f}s")
    print(f"    Saved → {v_dir}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three Gemini intent-attribution variants on the Mixed SOTA TwiBot-20 dataset."
    )
    parser.add_argument("--dirty",      default=DEFAULT_DIRTY)
    parser.add_argument("--clean",      default=DEFAULT_CLEAN)
    parser.add_argument("--mask",       default=DEFAULT_MASK)
    parser.add_argument("--n_records",  type=int, default=None)
    parser.add_argument("--out_dir",    default=None)
    parser.add_argument("--variants",   default="zero_shot,info,few_shot")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    # ── Validate variants ────────────────────────────────────────────────────
    requested = [v.strip() for v in args.variants.split(",")]
    unknown   = [v for v in requested if v not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Choose from: {list(VARIANTS)}")

    # ── Output dir ───────────────────────────────────────────────────────────
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(_SCRIPT_DIR, "output", f"twitterbot_gemini_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Init Gemini ───────────────────────────────────────────────────────────
    if not GEMINI_API_KEY:
        sys.exit("[ERROR] No GEMINI_API_KEY. Set env var or add to config.py.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")  # tier 1; ignores args.model on purpose -- cost policy
    model_flash = genai.GenerativeModel("gemini-2.5-flash")  # tier 2 fallback

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading data…")
    dirty_df = pd.read_csv(args.dirty, dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    mask_df  = pd.read_csv(args.mask,  keep_default_na=False)

    for df, name in [(dirty_df, "dirty"), (clean_df, "clean")]:
        missing = [c for c in TWITTER_BOT_COLS if c not in df.columns]
        if missing:
            sys.exit(f"[{name}] missing columns: {missing}")

    if not (len(dirty_df) == len(clean_df) == len(mask_df)):
        sys.exit(f"Row count mismatch: dirty={len(dirty_df)}, clean={len(clean_df)}, mask={len(mask_df)}")

    n_total   = len(dirty_df)
    n_records = min(args.n_records, n_total) if args.n_records else n_total

    # Count total changed features for info
    _feat_cols = [c for c in TWITTER_BOT_COLS if c not in _SKIP_COLS]
    n_changed_feats = int((mask_df[_feat_cols] != 0).sum().sum())

    print(f"  Total rows          : {n_total}")
    print(f"  Rows to process     : {n_records}")
    print(f"  Changed features    : {n_changed_feats} (across all rows)")
    print(f"  Variants to run     : {requested}")
    print(f"  Gemini model        : gemini-2.5-flash-lite (fallback: gemini-2.5-flash)")
    print(f"  Chunk size          : {args.chunk_size}")
    print(f"  Output dir          : {args.out_dir}")

    # ── Run variants ─────────────────────────────────────────────────────────
    t_start     = time.perf_counter()
    all_metrics: List[Dict] = []

    for variant_name in requested:
        metrics = run_variant(
            variant_name=variant_name,
            prompt_fn=VARIANTS[variant_name],
            model=model,
            dirty_df=dirty_df.iloc[:n_records].reset_index(drop=True),
            clean_df=clean_df.iloc[:n_records].reset_index(drop=True),
            mask_df=mask_df.iloc[:n_records].reset_index(drop=True),
            n_records=n_records,
            chunk_size=args.chunk_size,
            out_dir=args.out_dir,
        )
        all_metrics.append(metrics)

    # ── Cross-variant comparison ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("CROSS-VARIANT COMPARISON  (feature-level)")
    print(f"{'='*65}")
    header = f"{'Variant':<14} {'Acc':>6} {'F1_int':>8} {'F1_unint':>10} {'F1_w':>7} {'Fall':>5} {'AvgT':>7}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m['variant']:<14} "
            f"{m['accuracy']:>6.4f} "
            f"{m['f1_intentional']:>8.4f} "
            f"{m['f1_unintentional']:>10.4f} "
            f"{m['f1_weighted']:>7.4f} "
            f"{m['fallback_chunks']:>5} "
            f"{m['avg_llm_time_s']:>7.2f}s"
        )

    comp_df   = pd.DataFrame(all_metrics)
    comp_path = os.path.join(args.out_dir, "comparison_table.csv")
    comp_df.to_csv(comp_path, index=False)

    meta = {
        "run_timestamp":   datetime.datetime.now().isoformat(),
        "model":           "gemini-2.5-flash-lite+fallback:gemini-2.5-flash",
        "n_records":       n_records,
        "n_total_rows":    n_total,
        "dirty_path":      args.dirty,
        "clean_path":      args.clean,
        "mask_path":       args.mask,
        "variants_run":    requested,
        "chunk_size":      args.chunk_size,
        "total_wall_time": round(time.perf_counter() - t_start, 1),
        "out_dir":         args.out_dir,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  Comparison table → {comp_path}")
    print(f"  Total wall time  : {meta['total_wall_time']:.0f}s")
    print(f"\n✓ Done. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
