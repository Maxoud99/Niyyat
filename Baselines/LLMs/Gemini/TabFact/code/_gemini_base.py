#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared base module for TabFact Intent Attribution via Google Gemini.
====================================================================

Provides:
  - Data loading (dirty dataset + unified mask + corrected values from
    the earlier Gemini error-detection explanations.json).
  - Gemini API call + robust JSON parsing.
  - Feature-level decision extraction.
  - A ``run_pipeline`` orchestrator that takes a *prompt builder* and
    runs the attribution over all records in fixed-size chunks.

Each variant script (baremin / info / few-shots) only defines its own
ROLE / RUBRIC / FEW-SHOT blocks and a ``make_chunk_prompt`` function, and
then calls ``run_pipeline``.

Intent labels (per changed cell):
    1  → INTENTIONAL   (deliberate misinformation)
   -1  → UNINTENTIONAL (typo, drift, formatting)
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

# Default paths (can be overridden from CLI)
_BASE = "/home/mohamed/error_injector/llms_baseline/tabfact"
DIRTY_CSV        = f"{_BASE}/datasets/final/combined_dataset_TRF_no_claim.csv"
ORIG_DIRTY_CSV   = f"{_BASE}/datasets/final/combined_dataset_TRF.csv"
MASK_CSV         = f"{_BASE}/outputs/error_detection/run_20251222_163837/error_mask_updated.csv"
EXPLANATIONS_JSON = f"{_BASE}/outputs/error_detection/run_20251222_163837/explanations.json"

# Attribution covers these feature columns (metric has no errors; is_factual is the label)
ATTR_COLS: List[str] = [
    "subject_entity",
    "subject_type",
    "subject_location",
    "value",
    "claim_domain",
]

CHUNK_SIZE  = 10     # records per LLM call
MAX_RECORDS: Optional[int] = None   # None → full dataset


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _row_to_dict(df: pd.DataFrame, idx: int, cols: List[str]) -> Dict[str, str]:
    return {c: _to_str(df.at[idx, c]) for c in cols}


def _mask_cell_is_error(v) -> bool:
    return str(v).strip() in ("1", "1.0")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading: build (dirty, corrected, mask) aligned on the cleaned dataset
# ─────────────────────────────────────────────────────────────────────────────
def _align_new_to_orig(new_df: pd.DataFrame, orig_df: pd.DataFrame,
                       lookahead: int = 20) -> List[int]:
    """
    Recover which original row each cleaned-dataset row corresponds to.
    Order is preserved; some rows were dropped; some were stylistic updates.
    Uses value + is_factual as a composite key, with look-ahead to skip
    dropped rows.  Returns a list ``map_idx`` with len == len(new_df),
    where ``map_idx[n]`` is the original row index for new row n.
    """
    def norm(s):  # "0.0" → "0"
        return str(s).strip().lower().split(".")[0]

    o_val = [norm(v) for v in orig_df["value"]]
    o_fac = [norm(v) for v in orig_df["is_factual"]]
    n_val = [norm(v) for v in new_df["value"]]
    n_fac = [norm(v) for v in new_df["is_factual"]]

    out: List[int] = []
    o = 0
    for n in range(len(new_df)):
        if o >= len(orig_df):
            out.append(o - 1)
            continue
        if o_val[o] == n_val[n] and o_fac[o] == n_fac[n]:
            out.append(o);   o += 1
            continue
        # look ahead for a match (orig rows in between were DROPPED)
        found = -1
        for la in range(1, lookahead + 1):
            if o + la >= len(orig_df):
                break
            if o_val[o + la] == n_val[n] and o_fac[o + la] == n_fac[n]:
                found = o + la
                break
        if found >= 0:
            o = found
            out.append(o); o += 1
        else:
            # stylistic update → same position
            out.append(o); o += 1
    return out


def load_tabfact_attribution_data(
    dirty_path: str = DIRTY_CSV,
    mask_path: str  = MASK_CSV,
    expl_path: str  = EXPLANATIONS_JSON,
    orig_path: str  = ORIG_DIRTY_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (dirty_df, corrected_df, mask_df), all indexed 0..N-1 with
    the same row count.

    ``corrected_df`` is synthesized from ``explanations.json``:
    for every cell the mask marks as an error, the corresponding
    ``correct_value`` from the Gemini detection is inserted;  all other
    cells are copied verbatim from ``dirty_df``.
    """
    dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False,
                        quotechar='"', engine='python')
    mask  = pd.read_csv(mask_path,  dtype=str, keep_default_na=False)
    orig  = pd.read_csv(orig_path,  dtype=str, keep_default_na=False,
                        quotechar='"', engine='python')
    with open(expl_path, "r", encoding="utf-8") as f:
        explanations = json.load(f)

    if len(dirty) != len(mask):
        raise ValueError(
            f"dirty rows ({len(dirty)}) != mask rows ({len(mask)})")
    for c in ATTR_COLS:
        if c not in dirty.columns:
            raise ValueError(f"dirty CSV missing column: {c}")
        if c not in mask.columns:
            raise ValueError(f"mask CSV missing column: {c}")

    # Align new→orig to look up correct_value per row from explanations.json
    new2orig = _align_new_to_orig(dirty, orig)

    # Build corrected dataset
    corrected = dirty.copy(deep=True)
    n_applied_corrections = 0
    n_missing_corrections = 0
    for n in range(len(dirty)):
        orig_idx = new2orig[n]
        entry = explanations.get(str(orig_idx), None)
        # Map feature → correct_value from explanations (if any)
        corrections: Dict[str, str] = {}
        if entry and isinstance(entry.get("errors"), list):
            for err in entry["errors"]:
                f = str(err.get("feature", "")).strip()
                if f in ATTR_COLS:
                    corrections[f] = str(err.get("correct_value", "")).strip()

        # For each mask=1 cell, insert the correction (if found)
        for c in ATTR_COLS:
            if _mask_cell_is_error(mask.at[n, c]):
                if c in corrections and corrections[c]:
                    corrected.at[n, c] = corrections[c]
                    n_applied_corrections += 1
                else:
                    # No correct_value in explanations → leave dirty value
                    # (rare; happens if mask had an extra flagged cell)
                    n_missing_corrections += 1

    print(f"[data] dirty={len(dirty)}  mask={len(mask)}  cols={ATTR_COLS}")
    print(f"[data] corrections applied: {n_applied_corrections}"
          f"   missing: {n_missing_corrections}")
    return dirty, corrected, mask


# ─────────────────────────────────────────────────────────────────────────────
# Gemini call
# ─────────────────────────────────────────────────────────────────────────────
def call_gemini(model, prompt: str, max_output_tokens: int = 4096):
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
                "max_output_tokens": max_output_tokens,
                "candidate_count": 1,
            },
            safety_settings=safety,
            request_options={"timeout": 180},
        )
        if not resp.candidates:
            return ""
        cand = resp.candidates[0]
        if cand.finish_reason.value not in (1, 2):   # 1=STOP, 2=MAX_TOKENS
            return ""
        if not cand.content or not getattr(cand.content, "parts", None):
            return ""
        return cand.content.parts[0].text or ""
    except Exception:
        return None  # signals caller: fall back to next model tier


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

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

def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()
    for junk in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(junk, "")
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = json.loads(s)
        except Exception:
            pass
        s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

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
                    frag = s[start:i+1]
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


def decisions_from_obj(obj: dict) -> Optional[Dict[Tuple[int, str], Tuple[int, str]]]:
    """Extract {(row_id, column): (decision, reason)} from LLM JSON."""
    if not isinstance(obj, dict):
        return None
    feat = obj.get("feature_decisions")
    if not isinstance(feat, list):
        return None
    out: Dict[Tuple[int, str], Tuple[int, str]] = {}
    for item in feat:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item.get("row_id"))
            col = str(item.get("column", "")).strip()
            dv  = int(item.get("decision", -1))
            rsn = str(item.get("reason", "")).strip() or "reason_not_provided"
            out[(rid, col)] = (1 if dv >= 1 else -1, rsn)
        except (TypeError, ValueError, KeyError):
            continue
    return out or None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────
PromptBuilder = Callable[
    [List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[int]],
    str,
]


def run_pipeline(
    make_chunk_prompt: PromptBuilder,
    out_dir: str,
    variant_name: str,
    dirty_path: str = DIRTY_CSV,
    mask_path:  str = MASK_CSV,
    expl_path:  str = EXPLANATIONS_JSON,
    orig_path:  str = ORIG_DIRTY_CSV,
    max_records: Optional[int] = MAX_RECORDS,
    chunk_size: int = CHUNK_SIZE,
    max_output_tokens: int = 4096,
) -> str:
    """
    Execute the three-way attribution pipeline for one Gemini variant.
    Writes intent_labels.csv, intent_explanations.csv, run_stats.json,
    per_chunk_times.csv to ``out_dir``.
    Returns the path of ``intent_labels.csv``.
    """
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    labels_csv   = os.path.join(out_dir, "intent_labels.csv")
    expl_csv     = os.path.join(out_dir, "intent_explanations.csv")
    stats_json   = os.path.join(out_dir, "run_stats.json")
    per_chunk_csv = os.path.join(out_dir, "per_chunk_times.csv")

    # ── Load & align
    dirty, corrected, mask = load_tabfact_attribution_data(
        dirty_path, mask_path, expl_path, orig_path
    )
    n_records = len(dirty)
    if max_records is not None and max_records < n_records:
        print(f"[run] limiting to first {max_records} rows (of {n_records})")
        dirty     = dirty.iloc[:max_records].reset_index(drop=True)
        corrected = corrected.iloc[:max_records].reset_index(drop=True)
        mask      = mask.iloc[:max_records].reset_index(drop=True)
        n_records = max_records

    # ── Gemini client
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME_LITE)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)

    # ── Prepare output mask (same cols, start as a copy of mask)
    out_mask = mask.copy(deep=True)
    explanations_rows: List[Dict] = []
    chunk_times: List[float] = []
    llm_times:   List[float] = []
    status = {"ok": 0, "parse_error": 0, "api_error": 0}
    successful_chunks = 0

    n_chunks = (n_records + chunk_size - 1) // chunk_size
    print(f"[run] variant={variant_name}  records={n_records}  "
          f"chunk_size={chunk_size}  chunks={n_chunks}")

    for ci in tqdm(range(n_chunks), desc=f"{variant_name}", ncols=100):
        ct0 = time.perf_counter()
        s = ci * chunk_size
        e = min(s + chunk_size, n_records)

        dirty_rows     = [_row_to_dict(dirty,     i, ATTR_COLS) for i in range(s, e)]
        corrected_rows = [_row_to_dict(corrected, i, ATTR_COLS) for i in range(s, e)]
        mask_rows      = [{c: ("1" if _mask_cell_is_error(mask.at[i, c]) else "0")
                           for c in ATTR_COLS}
                          for i in range(s, e)]
        global_ids = list(range(s, e))

        prompt = make_chunk_prompt(corrected_rows, dirty_rows, mask_rows, global_ids)

        l0 = time.perf_counter()
        llm_text = call_gemini_with_fallback(model, model_flash, prompt, max_output_tokens=max_output_tokens)
        obj = parse_llm_json(llm_text)
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)

        parsed = decisions_from_obj(obj) if isinstance(obj, dict) else None
        if parsed is None:
            status["parse_error" if llm_text else "api_error"] += 1
            parsed = {}
            for li, rid in enumerate(global_ids):
                for c in ATTR_COLS:
                    if mask_rows[li][c] == "1":
                        parsed[(rid, c)] = (-1, "llm_fallback")
        else:
            status["ok"] += 1
            successful_chunks += 1

        # Fill output mask + per-cell explanations
        for li, rid in enumerate(global_ids):
            for c in ATTR_COLS:
                if mask_rows[li][c] == "1":
                    dec, reason = parsed.get((rid, c), (-1, "guard_fallback"))
                    out_mask.at[rid, c] = str(dec)
                    explanations_rows.append({
                        "row_id":        rid,
                        "column":        c,
                        "dirty_value":   dirty_rows[li][c],
                        "correct_value": corrected_rows[li][c],
                        "mask":          1,
                        "decision":      dec,
                        "reason":        reason,
                    })
                else:
                    out_mask.at[rid, c] = "0"

        chunk_times.append(time.perf_counter() - ct0)

    # ── Save outputs
    out_mask.to_csv(labels_csv, index=False)
    pd.DataFrame(explanations_rows).to_csv(expl_csv, index=False)
    pd.DataFrame({
        "chunk_id": list(range(n_chunks)),
        "chunk_time_sec": chunk_times,
        "llm_time_sec":   llm_times,
    }).to_csv(per_chunk_csv, index=False)

    # Flat cell counts (excluding 'metric' and 'is_factual' which are not here)
    flat = out_mask[ATTR_COLS].values.flatten().tolist()
    counts = pd.Series([str(x) for x in flat]).value_counts().to_dict()

    t1 = time.perf_counter()
    stats = {
        "variant":        variant_name,
        "model":          MODEL_NAME,
        "dataset":        "tabfact",
        "records":        n_records,
        "chunk_size":     chunk_size,
        "n_chunks":       n_chunks,
        "total_time_sec": t1 - t0,
        "avg_chunk_sec":  float(pd.Series(chunk_times).mean()) if chunk_times else None,
        "avg_llm_sec":    float(pd.Series(llm_times).mean()) if llm_times else None,
        "status_hist":    status,
        "successful_chunks": successful_chunks,
        "decision_counts": {
            "intentional (1)":   counts.get("1", 0),
            "unintentional (-1)": counts.get("-1", 0),
            "unchanged (0)":      counts.get("0", 0),
        },
        "outputs": {
            "intent_labels_csv":       labels_csv,
            "intent_explanations_csv": expl_csv,
            "per_chunk_times_csv":     per_chunk_csv,
        },
    }
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {variant_name} done in {t1-t0:.1f}s")
    print(f"   labels: {labels_csv}")
    print(f"   stats:  {stats_json}")
    print(f"   decisions: 1={counts.get('1',0)}  -1={counts.get('-1',0)}  "
          f"0={counts.get('0',0)}")
    return labels_csv
