#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Attribution — Three Gemini Variants — Mixed SOTA Adult Income Dataset
=============================================================================

Runs three Gemini prompt strategies back-to-back on the Mixed SOTA
Adult Income dataset (mixed error pipeline output, 1:1 dirty-to-clean ratio):

  Variant 1 — zero-shot    : plain JSON bundle, no rubric, no diffs
  Variant 2 — info         : adds per-cell DIFFS + decision rubric
  Variant 3 — few-shot     : info + two in-context examples

Key differences from the LLM-dataset pipelines:
  • The mixed dataset has a 1:1 dirty-to-clean ratio (no 3-variant grouping).
    Each dirty record is an independent query.
  • The ground-truth mask already contains 1/−1/0; we present a "blind" mask
    (0/1 — was this cell changed?) to the model, then compare prediction vs truth.
  • Metrics (accuracy, F1, etc.) are computed at the end for each variant.

Usage
-----
  # Quick test on first 100 records (= 100 API calls per variant)
  python run_gemini_mixed_sota.py --n_records 100

  # Full run
  python run_gemini_mixed_sota.py

  # Custom paths + limit
  python run_gemini_mixed_sota.py --dirty /path/to/dirty.csv \\
                                   --clean /path/to/clean.csv \\
                                   --mask  /path/to/mask.csv  \\
                                   --n_records 200 \\
                                   --out_dir   /tmp/my_run

Arguments
---------
  --dirty       Path to dirty CSV  (default: mixed_error_pipeline output)
  --clean       Path to clean CSV  (default: mixed_error_pipeline output)
  --mask        Path to combined mask CSV (1=intentional, -1=unintentional, 0=clean)
  --n_records   How many dirty records to process (default: ALL)
  --out_dir     Output directory   (default: auto-generated timestamped folder)
  --variants    Comma-separated subset of variants to run: zero_shot,info,few_shot
                (default: all three)
  --model       Gemini model name  (default: gemini-2.5-pro)
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ── API key ──────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

import google.generativeai as genai

# ── defaults ─────────────────────────────────────────────────────────────────
BASE = "/home/mohamed/error_injector/llms_baseline/mixed_error_pipeline/output"

DEFAULT_DIRTY = os.path.join(BASE, "adult_phase2_final.csv")
DEFAULT_CLEAN = os.path.join(BASE, "adult_clean.csv")
DEFAULT_MASK  = os.path.join(BASE, "mask_combined.csv")

DEFAULT_MODEL = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}

def blind_mask_row(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    """Returns 1 if cell was changed (|mask| != 0), 0 otherwise — no intent info."""
    return {
        c: ("1" if str(mask_df.at[idx, c]).strip() not in ("0", "0.0", "") else "0")
        for c in ADULT_COLS
    }

def compute_diffs(
    clean_row: Dict[str, str],
    dirty_row: Dict[str, str],
    blind: Dict[str, str],
) -> List[Dict[str, str]]:
    diffs = []
    for c in ADULT_COLS:
        if c == "class":
            continue
        if blind.get(c) == "1":
            diffs.append({"column": c, "from": clean_row.get(c, ""), "to": dirty_row.get(c, "")})
    return diffs

# ─────────────────────────────────────────────────────────────────────────────
# Gemini call (shared across all variants)
# ─────────────────────────────────────────────────────────────────────────────

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",  "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",  "threshold": "BLOCK_NONE"},
]

def call_gemini(model, prompt: str, timeout: int = 60) -> str:
    """Return model response text, or '' on any failure."""
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 1024,
                "candidate_count": 1,
            },
            safety_settings=SAFETY_SETTINGS,
            request_options={"timeout": timeout},
        )
        if not response.candidates:
            return ""
        cand = response.candidates[0]
        # finish_reason: 1=STOP, 2=MAX_TOKENS — both acceptable
        if int(cand.finish_reason) not in (1, 2):
            return ""
        if not cand.content or not cand.content.parts:
            return ""
        # Concatenate all text parts (thinking models may split across parts)
        texts = []
        for part in cand.content.parts:
            try:
                t = part.text
                if t:
                    texts.append(t)
            except Exception:
                pass
        return "\n".join(texts)
    except Exception as e:
        return None  # signals caller: fall back to next model tier


def call_gemini_with_fallback(model_lite, model_flash, prompt: str, timeout: int = 60) -> str:
    """Cost policy (2026-06-26): flash-lite first, then flash, then give up.
    Never escalates to gemini-2.5-pro."""
    result = call_gemini(model_lite, prompt, timeout=timeout)
    if result is not None:
        return result
    result = call_gemini(model_flash, prompt, timeout=timeout)
    if result is not None:
        return result
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing (robust, handles markdown fences + multiple schemas)
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_json(text: str) -> Optional[dict]:
    """Robustly extract first top-level JSON object or array from model text."""
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

    def _first_fragment(txt: str) -> Optional[str]:
        stack, start = [], None
        for i, ch in enumerate(txt):
            if ch in "{[":
                if not stack:
                    start = i
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if stack and ch == stack[-1]:
                    stack.pop()
                    if not stack and start is not None:
                        return txt[start : i + 1]
        return None

    frag = _first_fragment(s)
    if not frag:
        return None
    try:
        return json.loads(frag)
    except Exception:
        try:
            obj = ast.literal_eval(frag)
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None

def extract_decision(obj: Optional[dict], raw: str) -> Optional[int]:
    """
    Extract a single integer decision (1 or -1) from model output.

    Accepts multiple schemas:
      {"decision": 1}
      {"intent": 1}
      {"intentional": true/1}
      1  or  -1  (bare integer)
      "1" or "-1" (bare string)
    """
    # Bare scalar from raw text first
    stripped = raw.strip().lstrip("\ufeff")
    for fence in ("```json", "```"):
        stripped = stripped.replace(fence, "").strip()
    if stripped in ("1", "-1"):
        return int(stripped)

    if obj is None:
        return None

    # dict with known keys
    if isinstance(obj, dict):
        for key in ("decision", "intent", "intentional", "label", "result"):
            v = obj.get(key)
            if v is not None:
                try:
                    vi = int(v) if not isinstance(v, bool) else (1 if v else -1)
                    return 1 if vi >= 1 else -1
                except Exception:
                    pass
        # fallback: single-key dict
        if len(obj) == 1:
            v = list(obj.values())[0]
            try:
                vi = int(v) if not isinstance(v, bool) else (1 if v else -1)
                return 1 if vi >= 1 else -1
            except Exception:
                pass

    # bare integer / list[int]
    if isinstance(obj, (int, float)):
        return 1 if int(obj) >= 1 else -1
    if isinstance(obj, list) and len(obj) >= 1:
        try:
            return 1 if int(obj[0]) >= 1 else -1
        except Exception:
            pass

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders  (one per variant)
# ─────────────────────────────────────────────────────────────────────────────

_ROLE = (
    "You are a precise data-forensics assistant. "
    "Decide whether the changes in a single manipulated record are "
    "INTENTIONAL (1) or UNINTENTIONAL (-1). No zeros. Deterministic."
)

_RUBRIC = """\
Rubric:
- INTENTIONAL (1)  : large utility-seeking edits (education-num upgrade ≥2, big capital-gain from 0,
  hours-per-week +≥10, coordinated multi-field edits, fairness gaming, privacy masking).
- UNINTENTIONAL (-1): tiny numeric drift (±1-2), case-only / formatting-only, single trivial change.
- When uncertain: prefer -1 only for clearly small/benign edits; otherwise choose 1."""

_FEW_SHOT_EXAMPLES = """\
--- EXAMPLE 1 ---
DIFFS: education-num 9→13, education HS-grad→Bachelors, capital-gain 0→12000, hours-per-week 40→52
ANSWER: {"decision":1,"reason":"multiple strong utility upgrades"}

--- EXAMPLE 2 ---
DIFFS: age 36→35 (±1), workclass Private→private (case only)
ANSWER: {"decision":-1,"reason":"tiny numeric drift and case change"}

--- EXAMPLE 3 ---
DIFFS: sex Female→Male, race Black→White, hours-per-week 35→46
ANSWER: {"decision":1,"reason":"protected-attribute edits + workload increase"}

--- EXAMPLE 4 ---
DIFFS: capital-loss 0→1
ANSWER: {"decision":-1,"reason":"trivial single-cell change"}
"""

_OUTPUT_RULES = """\
Rules:
- Use ONLY the clean row, the dirty row, the change mask, and the diffs.
- Return ONLY one compact JSON object: {"decision": 1 or -1, "reason": "<≤10 words>"}
- No extra text outside the JSON."""


def prompt_zero_shot(
    clean: Dict[str, str],
    dirty: Dict[str, str],
    blind: Dict[str, str],
    row_id: int,
) -> str:
    bundle = {
        "columns":  ADULT_COLS,
        "clean_row": clean,
        "dirty_row": dirty,
        "change_mask": blind,
    }
    return (
        f"{_ROLE}\n\n"
        "Return ONLY: {\"decision\": 1 or -1, \"reason\": \"<short>\"}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return the JSON now."
    )


def prompt_info(
    clean: Dict[str, str],
    dirty: Dict[str, str],
    blind: Dict[str, str],
    row_id: int,
) -> str:
    diffs = compute_diffs(clean, dirty, blind)
    bundle = {
        "columns":    ADULT_COLS,
        "clean_row":  clean,
        "dirty_row":  dirty,
        "change_mask": blind,
        "diffs":      diffs,
    }
    return (
        f"{_ROLE}\n\n"
        f"{_RUBRIC}\n\n"
        f"{_OUTPUT_RULES}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return the JSON now."
    )


def prompt_few_shot(
    clean: Dict[str, str],
    dirty: Dict[str, str],
    blind: Dict[str, str],
    row_id: int,
) -> str:
    diffs = compute_diffs(clean, dirty, blind)
    bundle = {
        "columns":    ADULT_COLS,
        "clean_row":  clean,
        "dirty_row":  dirty,
        "change_mask": blind,
        "diffs":      diffs,
    }
    return (
        f"{_ROLE}\n\n"
        f"{_RUBRIC}\n\n"
        f"{_OUTPUT_RULES}\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        "--- NOW YOUR TURN ---\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return the JSON now."
    )


VARIANTS: Dict[str, callable] = {
    "zero_shot": prompt_zero_shot,
    "info":      prompt_info,
    "few_shot":  prompt_few_shot,
}

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Cell-level binary metrics: intentional=1, unintentional=-1."""
    from collections import Counter
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == -1)
    acc   = (tp + tn) / len(y_true) if y_true else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    prec_u = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_u  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_u   = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0
    n     = len(y_true)
    f1_w  = (f1 * (tp + fn) + f1_u * (tn + fp)) / n if n > 0 else 0
    return {
        "n_records": n,
        "accuracy":           round(acc, 4),
        "precision_int":      round(prec, 4),
        "recall_int":         round(rec, 4),
        "f1_intentional":     round(f1, 4),
        "precision_unint":    round(prec_u, 4),
        "recall_unint":       round(rec_u, 4),
        "f1_unintentional":   round(f1_u, 4),
        "f1_weighted":        round(f1_w, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_counts": dict(Counter(y_pred)),
        "true_counts": dict(Counter(y_true)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Single-variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant_name: str,
    prompt_fn: callable,
    model,
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    n_records: int,
    out_dir: str,
) -> Dict:
    """
    Run one Gemini variant over n_records dirty rows.

    For the 1:1 mixed dataset, dirty_df row i corresponds to clean_df row i.
    The ground-truth intent for row i is read from mask_df row i:
      any cell == 1  → record intent = 1  (intentional)
      any cell == -1 → record intent = -1 (unintentional)
      (A record cannot be both; mixed records are possible in theory but
       the Kireev pipeline marks all changed cells with the same sign.)

    Returns a summary dict.
    """
    v_dir = os.path.join(out_dir, variant_name)
    os.makedirs(v_dir, exist_ok=True)

    pred_rows:  List[Dict] = []
    y_true:     List[int]  = []
    y_pred:     List[int]  = []
    llm_times:  List[float] = []
    fallback_count = 0
    status_hist = {"ok": 0, "api_error": 0, "parse_error": 0, "fallback": 0}

    print(f"\n{'='*65}")
    print(f"  VARIANT: {variant_name.upper()}  ({n_records} ERRONEOUS records)")
    print(f"{'='*65}")

    # Pre-collect only the rows that actually have errors (mask != 0 somewhere)
    # so --n_records means "n_records with errors", not n_records total rows.
    error_row_indices = [
        i for i in range(len(dirty_df))
        if (mask_df.iloc[i] != 0).any()
    ]
    indices_to_run = error_row_indices[:n_records]
    print(f"  (Total rows with errors in dataset: {len(error_row_indices)})")

    for i in tqdm(indices_to_run, desc=f"{variant_name}", ncols=90):
        dirty_row = row_to_dict(dirty_df, i)
        clean_row = row_to_dict(clean_df, i)
        blind     = blind_mask_row(mask_df, i)

        # Ground-truth intent: majority sign of non-zero mask cells
        mask_vals = [
            int(mask_df.at[i, c])
            for c in ADULT_COLS
            if str(mask_df.at[i, c]).strip() not in ("0", "0.0", "")
        ]
        from collections import Counter
        true_intent = Counter(mask_vals).most_common(1)[0][0]
        true_intent = 1 if true_intent > 0 else -1

        # Build prompt
        prompt_text = prompt_fn(clean_row, dirty_row, blind, row_id=i)

        # Call model
        t0 = time.perf_counter()
        raw = call_gemini_with_fallback(model, model_flash, prompt_text)
        elapsed = time.perf_counter() - t0
        llm_times.append(elapsed)

        if not raw:
            status_hist["api_error"] += 1
        else:
            status_hist["ok"] += 1

        # Parse
        obj = parse_llm_json(raw)
        pred_intent = extract_decision(obj, raw)

        if pred_intent is None:
            pred_intent = -1   # conservative fallback
            fallback_count += 1
            status_hist["fallback"] += 1
            reason = "parse_fallback"
        else:
            reason = str(obj.get("reason", "")) if isinstance(obj, dict) else "scalar"

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        pred_rows.append({
            "row_id":       i,
            "true_intent":  true_intent,
            "pred_intent":  pred_intent,
            "reason":       reason,
            "llm_time_s":   round(elapsed, 3),
            "raw_response": raw[:300] if raw else "",
        })

    # Save per-record predictions
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(v_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["variant"] = variant_name
    metrics["fallback_count"] = fallback_count
    metrics["status_hist"] = status_hist
    metrics["avg_llm_time_s"] = round(sum(llm_times) / len(llm_times), 3) if llm_times else 0
    metrics["total_llm_time_s"] = round(sum(llm_times), 2)

    metrics_path = os.path.join(v_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Console summary
    print(f"\n  Results — {variant_name}:")
    print(f"    Records processed : {metrics['n_records']}")
    print(f"    Accuracy          : {metrics['accuracy']:.4f}")
    print(f"    F1 intentional    : {metrics['f1_intentional']:.4f}")
    print(f"    F1 unintentional  : {metrics['f1_unintentional']:.4f}")
    print(f"    F1 weighted       : {metrics['f1_weighted']:.4f}")
    print(f"    Fallbacks         : {fallback_count}")
    print(f"    Avg LLM time      : {metrics['avg_llm_time_s']:.2f}s")
    print(f"    Saved → {v_dir}")

    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run three Gemini intent-attribution variants on the Mixed SOTA Adult Income dataset."
    )
    parser.add_argument("--dirty",     default=DEFAULT_DIRTY, help="Path to dirty CSV")
    parser.add_argument("--clean",     default=DEFAULT_CLEAN, help="Path to clean CSV")
    parser.add_argument("--mask",      default=DEFAULT_MASK,  help="Path to combined mask CSV")
    parser.add_argument(
        "--n_records", type=int, default=None,
        help="Number of dirty records to process (default: ALL). Use 100 for a quick test.",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: auto-generated timestamped folder next to this script)",
    )
    parser.add_argument(
        "--variants", default="zero_shot,info,few_shot",
        help="Comma-separated variants to run (default: zero_shot,info,few_shot)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name")
    args = parser.parse_args()

    # Output directory — all runs live under a fixed output/ folder
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "output",
            f"mixed_sota_gemini_{ts}",
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # Which variants to run
    requested = [v.strip() for v in args.variants.split(",")]
    unknown   = [v for v in requested if v not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Choose from: {list(VARIANTS)}")

    # Load data
    print(f"\nLoading data...")
    dirty_df = pd.read_csv(args.dirty, dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    mask_df  = pd.read_csv(args.mask,  keep_default_na=False)   # numeric mask

    for df, name in [(dirty_df, "dirty"), (clean_df, "clean")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            sys.exit(f"[{name}] missing columns: {missing}")

    if len(dirty_df) != len(clean_df) or len(dirty_df) != len(mask_df):
        sys.exit(
            f"Row count mismatch: dirty={len(dirty_df)}, "
            f"clean={len(clean_df)}, mask={len(mask_df)}"
        )

    n_total   = len(dirty_df)
    n_errors  = int((mask_df != 0).any(axis=1).sum())
    n_records = min(args.n_records, n_errors) if args.n_records else n_errors

    print(f"  Total rows in dataset : {n_total}")
    print(f"  Rows with errors      : {n_errors}")
    print(f"  Records to process    : {n_records}")
    print(f"  Variants to run       : {requested}")
    print(f"  Model                 : gemini-2.5-flash-lite (fallback: gemini-2.5-flash)")
    print(f"  Output dir            : {args.out_dir}")

    # Configure Gemini
    if not GEMINI_API_KEY:
        sys.exit("GEMINI_API_KEY is not set. Check config.py or set the environment variable.")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")  # tier 1; ignores args.model on purpose -- cost policy
    model_flash = genai.GenerativeModel("gemini-2.5-flash")  # tier 2 fallback

    # Run each variant
    t_start    = time.perf_counter()
    all_metrics: List[Dict] = []

    for variant_name in requested:
        prompt_fn = VARIANTS[variant_name]
        metrics   = run_variant(
            variant_name=variant_name,
            prompt_fn=prompt_fn,
            model=model,
            dirty_df=dirty_df,
            clean_df=clean_df,
            mask_df=mask_df,
            n_records=n_records,
            out_dir=args.out_dir,
        )
        all_metrics.append(metrics)

    # Cross-variant comparison table
    print(f"\n{'='*65}")
    print("CROSS-VARIANT COMPARISON")
    print(f"{'='*65}")
    header = f"{'Variant':<14} {'Acc':>6} {'F1_int':>8} {'F1_unint':>10} {'F1_w':>7} {'Fall':>5} {'AvgT':>6}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        print(
            f"{m['variant']:<14} "
            f"{m['accuracy']:>6.4f} "
            f"{m['f1_intentional']:>8.4f} "
            f"{m['f1_unintentional']:>10.4f} "
            f"{m['f1_weighted']:>7.4f} "
            f"{m['fallback_count']:>5} "
            f"{m['avg_llm_time_s']:>6.2f}s"
        )

    # Save comparison table
    comp_df   = pd.DataFrame(all_metrics)
    comp_path = os.path.join(args.out_dir, "comparison_table.csv")
    comp_df.to_csv(comp_path, index=False)

    # Master run metadata
    meta = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "model": "gemini-2.5-flash-lite+fallback:gemini-2.5-flash",
        "n_records_requested": n_records,
        "n_total_rows":        n_total,
        "n_error_rows":        n_errors,
        "dirty_path":  args.dirty,
        "clean_path":  args.clean,
        "mask_path":   args.mask,
        "variants_run": requested,
        "total_wall_time_s": round(time.perf_counter() - t_start, 1),
        "out_dir": args.out_dir,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  Comparison table → {comp_path}")
    print(f"  Total wall time  : {meta['total_wall_time_s']:.0f}s")
    print(f"\n✓ Done. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
