#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Attribution — Three R1-Qwen Variants — Mixed SOTA Adult Income Dataset
==============================================================================

Starts the DeepSeek-R1-Distill-Qwen-32B TGI Docker container, waits for the
server to be ready, then runs three prompt strategies back-to-back on the
Mixed SOTA Adult Income dataset (1:1 dirty-to-clean ratio):

  Variant 1 — zero-shot : plain JSON bundle, no rubric, no diffs
  Variant 2 — info      : adds per-cell DIFFS + decision rubric
  Variant 3 — few-shot  : info + two in-context examples

Model : deepseek-ai/DeepSeek-R1-Distill-Qwen-32B served by TGI on port 6800
Template: Qwen ChatML  (<|im_start|> ... <|im_end|>)
Seed trick: assistant turn opened with '{"0":' for info/few-shot variants.
Thinking: model emits <think>...</think> before the JSON — stripped before parsing.

Usage
-----
  python run_r1qwen_mixed_sota.py --n_records 100   # quick test
  python run_r1qwen_mixed_sota.py                   # full run
  python run_r1qwen_mixed_sota.py --no_start_server --no_stop_server

Arguments
---------
  --dirty            Path to dirty CSV
  --clean            Path to clean CSV
  --mask             Path to combined mask CSV (1=intentional, -1=unintentional, 0=clean)
  --n_records        How many error records to process (default: ALL)
  --out_dir          Output directory (default: auto-generated timestamped folder)
  --variants         Comma-separated: zero_shot,info,few_shot  (default: all)
  --no_start_server  Skip Docker Compose startup
  --no_stop_server   Skip Docker Compose teardown after run
  --server_url       TGI endpoint (default: http://127.0.0.1:6800/generate)
  --compose_file     Path to docker-compose YAML (default: auto-detected)
  --server_wait      Seconds to wait for readiness (default: 600)
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ── defaults ─────────────────────────────────────────────────────────────────
BASE = "/home/mohamed/error_injector/llms_baseline/mixed_error_pipeline/output"

DEFAULT_DIRTY = os.path.join(BASE, "adult_phase2_final.csv")
DEFAULT_CLEAN = os.path.join(BASE, "adult_clean.csv")
DEFAULT_MASK  = os.path.join(BASE, "mask_combined.csv")

DEFAULT_SERVER_URL = "http://127.0.0.1:6800/generate"
DEFAULT_HEALTH_URL = "http://127.0.0.1:6800/health"

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
DEFAULT_COMPOSE_FILE = os.path.join(_REPO_ROOT, "docker-compose", "docker-compose-deepseek-R1.yml")

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]

# Qwen ChatML tokens
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"

# JSON seed — coaxes compact map output, same trick as Qwen
_SEED_MAP = '{"0":'

# Per-variant token budgets (R1 thinks first, so we allow more tokens)
_TOKENS = {
    "zero_shot": 1024,
    "info":       512,
    "few_shot":   512,
}

REQUEST_TIMEOUT_SEC = 180   # R1 spends time thinking — allow extra


# ─────────────────────────────────────────────────────────────────────────────
# Docker Compose helpers
# ─────────────────────────────────────────────────────────────────────────────

def start_server(compose_file: str, health_url: str, wait_sec: int = 600) -> None:
    if not os.path.exists(compose_file):
        sys.exit(f"[ERROR] docker-compose file not found: {compose_file}\n"
                 f"Pass --compose_file <path> to specify it explicitly.")
    print(f"\n[Server] Starting DeepSeek-R1-Qwen TGI via Docker Compose…")
    print(f"         compose file : {compose_file}")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "-d"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] docker compose up stderr:\n{result.stderr}")
    else:
        print(f"[Server] Container started (or already running).")
    _wait_for_server(health_url, wait_sec)


def stop_server(compose_file: str) -> None:
    if not os.path.exists(compose_file):
        print(f"[WARN] compose file not found, skipping teardown: {compose_file}")
        return
    print(f"\n[Server] Stopping DeepSeek-R1-Qwen TGI container…")
    result = subprocess.run(
        ["docker", "compose", "-f", compose_file, "down"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[WARN] docker compose down stderr:\n{result.stderr}")
    else:
        print("[Server] Container stopped.")


def _wait_for_server(health_url: str, timeout_sec: int) -> None:
    print(f"[Server] Waiting for TGI to be ready (up to {timeout_sec}s)…", flush=True)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                print(f"\n[Server] TGI is ready ✓")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    sys.exit(
        f"\n[ERROR] TGI did not become ready within {timeout_sec}s. "
        "Check Docker logs: docker compose logs text-generation"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}

def blind_mask_row(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {
        c: ("1" if str(mask_df.at[idx, c]).strip() not in ("0", "0.0", "") else "0")
        for c in ADULT_COLS
    }

def compute_diffs(
    clean: Dict[str, str], dirty: Dict[str, str], blind: Dict[str, str],
) -> List[Dict[str, str]]:
    return [
        {"column": c, "from": clean.get(c, ""), "to": dirty.get(c, "")}
        for c in ADULT_COLS
        if c != "class" and blind.get(c) == "1"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Qwen ChatML templates
# ─────────────────────────────────────────────────────────────────────────────

def qwen_chat(system_text: str, user_text: str) -> str:
    """Standard Qwen ChatML — no seed (zero-shot)."""
    return (
        f"{_IM_START}system\n{system_text}{_IM_END}\n"
        f"{_IM_START}user\n{user_text}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )

def qwen_chat_seed(system_text: str, user_text: str, seed: str = _SEED_MAP) -> str:
    """Qwen ChatML with assistant turn seeded to force compact JSON (info/few-shot)."""
    return (
        f"{_IM_START}system\n{system_text}{_IM_END}\n"
        f"{_IM_START}user\n{user_text}{_IM_END}\n"
        f"{_IM_START}assistant\n{seed}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TGI call
# ─────────────────────────────────────────────────────────────────────────────

def call_tgi(
    inputs_str: str,
    max_new_tokens: int,
    server_url: str,
    stop_extra: Optional[List[str]] = None,
) -> str:
    stop_tokens = [_IM_END]
    if stop_extra:
        stop_tokens.extend(stop_extra)
    payload = {
        "inputs": inputs_str,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
            "details": True,
            "stop": stop_tokens,
        },
    }
    r = requests.post(server_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, list) and data:
        return data[0].get("generated_text") or data[0].get("text", "")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Seed reattachment + think-block stripping
# ─────────────────────────────────────────────────────────────────────────────

def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by R1 models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()

def reattach_seed(continuation: str, seed: str = _SEED_MAP) -> str:
    """
    Strip think blocks, reattach the JSON seed, trim after first '}'.
    Returns '' if nothing usable remains.
    """
    s = strip_think(continuation or "")
    for junk in ("```json", "```", _IM_END):
        s = s.replace(junk, "")
    idx = s.find("}")
    if idx != -1:
        s = s[: idx + 1].strip()
    else:
        s = s.strip()
    return (seed + s) if s else ""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = strip_think(text.lstrip("\ufeff").strip())
    for junk in ("```json", "```", "~~~json", "~~~", _IM_END):
        s = s.replace(junk, "")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    # Extract first balanced JSON fragment
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
                    frag = s[start : i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        try:
                            obj = ast.literal_eval(frag)
                            return obj if isinstance(obj, (dict, list)) else None
                        except Exception:
                            return None
    return None


def extract_decision(obj: Optional[dict], raw: str) -> Optional[int]:
    """Extract single decision (1 or -1) from parsed JSON or raw text."""
    # Strip think block from raw before checking bare scalar
    stripped = strip_think(raw.strip().lstrip("\ufeff"))
    for fence in ("```json", "```"):
        stripped = stripped.replace(fence, "").strip()
    if stripped in ("1", "-1"):
        return int(stripped)

    if obj is None:
        return None

    if isinstance(obj, dict):
        # Compact-map seed schema: {"0": 1}
        if "0" in obj:
            try:
                v = obj["0"]
                vi = int(v) if not isinstance(v, bool) else (1 if v else -1)
                return 1 if vi >= 1 else -1
            except Exception:
                pass
        # Named-key schemas
        for key in ("decision", "intent", "intentional", "label", "result"):
            v = obj.get(key)
            if v is not None:
                try:
                    vi = int(v) if not isinstance(v, bool) else (1 if v else -1)
                    return 1 if vi >= 1 else -1
                except Exception:
                    pass
        # Single-key fallback
        if len(obj) == 1:
            v = list(obj.values())[0]
            try:
                vi = int(v) if not isinstance(v, bool) else (1 if v else -1)
                return 1 if vi >= 1 else -1
            except Exception:
                pass

    if isinstance(obj, (int, float)):
        return 1 if int(obj) >= 1 else -1
    if isinstance(obj, list) and obj:
        try:
            return 1 if int(obj[0]) >= 1 else -1
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
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

_OUTPUT_RULES_PLAIN = (
    "Return ONLY one compact JSON object: "
    '{"decision": 1 or -1, "reason": "<≤10 words>"}\n'
    "No extra text outside the JSON."
)

_OUTPUT_RULES_SEED = (
    "Rules:\n"
    "- Return ONLY compact JSON. Use exactly this format:\n"
    '  {"0": 1 or -1}\n'
    "  (key \"0\" = decision for this record)\n"
    "- No extra text outside the JSON."
)


def _bundle(clean, dirty, blind, with_diffs: bool = False) -> dict:
    b = {"columns": ADULT_COLS, "clean_row": clean, "dirty_row": dirty, "change_mask": blind}
    if with_diffs:
        b["diffs"] = compute_diffs(clean, dirty, blind)
    return b


def prompt_zero_shot(
    clean: Dict[str, str], dirty: Dict[str, str], blind: Dict[str, str], row_id: int,
) -> Tuple[str, bool]:
    user_text = (
        f"{_OUTPUT_RULES_PLAIN}\n\n"
        f"INPUT:\n{json.dumps(_bundle(clean, dirty, blind), ensure_ascii=False)}\n\n"
        "Return the JSON now."
    )
    return qwen_chat(_ROLE, user_text), False


def prompt_info(
    clean: Dict[str, str], dirty: Dict[str, str], blind: Dict[str, str], row_id: int,
) -> Tuple[str, bool]:
    system_text = f"{_ROLE}\n\n{_RUBRIC}\n\n{_OUTPUT_RULES_SEED}"
    user_text   = (
        f"INPUT:\n{json.dumps(_bundle(clean, dirty, blind, with_diffs=True), ensure_ascii=False)}\n"
        "Return ONLY one JSON now."
    )
    return qwen_chat_seed(system_text, user_text, seed=_SEED_MAP), True


def prompt_few_shot(
    clean: Dict[str, str], dirty: Dict[str, str], blind: Dict[str, str], row_id: int,
) -> Tuple[str, bool]:
    system_text = f"{_ROLE}\n\n{_RUBRIC}\n\n{_OUTPUT_RULES_SEED}"
    user_text   = (
        f"{_FEW_SHOT_EXAMPLES}\n"
        "--- NOW YOUR TURN ---\n"
        f"INPUT:\n{json.dumps(_bundle(clean, dirty, blind, with_diffs=True), ensure_ascii=False)}\n"
        "Return ONLY one JSON now."
    )
    return qwen_chat_seed(system_text, user_text, seed=_SEED_MAP), True


VARIANTS: Dict[str, callable] = {
    "zero_shot": prompt_zero_shot,
    "info":      prompt_info,
    "few_shot":  prompt_few_shot,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    assert len(y_true) == len(y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == -1)
    n  = len(y_true)
    acc    = (tp + tn) / n if n else 0
    prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_i  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_i   = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) > 0 else 0
    prec_u = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_u  = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_u   = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) > 0 else 0
    f1_w   = (f1_i * (tp + fn) + f1_u * (tn + fp)) / n if n > 0 else 0
    return {
        "n_records":          n,
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
    prompt_fn: callable,
    server_url: str,
    dirty_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    n_records: int,
    out_dir: str,
) -> Dict:
    v_dir = os.path.join(out_dir, variant_name)
    os.makedirs(v_dir, exist_ok=True)

    max_new_tokens = _TOKENS[variant_name]

    pred_rows:   List[Dict]  = []
    y_true:      List[int]   = []
    y_pred:      List[int]   = []
    llm_times:   List[float] = []
    fallback_count = 0
    status_hist = {"ok": 0, "http_error": 0, "parse_error": 0, "fallback": 0}

    print(f"\n{'='*65}")
    print(f"  VARIANT: {variant_name.upper()}  ({n_records} ERRONEOUS records)")
    print(f"  max_new_tokens={max_new_tokens}")
    print(f"{'='*65}")

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

        mask_vals = [
            int(mask_df.at[i, c])
            for c in ADULT_COLS
            if str(mask_df.at[i, c]).strip() not in ("0", "0.0", "")
        ]
        true_intent = Counter(mask_vals).most_common(1)[0][0]
        true_intent = 1 if true_intent > 0 else -1

        inputs_str, uses_seed = prompt_fn(clean_row, dirty_row, blind, row_id=i)

        t0 = time.perf_counter()
        raw_continuation = ""
        try:
            stop_extra = ["```", "\n\n"] if uses_seed else None
            raw_continuation = call_tgi(inputs_str, max_new_tokens, server_url, stop_extra)
            status_hist["ok"] += 1
        except requests.HTTPError:
            status_hist["http_error"] += 1
        except Exception:
            status_hist["parse_error"] += 1
        elapsed = time.perf_counter() - t0
        llm_times.append(elapsed)

        if uses_seed:
            llm_text = reattach_seed(raw_continuation, seed=_SEED_MAP)
        else:
            llm_text = strip_think(raw_continuation)

        obj = parse_llm_json(llm_text)
        pred_intent = extract_decision(obj, llm_text)

        if pred_intent is None:
            pred_intent = -1
            fallback_count += 1
            status_hist["fallback"] += 1
            reason = "parse_fallback"
        else:
            if isinstance(obj, dict):
                reason = str(obj.get("reason", obj.get("0", ""))).strip() or "ok"
            else:
                reason = "scalar"

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        pred_rows.append({
            "row_id":       i,
            "true_intent":  true_intent,
            "pred_intent":  pred_intent,
            "reason":       reason,
            "llm_time_s":   round(elapsed, 3),
            "raw_response": (raw_continuation or "")[:300],
        })

    pred_df   = pd.DataFrame(pred_rows)
    pred_path = os.path.join(v_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(y_true, y_pred)
    metrics["variant"]          = variant_name
    metrics["fallback_count"]   = fallback_count
    metrics["status_hist"]      = status_hist
    metrics["avg_llm_time_s"]   = round(sum(llm_times) / len(llm_times), 3) if llm_times else 0
    metrics["total_llm_time_s"] = round(sum(llm_times), 2)

    metrics_path = os.path.join(v_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run three R1-Qwen intent-attribution variants on the Mixed SOTA Adult Income dataset."
    )
    parser.add_argument("--dirty",     default=DEFAULT_DIRTY)
    parser.add_argument("--clean",     default=DEFAULT_CLEAN)
    parser.add_argument("--mask",      default=DEFAULT_MASK)
    parser.add_argument("--n_records", type=int, default=None)
    parser.add_argument("--out_dir",   default=None)
    parser.add_argument("--variants",  default="zero_shot,info,few_shot")
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--compose_file", default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--server_wait", type=int, default=600)
    parser.add_argument("--no_start_server", action="store_true")
    parser.add_argument("--no_stop_server",  action="store_true")
    args = parser.parse_args()

    # ── Output directory ─────────────────────────────────────────────────────
    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(_SCRIPT_DIR, "output", f"mixed_sota_r1qwen_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Variants ─────────────────────────────────────────────────────────────
    requested = [v.strip() for v in args.variants.split(",")]
    unknown   = [v for v in requested if v not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Choose from: {list(VARIANTS)}")

    # ── Start server ─────────────────────────────────────────────────────────
    health_url = args.server_url.replace("/generate", "/health")
    if not args.no_start_server:
        start_server(args.compose_file, health_url, wait_sec=args.server_wait)
    else:
        print("[Server] Skipping startup (--no_start_server). Assuming TGI is already running.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data…")
    dirty_df = pd.read_csv(args.dirty, dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    mask_df  = pd.read_csv(args.mask,  keep_default_na=False)

    for df, name in [(dirty_df, "dirty"), (clean_df, "clean")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            sys.exit(f"[{name}] missing columns: {missing}")

    if not (len(dirty_df) == len(clean_df) == len(mask_df)):
        sys.exit(f"Row count mismatch: dirty={len(dirty_df)}, clean={len(clean_df)}, mask={len(mask_df)}")

    n_total   = len(dirty_df)
    n_errors  = int((mask_df != 0).any(axis=1).sum())
    n_records = min(args.n_records, n_errors) if args.n_records else n_errors

    print(f"  Total rows in dataset : {n_total}")
    print(f"  Rows with errors      : {n_errors}")
    print(f"  Records to process    : {n_records}")
    print(f"  Variants to run       : {requested}")
    print(f"  TGI server            : {args.server_url}")
    print(f"  Output dir            : {args.out_dir}")

    # ── Run each variant ─────────────────────────────────────────────────────
    t_start     = time.perf_counter()
    all_metrics: List[Dict] = []

    try:
        for variant_name in requested:
            metrics = run_variant(
                variant_name=variant_name,
                prompt_fn=VARIANTS[variant_name],
                server_url=args.server_url,
                dirty_df=dirty_df,
                clean_df=clean_df,
                mask_df=mask_df,
                n_records=n_records,
                out_dir=args.out_dir,
            )
            all_metrics.append(metrics)

    finally:
        if not args.no_stop_server and not args.no_start_server:
            stop_server(args.compose_file)

    # ── Cross-variant comparison ──────────────────────────────────────────────
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

    comp_df   = pd.DataFrame(all_metrics)
    comp_path = os.path.join(args.out_dir, "comparison_table.csv")
    comp_df.to_csv(comp_path, index=False)

    meta = {
        "run_timestamp":       datetime.datetime.now().isoformat(),
        "model":               "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B (TGI)",
        "server_url":          args.server_url,
        "compose_file":        args.compose_file,
        "n_records_requested": n_records,
        "n_total_rows":        n_total,
        "n_error_rows":        n_errors,
        "dirty_path":          args.dirty,
        "clean_path":          args.clean,
        "mask_path":           args.mask,
        "variants_run":        requested,
        "total_wall_time_s":   round(time.perf_counter() - t_start, 1),
        "out_dir":             args.out_dir,
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  Comparison table → {comp_path}")
    print(f"  Total wall time  : {meta['total_wall_time_s']:.0f}s")
    print(f"\n✓ Done. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
