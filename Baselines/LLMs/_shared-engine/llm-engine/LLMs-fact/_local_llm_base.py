#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared base module for TabFact Intent Attribution via LOCAL LLMs (TGI).
=======================================================================

Supported models (each served by its own TGI container on a different port):
    mixtral   → port 6000, [INST] ... [/INST]
    llama     → port 6100, Llama-3 chat template
    qwen      → port 6300, ChatML (<|im_start|>...<|im_end|>)
    r1-qwen   → port 6800, DeepSeek-R1 chat template

This module provides:
    • call_tgi(server_url, prompt, ...)            – HTTP call to TGI
    • MODELS                                       – per-model config table
    • wrap_for_model(model_key, user_content)      – wrap raw prompt text
    • run_pipeline_tgi(make_core_prompt, model_key, out_dir, variant_name, ...)
      Mirrors _gemini_base.run_pipeline but uses the local TGI endpoint.

The data-loading machinery (dirty+corrected+mask alignment) is imported
from _gemini_base, so the ground-truth stays identical across all
models.
"""

from __future__ import annotations
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from _gemini_base import (
    ATTR_COLS,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
    CHUNK_SIZE, MAX_RECORDS,
    _row_to_dict, _mask_cell_is_error,
    load_tabfact_attribution_data,
    parse_llm_json, decisions_from_obj,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-model configuration
# ─────────────────────────────────────────────────────────────────────────────
MODELS: Dict[str, Dict] = {
    "mixtral": {
        "server_url":      "http://127.0.0.1:6000/generate",
        "display_name":    "Mixtral",
        "max_new_tokens":  1500,
        "timeout":         240,
        "stop_tokens":     [],
    },
    "llama": {
        "server_url":      "http://127.0.0.1:6100/generate",
        "display_name":    "Llama-3",
        "max_new_tokens":  1500,
        "timeout":         240,
        "stop_tokens":     ["<|eot_id|>"],
    },
    "qwen": {
        "server_url":      "http://127.0.0.1:6300/generate",
        "display_name":    "Qwen-2.5",
        "max_new_tokens":  1500,
        "timeout":         240,
        "stop_tokens":     ["<|im_end|>"],
    },
    "r1-qwen": {
        "server_url":      "http://127.0.0.1:6800/generate",
        "display_name":    "DeepSeek-R1-Qwen",
        "max_new_tokens":  3000,   # R1 thinks → needs more tokens
        "timeout":         360,
        "stop_tokens":     ["<｜end▁of▁sentence｜>"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Chat templates
# ─────────────────────────────────────────────────────────────────────────────
def _wrap_mixtral(user_content: str) -> str:
    return f"[INST] {user_content} [/INST]"


def _wrap_llama(user_content: str) -> str:
    system_text = ("You are a precise data-forensics assistant. Respond with "
                   "ONLY valid JSON, no prose, no markdown fences.")
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system_text}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _wrap_qwen(user_content: str) -> str:
    system_text = ("You are a precise data-forensics assistant. Respond with "
                   "ONLY valid JSON, no prose, no markdown fences.")
    return (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _wrap_r1(user_content: str) -> str:
    # DeepSeek-R1 uses <｜User｜> / <｜Assistant｜> markers
    return (
        f"<｜begin▁of▁sentence｜><｜User｜>{user_content}<｜Assistant｜>"
    )


_WRAPPERS: Dict[str, Callable[[str], str]] = {
    "mixtral": _wrap_mixtral,
    "llama":   _wrap_llama,
    "qwen":    _wrap_qwen,
    "r1-qwen": _wrap_r1,
}


def wrap_for_model(model_key: str, user_content: str) -> str:
    if model_key not in _WRAPPERS:
        raise ValueError(f"Unknown model: {model_key}. Known: {list(_WRAPPERS)}")
    return _WRAPPERS[model_key](user_content)


# ─────────────────────────────────────────────────────────────────────────────
# TGI call
# ─────────────────────────────────────────────────────────────────────────────
def call_tgi(server_url: str, prompt: str, max_new_tokens: int,
             timeout: int, stop_tokens: List[str],
             retries: int = 3, retry_delay: float = 10.0) -> str:
    params = {
        "max_new_tokens":   max_new_tokens,
        "do_sample":        False,
        "return_full_text": False,
    }
    if stop_tokens:
        params["stop"] = stop_tokens
    payload = {"inputs": prompt, "parameters": params}
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(server_url, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"] or ""
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return data[0].get("generated_text") or data[0].get("text", "") or ""
            return ""
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(retry_delay)
    # All retries exhausted
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# For R1 specifically: strip the <think>...</think> block if present
# ─────────────────────────────────────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    if not text:
        return text
    # Remove <think>…</think> (R1 chain-of-thought)
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also handle unclosed <think>…
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator (parallels _gemini_base.run_pipeline)
# ─────────────────────────────────────────────────────────────────────────────
PromptBuilder = Callable[
    [List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[int]],
    str,
]


def run_pipeline_tgi(
    make_core_prompt: PromptBuilder,
    model_key: str,
    out_dir: str,
    variant_name: str,
    dirty_path: str = DIRTY_CSV,
    mask_path:  str = MASK_CSV,
    expl_path:  str = EXPLANATIONS_JSON,
    orig_path:  str = ORIG_DIRTY_CSV,
    max_records: Optional[int] = MAX_RECORDS,
    chunk_size: int = CHUNK_SIZE,
) -> str:
    """
    Run one (model, variant) combination end-to-end.

    ``make_core_prompt`` is the same kind of builder used with Gemini:
    it returns the plain user content (role+rubric+rules+input JSON).
    We then wrap it in the model's chat template before POSTing to TGI.

    Writes intent_labels.csv, intent_explanations.csv, run_stats.json,
    per_chunk_times.csv into ``out_dir``.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model: {model_key}. Known: {list(MODELS)}")
    cfg = MODELS[model_key]

    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    labels_csv    = os.path.join(out_dir, "intent_labels.csv")
    expl_csv      = os.path.join(out_dir, "intent_explanations.csv")
    stats_json    = os.path.join(out_dir, "run_stats.json")
    per_chunk_csv = os.path.join(out_dir, "per_chunk_times.csv")

    # ── Load aligned data
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

    # ── Prepare outputs
    out_mask = mask.copy(deep=True)
    explanations_rows: List[Dict] = []
    chunk_times: List[float] = []
    llm_times:   List[float] = []
    status = {"ok": 0, "parse_error": 0, "api_error": 0}
    successful_chunks = 0

    n_chunks = (n_records + chunk_size - 1) // chunk_size
    print(f"[run] model={cfg['display_name']}  variant={variant_name}  "
          f"records={n_records}  chunks={n_chunks}  endpoint={cfg['server_url']}")

    for ci in tqdm(range(n_chunks), desc=f"{model_key}/{variant_name}", ncols=100):
        ct0 = time.perf_counter()
        s = ci * chunk_size
        e = min(s + chunk_size, n_records)

        dirty_rows     = [_row_to_dict(dirty,     i, ATTR_COLS) for i in range(s, e)]
        corrected_rows = [_row_to_dict(corrected, i, ATTR_COLS) for i in range(s, e)]
        mask_rows      = [{c: ("1" if _mask_cell_is_error(mask.at[i, c]) else "0")
                           for c in ATTR_COLS}
                          for i in range(s, e)]
        global_ids = list(range(s, e))

        core_prompt = make_core_prompt(corrected_rows, dirty_rows, mask_rows, global_ids)
        full_prompt = wrap_for_model(model_key, core_prompt)

        l0 = time.perf_counter()
        llm_text = call_tgi(
            cfg["server_url"], full_prompt,
            max_new_tokens=cfg["max_new_tokens"],
            timeout=cfg["timeout"],
            stop_tokens=cfg["stop_tokens"],
        )
        # Strip R1 chain-of-thought
        if model_key == "r1-qwen":
            llm_text = _strip_thinking(llm_text)
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
        "chunk_id":       list(range(n_chunks)),
        "chunk_time_sec": chunk_times,
        "llm_time_sec":   llm_times,
    }).to_csv(per_chunk_csv, index=False)

    flat = out_mask[ATTR_COLS].values.flatten().tolist()
    counts = pd.Series([str(x) for x in flat]).value_counts().to_dict()

    t1 = time.perf_counter()
    stats = {
        "variant":        variant_name,
        "model_key":      model_key,
        "model":          cfg["display_name"],
        "endpoint":       cfg["server_url"],
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
            "intentional (1)":    counts.get("1", 0),
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

    print(f"\n✅ {model_key}/{variant_name} done in {t1-t0:.1f}s")
    print(f"   labels: {labels_csv}")
    print(f"   stats:  {stats_json}")
    print(f"   decisions: 1={counts.get('1',0)}  -1={counts.get('-1',0)}  "
          f"0={counts.get('0',0)}")
    return labels_csv
