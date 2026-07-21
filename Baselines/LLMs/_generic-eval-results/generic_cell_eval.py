#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generic_cell_eval.py
=====================
Dataset-agnostic, cell-level LLM intent-attribution baseline (bare-minimum
prompt). Reuses fraud_baseline.LOADERS so it runs identically on any of the
8 registered datasets (including eBay) instead of depending on per-dataset
hardcoded scripts that can silently drift out of sync with the canonical
dataset (e.g. the older klim-kireev twitter-bot scripts point at a stale
Dec-2025 "combined_dataset.csv" rather than the current gemini-run_v2 data).

Mirrors the existing bare-min prompt convention (see
klim-kireev/datasets/twitter-bot/intent-attribution/intent-attribution-baremin-qwen.py):
JSON bundle of {clean, changed, mask} per record, one feature_decisions
entry per changed cell, decision in {+1, -1}.

Backends: Gemini (API) and any TGI-hosted local model (Qwen/Llama/Mixtral/
DeepSeek-R1) via a generic chat-template + /generate endpoint.

Usage:
  python generic_cell_eval.py --dataset ebay --model gemini
  python generic_cell_eval.py --dataset ebay --model qwen --port 6300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
FRAUD = os.path.join(ROOT, "fraud_baseline")
sys.path.insert(0, FRAUD)
from datasets import LOADERS  # noqa: E402

CHUNK_SIZE = 7
# Gemini 2.5 Pro's internal "thinking" tokens count against max_output_tokens
# and leave none for the visible answer if this is set too low (observed:
# finish_reason=MAX_TOKENS / SAFETY with empty .text at 1400) -- 1400 was
# fine for local TGI models (no hidden thinking budget) but not for Gemini.
MAX_NEW_TOKENS = 1400
GEMINI_MAX_NEW_TOKENS = 8192
REQUEST_TIMEOUT_SEC = 240
# Concurrent in-flight requests. Gemini: paid-tier API tolerates this fine.
# Local TGI: continuous batching means concurrent requests are *more*
# efficient, not less -- this is the intended usage pattern, not a hack.
MAX_CONCURRENCY = int(os.environ.get("LLM_EVAL_CONCURRENCY", "12"))

CHAT_TEMPLATES = {
    "qwen":    ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", ["<|im_end|>"]),
    "llama":   ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n", ["<|eot_id|>"]),
    "mixtral": ("[INST] {system}\n\n{user} [/INST]", ["</s>"]),
    "r1qwen":  ("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n", ["<|im_end|>"]),
}

ROLE_BLOCK = ("You are a data-quality analyst attributing the INTENT behind cell-level "
              "changes in a tabular dataset: was each change made DELIBERATELY to achieve "
              "some goal (intentional), or did it arise by ACCIDENT / data corruption "
              "(unintentional)?")

# Dataset-specific prompt variants, mirroring adult_row_level_eval.py's
# zero/info/few convention. Mined from the existing declarative/configs/*.txt
# domain descriptions (already written for User-Guided constraint extraction)
# rather than a single schema-free rubric shared across datasets -- a generic
# rubric with no column names defeats the point of testing "info"/"few"
# (whether domain-specific guidance helps), since it gives the LLM nothing
# more specific than the zero-shot prompt already implies.
_GENERIC_FALLBACK_INFO = (
    "\n\nDecision rubric:\n"
    "- Intentional: changes that plausibly serve a goal -- e.g. several "
    "fields on the same record changed together in a coordinated, "
    "self-consistent direction, or a field replaced with a "
    "placeholder/generic value that hides information.\n"
    "- Unintentional: isolated, incoherent changes with no apparent "
    "goal -- typos, truncation, encoding artefacts, a single value "
    "perturbed with no consistent direction relative to the rest of "
    "the record."
)
_GENERIC_FALLBACK_FEW = _GENERIC_FALLBACK_INFO + (
    "\n\nWorked examples:\n"
    "- Several fields shifted together in one consistent direction: intentional.\n"
    "- A field replaced with a placeholder/empty value, nothing else changed: "
    "intentional (obfuscation).\n"
    "- A single character substituted or truncated, unrelated to the rest "
    "of the record: unintentional.\n"
)

_TWITTERBOT_INFO = (
    "\n\nDecision rubric (TwiBot-20 profile features; a bot operator's goal is "
    "to make the account look more like a genuine human account to a bot "
    "classifier):\n"
    "- Intentional (bot-evasion): increasing followers_count, friends_count, "
    "favourites_count, statuses_count, or listed_count; changing "
    "default_profile or default_profile_image from 1 to 0; changing "
    "geo_enabled, profile_use_background_image, or has_location from 0 to 1; "
    "changing has_description from 0 to 1 together with description_length "
    "rising to a substantial positive value (up to 160).\n"
    "- NOT evasion-directed even though changed (treat as unintentional unless "
    "another evasion signal co-occurs): verified 0->1, protected 0->1 (these "
    "would invite more scrutiny, not less).\n"
    "- Unintentional (no plausible bot motive, implausible given platform "
    "limits): a negative value for any *_count field; description_length "
    "outside [0,160]; screen_name_length outside [1,15]; name_length outside "
    "[1,50]; has_description=1 with description_length=0 (or vice versa); any "
    "binary flag (verified, protected, default_profile, default_profile_image, "
    "geo_enabled, profile_use_background_image, has_created_date, "
    "has_description, has_location) taking a value other than 0 or 1."
)
_TWITTERBOT_FEW = _TWITTERBOT_INFO + (
    "\n\nWorked examples:\n"
    "- followers_count 120 -> 8500, friends_count 80 -> 3200, both rising "
    "together with no other change: intentional (bot-evasion).\n"
    "- default_profile_image 1 -> 0 together with has_description 0 -> 1 and "
    "description_length 0 -> 95: intentional (bot-evasion, coordinated).\n"
    "- followers_count -> -42 (negative): unintentional (implausible, not "
    "evasion-directed).\n"
    "- description_length -> 312 (exceeds Twitter's 160-char bio limit): "
    "unintentional.\n"
    "- verified 0 -> 1 alone, nothing else changed: unintentional (claiming "
    "verification invites scrutiny, not evasion).\n"
)

_EBAY_INFO = (
    "\n\nDecision rubric (real eBay listings; deliberate seller manipulation "
    "is hard to establish from a single row, since it is normally evidenced "
    "by cross-listing seller behaviour you cannot see here -- use only the "
    "row-visible signals below, and default to unintentional when none apply):\n"
    "- Unintentional (structural/parsing defects): HTML tags, encoding "
    "artefacts (mojibake), truncation markers, or control characters inside "
    "title/category/condition/spec_brand/spec_model/spec_type/spec_style; a "
    "placeholder string ('N/A', 'TBD', 'unknown', empty-but-non-null) used as "
    "a structured field's actual value; spec_upc not a valid 12-digit GS1 "
    "code, or spec_ean not a valid 13-digit GS1 code; negative or "
    "implausibly large (>20000) description_length; title under 5 or over 80 "
    "characters; seller_feedback_score negative; spec_brand differing from "
    "the dominant spelling only by letter case.\n"
    "- Possibly intentional (row-visible behavioural signal, weak evidence "
    "alone): price far below the typical price for the item's category and "
    "condition combined with the brand name absent from the title while a "
    "specific spec_upc/spec_ean is present (signature of a counterfeit-prone "
    "listing); returns_accepted=True with return_period_days outside eBay's "
    "standard options {7,14,30,60}.\n"
    "- Prefer unintentional for returns_accepted/return_period_days "
    "mismatches (e.g. accepted=True with period missing, or accepted=False "
    "with a period present) unless another low-trust signal co-occurs in the "
    "same row -- this is normally an incomplete form, not a deliberate choice."
)
_EBAY_FEW = _EBAY_INFO + (
    "\n\nWorked examples:\n"
    "- spec_upc changed to '12-345' (not 12 digits): unintentional (parsing "
    "defect).\n"
    "- title changed to include '&amp;' / mojibake characters: unintentional "
    "(scraping/encoding artefact).\n"
    "- spec_brand 'Nike' -> 'nike' (case-only): unintentional (eBay search is "
    "case-insensitive, this is a normalisation slip).\n"
    "- price dropped to $0.99 on a category where typical price is $80-120, "
    "AND spec_brand missing from title while spec_upc is present: intentional "
    "(counterfeit-prone listing pattern).\n"
    "- returns_accepted=True, return_period_days missing, no other unusual "
    "signal on the row: unintentional (incomplete form).\n"
)

DATASET_PROMPT_VARIANTS: Dict[str, Dict[str, str]] = {
    "twitterbot_llm": {"zero": "", "info": _TWITTERBOT_INFO, "few": _TWITTERBOT_FEW},
    "twitterbot_tfm": {"zero": "", "info": _TWITTERBOT_INFO, "few": _TWITTERBOT_FEW},
    "ebay": {"zero": "", "info": _EBAY_INFO, "few": _EBAY_FEW},
}

PROMPT_VARIANTS = {"zero": "", "info": _GENERIC_FALLBACK_INFO, "few": _GENERIC_FALLBACK_FEW}


def get_prompt_variant_text(dataset_key: str, prompt_variant: str) -> str:
    return DATASET_PROMPT_VARIANTS.get(dataset_key, PROMPT_VARIANTS)[prompt_variant]

SYS_RULES = """Rules:
- Analyze each pair of (clean, changed) records.
- For EACH CHANGED FEATURE (where mask=1), provide a separate decision.
- Decide 1 (intentional) or -1 (unintentional) PER FEATURE.
- Each feature should have its own specific explanation.
- Be deterministic - same input should yield same output.
- Return ONLY a JSON object, no additional text.

Required JSON format:
{
  "feature_decisions": [
    {"row_id": <global_id>, "column": "<feature_name>", "decision": 1 or -1, "reason": "<short explanation>"},
    ...
  ]
}
"""


def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.lstrip("﻿").strip()
    for junk in ("</think>", "<|im_end|>", "<|eot_id|>", "</s>", "```json", "```", "~~~json", "~~~"):
        s = s.replace(junk, "")
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    def extract_top_level_fragment(txt: str) -> Optional[str]:
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
                        return txt[start:i + 1]
        return None

    frag = extract_top_level_fragment(s)
    if not frag:
        return None
    try:
        return json.loads(frag)
    except Exception:
        return None


def decisions_from_obj(obj: dict) -> Dict[Tuple[int, str], int]:
    out: Dict[Tuple[int, str], int] = {}
    if not isinstance(obj, dict):
        return out
    feat_decs = obj.get("feature_decisions")
    if not isinstance(feat_decs, list):
        return out
    for item in feat_decs:
        if not isinstance(item, dict):
            continue
        try:
            rid = int(item["row_id"])
            col = str(item["column"])
            dec = int(item["decision"])
            if dec not in (1, -1):
                continue
            out[(rid, col)] = dec
        except Exception:
            continue
    return out


class GeminiBackend:
    # Cost policy (2026-06-26): gemini-2.5-pro is never called from this
    # codebase again after its usage ran up a ~550 EUR bill. Always try
    # flash-lite first, fall back to flash on failure, then give up --
    # see error_detection_system/gemini_safe.py.
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    def __init__(self):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(ROOT, "error_detection_system")
        sys.path.insert(0, config_path)
        from config import GEMINI_API_KEY
        from gemini_safe import SafeGeminiModel
        self.model = SafeGeminiModel(
            api_key=GEMINI_API_KEY,
            generation_config={"temperature": 0, "max_output_tokens": GEMINI_MAX_NEW_TOKENS},
            safety_settings=self.SAFETY_SETTINGS,
        )

    def call(self, system_text: str, user_text: str) -> str:
        return self.model.generate_content_safe(
            f"{system_text}\n\n{user_text}", retries_per_tier=2, timeout=180
        )


class TGIBackend:
    # TGI containers running large multi-GPU-sharded models (e.g. Llama-3-70B)
    # have been observed to crash with a CUDA "illegal memory access" and
    # auto-restart (docker restart: unless-stopped) -- a full reload of a 70B
    # model takes minutes. Without a wait-for-recovery retry, every in-flight
    # concurrent request during that window fails instantly and the whole
    # remaining queue races through empty responses instead of waiting the
    # crash out, which is what caused most of a run's attempts to be wasted
    # rather than just the in-flight batch at the moment of the crash.
    MAX_RETRIES = 6
    RETRY_BACKOFF_SEC = 30

    def __init__(self, model_key: str, port: int):
        self.url = f"http://127.0.0.1:{port}/generate"
        self.template, self.stop = CHAT_TEMPLATES[model_key]

    def call(self, system_text: str, user_text: str) -> str:
        prompt = self.template.format(system=system_text, user=user_text)
        payload = {"inputs": prompt, "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False,
            "return_full_text": False, "details": True, "stop": self.stop,
        }}
        for attempt in range(self.MAX_RETRIES):
            try:
                r = requests.post(self.url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                return ""
            except Exception:
                if attempt == self.MAX_RETRIES - 1:
                    return ""
                time.sleep(self.RETRY_BACKOFF_SEC * (attempt + 1))
        return ""


def make_chunk_prompt(feature_cols: List[str], clean_rows, dirty_rows, mask_rows, global_ids,
                       prompt_variant: str = "zero", dataset_key: str = "") -> Tuple[str, str]:
    records = []
    for i in range(len(clean_rows)):
        changed = []
        for col in feature_cols:
            if mask_rows[i][col] != 0:
                changed.append({"column": col, "original_value": to_str(clean_rows[i][col]),
                                 "changed_value": to_str(dirty_rows[i][col])})
        if changed:
            records.append({"row_id": global_ids[i], "changed_features": changed})
    bundle = {"columns": feature_cols, "records": records}
    user_content = (f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}\n\n"
                     "Analyze each changed feature and return your feature-level decisions in JSON format now.")
    system_prompt = f"{ROLE_BLOCK}{get_prompt_variant_text(dataset_key, prompt_variant)}\n\n{SYS_RULES}"
    return system_prompt, user_content


def load_cache(out_dir: str) -> Dict[Tuple[int, ...], Dict[Tuple[int, str], int]]:
    """Resume support: returns {chunk_tuple: decs} for chunks whose raw
    response already parsed successfully in a prior (e.g. crashed-TGI-server)
    run. Chunks with an empty or unparseable raw response are excluded, so
    they get retried below rather than permanently recorded as failures."""
    path = os.path.join(out_dir, "raw_responses.jsonl")
    cache: Dict[Tuple[int, ...], Dict[Tuple[int, str], int]] = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue  # tolerate a truncated last line from a killed process
            raw = d.get("raw")
            if not raw:
                continue
            obj = parse_llm_json(raw)
            decs = decisions_from_obj(obj) if obj else {}
            if decs:
                cache[tuple(d["chunk"])] = decs
    return cache


def run(dataset_key: str, model_key: str, port: int, out_dir: str, max_records: Optional[int] = None,
        prompt_variant: str = "zero"):
    ds = LOADERS[dataset_key]()
    dirty, mask, feature_cols = ds["dirty"], ds["mask"], ds["feature_cols"]
    blind = (mask != 0)

    if max_records:
        dirty, mask, blind = dirty.iloc[:max_records], mask.iloc[:max_records], blind.iloc[:max_records]

    backend = GeminiBackend() if model_key == "gemini" else TGIBackend(model_key, port)

    n_rows = len(dirty)
    row_groups = [i for i in range(n_rows) if blind.iloc[i].any()]
    print(f"[{dataset_key}/{model_key}] {n_rows} rows, {len(row_groups)} rows with >=1 flagged cell, "
          f"{int(blind.values.sum())} flagged cells")

    os.makedirs(out_dir, exist_ok=True)

    chunks = [row_groups[i:i + CHUNK_SIZE] for i in range(0, len(row_groups), CHUNK_SIZE)]

    # Resume support: skip chunks that already succeeded in a prior (e.g.
    # crashed-TGI-server) run of this exact (dataset, model) config. Failed/
    # empty chunks from before are NOT in the cache, so they're retried below.
    all_success = load_cache(out_dir)
    todo = [c for c in chunks if tuple(c) not in all_success]
    if all_success:
        print(f"[{dataset_key}/{model_key}] resuming: "
              f"{len(all_success)}/{len(chunks)} chunks already cached, {len(todo)} to do")

    debug_lock = threading.Lock()
    debug_f = open(os.path.join(out_dir, "raw_responses.jsonl"), "a")

    def process_chunk(chunk):
        clean_rows = [ds["clean"].iloc[i] if "clean" in ds and i < len(ds["clean"]) else dirty.iloc[i] for i in chunk]
        dirty_rows = [dirty.iloc[i] for i in chunk]
        mask_rows = [mask.iloc[i] for i in chunk]
        sys_p, user_p = make_chunk_prompt(feature_cols, clean_rows, dirty_rows, mask_rows, chunk, prompt_variant, dataset_key)
        raw = backend.call(sys_p, user_p)
        with debug_lock:
            debug_f.write(json.dumps({"chunk": chunk, "raw": raw}) + "\n")
            debug_f.flush()
        obj = parse_llm_json(raw)
        decs = decisions_from_obj(obj) if obj else {}
        return chunk, decs

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(process_chunk, c) for c in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset_key}/{model_key}"):
            chunk, decs = fut.result()
            if decs:
                all_success[tuple(chunk)] = decs

    debug_f.close()

    n_parse_fail = len(chunks) - len(all_success)
    predictions: Dict[Tuple[int, str], int] = {}
    for chunk, decs in all_success.items():
        for i in chunk:
            for col in feature_cols:
                if blind.iloc[i][col]:
                    predictions[(i, col)] = decs.get((i, col), -1)  # default unintentional on parse failure
    # Any chunk that never succeeded still needs its flagged cells defaulted.
    for chunk in chunks:
        if tuple(chunk) in all_success:
            continue
        for i in chunk:
            for col in feature_cols:
                if blind.iloc[i][col]:
                    predictions.setdefault((i, col), -1)

    y_true, y_pred = [], []
    for (i, col), pred in predictions.items():
        gt = int(mask.iloc[i][col])
        if gt == 0:
            continue
        y_true.append(1 if gt == 1 else 0)
        y_pred.append(1 if pred == 1 else 0)

    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    f1_int = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if 1 in labels else float("nan")
    f1_unint = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)) if 0 in labels else float("nan")
    metrics = {
        "n_cells": len(y_true),
        "n_parse_fail_chunks": n_parse_fail,
        "n_chunks": len(chunks),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)) if len(y_true) else float("nan"),
        "f1_intentional": f1_int,
        "f1_unintentional": f1_unint,
        "f1_macro": (f1_int + f1_unint) / 2 if len(y_true) else float("nan"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{dataset_key}/{model_key}] {metrics}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    p.add_argument("--model", required=True, choices=["gemini", "qwen", "llama", "mixtral", "r1qwen"])
    p.add_argument("--prompt", default="zero", choices=list(PROMPT_VARIANTS.keys()))
    p.add_argument("--port", type=int, default=6300)
    p.add_argument("--out", default=None)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "generic_cell_eval_output", args.dataset, args.model, args.prompt)
    run(args.dataset, args.model, args.port, out_dir, args.max_records, args.prompt)
