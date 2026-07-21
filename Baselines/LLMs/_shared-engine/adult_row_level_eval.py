#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adult_row_level_eval.py
=========================
Concurrent, row-level LLM intent-attribution baseline for Adult-LLM,
matching the paper's stated methodology (one decision per manipulated
RECORD, not per cell -- see Intent_Paper chapter 5/8: "Adult-LLM is
row-level (Macro F1)").

Replaces the older per-model scripts (intent_attribution_pipeline-{model}.py)
for this purpose: same prompt content and JSON contract, but issues requests
CONCURRENTLY instead of one-group-at-a-time, which is what made the original
scripts project a ~25-hour runtime for a single (model, prompt) config.

Each "group" is 1 clean record + its 3 manipulated variants (gain-targeted /
fairness-masking / obfuscation or unintentional), exactly as in the existing
scripts -- only the orchestration (concurrency) and dataset path (defaults
to the canonical fraud_baseline.LOADERS path) changed.

Usage:
  python adult_row_level_eval.py --model gemini --prompt zero
  python adult_row_level_eval.py --model qwen --prompt info --port 6300
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
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(THIS_DIR))))
sys.path.insert(0, THIS_DIR)
from generic_cell_eval import GeminiBackend, TGIBackend, parse_llm_json  # noqa: E402

MAX_CONCURRENCY = int(os.environ.get("LLM_EVAL_CONCURRENCY", "12"))

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]

ROLE_BLOCK = ("You are a data-quality analyst attributing the INTENT behind a record-level "
              "manipulation in the UCI Adult Income dataset: was the manipulated record "
              "changed DELIBERATELY to achieve some goal (intentional: gain-targeted, "
              "fairness-masking, or obfuscation), or did it arise by ACCIDENT / data "
              "corruption (unintentional)?")

PROMPT_VARIANTS = {
    "zero": "",
    "info": (
        "\n\nDecision rubric:\n"
        "- Intentional (gain-targeted): changes that would flip predicted income to >50K "
        "(e.g. raising capital-gain, hours-per-week, or education level).\n"
        "- Intentional (fairness-masking): sensitive attributes (race, sex) changed to the "
        "dataset's majority value.\n"
        "- Intentional (obfuscation): categorical fields replaced with 'Unknown'/'N/A'/'---'.\n"
        "- Unintentional: typos, OCR-style artefacts, random noise with no coherent direction."
    ),
    "few": (
        "\n\nDecision rubric (same as info) plus worked examples:\n"
        "- capital-gain 0 -> 99999, hours-per-week 40 -> 80: intentional (gain-targeted).\n"
        "- race 'Black' -> 'White' alone, no other change: intentional (fairness-masking).\n"
        "- workclass 'Private' -> 'Unknown': intentional (obfuscation).\n"
        "- age 35 -> 35l (typo-like corruption), native-country 'United-States' -> 'Uniteed-States': "
        "unintentional.\n"
    ),
}


def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}


def mask_row_to_dict(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: ("1" if str(mask_df.at[idx, c]).strip() in ("1", "1.0") else "0") for c in ADULT_COLS}


def make_group_prompt(prompt_variant: str, clean_row, group, group_masks, group_global_ids) -> Tuple[str, str]:
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the provided clean row, three manipulated rows, and their masks.\n"
        "- Decide 1 (intentional) or -1 (unintentional) PER manipulated record. No zeros.\n"
        "- Deterministic, no randomness.\n"
        "- Return ONLY a JSON object: "
        '{"decisions":[{"row_id": <global>, "decision": 1 or -1, "reason": "<short>"}, ...]}'
        " No extra text."
    )
    bundle = {
        "columns": ADULT_COLS,
        "clean": clean_row,
        "manipulated": [
            {"local_id": k, "row_id": group_global_ids[k], "row": group[k], "mask": group_masks[k]}
            for k in range(len(group))
        ],
    }
    system = f"{ROLE_BLOCK}{PROMPT_VARIANTS[prompt_variant]}\n\n{sys_rules}"
    user = f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\nReturn the JSON object now."
    return system, user


def decisions_from_obj(obj: dict) -> Dict[int, int]:
    out = {}
    if not isinstance(obj, dict):
        return out
    decs = obj.get("decisions")
    if isinstance(decs, list):
        for item in decs:
            if isinstance(item, dict) and "row_id" in item and "decision" in item:
                try:
                    rid, dec = int(item["row_id"]), int(item["decision"])
                    if dec in (1, -1):
                        out[rid] = dec
                except Exception:
                    continue
    elif isinstance(decs, dict):
        for k, v in decs.items():
            try:
                rid, dec = int(k), int(v)
                if dec in (1, -1):
                    out[rid] = dec
            except Exception:
                continue
    return out


def run(model_key: str, prompt_variant: str, manip_path: str, correct_path: str, masks_path: str,
        port: int, out_dir: str, max_groups: Optional[int] = None):
    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)[ADULT_COLS]
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)[ADULT_COLS]
    masks = pd.read_csv(masks_path, dtype=str, keep_default_na=False)[ADULT_COLS]

    n_clean = len(correct)
    assert len(manip) == 3 * n_clean, f"Expected manip == 3*clean; got {len(manip)} vs {3*n_clean}"
    if max_groups:
        n_clean = min(n_clean, max_groups)

    backend = GeminiBackend() if model_key == "gemini" else TGIBackend(model_key, port)

    os.makedirs(out_dir, exist_ok=True)
    debug_lock = threading.Lock()
    debug_f = open(os.path.join(out_dir, "raw_responses.jsonl"), "w")

    def process_group(j):
        clean_row = row_to_dict(correct, j)
        group, group_masks, gids = [], [], []
        for k in range(3):
            mi = 3 * j + k
            group.append(row_to_dict(manip, mi))
            group_masks.append(mask_row_to_dict(masks, mi))
            gids.append(mi)
        sys_p, user_p = make_group_prompt(prompt_variant, clean_row, group, group_masks, gids)
        raw = backend.call(sys_p, user_p)
        with debug_lock:
            debug_f.write(json.dumps({"group": j, "gids": gids, "raw": raw}) + "\n")
            debug_f.flush()
        obj = parse_llm_json(raw)
        decs = decisions_from_obj(obj) if obj else {}
        return gids, decs

    predictions: Dict[int, int] = {}
    n_parse_fail = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        futures = [pool.submit(process_group, j) for j in range(n_clean)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"adult/{model_key}/{prompt_variant}"):
            gids, decs = fut.result()
            if not decs:
                n_parse_fail += 1
            for rid in gids:
                predictions[rid] = decs.get(rid, -1)

    debug_f.close()

    # Ground truth: a record is intentional if ANY of its masked cells == 1
    y_true, y_pred = [], []
    for rid in range(3 * n_clean):  # only records actually processed (respects --max-groups)
        row_mask = masks.iloc[rid]
        flagged_vals = [int(float(row_mask[c])) for c in ADULT_COLS if row_mask[c] not in ("", "0", "0.0")]
        if not flagged_vals:
            continue
        gt = 1 if any(v == 1 for v in flagged_vals) else 0
        y_true.append(gt)
        y_pred.append(1 if predictions.get(rid, -1) == 1 else 0)

    from sklearn.metrics import f1_score, accuracy_score
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    metrics = {
        "n_records": len(y_true),
        "n_parse_fail_groups": n_parse_fail,
        "n_groups": n_clean,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)) if len(y_true) else float("nan"),
        "f1_intentional": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)) if 1 in labels else float("nan"),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[adult/{model_key}/{prompt_variant}] {metrics}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["gemini", "qwen", "llama", "mixtral", "r1qwen"])
    p.add_argument("--prompt", required=True, choices=["zero", "info", "few"])
    p.add_argument("--port", type=int, default=6300)
    p.add_argument("--max-groups", type=int, default=None)
    p.add_argument("--run-dir", default="run_v2_20260617_173016")
    args = p.parse_args()

    base = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw"
    out_dir = os.path.join(THIS_DIR, "adult_row_level_output", args.model, args.prompt)
    run(args.model, args.prompt,
        os.path.join(base, args.run_dir, "manipulated_records.csv"),
        os.path.join(base, "correct_records.csv"),
        os.path.join(base, args.run_dir, "masks_blind.csv"),
        args.port, out_dir, args.max_groups)
