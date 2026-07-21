#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Local LLM (Llama-3 via TGI)
---------------------------------------------------------------
This is a Llama-3 drop-in of your Mixtral script, preserving the SAME input prompt
content, but wrapped in the Llama-3 chat template and with a stop token.

Key changes vs. Mixtral:
- Chat template: <|begin_of_text|><system>...<|eot_id|><user>...<|eot_id|><assistant>
- params: do_sample=False, return_full_text=False, stop=["<|eot_id|>"]
- Robust parsing tolerates code fences and stray tokens.

Everything else (prompt shape, outputs, counters) stays the same.
"""

from __future__ import annotations
import json
import sys
import os
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# ======== CONFIG =========
# =========================
SERVER_URL = "http://127.0.0.1:6100/generate"  # Llama-3 TGI endpoint
MAX_NEW_TOKENS = 256                           # small: we only need 3 decisions + reasons
DO_SAMPLE = False                              # greedy, deterministic
REQUEST_TIMEOUT_SEC = 120

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/manipulated_records.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/masks-blind.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/first-trial-llama/intent_labels.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/first-trial-llama/intent_explanations.csv"

# Adult schema
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","class"
]

# =========================
# ====== UTILITIES ========
# =========================
def to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}

def mask_row_to_dict(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: "1" if str(mask_df.at[idx, c]).strip() in ("1","1.0") else "0" for c in ADULT_COLS}

# =========================
# ==== Llama-3 template ===
# =========================
LLAMA3_EOT = "<|eot_id|>"
LLAMA3_BOT = "<|begin_of_text|>"
LLAMA3_SYS = "<|start_header_id|>system<|end_header_id|>\n"
LLAMA3_USR = "<|start_header_id|>user<|end_header_id|>\n"
LLAMA3_AST = "<|start_header_id|>assistant<|end_header_id|>\n"

def llama3_chat(system_text: str, user_text: str) -> str:
    """Single-turn chat template; assistant header left open for generation."""
    return f"{LLAMA3_BOT}{LLAMA3_SYS}{system_text}{LLAMA3_EOT}{LLAMA3_USR}{user_text}{LLAMA3_EOT}{LLAMA3_AST}"

def call_tgi(inputs: str) -> str:
    """Greedy call to TGI with Llama stop token; returns generated_text."""
    params = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "return_full_text": False,
        "details": True,
        "stop": [LLAMA3_EOT],
    }
    payload = {"inputs": inputs, "parameters": params}
    r = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    # Unexpected schema: stringify as last resort (caller will parse/decide)
    return json.dumps(data)

# =========================
# ======= PARSING =========
# =========================
def parse_llm_json(text: str) -> Optional[dict]:
    """Robustly extract first top-level JSON (object/array) from model text."""
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()

    # Remove common wrappers/fences/markers
    for junk in ("</think>", "<|eot_id|>", "<|im_end|>"):
        s = s.replace(junk, "")
    # strip code fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
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

def decisions_from_obj(obj: dict, global_ids: List[int]) -> Optional[List[Tuple[int, Optional[int], str]]]:
    """
    Accepts:
      A) {"decisions": [ {"local_id":..,"row_id":..,"decision":..,"reason":..}, ... ]}
      B) {"decisions": [ 1, -1, -1 ]}  # by local_id order 0,1,2
      C) {"decisions": {"0": 1, "1": -1, "2": -1}}  # map by local_id
    Returns [(rid, decision|None, reason), ...] for lids 0..2, or None if unusable.
    """
    decs = obj.get("decisions", None)
    if decs is None:
        return None

    # -------- Case A: list of dicts (original schema) --------
    if isinstance(decs, list) and (len(decs) == 0 or isinstance(decs[0], dict)):
        out: List[Tuple[int, Optional[int], str]] = []
        by_lid: Dict[int, dict] = {}
        for it in decs:
            try:
                lid = int(it.get("local_id"))
                by_lid[lid] = it
            except Exception:
                continue
        for lid in (0, 1, 2):
            rid = global_ids[lid]
            it = by_lid.get(lid)
            if it is None:
                out.append((rid, None, "json_missing_entry"))
                continue
            try:
                dec = 1 if int(it.get("decision", -1)) >= 1 else -1
            except Exception:
                dec = None
            reason = (str(it.get("reason", "")).strip() or "reason_not_provided")
            out.append((rid, dec, reason))
        return out

    # -------- Case B: list of 3 scalars --------
    if isinstance(decs, list) and len(decs) >= 3 and all(not isinstance(x, dict) for x in decs[:3]):
        out: List[Tuple[int, Optional[int], str]] = []
        for lid in (0, 1, 2):
            v = decs[lid]
            try:
                dec = 1 if int(v) >= 1 else -1
                out.append((global_ids[lid], dec, "array_schema"))
            except Exception:
                out.append((global_ids[lid], None, "array_schema_bad_value"))
        return out

    # -------- Case C: map/dict inside "decisions" --------
    if isinstance(decs, dict):
        out: List[Tuple[int, Optional[int], str]] = []
        have_any = False
        for lid in (0, 1, 2):
            v = decs.get(str(lid), decs.get(lid))
            if v is None:
                out.append((global_ids[lid], None, "map_schema_missing"))
                continue
            try:
                dec = 1 if int(v) >= 1 else -1
                have_any = True
                out.append((global_ids[lid], dec, "map_schema"))
            except Exception:
                out.append((global_ids[lid], None, "map_schema_bad_value"))
        return out if have_any else None

    # Unknown shape
    return None

def _from_simple_triplet_map_text(llm_text: str, global_ids: List[int]) -> Optional[List[Tuple[int,int,str]]]:
    """
    Accepts outputs like: {"0": 1, "1": -1, "2": -1} (or with int keys).
    Returns list[(row_id, decision, reason)] or None.
    """
    s = llm_text.strip().lstrip("\ufeff")
    # slice outermost JSON object if extra text exists
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    out = []
    for lid in (0, 1, 2):
        v = obj.get(str(lid), obj.get(lid))
        if v is None:
            return None
        try:
            dec = 1 if int(v) >= 1 else -1
        except Exception:
            return None
        out.append((global_ids[lid], dec, "simple_schema"))
    return out

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "ROLE\n"
    "You are a precise data-forensics assistant. Your task is to decide, for each manipulated RECORD, "
    "whether the changes are INTENTIONAL (1) or UNINTENTIONAL (-1). Deterministic behavior is required."
)

def make_group_prompt(clean_row: Dict[str, str],
                      group: List[Dict[str, str]],
                      group_masks: List[Dict[str, str]],
                      group_global_ids: List[int]) -> str:
    """
    SAME content as Mixtral version — strict JSON requested:

    {
      "decisions": [
        {"row_id": <global>, "local_id": 0, "decision": 1|-1, "reason": "<short>"},
        {"row_id": <global>, "local_id": 1, "decision": 1|-1, "reason": "<short>"},
        {"row_id": <global>, "local_id": 2, "decision": 1|-1, "reason": "<short>"}
      ]
    }
    """
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the provided clean row, three manipulated rows, and their masks.\n"
        "- Decide 1 (intentional) or -1 (unintentional) PER manipulated record. No zeros.\n"
        "- Deterministic, no randomness.\n"
        "- Return ONLY a strict JSON object matching the required schema. No extra text."
    )
    bundle = {
        "columns": ADULT_COLS,
        "clean": clean_row,
        "manipulated": [
            {
                "local_id": k,
                "row_id": group_global_ids[k],
                "row": group[k],
                "mask": group_masks[k]
            } for k in range(len(group))
        ]
    }
    prompt = (
        f"{ROLE_BLOCK}\n\n{sys_rules}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return the JSON object now."
    )
    return prompt

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str):
    t0 = time.perf_counter()

    # Ensure output directory exists
    out_dir = os.path.dirname(OUT_LABELS_CSV) if OUT_LABELS_CSV else "."
    os.makedirs(out_dir, exist_ok=True)
    stats_json_path = os.path.join(out_dir, "run_stats.json")
    per_group_csv_path = os.path.join(out_dir, "per_group_times.csv")

    # Load CSVs as strings
    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)
    masks  = pd.read_csv(masks_path,  dtype=str, keep_default_na=False)

    # Ensure columns & order
    for df, name in [(manip,"manipulated_records"), (correct,"correct_records"), (masks,"masks")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        df[:] = df[ADULT_COLS]

    # Sanity: manipulated must be 3x clean
    n_clean = len(correct)
    n_manip = len(manip)
    if n_manip != 3 * n_clean:
        raise ValueError(f"Expected manipulated rows == 3 * clean rows; got {n_manip} vs 3*{n_clean}={3*n_clean}")

    # Prepare outputs
    out_mask_df = masks.copy(deep=True)  # generated mask (1 -> decision; 0 stays 0)
    explanations_rows: List[Dict[str, str]] = []

    # Timing & status trackers
    group_times: List[float] = []
    llm_times:   List[float] = []
    llm_success_groups = 0
    fallback_groups = 0
    llm_status_hist = {"ok_decisions_list": 0, "ok_simple_map": 0, "http_error": 0, "parse_error": 0}

    # Iterate groups: clean j with manipulated 3j..3j+2
    for j in tqdm(range(n_clean), desc="Groups (clean + 3 manipulated)", ncols=100):
        g_t0 = time.perf_counter()

        clean_row = row_to_dict(correct, j)
        manip_rows, mask_rows, global_ids = [], [], []
        for k in range(3):
            mi = 3*j + k
            manip_rows.append(row_to_dict(manip, mi))
            mask_rows.append(mask_row_to_dict(masks, mi))
            global_ids.append(mi)

        # SAME prompt as Mixtral, just wrapped in Llama chat template
        prompt_text = make_group_prompt(clean_row, manip_rows, mask_rows, global_ids)
        chat_inputs = llama3_chat(
            system_text="You are a precise data-forensics assistant. Output strict JSON only.",
            user_text=prompt_text
        )

        obj = None
        llm_text = ""
        l0 = time.perf_counter()
        try:
            llm_text = call_tgi(chat_inputs)  # returns generated_text string
            obj = parse_llm_json(llm_text)
        except requests.HTTPError:
            llm_status_hist["http_error"] += 1
            obj = None
            llm_text = ""
        except Exception:
            llm_status_hist["parse_error"] += 1
            obj = None
            llm_text = ""
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)

        decisions: Dict[int, Tuple[int, str]] = {}

        # ----- accept either schema -----
        parsed: Optional[List[Tuple[int, Optional[int], str]]] = None
        if isinstance(obj, dict):
            parsed = decisions_from_obj(obj, global_ids)
            if parsed:
                llm_success_groups += 1
                llm_status_hist["ok_decisions_list"] += 1
        if parsed is None and llm_text:
            parsed2 = _from_simple_triplet_map_text(llm_text, global_ids)
            if parsed2 is not None:
                parsed = parsed2
                llm_success_groups += 1
                llm_status_hist["ok_simple_map"] += 1
        # --------------------------------

        if parsed is None:
            # Final fallback: mark all three rows -1 with reason
            fallback_groups += 1
            for rid in global_ids:
                decisions[rid] = (-1, "llm_fallback")
        else:
            for rid, dec, reason in parsed:
                # In rare cases dec can be None (bad value); fallback to -1 here.
                if dec is None:
                    decisions[rid] = (-1, f"{reason}_fallback")
                else:
                    decisions[rid] = (dec, reason)

        # Fill generated mask + explanations
        for k, rid in enumerate(global_ids):
            decision, reason = decisions[rid]
            for c in ADULT_COLS:
                if c == "class":
                    out_mask_df.at[rid, c] = "0"
                else:
                    out_mask_df.at[rid, c] = (
                        str(decision) if str(masks.at[rid, c]).strip() in ("1","1.0") else "0"
                    )
            # per-cell explanations (record-level reason)
            for c in ADULT_COLS:
                if str(masks.at[rid, c]).strip() in ("1","1.0"):
                    explanations_rows.append({
                        "row_id": rid,
                        "column": c,
                        "manipulated_value": manip_rows[k][c],
                        "original_value": clean_row[c],
                        "mask": 1,
                        "diagnosis": str(decision),
                        "rule": "record_intent_decision",
                        "details": json.dumps({"reason": reason}, ensure_ascii=False),
                        "mapping_status": "matched",
                        "mapping_score": ""
                    })

        g_t1 = time.perf_counter()
        group_times.append(g_t1 - g_t0)

    # Save outputs
    out_mask_df.to_csv(OUT_LABELS_CSV, index=False)

    expl_df = pd.DataFrame(explanations_rows, columns=[
        "row_id","column","manipulated_value","original_value","mask",
        "diagnosis","rule","details","mapping_status","mapping_score"
    ])
    expl_df.to_csv(OUT_EXPL_CSV, index=False)

    # Summary
    flat_vals = out_mask_df.values.flatten().tolist()
    counts = pd.Series(flat_vals).value_counts().to_dict()
    print("\n=== Generated Mask Counts ===")
    for k in ["1","-1","0"]:
        print(f"{k:>3}: {counts.get(k,0)}")
    print(f"ALL: {len(flat_vals)}")

    # Runtime stats
    t1 = time.perf_counter()
    stats_payload = {
        "clean_rows": n_clean,
        "manip_rows": n_manip,
        "groups": n_clean,
        "total_time_sec": t1 - t0,
        "avg_group_time_sec": float(pd.Series(group_times).mean()) if group_times else None,
        "median_group_time_sec": float(pd.Series(group_times).median()) if group_times else None,
        "total_llm_time_sec": float(pd.Series(llm_times).sum()) if llm_times else None,
        "avg_llm_time_sec": float(pd.Series(llm_times).mean()) if llm_times else None,
        "llm_success_groups": llm_success_groups,
        "fallback_groups": fallback_groups,
        "llm_status_hist": llm_status_hist,
        "outputs": {
            "generated_mask_csv": OUT_LABELS_CSV,
            "explanations_csv": OUT_EXPL_CSV,
            "per_group_times_csv": per_group_csv_path
        }
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)

    per_group_df = pd.DataFrame({
        "group_id": list(range(n_clean)),
        "group_time_sec": group_times,
        "llm_time_sec": llm_times
    })
    per_group_df.to_csv(per_group_csv_path, index=False)

    print(f"\nSaved runtime stats to:\n  - {stats_json_path}\n  - {per_group_csv_path}")
    print(f"Outputs:\n  - {OUT_LABELS_CSV}\n  - {OUT_EXPL_CSV}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else MANIPULATED_CSV
    c = sys.argv[2] if len(sys.argv) > 2 else CORRECT_CSV
    k = sys.argv[3] if len(sys.argv) > 3 else MASKS_CSV
    run_pipeline(m, c, k)