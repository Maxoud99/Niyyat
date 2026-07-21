#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Local LLM (Llama-3 Instruct) — Few-shot, TGI-fixed
--------------------------------------------------------------------------------------
Surgical fixes for TGI:
- Use Llama-3 chat template (<|begin_of_text|> ... <|eot_id|>) so /generate returns text.
- Remove unsupported `parameters.grammar`.
- Add stop token "<|eot_id|>".
- Deterministic: do_sample=False, no explicit temperature.

Few-shot remains ON (USE_FEW_SHOT=True). Demos are included in the user message.

Additional surgical fixes:
- Robust JSON parsing tolerant to wrappers and quoted JSON.
- Consume LLM output by local_id (0/1/2) to map to our global_ids.
- Switchable fallback policy: PURE_FALLBACK_MINUS_ONE forces all fallbacks to -1.
"""

from __future__ import annotations
import json
import sys
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# ======== CONFIG =========
# =========================
SERVER_URL = "http://127.0.0.1:6100/generate"  # TGI endpoint backing your few-shot run
MAX_NEW_TOKENS = 96
REQUEST_TIMEOUT_SEC = 120

# File paths
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/manipulated_records.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/masks-blind.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-llama/intent_labels.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-llama/intent_explanations.csv"

# Adult schema
ADULT_COLS = [
    "age","workclass","fnlwgt","education","education-num","marital-status",
    "occupation","relationship","race","sex","capital-gain","capital-loss",
    "hours-per-week","native-country","class"
]

# ======= Few-shot config (ON by default) =======
USE_FEW_SHOT = True

FEW_SHOT_BLOCK = """
EXAMPLE 1 — Utility-seeking vs. benign slips (abridged)
Context: Clean row vs. 3 manipulated rows. DIFFS show only changed cells.

local_id 0 DIFFS:
- education-num: 9 → 13
- education: HS-grad → Bachelors
- capital-gain: 0 → 12000
- hours-per-week: 40 → 52
Rationale: multiple strong utility signals → INTENTIONAL (1).

local_id 1 DIFFS:
- age: 36 → 35 (±1)
- workclass: "Private" → "private" (case-only)
Rationale: tiny numeric drift + case change → UNINTENTIONAL (-1).

local_id 2 DIFFS:
- capital-loss: 0 → 1 (tiny)
Rationale: trivial change → UNINTENTIONAL (-1).

Gold decisions:
{"decisions":[
  {"row_id":100,"local_id":0,"decision":1,"reason":"large utility-seeking upgrades"},
  {"row_id":101,"local_id":1,"decision":-1,"reason":"small numeric drift & case-only"},
  {"row_id":102,"local_id":2,"decision":-1,"reason":"tiny change; no advantage"}
]}

EXAMPLE 2 — Fairness/attribute masking & privacy (abridged)
local_id 0 DIFFS:
- sex: Female → Male
- race: Black → White
- hours-per-week: 35 → 46
Rationale: edits likely intended to escape bias / seek advantage + larger hours → INTENTIONAL (1).

local_id 1 DIFFS:
- native-country: "United-States" → "?"
- occupation: "Tech-support" → "Tech support" (spacing)
Rationale: disguised missing value (privacy) is intentional; spacing change is incidental. Net effect → INTENTIONAL (1).

local_id 2 DIFFS:
- age: 28 → 29 (+1)
Rationale: tiny numeric slips → UNINTENTIONAL (-1).

Gold decisions:
{"decisions":[
  {"row_id":200,"local_id":0,"decision":1,"reason":"protected-attribute edits + workload increase"},
  {"row_id":201,"local_id":1,"decision":1,"reason":"DMV/privacy masking dominates"},
  {"row_id":202,"local_id":2,"decision":-1,"reason":"minor numeric slips"}
]}
""".strip()

# ======= Fallback policy =======
PURE_FALLBACK_MINUS_ONE = True  # True → every fallback = -1 ; False → use heuristic below

# ======= Heuristic fallback parameters (used only if PURE_FALLBACK_MINUS_ONE=False) =======
HEUR_EDU_RANK = {
    "Preschool":0,"1st-4th":1,"2nd-4th":1,"5th-6th":2,"7th-8th":3,"9th":4,"10th":5,
    "11th":6,"12th":7,"HS-grad":8,"Some-college":9,"Assoc-voc":10,"Assoc-acdm":10,
    "Bachelors":11,"Masters":12,"Prof-school":13,"Doctorate":14
}
FALLBACK_INTENT_THRESHOLD = 2  # score ≥ 2 → label 1 else -1
BIG_GAIN = 5000
BIG_HOURS = 10
SMALL_NUM_DRIFT = 2

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

def parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def compute_diffs(clean_row: Dict[str,str], manip_row: Dict[str,str], mask_row: Dict[str,str]) -> List[Dict[str,str]]:
    diffs = []
    for c in ADULT_COLS:
        if c == "class":
            continue
        if mask_row.get(c) == "1":
            diffs.append({"column": c, "from": clean_row.get(c,""), "to": manip_row.get(c,"")})
    return diffs

def education_rank(val: str) -> Optional[int]:
    return HEUR_EDU_RANK.get(val, None)

def heuristic_record_score(clean_row: Dict[str,str], manip_row: Dict[str,str], mask_row: Dict[str,str]) -> int:
    # Only used if PURE_FALLBACK_MINUS_ONE=False
    score = 0
    if mask_row.get("education-num") == "1":
        a = parse_float(clean_row.get("education-num","")); b = parse_float(manip_row.get("education-num",""))
        if a is not None and b is not None:
            if b - a >= 2: score += 2
            elif abs(b - a) <= SMALL_NUM_DRIFT: score -= 1
    if mask_row.get("education") == "1":
        r0, r1 = education_rank(clean_row.get("education","")), education_rank(manip_row.get("education",""))
        if r0 is not None and r1 is not None:
            if r1 > r0: score += 2
            elif r1 < r0: score -= 1
    if mask_row.get("capital-gain") == "1":
        a = parse_float(clean_row.get("capital-gain","")); b = parse_float(manip_row.get("capital-gain",""))
        if a is not None and b is not None:
            if (a == 0) and (b >= BIG_GAIN): score += 3
            elif abs(b - a) <= BIG_GAIN/100: score -= 1
    if mask_row.get("hours-per-week") == "1":
        a = parse_float(clean_row.get("hours-per-week","")); b = parse_float(manip_row.get("hours-per-week",""))
        if a is not None and b is not None:
            if b - a >= BIG_HOURS: score += 1
            elif abs(b - a) <= SMALL_NUM_DRIFT: score -= 1
    for c in ("age","fnlwgt","capital-loss"):
        if mask_row.get(c) == "1":
            a = parse_float(clean_row.get(c,"")); b = parse_float(manip_row.get(c,""))
            if a is not None and b is not None and abs(b - a) <= SMALL_NUM_DRIFT:
                score -= 1
    for c in ("workclass","occupation","relationship","native-country"):
        if mask_row.get(c) == "1":
            if clean_row.get(c,"").lower() == manip_row.get(c,"").lower():
                score -= 1
    return score

# =========================
# ==== Llama-3 templating =
# =========================
LLAMA3_EOT = "<|eot_id|>"
LLAMA3_BOT = "<|begin_of_text|>"
LLAMA3_SYS = "<|start_header_id|>system<|end_header_id|>\n"
LLAMA3_USR = "<|start_header_id|>user<|end_header_id|>\n"
LLAMA3_AST = "<|start_header_id|>assistant<|end_header_id|>\n"

def llama3_chat(system_text: str, user_text: str) -> str:
    """Single-turn chat template, assistant header left open for generation."""
    return f"{LLAMA3_BOT}{LLAMA3_SYS}{system_text}{LLAMA3_EOT}{LLAMA3_USR}{user_text}{LLAMA3_EOT}{LLAMA3_AST}"

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "You are a precise data-forensics assistant. Your task is to decide, for each manipulated RECORD, "
    "whether the changes are INTENTIONAL (1) or UNINTENTIONAL (-1). Deterministic behavior is required."
)

RUBRIC = (
    "Rubric:\n"
    "- Mark as INTENTIONAL (1) when edits plausibly seek advantage in screening systems:\n"
    "  • Higher education level (education-num increase ≥2, or clear categorical upgrade)\n"
    "  • Large jump in capital-gain from 0 to a high value\n"
    "  • Big increase in hours-per-week (e.g., +10 or more)\n"
    "  • Coordinated multiple benefit-linked edits\n"
    "- Also INTENTIONAL when edits plausibly try to escape system bias (fairness gaming):\n"
    "  • e.g., flipping protected attributes (sex/race) toward majority categories\n"
    "- Also INTENTIONAL when changes plausibly aim to preserve privacy:\n"
    "  • Disguised missing values (DMV), e.g., replacing with '?' or blanks in identifying fields\n"
    "- Mark as UNINTENTIONAL (-1) when edits look like benign human error:\n"
    "  • Small numeric drift (±1–2)\n"
    "  • Case-only or trivial formatting changes\n"
    "  • Single isolated change with low utility impact\n"
    "- If uncertain after applying the rubric, prefer -1 only when changes are small/benign; otherwise 1."
)

def build_chat_input(clean_row: Dict[str, str],
                     group: List[Dict[str, str]],
                     group_masks: List[Dict[str, str]],
                     group_global_ids: List[int]) -> str:
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the provided clean row, three manipulated rows, their masks, and DIFFS.\n"
        "- Decide 1 (intentional) or -1 (unintentional) PER manipulated record. No zeros.\n"
        "- Keep each reason ≤ 10 words.\n"
        '- Return ONLY this JSON:\n'
        '{ "decisions": ['
        '{"row_id":<global>,"local_id":0,"decision":1|-1,"reason":"<short>"},'
        '{"row_id":<global>,"local_id":1,"decision":1|-1,"reason":"<short>"},'
        '{"row_id":<global>,"local_id":2,"decision":1|-1,"reason":"<short>"} ] }\n'
        'If you cannot produce the long schema, return ONLY {"0":1,"1":-1,"2":-1}.'
    )

    manipulated_bundle = []
    for k in range(len(group)):
        manipulated_bundle.append({
            "local_id": k,
            "row_id": group_global_ids[k],
            "row": group[k],
            "mask": group_masks[k],
            "diffs": compute_diffs(clean_row, group[k], group_masks[k]),
        })

    bundle = {
        "columns": ADULT_COLS,
        "clean": clean_row,
        "manipulated": manipulated_bundle
    }

    system_text = f"{ROLE_BLOCK}\n\n{RUBRIC}\n\n{sys_rules}"
    fewshot = (FEW_SHOT_BLOCK + "\n\n") if USE_FEW_SHOT else ""
    user_text = f"{fewshot}INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\nReturn the JSON now."
    return llama3_chat(system_text, user_text)

# =========================
# == TGI call & parsing ===
# =========================
def call_tgi(inputs_str: str) -> str:
    """Call TGI; returns generated_text string."""
    params = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "return_full_text": False,
        "details": True,
        "stop": [LLAMA3_EOT],
    }
    payload = {"inputs": inputs_str, "parameters": params}
    r = requests.post(SERVER_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0].get("generated_text") or data[0].get("text", "")
    return ""

def parse_llm_json(text: str) -> Optional[dict]:
    """Extract the first top-level JSON object/array from LLM text."""
    if not text:
        return None

    s = text

    # 1) Strip BOM + common wrappers & markers
    s = s.lstrip("\ufeff").strip()
    for junk in ("</think>", "<|eot_id|>", "<|im_end|>", "```json", "```"):
        s = s.replace(junk, "")
    s = s.strip()

    # 2) If it's JSON-as-a-string (whole thing quoted), unescape once
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            s = json.loads(s)
        except Exception:
            pass
        s = (s or "").strip()

    # 3) Fast path: direct decode
    try:
        return json.loads(s)
    except Exception:
        pass

    # 4) Extract the first balanced top-level JSON fragment (object or array)
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
    if frag:
        try:
            return json.loads(frag)
        except Exception:
            # 5) Last resort: Python-literal style (single quotes etc.)
            try:
                import ast
                obj = ast.literal_eval(frag)
                if isinstance(obj, (dict, list)):
                    return obj
            except Exception:
                pass

    return None

# -------- consume LLM obj by local_id (robust to bad/missing row_id) --------
def decisions_from_obj(obj: dict, global_ids: List[int]) -> List[Tuple[int, Optional[int], str]]:
    """
    Returns [(rid, decision|None, reason), ...] for lids 0..2
    """
    out: List[Tuple[int, Optional[int], str]] = []
    by_lid: Dict[int, dict] = {}
    for it in obj.get("decisions", []):
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

def _from_simple_triplet_map_text(llm_text: str, global_ids: List[int]) -> Optional[List[Tuple[int,int,str]]]:
    s = llm_text.strip().lstrip("\ufeff")
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    out: List[Tuple[int, Optional[int], str]] = []
    any_ok = False
    for lid in (0, 1, 2):
        v = obj.get(str(lid), obj.get(lid))
        if v is None:
            out.append((global_ids[lid], None, "simple_schema_missing"))
            continue
        try:
            dec = 1 if int(v) >= 1 else -1
            any_ok = True
            out.append((global_ids[lid], dec, "simple_schema"))
        except Exception:
            out.append((global_ids[lid], None, "simple_schema_bad_value"))
    return out if any_ok else None

# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str):
    t0 = time.perf_counter()

    out_dir = os.path.dirname(OUT_LABELS_CSV) if OUT_LABELS_CSV else "."
    os.makedirs(out_dir, exist_ok=True)
    stats_json_path = os.path.join(out_dir, "run_stats.json")
    per_group_csv_path = os.path.join(out_dir, "per_group_times.csv")

    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)
    masks  = pd.read_csv(masks_path,  dtype=str, keep_default_na=False)

    for df, name in [(manip,"manipulated_records"), (correct,"correct_records"), (masks,"masks")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        df[:] = df[ADULT_COLS]

    n_clean = len(correct)
    n_manip = len(manip)
    if n_manip != 3 * n_clean:
        raise ValueError(f"Expected manipulated rows == 3 * clean rows; got {n_manip} vs 3*{n_clean}={3*n_clean}")

    out_mask_df = masks.copy(deep=True)
    explanations_rows: List[Dict[str, str]] = []

    group_times: List[float] = []
    llm_times:   List[float] = []
    sources_hist = {"llm":0, "fallback":0}
    llm_status_hist = {"ok": 0, "http_error": 0, "parse_error": 0,
                       "ok_decisions_list": 0, "ok_simple_map": 0}

    for j in tqdm(range(n_clean), desc="Groups (clean + 3 manipulated)", ncols=100):
        g_t0 = time.perf_counter()

        clean_row = row_to_dict(correct, j)
        manip_rows, mask_rows, global_ids = [], [], []
        for k in range(3):
            mi = 3*j + k
            manip_rows.append(row_to_dict(manip, mi))
            mask_rows.append(mask_row_to_dict(masks, mi))
            global_ids.append(mi)

        # Build chat-templated input + call TGI
        inputs_str = build_chat_input(clean_row, manip_rows, mask_rows, global_ids)

        obj = None
        llm_text = ""
        l0 = time.perf_counter()
        try:
            llm_text = call_tgi(inputs_str)
            obj = parse_llm_json(llm_text)
            llm_status_hist["ok"] += 1
        except requests.HTTPError:
            llm_status_hist["http_error"] += 1
            obj = None
        except Exception:
            llm_status_hist["parse_error"] += 1
            obj = None
        l1 = time.perf_counter()
        llm_times.append(l1 - l0)

        # Accept either schema
        parsed: Optional[List[Tuple[int, Optional[int], str]]] = None
        if isinstance(obj, dict) and "decisions" in obj:
            parsed = decisions_from_obj(obj, global_ids)
            llm_status_hist["ok_decisions_list"] += 1
        if parsed is None and llm_text:
            parsed = _from_simple_triplet_map_text(llm_text, global_ids)
            if parsed is not None:
                llm_status_hist["ok_simple_map"] += 1

        # Decisions per rid
        decisions: Dict[int, Tuple[int,str,str]] = {}

        if parsed is None:
            for k, rid in enumerate(global_ids):
                if PURE_FALLBACK_MINUS_ONE:
                    label = -1
                    reason = "fallback_pure_minus_one"
                else:
                    sc = heuristic_record_score(clean_row, manip_rows[k], mask_rows[k])
                    label = 1 if sc >= FALLBACK_INTENT_THRESHOLD else -1
                    reason = f"heuristic_score={sc}"
                decisions[rid] = (label, reason, "fallback")
            sources_hist["fallback"] += 1
        else:
            any_from_llm = False
            for k, rid in enumerate(global_ids):
                tup = next((t for t in parsed if t[0] == rid), None)
                if tup is None or tup[1] is None:
                    if PURE_FALLBACK_MINUS_ONE:
                        label = -1
                        reason = "fallback_missing_pure_minus_one"
                    else:
                        sc = heuristic_record_score(clean_row, manip_rows[k], mask_rows[k])
                        label = 1 if sc >= FALLBACK_INTENT_THRESHOLD else -1
                        reason = f"heuristic_missing_score={sc}"
                    decisions[rid] = (label, reason, "fallback_missing")
                else:
                    label = tup[1]; reason = tup[2]
                    decisions[rid] = (label, reason, "llm")
                    any_from_llm = True
            sources_hist["llm" if any_from_llm else "fallback"] += 1

        # Fill outputs
        for k, rid in enumerate(global_ids):
            decision, reason, src = decisions[rid]
            for c in ADULT_COLS:
                if c == "class":
                    out_mask_df.at[rid, c] = "0"
                else:
                    out_mask_df.at[rid, c] = (
                        str(decision) if str(masks.at[rid, c]).strip() in ("1","1.0") else "0"
                    )
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
                        "details": json.dumps({"reason": reason, "source": src}, ensure_ascii=False),
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

    # Summary & instrumentation
    gen_vals = out_mask_df.replace({"1.0":"1","-1.0":"-1"}).values.flatten().tolist()
    counts = pd.Series(gen_vals).value_counts().to_dict()

    print("\n=== Generated Mask Counts (all cells) ===")
    for k in ["1","-1","0"]:
        print(f"{k:>3}: {counts.get(k,0)}")
    print(f"ALL: {len(gen_vals)}")

    print("\n=== Instrumentation ===")
    print("LLM status histogram:", json.dumps(llm_status_hist, indent=2))
    print("Decision sources histogram:", json.dumps(sources_hist, indent=2))

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
        "llm_status_hist": llm_status_hist,
        "decision_sources_hist": sources_hist,
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
