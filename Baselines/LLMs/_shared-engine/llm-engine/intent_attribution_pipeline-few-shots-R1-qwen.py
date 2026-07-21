#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Record-level Intent Attribution via Local LLM (Qwen via TGI) — Few-shot, Robust JSON
------------------------------------------------------------------------------------
TGI/Qwen fixes:
- Qwen ChatML template (<|im_start|> ... <|im_end|>), assistant turn left open.
- Seed assistant with '{"0":' so Qwen continues compact JSON map.
- Strip '</think>' / code fences; trim after first '}' and reattach seed before parsing.
- Accept either schema:
    A) {"0":1|-1,"1":1|-1,"2":1|-1}
    B) {"decisions":[{"row_id":...,"local_id":i,"decision":±1,"reason":"..."}]}
- Salvage decisions from prose if needed.
- Pure fallback: if still no parse, set all three to -1 (no heuristics).

Few-shot demos are included in the user message (USE_FEW_SHOT=True).
"""

from __future__ import annotations
import json, sys, os, time, re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# =========================
# ======== CONFIG =========
# =========================
SERVER_URL = "http://127.0.0.1:6800/generate"  # TGI endpoint serving Qwen
MAX_NEW_TOKENS = 96
REQUEST_TIMEOUT_SEC = 120

# File paths (adjust if needed)
MANIPULATED_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/manipulated_records.csv"
CORRECT_CSV     = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv"
MASKS_CSV       = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812/masks-blind.csv"

# Outputs
OUT_LABELS_CSV = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-qwen-deepseek/intent_labels.csv"
OUT_EXPL_CSV   = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/third-trial-qwen-deepseek/intent_explanations.csv"

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

def compute_diffs(clean_row: Dict[str,str], manip_row: Dict[str,str], mask_row: Dict[str,str]) -> List[Dict[str,str]]:
    diffs = []
    for c in ADULT_COLS:
        if c == "class":
            continue
        if mask_row.get(c) == "1":
            diffs.append({"column": c, "from": clean_row.get(c,""), "to": manip_row.get(c,"")})
    return diffs

# =========================
# ==== Qwen templating ====
# =========================
QWEN_IM_START = "<|im_start|>"
QWEN_IM_END   = "<|im_end|>"
ASSISTANT_JSON_SEED = '{"0":'  # seed so Qwen continues a compact map

def qwen_chat_seed(system_text: str, user_text: str, assistant_seed: str = "") -> str:
    return (
        f"{QWEN_IM_START}system\n{system_text}{QWEN_IM_END}\n"
        f"{QWEN_IM_START}user\n{user_text}{QWEN_IM_END}\n"
        f"{QWEN_IM_START}assistant\n{assistant_seed}"
    )

# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "You are a precise data-forensics assistant. Decide for each manipulated RECORD "
    "whether changes are INTENTIONAL (1) or UNINTENTIONAL (-1). Deterministic behavior."
)

RUBRIC = (
    "Rubric:\n"
    "- INTENTIONAL (1): large utility-seeking edits (education-num +≥2 or categorical upgrade), "
    "big jump in capital-gain from 0, hours-per-week +≥10, coordinated multiple edits, fairness gaming, privacy DMV.\n"
    "- UNINTENTIONAL (-1): tiny numeric drift (±1–2), case-only/formatting, single trivial change.\n"
    "- If uncertain: pick -1 only when edits are small/benign; otherwise 1."
)

def build_chat_input(clean_row: Dict[str, str],
                     group: List[Dict[str, str]],
                     group_masks: List[Dict[str, str]],
                     group_global_ids: List[int]) -> str:
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the clean row + three manipulated rows with their masks and DIFFS.\n"
        "- Return ONLY ONE JSON. Choose exactly ONE format:\n"
        '  A) {"0":1|-1,"1":1|-1,"2":1|-1}\n'
        '  B) {"decisions":[{"row_id":<global>,"local_id":0,"decision":1|-1,"reason":"<≤10w>"},'
        '{"row_id":<global>,"local_id":1,"decision":1|-1,"reason":"<≤10w>"},'
        '{"row_id":<global>,"local_id":2,"decision":1|-1,"reason":"<≤10w>"}]}\n'
        "- No extra text outside the JSON."
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
    user_text = f"{fewshot}INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\nReturn ONLY one JSON now."
    return qwen_chat_seed(system_text, user_text, assistant_seed=ASSISTANT_JSON_SEED)

# =========================
# == TGI call & parsing ===
# =========================
def call_tgi(inputs_str: str) -> str:
    params = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "return_full_text": False,
        "details": True,
        "stop": [QWEN_IM_END, "</think>", "```", "\n\n"],  # stop early if it starts rambling
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

def clean_and_reattach_seed(continuation: str) -> str:
    """
    Reattach the JSON seed and trim any trailing junk after the first '}'.
    Also strip known stray markers (</think>, code fences).
    Example continuation: '1,\"1\":-1,\"2\":-1}\\n</think>' -> '{"0":1,"1":-1,"2":-1}'
    """
    s = continuation or ""
    s = s.replace("</think>", "").replace("```json", "").replace("```", "")
    # cut at first closing brace
    idx = s.find("}")
    if idx != -1:
        s = s[:idx+1]
    s = s.strip()
    return ASSISTANT_JSON_SEED + s if s else ""

def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    s = text.strip().lstrip("\ufeff")
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            return None
    return None

def salvage_decisions_from_prose(llm_text: str, global_ids: List[int]) -> Optional[List[Tuple[int,int,str]]]:
    """Best-effort salvage if model produced prose with embedded signals."""
    if not llm_text:
        return None
    s = llm_text

    # fenced JSON first
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.I | re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "decisions" in obj:
                out = []
                for item in obj["decisions"]:
                    lid = int(item.get("local_id"))
                    dec = 1 if int(item.get("decision")) >= 1 else -1
                    out.append((global_ids[lid], dec, "salvage_fence_long"))
                return out if len(out) == 3 else None
            if isinstance(obj, dict) and all(k in obj for k in ("0","1","2")):
                out = []
                for lid in (0,1,2):
                    dec = 1 if int(obj[str(lid)]) >= 1 else -1
                    out.append((global_ids[lid], dec, "salvage_fence_map"))
                return out
        except Exception:
            pass

    # simple compact map anywhere
    m = re.search(r'\{\s*"0"\s*:\s*(-?1)\s*,\s*"1"\s*:\s*(-?1)\s*,\s*"2"\s*:\s*(-?1)\s*\}', s)
    if m:
        vals = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        return [(global_ids[i], 1 if vals[i] >= 1 else -1, "salvage_simple_map") for i in (0,1,2)]

    # long schema anywhere
    jstart, jend = s.find("{"), s.rfind("}")
    if 0 <= jstart < jend:
        frag = s[jstart:jend+1]
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict) and "decisions" in obj:
                out = []
                for item in obj["decisions"]:
                    lid = int(item.get("local_id"))
                    dec = 1 if int(item.get("decision")) >= 1 else -1
                    out.append((global_ids[lid], dec, "salvage_long"))
                if len(out) == 3:
                    return out
        except Exception:
            pass

    # prose keywords (Decision: -1 / 1) as last resort
    out = [None, None, None]
    s_low = s.lower()
    for lid in (0,1,2):
        hit = re.search(rf'(manipulated\s*(record|row)\s*{lid}\b|local[_\s-]*id\s*{lid}\b)', s_low)
        if not hit:
            continue
        window = s_low[hit.end(): hit.end()+500]
        m_dec = re.search(r'decision\s*[:\-–]\s*(-?1)\b', window)
        if m_dec:
            out[lid] = (global_ids[lid], int(m_dec.group(1)), "salvage_prose_decision"); continue
        if "unintentional" in window:
            out[lid] = (global_ids[lid], -1, "salvage_prose_keywords")
        elif "intentional" in window:
            out[lid] = (global_ids[lid], 1, "salvage_prose_keywords")

    if any(x is not None for x in out):
        return [x if x is not None else (global_ids[lid], None, "salvage_missing") for lid, x in enumerate(out)]
    return None

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
                       "ok_decisions_list": 0, "ok_simple_map": 0, "ok_salvage": 0}

    for j in tqdm(range(n_clean), desc="Groups (clean + 3 manipulated)", ncols=100):
        g_t0 = time.perf_counter()

        clean_row = row_to_dict(correct, j)
        manip_rows, mask_rows, global_ids = [], [], []
        for k in range(3):
            mi = 3*j + k
            manip_rows.append(row_to_dict(manip, mi))
            mask_rows.append(mask_row_to_dict(masks, mi))
            global_ids.append(mi)

        inputs_str = build_chat_input(clean_row, manip_rows, mask_rows, global_ids)

        obj = None
        llm_text = ""
        l0 = time.perf_counter()
        try:
            cont = call_tgi(inputs_str)
            llm_text = clean_and_reattach_seed(cont)  # reattach '{"0":' and trim
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

        # ----- accept either schema -----
        parsed: Optional[List[Tuple[int,int,str]]] = None
        if isinstance(obj, dict):
            # Long schema
            if "decisions" in obj and isinstance(obj["decisions"], list):
                tmp = []
                try:
                    for item in obj["decisions"]:
                        lid = int(item.get("local_id"))
                        dec = 1 if int(item.get("decision")) >= 1 else -1
                        rsn = (str(item.get("reason","")).strip() or "reason_not_provided")
                        tmp.append((global_ids[lid], dec, rsn))
                    parsed = tmp
                    llm_status_hist["ok_decisions_list"] += 1
                except Exception:
                    parsed = None
            # Compact map
            if parsed is None:
                try:
                    vals = [obj.get("0", None), obj.get("1", None), obj.get("2", None)]
                    if all(v is not None for v in vals):
                        parsed = []
                        for lid, v in enumerate(vals):
                            parsed.append((global_ids[lid], 1 if int(v) >= 1 else -1, "map"))
                        llm_status_hist["ok_simple_map"] += 1
                except Exception:
                    parsed = None

        if parsed is None and llm_text:
            salv = salvage_decisions_from_prose(llm_text, global_ids)
            if salv is not None:
                tmp = []
                for rid, dec, src in salv:
                    if dec is None:
                        tmp.append((rid, -1, "salvage_missing->-1"))
                    else:
                        tmp.append((rid, dec, src))
                parsed = tmp
                llm_status_hist["ok_salvage"] += 1

        # decisions per rid (pure -1 fallback)
        decisions: Dict[int, Tuple[int,str,str]] = {}
        if parsed is None:
            for rid in global_ids:
                decisions[rid] = (-1, "fallback_pure_minus1", "fallback")
            sources_hist["fallback"] += 1
        else:
            any_from_llm = False
            by_rid = {rid: (dec, rsn) for (rid, dec, rsn) in parsed}
            for rid in global_ids:
                if rid in by_rid:
                    dec, rsn = by_rid[rid]
                    decisions[rid] = (dec, rsn, "llm"); any_from_llm = True
                else:
                    decisions[rid] = (-1, "missing_llm_entry->-1", "fallback")
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
            # Per-cell explanation rows (record-level reason)
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
        "clean_rows": len(correct),
        "manip_rows": len(manip),
        "groups": len(correct),
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
        "group_id": list(range(len(correct))),
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