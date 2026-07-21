#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cell-level Intent Attribution via Google Gemini
-----------------------------------------------------------
Addresses reviewer W6 (evaluation asymmetry): instead of making one
binary decision per manipulated RECORD, the LLM now makes one decision
per CHANGED CELL.  This matches NIYYAT's cell-level evaluation unit and
makes cell-level F1 scores directly comparable.

Expected dataset: generated with 50% pure-unintentional / 20% pure-
intentional / 30% mixed proportions (see generate-manipulated-data-gemini-v2.py
with updated P_PURE_INTENTIONAL=0.20, P_MIXED=0.30).

Input mask convention (masks_blind.csv):
   1  — cell was changed (no intent revealed)
   0  — cell is unchanged

Output mask (intent_labels.csv):
   1  — cell predicted INTENTIONAL
  -1  — cell predicted UNINTENTIONAL
   0  — cell not changed (pass-through)

Ground-truth mask (masks.csv):
   1  — intentional change
  -1  — unintentional change
   0  — no change

NOTE on rate limits: gemini-2.5-pro paid tier has a daily/per-minute request
cap. Making one API call per group of 3 manipulated rows (≈6500 calls for the
full Adult dataset) burns through quota almost immediately and right after
generation already consumed a large share of it. To stay within budget this
pipeline batches GROUPS_PER_CALL groups (default 7, i.e. 21 manipulated rows)
into a single request — matching the chunking strategy already used by the
dataset generator — and retries with exponential backoff on failure.
"""

from __future__ import annotations
import json
import sys
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import GEMINI_API_KEY

# =========================
# ======== CONFIG =========
# =========================
MODEL_NAME_LITE = "gemini-2.5-flash-lite"  # Cost policy 2026-06-26: never gemini-2.5-pro
MODEL_NAME_FLASH = "gemini-2.5-flash"

GROUPS_PER_CALL = int(os.getenv("GROUPS_PER_CALL", "7"))   # 7 groups = 21 manipulated rows / call
MAX_RETRIES_PER_CALL = int(os.getenv("MAX_RETRIES_PER_CALL", "6"))
RETRY_BACKOFF_BASE_SEC = float(os.getenv("RETRY_BACKOFF_BASE_SEC", "5.0"))
INTER_CALL_SLEEP_SEC = float(os.getenv("INTER_CALL_SLEEP_SEC", "1.0"))

MANIPULATED_CSV = (
    "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/"
    "tenth-trial/data/raw/run_v3_PLACEHOLDER/manipulated_records.csv"
)
CORRECT_CSV = (
    "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/"
    "tenth-trial/data/raw/correct_records.csv"
)
MASKS_CSV = (
    "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/"
    "tenth-trial/data/raw/run_v3_PLACEHOLDER/masks_blind.csv"
)

OUT_DIR_NAME = {
    "zero": "cell-level-gemini-pro",
    "info": "cell-level-gemini-pro-info",
    "few":  "cell-level-gemini-pro-few",
}

def out_paths(prompt_variant: str) -> Tuple[str, str]:
    d = OUT_DIR_NAME[prompt_variant]
    base = f"/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/model_outputs/local-llms/{d}"
    return f"{base}/intent_labels.csv", f"{base}/intent_explanations.csv"


# Same rubric as adult_row_level_eval.py's PROMPT_VARIANTS, adapted from a
# per-record to a per-cell decision (this pipeline decides per CHANGED CELL,
# not per manipulated record).
PROMPT_VARIANTS = {
    "zero": "",
    "info": (
        "\n\nDecision rubric:\n"
        "- Intentional (gain-targeted): a changed cell that would flip predicted income to >50K "
        "(e.g. raising capital-gain, hours-per-week, or education level).\n"
        "- Intentional (fairness-masking): a sensitive attribute (race, sex) changed to the "
        "dataset's majority value.\n"
        "- Intentional (obfuscation): a categorical cell replaced with 'Unknown'/'N/A'/'---'.\n"
        "- Unintentional: a typo, OCR-style artefact, or random noise with no coherent direction.\n"
        "- A single record may contain BOTH kinds of cell changes -- decide each cell on its own "
        "merits, not by what the rest of the record looks like."
    ),
    "few": (
        "\n\nDecision rubric (same as info) plus worked examples:\n"
        "- capital-gain 0 -> 99999: intentional (gain-targeted).\n"
        "- race 'Black' -> 'White', no other change in that cell: intentional (fairness-masking).\n"
        "- workclass 'Private' -> 'Unknown': intentional (obfuscation).\n"
        "- age 35 -> 35l (typo-like corruption): unintentional.\n"
        "- native-country 'United-States' -> 'Uniteed-States': unintentional.\n"
        "- In a record where capital-gain 0 -> 99999 (gain-targeted, intentional) AND "
        "native-country has a typo (unintentional), label each cell independently: the "
        "capital-gain cell is intentional, the native-country cell is unintentional.\n"
    ),
}

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]


# =========================
# ====== UTILITIES ========
# =========================
def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)


def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in ADULT_COLS}


def changed_cols_from_blind_mask(mask_df: pd.DataFrame, idx: int) -> List[str]:
    """Return columns where the blind mask == 1 (cell was changed)."""
    return [
        c for c in ADULT_COLS
        if c != "class" and str(mask_df.at[idx, c]).strip() in ("1", "1.0")
    ]


# =========================
# ===== GEMINI CALL =======
# =========================
def call_gemini_raw(model, prompt: str) -> Tuple[str, Optional[str]]:
    """Single attempt. Returns (text, error_message). error_message is None on success."""
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 16384,
                "candidate_count": 1,
            },
            safety_settings=safety_settings,
            request_options={"timeout": 180},
        )
        if not response.candidates:
            return "", "no_candidates"
        candidate = response.candidates[0]
        if candidate.finish_reason.value not in (1, 2):
            return "", f"finish_reason={candidate.finish_reason.value}"
        if not candidate.content or not getattr(candidate.content, "parts", None):
            return "", "no_parts"
        text = candidate.content.parts[0].text or ""
        if not text:
            return "", "empty_text"
        return text, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def call_gemini_with_retry(model, prompt: str) -> Tuple[str, Dict]:
    """Retry with exponential backoff. Returns (text, debug_info)."""
    last_err = None
    for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
        text, err = call_gemini_raw(model, prompt)
        if err is None and text:
            return text, {"attempts": attempt, "error": None}
        last_err = err
        backoff = RETRY_BACKOFF_BASE_SEC * (2 ** (attempt - 1))
        time.sleep(min(backoff, 60.0))
    return "", {"attempts": MAX_RETRIES_PER_CALL, "error": last_err}


def call_gemini_with_fallback(model_lite, model_flash, prompt: str) -> Tuple[str, Dict]:
    """Cost policy (2026-06-26): exhaust retries on flash-lite first, then
    fall back to flash, then give up. Never escalates to gemini-2.5-pro."""
    text, dbg = call_gemini_with_retry(model_lite, prompt)
    if text:
        dbg["model_tier"] = "flash-lite"
        return text, dbg
    text, dbg = call_gemini_with_retry(model_flash, prompt)
    dbg["model_tier"] = "flash" if text else "exhausted"
    return text, dbg


# =========================
# ======= PARSING =========
# =========================
def parse_llm_json(text: str) -> Optional[dict]:
    s = (text or "").lstrip("﻿").strip()
    for fence in ("```json", "```", "~~~json", "~~~"):
        s = s.replace(fence, "")
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
            import ast
            obj = ast.literal_eval(frag)
            return obj if isinstance(obj, (dict, list)) else None
        except Exception:
            return None


def extract_chunk_decisions(
    obj: dict,
    chunk_group_ids: List[int],          # local group indices 0..len(chunk)-1
    chunk_global_ids: List[List[int]],   # per-group [row_id0, row_id1, row_id2]
    chunk_changed_cols: List[List[List[str]]],  # per-group, per-local-row, list of changed cols
) -> Optional[Dict[int, Dict[str, int]]]:
    """
    Expected format:
      {"groups": {
          "0": {"decisions": {"0": {"age": 1}, "1": {"workclass": -1}, "2": {...}}},
          "1": {"decisions": {...}},
          ...
      }}

    Returns {global_row_id: {col: decision}} or None if the whole chunk is unusable.
    """
    if not isinstance(obj, dict):
        return None
    groups_raw = obj.get("groups")
    if not isinstance(groups_raw, dict):
        return None

    result: Dict[int, Dict[str, int]] = {}
    for gi in chunk_group_ids:
        g_entry = groups_raw.get(str(gi), groups_raw.get(gi))
        if not isinstance(g_entry, dict):
            return None
        decs_raw = g_entry.get("decisions", g_entry)
        if not isinstance(decs_raw, dict):
            return None

        for lid in (0, 1, 2):
            rid = chunk_global_ids[gi][lid]
            expected_cols = set(chunk_changed_cols[gi][lid])
            cell_map = decs_raw.get(str(lid), decs_raw.get(lid))
            if not isinstance(cell_map, dict):
                return None

            per_col: Dict[str, int] = {}
            for col, val in cell_map.items():
                if col not in expected_cols:
                    continue
                try:
                    dec = 1 if int(val) >= 1 else -1
                except Exception:
                    dec = -1
                per_col[col] = dec
            for col in expected_cols:
                if col not in per_col:
                    per_col[col] = -1

            result[rid] = per_col

    return result


# =========================
# ======== PROMPT =========
# =========================
ROLE_BLOCK = (
    "ROLE\n"
    "You are a precise data-forensics assistant. "
    "You will receive several GROUPS. Each group has one clean record and three "
    "manipulated variants of it, each with a list of changed cells. "
    "Your task is to decide, FOR EACH CHANGED CELL INDEPENDENTLY, whether that "
    "individual cell change is INTENTIONAL (1) or UNINTENTIONAL (-1). "
    "A single record may contain BOTH intentional and unintentional changes "
    "in different cells. Deterministic behavior is required."
)


def make_chunk_prompt(
    chunk_clean_rows: List[Dict[str, str]],
    chunk_manip_rows: List[List[Dict[str, str]]],      # per-group, 3 rows
    chunk_changed_cols: List[List[List[str]]],          # per-group, 3 lists of changed cols
    chunk_global_ids: List[List[int]],                  # per-group, 3 global row ids
    prompt_variant: str = "zero",
) -> str:
    sys_rules = (
        "Rules:\n"
        "- Use ONLY the provided clean row, three manipulated rows, and "
        "their changed-column lists, PER GROUP.\n"
        "- Decide 1 (intentional) or -1 (unintentional) PER CHANGED CELL. "
        "No zeros. No skipping.\n"
        "- Cells in different rows/groups are independent — do not force the "
        "same label across a row or group.\n"
        "- Deterministic, no randomness.\n"
        f"{PROMPT_VARIANTS[prompt_variant]}\n"
        "- Return ONLY one strict JSON object. No extra text.\n\n"
        "OUTPUT FORMAT (exactly, one entry per group_index):\n"
        '{"groups": {\n'
        '  "0": {"decisions": {"0": {"<col>": 1|-1, ...}, "1": {...}, "2": {...}}},\n'
        '  "1": {"decisions": {"0": {...}, "1": {...}, "2": {...}}},\n'
        "  ...\n"
        "}}\n"
        "Include ONLY columns that appear in changed_columns for that local_id."
    )

    groups_payload = []
    for gi, clean_row in enumerate(chunk_clean_rows):
        groups_payload.append({
            "group_index": gi,
            "clean": clean_row,
            "manipulated": [
                {
                    "local_id": k,
                    "row_id": chunk_global_ids[gi][k],
                    "row": chunk_manip_rows[gi][k],
                    "changed_columns": chunk_changed_cols[gi][k],
                }
                for k in range(3)
            ],
        })

    bundle = {"columns": ADULT_COLS, "groups": groups_payload}

    return (
        f"{ROLE_BLOCK}\n\n{sys_rules}\n\n"
        f"INPUT:\n{json.dumps(bundle, ensure_ascii=False)}\n\n"
        "Return the JSON object now."
    )


# =========================
# ========= MAIN ==========
# =========================
def run_pipeline(manip_path: str, correct_path: str, masks_path: str, prompt_variant: str = "zero") -> None:
    t0 = time.perf_counter()

    out_labels_csv, out_expl_csv = out_paths(prompt_variant)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME_LITE)
    model_flash = genai.GenerativeModel(MODEL_NAME_FLASH)

    out_dir = os.path.dirname(out_labels_csv)
    os.makedirs(out_dir, exist_ok=True)
    stats_path = os.path.join(out_dir, "run_stats.json")
    per_chunk_path = os.path.join(out_dir, "per_chunk_times.csv")
    fail_dir = os.path.join(out_dir, "failures")
    os.makedirs(fail_dir, exist_ok=True)

    manip = pd.read_csv(manip_path, dtype=str, keep_default_na=False)
    correct = pd.read_csv(correct_path, dtype=str, keep_default_na=False)
    masks = pd.read_csv(masks_path, dtype=str, keep_default_na=False)

    for df, name in [(manip, "manipulated"), (correct, "correct"), (masks, "masks_blind")]:
        missing = [c for c in ADULT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        df[:] = df[ADULT_COLS]

    n_clean = len(correct)
    n_manip = len(manip)
    if n_manip != 3 * n_clean:
        raise ValueError(
            f"Expected manipulated rows == 3 * clean rows; "
            f"got {n_manip} vs 3*{n_clean}={3*n_clean}"
        )

    out_mask_df = pd.DataFrame("0", index=range(n_manip), columns=ADULT_COLS)
    explanations: List[Dict] = []

    chunk_times: List[float] = []
    ok_groups = fallback_groups = 0

    num_chunks = (n_clean + GROUPS_PER_CALL - 1) // GROUPS_PER_CALL
    pbar = tqdm(range(num_chunks), desc="Chunks (cell-level attribution)", ncols=110)

    for chunk_idx in pbar:
        c0 = time.perf_counter()
        start_j = chunk_idx * GROUPS_PER_CALL
        end_j = min(start_j + GROUPS_PER_CALL, n_clean)
        group_js = list(range(start_j, end_j))

        chunk_clean_rows, chunk_manip_rows, chunk_changed_cols, chunk_global_ids = [], [], [], []
        for j in group_js:
            clean_row = row_to_dict(correct, j)
            chunk_clean_rows.append(clean_row)
            rows_3, cols_3, ids_3 = [], [], []
            for k in range(3):
                mi = 3 * j + k
                rows_3.append(row_to_dict(manip, mi))
                cols_3.append(changed_cols_from_blind_mask(masks, mi))
                ids_3.append(mi)
            chunk_manip_rows.append(rows_3)
            chunk_changed_cols.append(cols_3)
            chunk_global_ids.append(ids_3)

        local_group_ids = list(range(len(group_js)))  # 0..len(chunk)-1, indexes into chunk_* lists
        prompt_text = make_chunk_prompt(
            chunk_clean_rows, chunk_manip_rows, chunk_changed_cols, chunk_global_ids, prompt_variant
        )

        llm_text, dbg = call_gemini_with_fallback(model, model_flash, prompt_text)
        obj = parse_llm_json(llm_text) if llm_text else None

        decisions: Optional[Dict[int, Dict[str, int]]] = None
        if isinstance(obj, dict):
            decisions = extract_chunk_decisions(
                obj, local_group_ids, chunk_global_ids, chunk_changed_cols
            )

        if decisions is None:
            fallback_groups += len(group_js)
            decisions = {}
            for gi in local_group_ids:
                for k in range(3):
                    rid = chunk_global_ids[gi][k]
                    decisions[rid] = {col: -1 for col in chunk_changed_cols[gi][k]}

            fail_path = os.path.join(fail_dir, f"chunk_{chunk_idx}.json")
            with open(fail_path, "w", encoding="utf-8") as f:
                json.dump({
                    "chunk_index": chunk_idx,
                    "start_j": start_j, "end_j": end_j,
                    "debug": dbg,
                    "raw_response_snippet": (llm_text or "")[:2000],
                }, f, ensure_ascii=False, indent=2)
        else:
            ok_groups += len(group_js)

        for gi in local_group_ids:
            for k in range(3):
                rid = chunk_global_ids[gi][k]
                per_col = decisions.get(rid, {})
                for col, dec in per_col.items():
                    out_mask_df.at[rid, col] = str(dec)
                for col in chunk_changed_cols[gi][k]:
                    dec = per_col.get(col, -1)
                    explanations.append({
                        "row_id": rid,
                        "column": col,
                        "manipulated_value": chunk_manip_rows[gi][k][col],
                        "original_value": chunk_clean_rows[gi][col],
                        "diagnosis": str(dec),
                        "rule": "cell_intent_decision",
                        "details": json.dumps({"cell_level": True}, ensure_ascii=False),
                    })

        chunk_times.append(time.perf_counter() - c0)
        time.sleep(INTER_CALL_SLEEP_SEC)

    out_mask_df.to_csv(out_labels_csv, index=False)
    pd.DataFrame(explanations).to_csv(out_expl_csv, index=False)

    flat = out_mask_df.values.flatten().tolist()
    counts = pd.Series(flat).value_counts().to_dict()
    print("\n=== Generated Cell-level Mask Counts ===")
    for k in ["1", "-1", "0"]:
        print(f"  {k:>3}: {counts.get(k, 0)}")
    print(f"  ALL: {len(flat)}")

    t1 = time.perf_counter()
    stats = {
        "model": f"{MODEL_NAME_LITE}+fallback:{MODEL_NAME_FLASH}",
        "prompt_variant": prompt_variant,
        "evaluation_unit": "cell",
        "groups_per_call": GROUPS_PER_CALL,
        "clean_rows": n_clean,
        "manip_rows": n_manip,
        "total_time_sec": t1 - t0,
        "avg_chunk_time_sec": float(pd.Series(chunk_times).mean()) if chunk_times else None,
        "llm_ok_groups": ok_groups,
        "llm_fallback_groups": fallback_groups,
        "outputs": {
            "intent_labels_csv": out_labels_csv,
            "explanations_csv": out_expl_csv,
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "chunk_id": list(range(num_chunks)),
        "chunk_time_sec": chunk_times,
    }).to_csv(per_chunk_path, index=False)

    print(f"\nStats: {stats_path}")
    print(f"Output labels: {out_labels_csv}")
    print(f"Explanations: {out_expl_csv}")
    print(f"LLM ok groups: {ok_groups} / fallback groups: {fallback_groups}")


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else MANIPULATED_CSV
    c = sys.argv[2] if len(sys.argv) > 2 else CORRECT_CSV
    k = sys.argv[3] if len(sys.argv) > 3 else MASKS_CSV
    v = sys.argv[4] if len(sys.argv) > 4 else "zero"
    assert v in PROMPT_VARIANTS, f"--variant must be one of {list(PROMPT_VARIANTS)}, got {v!r}"
    run_pipeline(m, c, k, v)
