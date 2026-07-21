#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intent Attribution — Three Qwen Variants — Mixed SOTA Twitter Bot Dataset
=========================================================================

Runs three prompt strategies back-to-back on the Mixed SOTA TwiBot-20 dataset
using Qwen2.5-32B-Instruct served via TGI on port 6300.

  Variant 1 — zero_shot : plain JSON bundle, no rubric, no examples
  Variant 2 — info      : adds per-feature DIFFS + decision rubric
  Variant 3 — few_shot  : info + three in-context examples

Key design:
  - Feature-level decisions: each changed feature gets its own 1/-1
  - Blind mask: model sees 0/1 (was changed?) — NOT the true intent
  - Ground-truth mask: 1=intentional, -1=unintentional, 0=clean
  - Chunked processing (default 7 records/chunk to fit token budget)
  - Qwen ChatML template: <|im_start|>...<|im_end|>

Usage
-----
  python run_qwen_twitterbot_mixed_sota.py --n_records 50    # quick test
  python run_qwen_twitterbot_mixed_sota.py                   # full run
  python run_qwen_twitterbot_mixed_sota.py --no_start_server --no_stop_server
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ── Defaults ──────────────────────────────────────────────────────────────────
_BASE = "/home/mohamed/error_injector/llms_baseline/mixed_error_pipeline_twitter/output"

DEFAULT_DIRTY = os.path.join(_BASE, "twibot20_phase2_final.csv")
DEFAULT_CLEAN = os.path.join(_BASE, "twibot20_clean.csv")
DEFAULT_MASK  = os.path.join(_BASE, "mask_combined.csv")

DEFAULT_SERVER_URL = "http://127.0.0.1:6300/generate"
DEFAULT_HEALTH_URL = "http://127.0.0.1:6300/health"

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "..", ".."))
DEFAULT_COMPOSE_FILE = os.path.join(_REPO_ROOT, "docker-compose", "docker-compose-qwen.yml")

TWITTER_BOT_COLS = [
    "user_id", "followers_count", "friends_count", "favourites_count",
    "statuses_count", "listed_count", "verified", "protected",
    "default_profile", "default_profile_image", "geo_enabled",
    "profile_use_background_image", "has_created_date",
    "description_length", "screen_name_length", "name_length",
    "has_description", "has_location", "label",
]
_SKIP_COLS = {"user_id", "label"}

DEFAULT_CHUNK_SIZE  = 7
MAX_NEW_TOKENS      = 1200
REQUEST_TIMEOUT_SEC = 240

# Qwen ChatML tokens
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"


# ─────────────────────────────────────────────────────────────────────────────
# Server management
# ─────────────────────────────────────────────────────────────────────────────

def start_server(compose_file: str, health_url: str, wait_sec: int = 600) -> None:
    import subprocess
    if not os.path.exists(compose_file):
        sys.exit(f"[ERROR] Compose file not found: {compose_file}")
    print(f"\n[Server] Starting Qwen TGI…  compose: {compose_file}")
    subprocess.run(["docker", "compose", "-f", compose_file, "up", "-d"],
                   capture_output=True, text=True)
    _wait_for_server(health_url, wait_sec)

def stop_server(compose_file: str) -> None:
    import subprocess
    if not os.path.exists(compose_file):
        return
    print(f"\n[Server] Stopping Qwen TGI…")
    subprocess.run(["docker", "compose", "-f", compose_file, "down"],
                   capture_output=True, text=True)
    print("[Server] Stopped.")

def _wait_for_server(health_url: str, timeout_sec: int) -> None:
    print(f"[Server] Waiting for TGI (up to {timeout_sec}s)…", flush=True)
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if requests.get(health_url, timeout=5).status_code == 200:
                print(f"\n[Server] TGI ready ✓")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    sys.exit(f"\n[ERROR] TGI not ready after {timeout_sec}s.")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def to_str(x) -> str:
    return "" if pd.isna(x) else str(x)

def row_to_dict(df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {c: to_str(df.at[idx, c]) for c in TWITTER_BOT_COLS}

def blind_mask_row(mask_df: pd.DataFrame, idx: int) -> Dict[str, str]:
    return {
        c: ("1" if str(mask_df.at[idx, c]).strip() not in ("0", "0.0", "") else "0")
        for c in TWITTER_BOT_COLS
    }

def changed_features_list(clean_row, dirty_row, blind) -> List[Dict]:
    return [
        {"column": c, "original_value": clean_row[c], "changed_value": dirty_row[c]}
        for c in TWITTER_BOT_COLS
        if c not in _SKIP_COLS and blind.get(c) == "1"
    ]

def compute_diffs(clean_row, dirty_row, blind) -> List[Dict]:
    return [
        {"column": c, "from": clean_row.get(c, ""), "to": dirty_row.get(c, "")}
        for c in TWITTER_BOT_COLS
        if c not in _SKIP_COLS and blind.get(c) == "1"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Qwen ChatML templates
# ─────────────────────────────────────────────────────────────────────────────

def qwen_chat(system_text: str, user_text: str) -> str:
    return (
        f"{_IM_START}system\n{system_text}{_IM_END}\n"
        f"{_IM_START}user\n{user_text}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TGI call
# ─────────────────────────────────────────────────────────────────────────────

def call_tgi(inputs: str, server_url: str) -> str:
    payload = {
        "inputs": inputs,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "return_full_text": False,
            "details": True,
            "stop": [_IM_END],
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
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_json(text: str) -> Optional[dict]:
    if not text:
        return None
    s = text.lstrip("\ufeff").strip()
    for junk in ("</think>", _IM_END, "```json", "```", "~~~json", "~~~"):
        s = s.replace(junk, "")
    s = s.strip()
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
                    frag = s[start: i + 1]
                    try:
                        return json.loads(frag)
                    except Exception:
                        try:
                            obj = ast.literal_eval(frag)
                            return obj if isinstance(obj, (dict, list)) else None
                        except Exception:
                            return None
    return None


def decisions_from_obj(obj, global_ids) -> Optional[Dict[Tuple[int, str], Tuple[int, str]]]:
    if not isinstance(obj, dict):
        return None
    feat_decs = obj.get("feature_decisions")
    if not isinstance(feat_decs, list) or not feat_decs:
        return None
    decisions: Dict[Tuple[int, str], Tuple[int, str]] = {}
    for item in feat_decs:
        if not isinstance(item, dict):
            continue
        try:
            row_id   = int(item.get("row_id"))
            column   = str(item.get("column", "")).strip()
            dec_val  = int(item.get("decision", -1))
            reason   = str(item.get("reason", "")).strip() or "no_reason"
            if column:
                decisions[(row_id, column)] = (1 if dec_val >= 1 else -1, reason)
        except (ValueError, TypeError, KeyError):
            continue
    return decisions if decisions else None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt blocks
# ─────────────────────────────────────────────────────────────────────────────

_ROLE = """\
You are a precise data-forensics assistant specializing in Twitter bot detection evasion analysis.
Analyze changes made to individual features in Twitter bot profiles and determine whether each change was:
- INTENTIONAL (1): Deliberately made to make a bot profile appear more human-like or evade bot detection
- UNINTENTIONAL (-1): Accidental changes, or normal profile variations
IMPORTANT: Evaluate EACH CHANGED FEATURE INDEPENDENTLY."""

_RUBRIC = """\
Rubric for Twitter Bot Evasion Detection:
INTENTIONAL (1): Large follower/friend count increases, engagement gaming (favourites/statuses jumps),
  verification changes, adding description/location, disabling default_profile/default_profile_image,
  enabling geo_enabled or profile_use_background_image, coordinated multi-feature evasion edits.
UNINTENTIONAL (-1): Small numeric drift (≤5% change), minor description_length edits,
  small listed_count variations, formatting-only differences."""

_FEW_SHOT = """\
EXAMPLE 1 — Evasion: followers_count 45→15000, friends_count 32→12500, has_description 0→1, description_length 0→85
{"feature_decisions":[
  {"row_id":1001,"column":"followers_count","decision":1,"reason":"massive follower jump for popularity"},
  {"row_id":1001,"column":"friends_count","decision":1,"reason":"large increase for social appearance"},
  {"row_id":1001,"column":"has_description","decision":1,"reason":"profile completion strategy"},
  {"row_id":1001,"column":"description_length","decision":1,"reason":"coordinated with description addition"}
]}

EXAMPLE 2 — Mixed: verified 0→1 (intentional), statuses_count 1210→1215 (unintentional), followers_count 820→845 (unintentional), friends_count 150→50000 (intentional)
{"feature_decisions":[
  {"row_id":2001,"column":"verified","decision":1,"reason":"verification gaming"},
  {"row_id":2001,"column":"statuses_count","decision":-1,"reason":"tiny activity variation"},
  {"row_id":2002,"column":"followers_count","decision":-1,"reason":"minor natural growth"},
  {"row_id":2003,"column":"friends_count","decision":1,"reason":"extreme jump for evasion"}
]}

EXAMPLE 3 — Coordinated: default_profile 1→0, default_profile_image 1→0, has_location 0→1, profile_use_background_image 0→1
{"feature_decisions":[
  {"row_id":3001,"column":"default_profile","decision":1,"reason":"profile customization for authenticity"},
  {"row_id":3001,"column":"default_profile_image","decision":1,"reason":"image customization"},
  {"row_id":3001,"column":"has_location","decision":1,"reason":"location added for authenticity"},
  {"row_id":3001,"column":"profile_use_background_image","decision":1,"reason":"coordinated background activation"}
]}"""

_OUTPUT_RULES = """\
Rules:
- For EACH CHANGED FEATURE (mask=1), provide a separate decision.
- Return ONLY a JSON object, no extra text.
Required format:
{"feature_decisions":[{"row_id":<id>,"column":"<col>","changed_value":"<val>","original_value":"<val>","decision":1 or -1,"reason":"<≤15 words>"},...]}}"""


def _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids, with_diffs=False):
    records = []
    for i in range(len(clean_rows)):
        cf = changed_features_list(clean_rows[i], dirty_rows[i], blind_rows[i])
        if not cf:
            continue
        entry = {"row_id": global_ids[i], "clean_profile": clean_rows[i],
                 "changed_profile": dirty_rows[i], "changed_features": cf}
        if with_diffs:
            entry["diffs"] = compute_diffs(clean_rows[i], dirty_rows[i], blind_rows[i])
        records.append(entry)
    return {"columns": TWITTER_BOT_COLS, "records": records}


def make_prompt_zero_shot(clean_rows, dirty_rows, blind_rows, global_ids) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids)
    system = f"{_ROLE}\n\n{_OUTPUT_RULES}"
    user   = f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False)}\n\nReturn JSON now."
    return qwen_chat(system, user)


def make_prompt_info(clean_rows, dirty_rows, blind_rows, global_ids) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids, with_diffs=True)
    system = f"{_ROLE}\n\n{_RUBRIC}\n\n{_OUTPUT_RULES}"
    user   = f"INPUT DATA:\n{json.dumps(bundle, ensure_ascii=False)}\n\nReturn JSON now."
    return qwen_chat(system, user)


def make_prompt_few_shot(clean_rows, dirty_rows, blind_rows, global_ids) -> str:
    bundle = _build_bundle(clean_rows, dirty_rows, blind_rows, global_ids, with_diffs=True)
    system = f"{_ROLE}\n\n{_RUBRIC}\n\n{_OUTPUT_RULES}"
    user   = f"{_FEW_SHOT}\n\nINPUT DATA:\n{json.dumps(bundle, ensure_ascii=False)}\n\nReturn JSON now."
    return qwen_chat(system, user)


VARIANTS = {
    "zero_shot": make_prompt_zero_shot,
    "info":      make_prompt_info,
    "few_shot":  make_prompt_few_shot,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> Dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == -1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == -1)
    n  = len(y_true)
    acc    = (tp + tn) / n if n else 0
    prec_i = tp / (tp + fp) if (tp + fp) else 0
    rec_i  = tp / (tp + fn) if (tp + fn) else 0
    f1_i   = 2 * prec_i * rec_i / (prec_i + rec_i) if (prec_i + rec_i) else 0
    prec_u = tn / (tn + fn) if (tn + fn) else 0
    rec_u  = tn / (tn + fp) if (tn + fp) else 0
    f1_u   = 2 * prec_u * rec_u / (prec_u + rec_u) if (prec_u + rec_u) else 0
    f1_w   = (f1_i * (tp + fn) + f1_u * (tn + fp)) / n if n else 0
    return {
        "n_features": n, "accuracy": round(acc, 4),
        "precision_int": round(prec_i, 4), "recall_int": round(rec_i, 4),
        "f1_intentional": round(f1_i, 4),
        "precision_unint": round(prec_u, 4), "recall_unint": round(rec_u, 4),
        "f1_unintentional": round(f1_u, 4), "f1_weighted": round(f1_w, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "pred_counts": dict(Counter(y_pred)), "true_counts": dict(Counter(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Variant runner
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(variant_name, prompt_fn, server_url, dirty_df, clean_df, mask_df,
                n_records, chunk_size, out_dir) -> Dict:
    v_dir = os.path.join(out_dir, variant_name)
    os.makedirs(v_dir, exist_ok=True)

    out_mask_df = mask_df.copy(deep=True)
    pred_rows, y_true, y_pred = [], [], []
    chunk_times, llm_times = [], []
    llm_success, fallback_chunks = 0, 0
    status_hist = {"ok": 0, "http_error": 0, "fallback": 0}

    print(f"\n{'='*65}\n  VARIANT: {variant_name.upper()}  ({n_records} records, chunk={chunk_size})\n{'='*65}")
    n_chunks = (n_records + chunk_size - 1) // chunk_size

    for chunk_idx in tqdm(range(n_chunks), desc=variant_name, ncols=90):
        ct0 = time.perf_counter()
        start_i    = chunk_idx * chunk_size
        end_i      = min(start_i + chunk_size, n_records)
        clean_rows = [row_to_dict(clean_df, i) for i in range(start_i, end_i)]
        dirty_rows = [row_to_dict(dirty_df, i) for i in range(start_i, end_i)]
        blind_rows = [blind_mask_row(mask_df, i) for i in range(start_i, end_i)]
        global_ids = list(range(start_i, end_i))

        prompt_text = prompt_fn(clean_rows, dirty_rows, blind_rows, global_ids)
        obj = None
        lt0 = time.perf_counter()
        try:
            llm_text = call_tgi(prompt_text, server_url)
            obj = parse_llm_json(llm_text)
            status_hist["ok"] += 1
        except Exception:
            status_hist["http_error"] += 1
            llm_text = ""
        llm_times.append(time.perf_counter() - lt0)

        parsed = decisions_from_obj(obj, global_ids) if isinstance(obj, dict) else None
        if parsed is not None:
            llm_success += 1
            feature_decisions = parsed
        else:
            fallback_chunks += 1
            status_hist["fallback"] += 1
            feature_decisions = {
                (rid, col): (-1, "llm_fallback")
                for local_idx, rid in enumerate(global_ids)
                for col in TWITTER_BOT_COLS
                if col not in _SKIP_COLS and blind_rows[local_idx].get(col) == "1"
            }

        for local_idx, rid in enumerate(global_ids):
            for col in TWITTER_BOT_COLS:
                try:
                    raw = int(float(str(mask_df.at[rid, col]).strip()))
                except (ValueError, TypeError):
                    raw = 0
                if raw != 0:
                    true_intent = 1 if raw > 0 else -1
                    decision, reason = feature_decisions.get((rid, col), (-1, "guard_fallback"))
                    out_mask_df.at[rid, col] = decision
                    y_true.append(true_intent)
                    y_pred.append(decision)
                    pred_rows.append({"row_id": rid, "column": col,
                                      "true_intent": true_intent, "pred_intent": decision,
                                      "reason": reason,
                                      "clean_val": clean_rows[local_idx].get(col, ""),
                                      "dirty_val": dirty_rows[local_idx].get(col, "")})
                else:
                    out_mask_df.at[rid, col] = 0
        chunk_times.append(time.perf_counter() - ct0)

    pd.DataFrame(pred_rows).to_csv(os.path.join(v_dir, "predictions.csv"), index=False)
    out_mask_df.to_csv(os.path.join(v_dir, "intent_labels.csv"), index=False)

    metrics = compute_metrics(y_true, y_pred)
    metrics.update({"variant": variant_name, "llm_success_chunks": llm_success,
                    "fallback_chunks": fallback_chunks, "status_hist": status_hist,
                    "avg_llm_time_s": round(sum(llm_times) / len(llm_times), 3) if llm_times else 0,
                    "total_llm_time_s": round(sum(llm_times), 2)})
    with open(os.path.join(v_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n  Results — {variant_name}:")
    print(f"    Features    : {metrics['n_features']}  |  Acc: {metrics['accuracy']:.4f}  |  F1_w: {metrics['f1_weighted']:.4f}")
    print(f"    F1_int: {metrics['f1_intentional']:.4f}  |  F1_unint: {metrics['f1_unintentional']:.4f}  |  Fallbacks: {fallback_chunks}/{n_chunks}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirty",         default=DEFAULT_DIRTY)
    parser.add_argument("--clean",         default=DEFAULT_CLEAN)
    parser.add_argument("--mask",          default=DEFAULT_MASK)
    parser.add_argument("--n_records",     type=int, default=None)
    parser.add_argument("--out_dir",       default=None)
    parser.add_argument("--variants",      default="zero_shot,info,few_shot")
    parser.add_argument("--server_url",    default=DEFAULT_SERVER_URL)
    parser.add_argument("--compose_file",  default=DEFAULT_COMPOSE_FILE)
    parser.add_argument("--server_wait",   type=int, default=600)
    parser.add_argument("--chunk_size",    type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--no_start_server", action="store_true")
    parser.add_argument("--no_stop_server",  action="store_true")
    args = parser.parse_args()

    requested = [v.strip() for v in args.variants.split(",")]
    unknown   = [v for v in requested if v not in VARIANTS]
    if unknown:
        sys.exit(f"Unknown variants: {unknown}. Choose from: {list(VARIANTS)}")

    if args.out_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = os.path.join(_SCRIPT_DIR, "output", f"twitterbot_qwen_{ts}")
    os.makedirs(args.out_dir, exist_ok=True)

    health_url = args.server_url.replace("/generate", "/health")
    if not args.no_start_server:
        start_server(args.compose_file, health_url, wait_sec=args.server_wait)

    print(f"\nLoading data…")
    dirty_df = pd.read_csv(args.dirty, dtype=str, keep_default_na=False)
    clean_df = pd.read_csv(args.clean, dtype=str, keep_default_na=False)
    mask_df  = pd.read_csv(args.mask,  keep_default_na=False)
    n_records = min(args.n_records, len(dirty_df)) if args.n_records else len(dirty_df)

    print(f"  Rows: {len(dirty_df)}  |  Processing: {n_records}  |  Variants: {requested}")
    print(f"  Server: {args.server_url}  |  Chunk: {args.chunk_size}  |  Out: {args.out_dir}")

    t_start     = time.perf_counter()
    all_metrics = []
    try:
        for variant_name in requested:
            all_metrics.append(run_variant(
                variant_name, VARIANTS[variant_name], args.server_url,
                dirty_df.iloc[:n_records].reset_index(drop=True),
                clean_df.iloc[:n_records].reset_index(drop=True),
                mask_df.iloc[:n_records].reset_index(drop=True),
                n_records, args.chunk_size, args.out_dir,
            ))
    finally:
        if not args.no_stop_server and not args.no_start_server:
            stop_server(args.compose_file)

    print(f"\n{'='*65}\nCROSS-VARIANT COMPARISON\n{'='*65}")
    print(f"{'Variant':<14} {'Acc':>6} {'F1_int':>8} {'F1_unint':>10} {'F1_w':>7} {'Fall':>5} {'AvgT':>7}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['variant']:<14} {m['accuracy']:>6.4f} {m['f1_intentional']:>8.4f} "
              f"{m['f1_unintentional']:>10.4f} {m['f1_weighted']:>7.4f} "
              f"{m['fallback_chunks']:>5} {m['avg_llm_time_s']:>7.2f}s")

    comp_path = os.path.join(args.out_dir, "comparison_table.csv")
    pd.DataFrame(all_metrics).to_csv(comp_path, index=False)
    meta = {"run_timestamp": datetime.datetime.now().isoformat(),
            "model": "Qwen2.5-32B-Instruct (TGI port 6300)",
            "n_records": n_records, "variants_run": requested,
            "total_wall_time": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\n✓ Done in {meta['total_wall_time']:.0f}s. Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
