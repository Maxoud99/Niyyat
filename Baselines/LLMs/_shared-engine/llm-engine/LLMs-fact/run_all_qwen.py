#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ALL three intent-attribution variants on TabFact for Qwen-2.5.
====================================================================

Sequentially executes:
    1. bareminimum  →  outputs/baremin-qwen/
    2. info         →  outputs/info-qwen/
    3. few-shots    →  outputs/few-shots-qwen/

Requires the Qwen-2.5 TGI container to be running on port 6300.

Examples
--------
    python run_all_qwen.py                       # full dataset
    python run_all_qwen.py --max-records 30      # smoke test
    python run_all_qwen.py --only info           # one variant
    python run_all_qwen.py --skip bareminimum
"""

import argparse, json, os, sys, time

from _local_llm_base import (
    run_pipeline_tgi, MODELS,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)
from _prompts import make_prompt_baremin, make_prompt_info, make_prompt_fewshot

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_KEY = "qwen"

VARIANTS = {
    "bareminimum": {
        "prompt_fn": make_prompt_baremin,
        "out_dir":   os.path.join(HERE, "outputs", "baremin-qwen"),
    },
    "info": {
        "prompt_fn": make_prompt_info,
        "out_dir":   os.path.join(HERE, "outputs", "info-qwen"),
    },
    "few-shots": {
        "prompt_fn": make_prompt_fewshot,
        "out_dir":   os.path.join(HERE, "outputs", "few-shots-qwen"),
    },
}


def _bar(title: str) -> None:
    bar = "═" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirty", default=DIRTY_CSV)
    ap.add_argument("--mask",  default=MASK_CSV)
    ap.add_argument("--expl",  default=EXPLANATIONS_JSON)
    ap.add_argument("--orig",  default=ORIG_DIRTY_CSV)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--skip", action="append", default=[],
                    choices=list(VARIANTS.keys()))
    ap.add_argument("--only", action="append", default=[],
                    choices=list(VARIANTS.keys()))
    args = ap.parse_args()

    to_run = [v for v in VARIANTS if v not in args.skip]
    if args.only:
        to_run = [v for v in to_run if v in args.only]

    cfg = MODELS[MODEL_KEY]
    _bar(f"TabFact Intent Attribution — {cfg['display_name']} — {len(to_run)} variant(s)")
    print(f"endpoint:       {cfg['server_url']}")
    print(f"dirty:          {args.dirty}")
    print(f"mask:           {args.mask}")
    print(f"explanations:   {args.expl}")
    print(f"orig dirty:     {args.orig}")
    print(f"max records:    {args.max_records if args.max_records else 'ALL'}")
    print(f"variants:       {', '.join(to_run)}")

    results = {}
    t0 = time.perf_counter()
    for name in to_run:
        v = VARIANTS[name]
        _bar(f"VARIANT: {name}  ({MODEL_KEY})")
        try:
            run_pipeline_tgi(
                make_core_prompt= v["prompt_fn"],
                model_key=        MODEL_KEY,
                out_dir=          v["out_dir"],
                variant_name=     name,
                dirty_path=       args.dirty,
                mask_path=        args.mask,
                expl_path=        args.expl,
                orig_path=        args.orig,
                max_records=      args.max_records,
            )
            stats_path = os.path.join(v["out_dir"], "run_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    results[name] = json.load(f)
        except Exception as ex:
            print(f"❌ variant '{name}' crashed: {ex!r}")
            results[name] = {"error": repr(ex)}

    _bar("SUMMARY")
    print(f"{'variant':<14} {'rows':>6} {'time(s)':>9} "
          f"{'INT (1)':>9} {'UNINT (-1)':>11} {'ok chunks':>10}")
    print("-" * 72)
    for name in to_run:
        r = results.get(name, {})
        if "error" in r:
            print(f"{name:<14} ERROR: {r['error']}")
            continue
        dc = r.get("decision_counts", {})
        print(f"{name:<14} "
              f"{r.get('records','?'):>6} "
              f"{r.get('total_time_sec',0):>9.1f} "
              f"{dc.get('intentional (1)',0):>9} "
              f"{dc.get('unintentional (-1)',0):>11} "
              f"{r.get('successful_chunks',0):>10}")
    total = time.perf_counter() - t0
    print("-" * 72)
    print(f"Total wall-time: {total:.1f}s")
    print("\nOutputs:")
    for name in to_run:
        print(f"  • {name:<14} → {VARIANTS[name]['out_dir']}")


if __name__ == "__main__":
    main()
