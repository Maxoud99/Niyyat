#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ALL three Gemini intent-attribution variants on TabFact in one click.
=========================================================================

Sequentially executes:
    1. bareminimum   →  outputs/bareminimum-gemini/
    2. info          →  outputs/info-gemini/
    3. few-shots     →  outputs/few-shots-gemini/

Each variant produces:
    intent_labels.csv        — per-cell 1 / -1 / 0 mask
    intent_explanations.csv  — per-cell reason + dirty/correct values
    run_stats.json           — timing + decision counts
    per_chunk_times.csv      — per-chunk execution profile

After all variants finish, a short comparison table is printed.

Examples
--------
# Full dataset, default paths
python run_all_gemini.py

# Quick smoke-test on 30 records
python run_all_gemini.py --max-records 30

# Skip a variant
python run_all_gemini.py --skip info
"""

import argparse
import json
import os
import sys
import time

# Import variants as modules
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


baremin   = _load("tf_baremin",   "intent-attribution-baremin-gemini.py")
info_var  = _load("tf_info",      "intent-attribution-info-gemini.py")
fewshot   = _load("tf_few_shots", "intent-attribution-few-shots-gemini.py")

from _gemini_base import (
    run_pipeline,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)


VARIANTS = {
    "bareminimum": {
        "module":  baremin,
        "out_dir": baremin.OUT_DIR,
        "max_output_tokens": 4096,
    },
    "info": {
        "module":  info_var,
        "out_dir": info_var.OUT_DIR,
        "max_output_tokens": 6144,
    },
    "few-shots": {
        "module":  fewshot,
        "out_dir": fewshot.OUT_DIR,
        "max_output_tokens": 8192,
    },
}


def _print_header(title: str) -> None:
    bar = "═" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirty", default=DIRTY_CSV)
    ap.add_argument("--mask",  default=MASK_CSV)
    ap.add_argument("--expl",  default=EXPLANATIONS_JSON)
    ap.add_argument("--orig",  default=ORIG_DIRTY_CSV)
    ap.add_argument("--max-records", type=int, default=None,
                    help="Limit rows (default: full dataset)")
    ap.add_argument("--skip", action="append", default=[],
                    choices=list(VARIANTS.keys()),
                    help="Variant to skip (repeatable)")
    ap.add_argument("--only", action="append", default=[],
                    choices=list(VARIANTS.keys()),
                    help="Run only the given variant(s) (repeatable)")
    args = ap.parse_args()

    to_run = [v for v in VARIANTS if v not in args.skip]
    if args.only:
        to_run = [v for v in to_run if v in args.only]

    _print_header(f"TabFact Intent Attribution — {len(to_run)} variant(s)")
    print(f"dirty:         {args.dirty}")
    print(f"mask:          {args.mask}")
    print(f"explanations:  {args.expl}")
    print(f"orig dirty:    {args.orig}")
    print(f"max records:   {args.max_records if args.max_records else 'ALL'}")
    print(f"variants:      {', '.join(to_run)}")

    results = {}
    t0 = time.perf_counter()

    for name in to_run:
        cfg = VARIANTS[name]
        _print_header(f"VARIANT: {name}")
        try:
            run_pipeline(
                make_chunk_prompt=cfg["module"].make_chunk_prompt,
                out_dir=cfg["out_dir"],
                variant_name=name,
                dirty_path=args.dirty,
                mask_path=args.mask,
                expl_path=args.expl,
                orig_path=args.orig,
                max_records=args.max_records,
                max_output_tokens=cfg["max_output_tokens"],
            )
            stats_path = os.path.join(cfg["out_dir"], "run_stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r", encoding="utf-8") as f:
                    results[name] = json.load(f)
        except Exception as ex:
            print(f"❌ variant '{name}' crashed: {ex!r}")
            results[name] = {"error": repr(ex)}

    # ── Summary
    _print_header("SUMMARY")
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
