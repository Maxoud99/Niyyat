#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell-level Intent Attribution for eBay via DeepSeek-R1-Distill-Qwen-32B (info-enhanced prompt).

Thin wrapper around the shared, dataset-agnostic engine (generic_cell_eval.py
in the parent directory) -- all loading, prompting, retry, and parsing logic
lives there. This file just pins dataset/model/prompt for direct, familiar
invocation matching the per-model-per-prompt convention used elsewhere.

NOTE ON CLEAN VALUES: eBay has no oracle clean twin (real scraped
listings, not synthetic injection). "Clean" here is a pseudo-clean
reference (per category+marketplace median/mode, see
wdc_product_analysis/intentionality/build_pseudo_clean.py) -- the same
approximation used by NIYYAT Reference-Augmented and ECOD-LOO for this
dataset, not a real original value. This is a weaker condition than
Adult/TwitterBot's LLM baselines, which see a true clean original.

Usage:
  python intent-attribution-info-r1qwen.py [--port 6800] [--max-records N]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generic_cell_eval import run  # noqa: E402

DATASET = "ebay"
MODEL = "r1qwen"
PROMPT = "info"
DEFAULT_PORT = 6800

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "generic_cell_eval_output", DATASET, MODEL, PROMPT)
    run(DATASET, MODEL, args.port, out_dir, args.max_records, PROMPT)
