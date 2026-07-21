#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TabFact Intent Attribution — Llama-3 (port 6100), info variant.
Serves via TGI at port configured in ``_local_llm_base.MODELS["llama"]``.

Output directory: <this-folder>/outputs/info-llama/
"""

import os
from _local_llm_base import (
    run_pipeline_tgi,
    DIRTY_CSV, MASK_CSV, EXPLANATIONS_JSON, ORIG_DIRTY_CSV,
)
from _prompts import make_prompt_info

MODEL_KEY    = "llama"
VARIANT_NAME = "info"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs",
                       "info-llama")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dirty", default=DIRTY_CSV)
    p.add_argument("--mask",  default=MASK_CSV)
    p.add_argument("--expl",  default=EXPLANATIONS_JSON)
    p.add_argument("--orig",  default=ORIG_DIRTY_CSV)
    p.add_argument("--out",   default=OUT_DIR)
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args()

    run_pipeline_tgi(
        make_core_prompt=  make_prompt_info,
        model_key=         MODEL_KEY,
        out_dir=           args.out,
        variant_name=      VARIANT_NAME,
        dirty_path=        args.dirty,
        mask_path=         args.mask,
        expl_path=         args.expl,
        orig_path=         args.orig,
        max_records=       args.max_records,
    )
