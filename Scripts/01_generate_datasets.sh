#!/usr/bin/env bash
# Regenerate the dirty-data variants for Adult Income and TwitterBot.
# Run individual blocks as needed -- a full run touches LLM APIs / local
# TGI servers and can take hours, so this is not meant to be run blindly
# end-to-end without reading it first.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

echo "=== Adult / Mixed-SOTA (tab_err unintentional + Kireev greedy-search intentional) ==="
( cd "$MIXED_ADULT" && python run_pipeline.py )

echo "=== Adult / TFM-based (TabPFN-guided masking+imputation) ==="
( cd "$TFM" && python src/pipeline.py --model tabpfn --dataset adult )

echo "=== TwitterBot / Mixed-SOTA ==="
( cd "$MIXED_TWITTER" && python run_pipeline.py )

echo "=== TwitterBot / TFM-based ==="
( cd "$TFM" && python src/pipeline_twitter.py )

echo "=== Adult / LLM-based (Gemini cell-level manipulation, ~hours) ==="
( cd "$ADULT_LLM" && bash run_cell_level_pipeline.sh )

echo "=== TwitterBot / LLM-based (Gemini v2 manipulation) ==="
( cd "$KLIM/experiments" && python generate-manipulated-data-gemini-v2.py )
