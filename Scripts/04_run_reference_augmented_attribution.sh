#!/usr/bin/env bash
# System / Reference-Augmented Attribution:
#   - LLM engine (Gemini/Llama/Qwen/Mixtral/DeepSeek-R1) with clean
#     reference rows, across Adult / TabFact / TwitterBot
#   - Family B+ (fingerprint: 13 heuristic + 10 clean-reference statistics)
#   - Family C  (declarative: user NL description -> LLM-derived
#     integrity constraints, evaluated against each cell)
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

LLM_DIR="$ERR_SYS/src/attribution/llm-based"

echo "=== Reference-Augmented: LLM engine, Adult (per-model, mixed_sota/tfm_sota variants) ==="
( cd "$LLM_DIR" && python run_gemini_mixed_sota.py )
# python run_llama_mixed_sota.py / run_qwen_mixed_sota.py / run_mixtral_mixed_sota.py / run_r1qwen_mixed_sota.py
# python run_gemini_tfm_sota.py  (and llama/qwen/mixtral/r1qwen variants)

echo "=== Reference-Augmented: LLM engine, TabFact ==="
( cd "$LLM_DIR/LLMs-fact" && python run_all_gemini.py )
# run_all_llama.py / run_all_mixtral.py / run_all_qwen.py / run_all_r1_qwen.py

echo "=== Reference-Augmented: LLM engine, TwitterBot ==="
( cd "$LLM_DIR/twitterbot" && python run_gemini_twitterbot_mixed_sota.py )

echo "=== Reference-Augmented: Family B+ (fingerprint, clean-reference statistics) ==="
( cd "$ERR_SYS/src/attribution/fingerprint" && python pipeline.py )

echo "=== Reference-Augmented: Family C (declarative, LLM-derived constraints) ==="
( cd "$ERR_SYS/src/attribution/declarative" && python run_all_declarative.py )
