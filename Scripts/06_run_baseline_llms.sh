#!/usr/bin/env bash
# Baselines / LLMs: zero-label LLM intent attribution (Gemini, Llama, Qwen,
# Mixtral, DeepSeek-R1, GPT). Each model sees the clean row + dirty row +
# blind mask and must call intent without any labeled examples (besides the
# optional few-shot prompt variant). Used to benchmark against the System.
# No Claude implementation exists yet -- see Baselines/LLMs/Claude/NOTE.md.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

DATASET="${1:-adult}"   # adult | tabfact | twitterbot

case "$DATASET" in
  adult)
    cd "$ADULT_LLM"
    python "$SRC/adult_income_dataset/intent-attribution/intent_attribution_pipeline-gemini.py"
    # ...-llama.py / -mixtral.py / -qwen.py / -R1-qwen.py / -gpt.py
    ;;
  tabfact)
    cd "$SRC/error_detection_system/src/attribution/llm-based/LLMs-fact"
    python run_all_gemini.py
    # run_all_llama.py / run_all_mixtral.py / run_all_qwen.py / run_all_r1_qwen.py
    ;;
  twitterbot)
    cd "$KLIM/datasets/twitter-bot/intent-attribution"
    python evaluate_all_models.py
    ;;
  *)
    echo "usage: $0 [adult|tabfact|twitterbot]" >&2; exit 1 ;;
esac
