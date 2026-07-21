#!/usr/bin/env bash
# Runs Qwen then Llama, each doing zero+info+few x 3 datasets (twitterbot_llm,
# twitterbot_tfm, ebay), on the qwen-parallel container (port 6300, GPUs 1,2).
# Fully fresh: no reuse of any previously-computed result for these two
# models on these three datasets.
set -u
SCRIPT_DIR="/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based"
COMPOSE_DIR="/home/mohamed/error_injector/llms_baseline/docker-compose"
DATASETS="twitterbot_llm twitterbot_tfm ebay"
PROMPTS="zero info few"

cd "${SCRIPT_DIR}"

echo "==================== QWEN (port 6300) ===================="
for ds in $DATASETS; do
  for prompt in $PROMPTS; do
    echo "[Qwen] dataset=$ds prompt=$prompt"
    LLM_EVAL_CONCURRENCY=4 python3 generic_cell_eval.py --dataset "$ds" --model qwen --prompt "$prompt" --port 6300
    echo "[Qwen] dataset=$ds prompt=$prompt done (exit $?)"
  done
done
echo "QWEN DONE -- stopping qwen-parallel container"
cd "${COMPOSE_DIR}" && docker compose -p qwen-parallel -f docker-compose-qwen.yml -f docker-compose-qwen.gpu12.override.yml down
sleep 5

echo "==================== LLAMA (port 6100, GPUs 1,2) ===================="
cd "${COMPOSE_DIR}"
docker compose -p llama-parallel -f docker-compose-llama.yml -f docker-compose-llama.gpu12.override.yml up -d
attempt=0
while [ $attempt -lt 60 ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:6100/health 2>/dev/null || echo "000")
  if [ "$code" = "200" ]; then echo "[Llama] Ready!"; break; fi
  echo "[Llama] attempt $((attempt+1)): HTTP ${code}"
  sleep 10
  attempt=$((attempt+1))
done

cd "${SCRIPT_DIR}"
for ds in $DATASETS; do
  for prompt in $PROMPTS; do
    echo "[Llama] dataset=$ds prompt=$prompt"
    LLM_EVAL_CONCURRENCY=4 python3 generic_cell_eval.py --dataset "$ds" --model llama --prompt "$prompt" --port 6100
    echo "[Llama] dataset=$ds prompt=$prompt done (exit $?)"
  done
done
echo "LLAMA DONE -- stopping llama-parallel container"
cd "${COMPOSE_DIR}" && docker compose -p llama-parallel -f docker-compose-llama.yml -f docker-compose-llama.gpu12.override.yml down

echo "==================== QWEN+LLAMA GRID ALL DONE ===================="
