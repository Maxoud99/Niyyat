#!/usr/bin/env bash
# Runs Qwen3.6-35B-A3B (port 6500, GPU1) and Gemma-4-31B-it (port 6700, GPU2)
# in parallel, each doing zero+info+few x the 9 datasets in the NIYYAT paper's
# benchmark suite (Intent_Paper/chapters/5_Datasets.tex, Table 1). Chile
# Customs is deliberately excluded -- it was removed from the benchmark
# (see the \begin{comment} block wrapping sec:chile-customs in that file).
set -u
SCRIPT_DIR="/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based"
COMPOSE_DIR="/home/mohamed/error_injector/llms_baseline/docker-compose"
DATASETS="adult_llm twitterbot_llm adult_mixed twitterbot_mixed adult_tfm twitterbot_tfm tabfact ebay credit_card_fraud"
PROMPTS="zero info few"

cd "${COMPOSE_DIR}"
echo "==================== Bringing up Qwen3.6 (6500) + Gemma4 (6700) ===================="
docker compose -p qwen36-solo -f docker-compose-qwen3.6.yml up -d
docker compose -p gemma4-solo -f docker-compose-gemma4.yml up -d

wait_for_health () {
  local name="$1" port="$2" attempt=0
  while [ $attempt -lt 90 ]; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null || echo "000")
    if [ "$code" = "200" ]; then echo "[$name] Ready!"; return 0; fi
    echo "[$name] attempt $((attempt+1)): HTTP ${code}"
    sleep 10
    attempt=$((attempt+1))
  done
  echo "[$name] FAILED to become healthy"
  return 1
}

wait_for_health "Qwen3.6" 6500
QWEN_OK=$?
wait_for_health "Gemma4" 6700
GEMMA_OK=$?

cd "${SCRIPT_DIR}"

# run_grid_driver.py wraps the 27 (dataset x prompt) combinations per model in
# a single outer tqdm bar with ETA -- generic_cell_eval.py's own tqdm bar only
# covers chunks *within* one run and resets to 0 on every subprocess, so it
# can't show overall grid progress on its own.
if [ $QWEN_OK -eq 0 ]; then
  python3 -u run_grid_driver.py qwen36 6500 > "${SCRIPT_DIR}/qwen36_grid.log" 2>&1 &
  QWEN_PID=$!
else
  echo "Skipping Qwen3.6 grid -- container never became healthy"
  QWEN_PID=""
fi

if [ $GEMMA_OK -eq 0 ]; then
  python3 -u run_grid_driver.py gemma4 6700 > "${SCRIPT_DIR}/gemma4_grid.log" 2>&1 &
  GEMMA_PID=$!
else
  echo "Skipping Gemma4 grid -- container never became healthy"
  GEMMA_PID=""
fi

[ -n "$QWEN_PID" ] && wait "$QWEN_PID"
[ -n "$GEMMA_PID" ] && wait "$GEMMA_PID"

echo "==================== QWEN3.6 + GEMMA4 GRID ALL DONE ===================="
cd "${COMPOSE_DIR}"
docker compose -p qwen36-solo -f docker-compose-qwen3.6.yml down
docker compose -p gemma4-solo -f docker-compose-gemma4.yml down
