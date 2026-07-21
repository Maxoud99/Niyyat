#!/usr/bin/env bash
# =============================================================================
# run_queue_tfm.sh — Sequential TFM intent-attribution queue
#
# Runs all 4 local LLMs (Mixtral → Llama → Qwen → DeepSeek-R1) on the
# TFM-Inject Adult Income dataset, starting/stopping Docker containers
# between each model.
#
# Usage:
#   cd /home/mohamed/error_injector/llms_baseline/docker-compose
#   nohup bash /home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based/run_queue_tfm.sh \
#     > /home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based/run_queue_tfm.log 2>&1 &
#
# Options (set as env vars before calling):
#   N_RECORDS=100   # process only first 100 error records per model (quick test)
#   VARIANTS=zero_shot,info   # subset of variants
# =============================================================================

set -u

SCRIPT_DIR="/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based"
COMPOSE_DIR="/home/mohamed/error_injector/llms_baseline/docker-compose"
LOG_PREFIX="[TFM-Queue]"

N_RECORDS_ARG=""
if [ -n "${N_RECORDS:-}" ]; then
  N_RECORDS_ARG="--n_records ${N_RECORDS}"
fi

VARIANTS_ARG=""
if [ -n "${VARIANTS:-}" ]; then
  VARIANTS_ARG="--variants ${VARIANTS}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Helper: wait for TGI health endpoint
# ─────────────────────────────────────────────────────────────────────────────
wait_ready() {
  local port="$1"
  local label="$2"
  local max_attempts=60
  local attempt=0
  echo "${LOG_PREFIX} [${label}] Waiting for TGI on port ${port}..."
  while [ $attempt -lt $max_attempts ]; do
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${port}/health" 2>/dev/null || echo "000")
    if [ "$http_code" = "200" ]; then
      echo "${LOG_PREFIX} [${label}] Ready!"
      return 0
    fi
    echo "${LOG_PREFIX} [${label}] attempt $((attempt+1)): HTTP ${http_code}"
    sleep 10
    attempt=$((attempt+1))
  done
  echo "${LOG_PREFIX} [${label}] ERROR: TGI did not become ready after $((max_attempts*10))s. Aborting."
  return 1
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Mixtral (port 6000)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==================== [1/4] MIXTRAL — TFM ===================="
echo "${LOG_PREFIX} Starting Mixtral container..."
cd "${COMPOSE_DIR}"
docker compose -f docker-compose-mixtral.yml up -d

wait_ready 6000 "Mixtral"
if [ $? -ne 0 ]; then
  echo "${LOG_PREFIX} [Mixtral] Skipping due to server error."
else
  echo "${LOG_PREFIX} [Mixtral] Running python script..."
  python3 "${SCRIPT_DIR}/run_mixtral_tfm_sota.py" \
    --no_start_server --no_stop_server \
    ${N_RECORDS_ARG} ${VARIANTS_ARG}
  echo "${LOG_PREFIX} [Mixtral] Python done (exit $?)."
fi

echo "${LOG_PREFIX} [Mixtral] Stopping container..."
docker compose -f docker-compose-mixtral.yml down
echo "${LOG_PREFIX} [Mixtral] container stopped"
sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Llama (port 6100)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==================== [2/4] LLAMA — TFM ===================="
echo "${LOG_PREFIX} Starting Llama container..."
cd "${COMPOSE_DIR}"
docker compose -f docker-compose-llama.yml up -d

wait_ready 6100 "Llama"
if [ $? -ne 0 ]; then
  echo "${LOG_PREFIX} [Llama] Skipping due to server error."
else
  echo "${LOG_PREFIX} [Llama] Running python script..."
  python3 "${SCRIPT_DIR}/run_llama_tfm_sota.py" \
    --no_start_server --no_stop_server \
    ${N_RECORDS_ARG} ${VARIANTS_ARG}
  echo "${LOG_PREFIX} [Llama] Python done (exit $?)."
fi

echo "${LOG_PREFIX} [Llama] Stopping container..."
docker compose -f docker-compose-llama.yml down
echo "${LOG_PREFIX} [Llama] container stopped"
sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Qwen (port 6300)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==================== [3/4] QWEN — TFM ===================="
echo "${LOG_PREFIX} Starting Qwen container..."
cd "${COMPOSE_DIR}"
docker compose -f docker-compose-qwen.yml up -d

wait_ready 6300 "Qwen"
if [ $? -ne 0 ]; then
  echo "${LOG_PREFIX} [Qwen] Skipping due to server error."
else
  echo "${LOG_PREFIX} [Qwen] Running python script..."
  python3 "${SCRIPT_DIR}/run_qwen_tfm_sota.py" \
    --no_start_server --no_stop_server \
    ${N_RECORDS_ARG} ${VARIANTS_ARG}
  echo "${LOG_PREFIX} [Qwen] Python done (exit $?)."
fi

echo "${LOG_PREFIX} [Qwen] Stopping container..."
docker compose -f docker-compose-qwen.yml down
echo "${LOG_PREFIX} [Qwen] container stopped"
sleep 5

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: DeepSeek-R1 (port 6800)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==================== [4/4] DEEPSEEK-R1 — TFM ===================="
echo "${LOG_PREFIX} Starting DeepSeek-R1 container..."
cd "${COMPOSE_DIR}"
docker compose -f docker-compose-deepseek-R1.yml up -d

wait_ready 6800 "R1-Qwen"
if [ $? -ne 0 ]; then
  echo "${LOG_PREFIX} [R1-Qwen] Skipping due to server error."
else
  echo "${LOG_PREFIX} [R1-Qwen] Running python script..."
  python3 "${SCRIPT_DIR}/run_r1qwen_tfm_sota.py" \
    --no_start_server --no_stop_server \
    ${N_RECORDS_ARG} ${VARIANTS_ARG}
  echo "${LOG_PREFIX} [R1-Qwen] Python done (exit $?)."
fi

echo "${LOG_PREFIX} [R1-Qwen] Stopping container..."
docker compose -f docker-compose-deepseek-R1.yml down
echo "${LOG_PREFIX} [R1-Qwen] container stopped"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "==================== ALL DONE (TFM) ===================="
