#!/bin/bash
# Chain: Mixtral (re-run) → stop → Llama → stop → Qwen → stop → R1 → stop
HERE="/home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/llm-based/LLMs-fact"
DC="/home/mohamed/error_injector/llms_baseline/docker-compose"

wait_ready() {
    local port=$1; local name=$2
    echo "[$name] Waiting on port $port..."
    for i in $(seq 1 30); do
        sleep 10
        status=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null || echo "000")
        echo "  [$name] attempt $i: HTTP $status"
        if [ "$status" = "200" ]; then echo "  [$name] Ready!"; return 0; fi
    done
    echo "  [$name] ERROR: never became ready after 300s"; return 1
}

run_model() {
    local compose_file=$1 port=$2 model_name=$3 run_script=$4 log_file=$5
    echo ""; echo "==================== $model_name ===================="
    cd "$DC" && docker compose -f "$compose_file" up -d
    if ! wait_ready "$port" "$model_name"; then
        echo "[$model_name] SKIPPED — container failed to start"; return 1
    fi
    cd "$HERE" && python "$run_script" > "$log_file" 2>&1
    echo "[$model_name] done (exit=$?)"
    cd "$DC" && docker compose -f "$compose_file" down
    echo "[$model_name] container stopped"
}

# Re-run Mixtral (container is already up)
echo "==================== Mixtral (re-run) ===================="
wait_ready 6000 "Mixtral" || (cd "$DC" && docker compose -f docker-compose-mixtral.yml up -d && wait_ready 6000 "Mixtral")
cd "$HERE" && python run_all_mixtral.py > run_all_mixtral.log 2>&1
echo "[Mixtral] done (exit=$?)"
cd "$DC" && docker compose -f docker-compose-mixtral.yml down
echo "[Mixtral] container stopped"

# Llama → Qwen → R1
run_model "docker-compose-llama.yml"       6100 "Llama"   "run_all_llama.py"   "$HERE/run_all_llama.log"
run_model "docker-compose-qwen.yml"        6300 "Qwen"    "run_all_qwen.py"    "$HERE/run_all_qwen.log"
run_model "docker-compose-deepseek-R1.yml" 6800 "R1-Qwen" "run_all_r1_qwen.py" "$HERE/run_all_r1_qwen.log"

echo ""; echo "==================== ALL DONE ===================="
