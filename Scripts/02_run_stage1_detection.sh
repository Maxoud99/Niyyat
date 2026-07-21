#!/usr/bin/env bash
# Stage 1: record-level error detection (ensemble of LOF/IsolationForest/
# GMM/HDBSCAN/LSTM-autoencoder/statistical ensemble). Must run before any
# attribution method, since attribution operates on the cells Stage 1 flags.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

DATASET="${1:-adult_income}"   # adult_income | adult_income_v2 | twitter_bot | tabFact

( cd "$ERR_SYS" && python src/run_stage1.py --dataset "$DATASET" )
