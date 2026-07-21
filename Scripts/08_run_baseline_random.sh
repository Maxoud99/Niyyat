#!/usr/bin/env bash
# Baselines / Random: minimum-performance-threshold strategies (random 50/50,
# constant always-intentional/unintentional, biased probability guessing at
# 40-90%). Establishes the floor that System and LLM baselines must clear.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

DATASET="${1:-adult}"   # adult | twitterbot

case "$DATASET" in
  adult)      ( cd "$ADULT_LLM" && python baseline_guessing_strategies.py ) ;;
  twitterbot) ( cd "$KLIM/datasets/twitter-bot/intent-attribution" && python baseline_guessing_strategies.py ) ;;
  *) echo "usage: $0 [adult|twitterbot]" >&2; exit 1 ;;
esac
