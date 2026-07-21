#!/usr/bin/env bash
# System / Heuristics Attribution: H1-H8 feature extraction + RF classifier
# (Scenario B). Runs the per-dataset evaluation drivers; see
# error_detection_system/src/attribution/heuristics/run_ablation.py for the
# heuristic-group ablation used in the paper.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

cd "$ERR_SYS/src/attribution/heuristics"
python test_adult_income.py
python test_twitterbot.py
python test_scenario_b.py
# python run_ablation.py    # heuristic-group ablation (H1-H3 / H4-H6 / H7-H8)
