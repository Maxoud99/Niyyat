#!/usr/bin/env bash
# Master control script: runs the full pipeline end-to-end in order.
# Each step is also runnable standalone -- see README.md in this folder.
set -euo pipefail
cd "$(dirname "$0")"

./02_run_stage1_detection.sh adult_income
./03_run_heuristics_attribution.sh
./04_run_reference_augmented_attribution.sh
./05_run_user_guided_attribution.sh
./06_run_baseline_llms.sh adult
./07_run_baseline_frauddetector.sh
./08_run_baseline_random.sh adult

echo "Done. See Results/ for curated comparison tables, or the results/"
echo "and outputs/ folders nested inside each System/ and Baselines/ method"
echo "for full per-run artifacts."
