#!/usr/bin/env bash
# Baselines / FraudDetector: ECOD (Empirical Cumulative Distribution
# Outlier) fit on the clean distribution + per-cell Leave-One-Out score
# delta. Evaluated on all 7 dataset variants by default.
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

( cd "$FRAUD" && python run_all.py "$@" )
# e.g. ./07_run_baseline_frauddetector.sh --datasets adult_llm adult_mixed adult_tfm
