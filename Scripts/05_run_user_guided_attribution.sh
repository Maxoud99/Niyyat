#!/usr/bin/env bash
# System / User-guided Attribution: cluster the 13-feature fingerprint,
# sample ~1% of cells per cluster for a human/oracle to label, then
# propagate labels to the rest with Random Forest (or KNN / LabelProp /
# LabelSpread / MajorityVote). no_clustering/ is the uniform-random-sampling
# ablation of the same pipeline (isolates the effect of clustering).
set -euo pipefail
source "$(dirname "$0")/_paths.sh"

echo "=== User-guided: clustering-organized (6 clustering algos x 5 propagation methods) ==="
( cd "$ERR_SYS/src/attribution/clustering-organized" && ./run_comparison.sh )

echo "=== User-guided: no_clustering (uniform random 1% sampling, ablation) ==="
( cd "$ERR_SYS/src/attribution/no_clustering" && python run_all_no_clustering.py )
