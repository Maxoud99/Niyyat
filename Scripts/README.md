# Scripts

Orchestration layer for the whole NIYYAT project. Pipeline order:

| # | Script | What it does |
|---|--------|--------------|
| 01 | `01_generate_datasets.sh` | Regenerate the dirty-data variants (LLM/Mixed-SOTA/TFM) for Adult and TwitterBot |
| 02 | `02_run_stage1_detection.sh <dataset>` | Stage 1: ensemble record-level error detection (must run before attribution) |
| 03 | `03_run_heuristics_attribution.sh` | System / Heuristics Attribution (H1-H8 + RF) |
| 04 | `04_run_reference_augmented_attribution.sh` | System / Reference-Augmented (LLM engine, fingerprint B+, declarative C) |
| 05 | `05_run_user_guided_attribution.sh` | System / User-guided Attribution (clustering + 1% label budget + propagation) |
| 06 | `06_run_baseline_llms.sh <adult\|tabfact\|twitterbot>` | Baselines / LLMs (zero-label LLM intent attribution) |
| 07 | `07_run_baseline_frauddetector.sh` | Baselines / FraudDetector (ECOD + Leave-One-Out) |
| 08 | `08_run_baseline_random.sh <adult\|twitterbot>` | Baselines / Random (random/constant/probability guessing) |
| - | `run_all.sh` | Runs 02-08 in sequence with sane defaults |

## Important: these wrappers run code in its ORIGINAL location

`NIYYAT/Datasets`, `NIYYAT/System`, and `NIYYAT/Baselines` are **organized
copies** for browsing — the originals were left untouched, per the request
that produced this folder ("do not move, just copy"). The underlying Python
scripts have hardcoded absolute paths to their *original* input/output
locations (e.g. `.../tenth-trial/results/baselines/baseline_comparison.csv`),
scattered across many sibling project folders. Rewriting every one of those
paths to point at the new copy would be invasive and risk silently breaking
working pipelines, so these wrappers instead `cd` into the original folder
(see `_paths.sh`) and invoke the real entry point there. Output lands back
in the original `results/`/`outputs/` folders, not inside `NIYYAT/`.

If you want the copies under `NIYYAT/` to reflect a fresh run, re-copy the
relevant `results/`/`outputs/` subfolder afterwards.
