#!/usr/bin/env bash
# Shared path variables for all Scripts/*.sh wrappers.
# NOTE: these point at the ORIGINAL, still-live project folders -- not at the
# copies under NIYYAT/. The NIYYAT/ tree is a curated, organized COPY for
# browsing/presentation; the actual Python scripts have hardcoded absolute
# paths baked in (input CSVs, output dirs) that assume their original
# location, so we run them in place rather than re-pointing them at the copy.
# See NIYYAT/README.md for the full rationale.

SRC=/home/mohamed/error_injector/llms_baseline

ERR_SYS="$SRC/error_detection_system"
ADULT_LLM="$SRC/adult_income_dataset/tenth-trial"
MIXED_ADULT="$SRC/mixed_error_pipeline"
MIXED_TWITTER="$SRC/mixed_error_pipeline_twitter"
TFM="$SRC/tfm_error_injection"
KLIM="$SRC/klim-kireev"
TABFACT="$SRC/tabfact"
FRAUD="$SRC/fraud_baseline"
