#!/usr/bin/env bash
set -euo pipefail

SRC=/home/mohamed/error_injector/llms_baseline
DST=/home/mohamed/error_injector/llms_baseline/NIYYAT

CP=(rsync -a --exclude=__pycache__ --exclude=*.pyc --exclude=.DS_Store)

mkdir -p "$DST"

echo "############################################"
echo "# 1. Skeleton"
echo "############################################"
mkdir -p "$DST/Datasets/Adult/LLM-based" \
         "$DST/Datasets/Adult/Mixed-sota-based" \
         "$DST/Datasets/Adult/TFM-based" \
         "$DST/Datasets/TwitterBot/LLM-based" \
         "$DST/Datasets/TwitterBot/Mixed-sota-based" \
         "$DST/Datasets/TwitterBot/TFM-based" \
         "$DST/Datasets/TabFact" \
         "$DST/Datasets/eBay" \
         "$DST/Baselines/LLMs/Gemini" \
         "$DST/Baselines/LLMs/Llama" \
         "$DST/Baselines/LLMs/Qwen" \
         "$DST/Baselines/LLMs/Deepseek-R1" \
         "$DST/Baselines/LLMs/Mixtral" \
         "$DST/Baselines/LLMs/GPT" \
         "$DST/Baselines/LLMs/Claude" \
         "$DST/Baselines/FraudDetector" \
         "$DST/Baselines/Random" \
         "$DST/System/Heuristics Attribution" \
         "$DST/System/Reference-Augmented" \
         "$DST/System/User-guided Attribution" \
         "$DST/System/Shared-Core" \
         "$DST/Results/Adult" \
         "$DST/Results/TwitterBot" \
         "$DST/Results/TabFact" \
         "$DST/Scripts"

echo "############################################"
echo "# 2. Datasets"
echo "############################################"

### Adult / LLM-based  (most recent = run_v2_20260617_173016, last week)
A_LLM="$SRC/adult_income_dataset/tenth-trial"
mkdir -p "$DST/Datasets/Adult/LLM-based/clean" "$DST/Datasets/Adult/LLM-based/most_recent_run_v2_20260617" "$DST/Datasets/Adult/LLM-based/reference_run_20251031"
"${CP[@]}" "$A_LLM/data/raw/correct_records.csv" "$DST/Datasets/Adult/LLM-based/clean/"
"${CP[@]}" "$A_LLM/data/raw/run_v2_20260617_173016/" "$DST/Datasets/Adult/LLM-based/most_recent_run_v2_20260617/"
"${CP[@]}" "$A_LLM/data/raw/run_20251031_211812/" "$DST/Datasets/Adult/LLM-based/reference_run_20251031/"
"${CP[@]}" "$A_LLM/README.md" "$DST/Datasets/Adult/LLM-based/SOURCE_README.md" 2>/dev/null || true

### Adult / Mixed-SOTA-based
"${CP[@]}" "$SRC/mixed_error_pipeline/output/" "$DST/Datasets/Adult/Mixed-sota-based/"

### Adult / TFM-based
mkdir -p "$DST/Datasets/Adult/TFM-based"
"${CP[@]}" "$SRC/tfm_error_injection/output/adult/tabpfn/" "$DST/Datasets/Adult/TFM-based/tabpfn/"

### TwitterBot / LLM-based (most recent + complete = gemini-run_v2_20260617_112647)
TB_LLM="$SRC/klim-kireev/datasets/twitter-bot"
"${CP[@]}" "$TB_LLM/gemini-run_v2_20260617_112647/" "$DST/Datasets/TwitterBot/LLM-based/most_recent_run_v2_20260617/"

### TwitterBot / Mixed-SOTA-based
"${CP[@]}" "$SRC/mixed_error_pipeline_twitter/output/" "$DST/Datasets/TwitterBot/Mixed-sota-based/"

### TwitterBot / TFM-based
"${CP[@]}" "$SRC/tfm_error_injection/output/twitterbot/tabpfn/" "$DST/Datasets/TwitterBot/TFM-based/tabpfn/"

### TabFact
"${CP[@]}" "$SRC/tabfact/datasets/final/" "$DST/Datasets/TabFact/final/"
"${CP[@]}" "$SRC/tabfact/datasets/original_splits/" "$DST/Datasets/TabFact/original_splits/"

### eBay (just the ~10k row listings CSVs, per user decision)
"${CP[@]}" "$SRC/wdc_product_analysis/data/ebay_all_listings.csv" "$DST/Datasets/eBay/"
"${CP[@]}" "$SRC/wdc_product_analysis/data/ebay_all_listings_labelled.csv" "$DST/Datasets/eBay/"

echo "############################################"
echo "# 3. System"
echo "############################################"

### Heuristics Attribution (H1-H9)
"${CP[@]}" "$SRC/error_detection_system/src/attribution/heuristics/" "$DST/System/Heuristics Attribution/"

### Reference-Augmented (LLM-based engine across all 3 datasets + declarative + fingerprint + stage2 production attributor)
mkdir -p "$DST/System/Reference-Augmented/llm-engine" "$DST/System/Reference-Augmented/declarative" "$DST/System/Reference-Augmented/fingerprint" "$DST/System/Reference-Augmented/stage2-production"
"${CP[@]}" "$SRC/error_detection_system/src/attribution/llm-based/" "$DST/System/Reference-Augmented/llm-engine/"
"${CP[@]}" "$SRC/error_detection_system/src/attribution/declarative/" "$DST/System/Reference-Augmented/declarative/"
"${CP[@]}" "$SRC/error_detection_system/src/attribution/fingerprint/" "$DST/System/Reference-Augmented/fingerprint/"
"${CP[@]}" "$SRC/error_detection_system/src/detectors/stage2/" "$DST/System/Reference-Augmented/stage2-production/"

### User-guided Attribution (clustering + sampling + label propagation)
"${CP[@]}" "$SRC/error_detection_system/src/attribution/no_clustering/" "$DST/System/User-guided Attribution/no_clustering/"
"${CP[@]}" "$SRC/error_detection_system/src/attribution/clustering-organized/" "$DST/System/User-guided Attribution/clustering-organized/"

### Shared core (Stage 1 detection + preprocessing + evaluation + imputation + config) -- needed for the system to actually run
"${CP[@]}" "$SRC/error_detection_system/src/detectors/stage1/" "$DST/System/Shared-Core/stage1-detection/"
"${CP[@]}" "$SRC/error_detection_system/src/preprocessing/" "$DST/System/Shared-Core/preprocessing/"
"${CP[@]}" "$SRC/error_detection_system/src/evaluation/" "$DST/System/Shared-Core/evaluation/"
"${CP[@]}" "$SRC/error_detection_system/src/imputation/" "$DST/System/Shared-Core/imputation/"
"${CP[@]}" "$SRC/error_detection_system/src/run_stage1.py" "$DST/System/Shared-Core/"
"${CP[@]}" "$SRC/error_detection_system/config.py" "$DST/System/Shared-Core/" 2>/dev/null || true
"${CP[@]}" "$SRC/error_detection_system/requirements.txt" "$DST/System/Shared-Core/" 2>/dev/null || true
"${CP[@]}" "$SRC/error_detection_system/RUN_STAGE2_LLM.sh" "$DST/System/Shared-Core/" 2>/dev/null || true

echo "############################################"
echo "# 4. Baselines"
echo "############################################"

# ---- 4a. LLMs, per model, per dataset ----
AII="$SRC/adult_income_dataset/intent-attribution"
AIIRES="$SRC/adult_income_dataset/tenth-trial/results/model_outputs/local-llms"
TBII="$SRC/klim-kireev/datasets/twitter-bot/intent-attribution"
LFACT="$SRC/error_detection_system/src/attribution/llm-based/LLMs-fact"
TFII="$SRC/tabfact/intent-attribution"

copy_model() {
  local model="$1"; shift
  local dest="$DST/Baselines/LLMs/$model"
  mkdir -p "$dest/Adult/code" "$dest/Adult/results" "$dest/TwitterBot/code" "$dest/TwitterBot/results" "$dest/TabFact/code" "$dest/TabFact/results"
}

# Gemini
copy_model "Gemini"
for f in intent_attribution_pipeline-gemini.py intent_attribution_pipeline-batched-gemini.py intent_attribution_pipeline-cell-level-gemini.py intent_attribution_pipeline-few-shots-gemini.py intent_attribution_pipeline-few-shots-batched-gemini.py intent_attribution_pipeline-info-gemini.py intent_attribution_pipeline-info-batched-gemini.py; do
  "${CP[@]}" "$AII/$f" "$DST/Baselines/LLMs/Gemini/Adult/code/" 2>/dev/null || true
done
for d in first-trial-gemini-pro first-trial-gemini-pro_v2 second-trial-gemini-pro third-trial-gemini-pro gemini-flash-lite cell-level-gemini-pro; do
  "${CP[@]}" "$AIIRES/$d/" "$DST/Baselines/LLMs/Gemini/Adult/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-gemini.py intent-attribution-few-shots-gemini.py intent-attribution-info-gemini.py; do
  "${CP[@]}" "$TBII/$f" "$DST/Baselines/LLMs/Gemini/TwitterBot/code/" 2>/dev/null || true
done
for d in bare-min-gemini few-shots-gemini info-gemini; do
  "${CP[@]}" "$TBII/outputs/$d/" "$DST/Baselines/LLMs/Gemini/TwitterBot/results/$d/" 2>/dev/null || true
done
"${CP[@]}" "$TFII/"*gemini*.py "$DST/Baselines/LLMs/Gemini/TabFact/code/" 2>/dev/null || true
"${CP[@]}" "$TFII/run_all_gemini.py" "$DST/Baselines/LLMs/Gemini/TabFact/code/" 2>/dev/null || true
for d in bareminimum-gemini few-shots-gemini info-gemini; do
  "${CP[@]}" "$LFACT/outputs/$d/" "$DST/Baselines/LLMs/Gemini/TabFact/results/$d/" 2>/dev/null || true
done
"${CP[@]}" "$LFACT/run_all_gemini.py" "$DST/Baselines/LLMs/Gemini/TabFact/code/run_all_gemini_LLMsfact.py" 2>/dev/null || true

# Llama
copy_model "Llama"
for f in intent_attribution_pipeline-llama.py intent_attribution_pipeline-few-shots-llama.py intent_attribution_pipeline-info-llama.py; do
  "${CP[@]}" "$AII/$f" "$DST/Baselines/LLMs/Llama/Adult/code/" 2>/dev/null || true
done
for d in first-trial-llama second-trial-llama third-trial-llama third-trial-deep-llama; do
  "${CP[@]}" "$AIIRES/$d/" "$DST/Baselines/LLMs/Llama/Adult/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-llama.py intent-attribution-few-shots-llama.py intent-attribution-info-llama.py; do
  "${CP[@]}" "$TBII/$f" "$DST/Baselines/LLMs/Llama/TwitterBot/code/" 2>/dev/null || true
done
for d in bare-min-llama few-shots-llama info-llama; do
  "${CP[@]}" "$TBII/outputs/$d/" "$DST/Baselines/LLMs/Llama/TwitterBot/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-llama.py intent-attribution-few-shots-llama.py intent-attribution-info-llama.py run_all_llama.py run_all_llama.log; do
  "${CP[@]}" "$LFACT/$f" "$DST/Baselines/LLMs/Llama/TabFact/code/" 2>/dev/null || true
done
for d in baremin-llama few-shots-llama info-llama; do
  "${CP[@]}" "$LFACT/outputs/$d/" "$DST/Baselines/LLMs/Llama/TabFact/results/$d/" 2>/dev/null || true
done

# Mixtral
copy_model "Mixtral"
for f in intent_attribution_pipeline-mixtral.py intent_attribution_pipeline-few-shots-mixtral.py intent_attribution_pipeline-info-mixtral.py; do
  "${CP[@]}" "$AII/$f" "$DST/Baselines/LLMs/Mixtral/Adult/code/" 2>/dev/null || true
done
for d in first-trial-mixtral second-trial-mixtral third-trial-mixtral; do
  "${CP[@]}" "$AIIRES/$d/" "$DST/Baselines/LLMs/Mixtral/Adult/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-mixtral.py intent-attribution-few-shots-mixtral.py intent-attribution-info-mixtral.py; do
  "${CP[@]}" "$TBII/$f" "$DST/Baselines/LLMs/Mixtral/TwitterBot/code/" 2>/dev/null || true
done
for d in bare-min-mixtral few-shots-mixtral info-mixtral; do
  "${CP[@]}" "$TBII/outputs/$d/" "$DST/Baselines/LLMs/Mixtral/TwitterBot/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-mixtral.py intent-attribution-few-shots-mixtral.py intent-attribution-info-mixtral.py run_all_mixtral.py run_all_mixtral.log; do
  "${CP[@]}" "$LFACT/$f" "$DST/Baselines/LLMs/Mixtral/TabFact/code/" 2>/dev/null || true
done
for d in baremin-mixtral few-shots-mixtral info-mixtral; do
  "${CP[@]}" "$LFACT/outputs/$d/" "$DST/Baselines/LLMs/Mixtral/TabFact/results/$d/" 2>/dev/null || true
done

# Qwen  (exclude R1-qwen variants explicitly)
copy_model "Qwen"
for f in intent_attribution_pipeline-qwen.py intent_attribution_pipeline-few-shots-qwen.py intent_attribution_pipeline-info-qwen.py; do
  "${CP[@]}" "$AII/$f" "$DST/Baselines/LLMs/Qwen/Adult/code/" 2>/dev/null || true
done
for d in first-trial-qwen second-trial-qwen third-trial-qwen; do
  "${CP[@]}" "$AIIRES/$d/" "$DST/Baselines/LLMs/Qwen/Adult/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-qwen.py intent-attribution-few-shots-qwen.py intent-attribution-info-qwen.py; do
  "${CP[@]}" "$TBII/$f" "$DST/Baselines/LLMs/Qwen/TwitterBot/code/" 2>/dev/null || true
done
for d in bare-min-qwen few-shots-qwen info-qwen; do
  "${CP[@]}" "$TBII/outputs/$d/" "$DST/Baselines/LLMs/Qwen/TwitterBot/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-qwen.py intent-attribution-few-shots-qwen.py intent-attribution-info-qwen.py run_all_qwen.py run_all_qwen.log; do
  "${CP[@]}" "$LFACT/$f" "$DST/Baselines/LLMs/Qwen/TabFact/code/" 2>/dev/null || true
done
for d in baremin-qwen few-shots-qwen info-qwen; do
  "${CP[@]}" "$LFACT/outputs/$d/" "$DST/Baselines/LLMs/Qwen/TabFact/results/$d/" 2>/dev/null || true
done

# Deepseek-R1 (R1-qwen / R1 variants -> DeepSeek-R1-Distill-Qwen)
copy_model "Deepseek-R1"
for f in intent_attribution_pipeline-R1-qwen.py intent_attribution_pipeline-few-shots-R1-qwen.py intent_attribution_pipeline-info-R1-qwen.py; do
  "${CP[@]}" "$AII/$f" "$DST/Baselines/LLMs/Deepseek-R1/Adult/code/" 2>/dev/null || true
done
for d in first-trial-R1-deepseek second-trial-R1-deepseek third-trial-R1-deepseek; do
  "${CP[@]}" "$AIIRES/$d/" "$DST/Baselines/LLMs/Deepseek-R1/Adult/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-R1.py intent-attribution-few-shots-R1.py intent-attribution-info-R1.py; do
  "${CP[@]}" "$TBII/$f" "$DST/Baselines/LLMs/Deepseek-R1/TwitterBot/code/" 2>/dev/null || true
done
for d in bare-min-R1 few-shots-R1 info-R1; do
  "${CP[@]}" "$TBII/outputs/$d/" "$DST/Baselines/LLMs/Deepseek-R1/TwitterBot/results/$d/" 2>/dev/null || true
done
for f in intent-attribution-baremin-r1-qwen.py intent-attribution-few-shots-r1-qwen.py intent-attribution-info-r1-qwen.py run_all_r1_qwen.py run_all_r1_qwen.log; do
  "${CP[@]}" "$LFACT/$f" "$DST/Baselines/LLMs/Deepseek-R1/TabFact/code/" 2>/dev/null || true
done
for d in baremin-r1-qwen few-shots-r1-qwen info-r1-qwen; do
  "${CP[@]}" "$LFACT/outputs/$d/" "$DST/Baselines/LLMs/Deepseek-R1/TabFact/results/$d/" 2>/dev/null || true
done

# GPT (Adult only -- no TabFact/TwitterBot implementation found)
copy_model "GPT"
"${CP[@]}" "$AII/intent_attribution_pipeline-gpt.py" "$DST/Baselines/LLMs/GPT/Adult/code/" 2>/dev/null || true
rmdir "$DST/Baselines/LLMs/GPT/TwitterBot/code" "$DST/Baselines/LLMs/GPT/TwitterBot/results" "$DST/Baselines/LLMs/GPT/TwitterBot" "$DST/Baselines/LLMs/GPT/TabFact/code" "$DST/Baselines/LLMs/GPT/TabFact/results" "$DST/Baselines/LLMs/GPT/TabFact" 2>/dev/null || true

# Claude -- placeholder, no implementation exists anywhere in the source codebase yet
rm -rf "$DST/Baselines/LLMs/Claude"
mkdir -p "$DST/Baselines/LLMs/Claude"
cat > "$DST/Baselines/LLMs/Claude/NOTE.md" << 'EOF'
# Claude — not yet implemented

No Claude-based intent-attribution pipeline exists anywhere in the source
codebase as of this reorganization (2026-06-25). This folder is a
placeholder so the LLM baseline lineup matches the intended model roster
(Gemini, Llama, Qwen, Deepseek-R1, Mixtral, GPT, Claude).

To add it, mirror the structure used by the other models, e.g. by adapting:
- Baselines/LLMs/GPT/Adult/code/intent_attribution_pipeline-gpt.py (Adult, row/cell-level)
- Baselines/LLMs/Gemini/TwitterBot/code/*.py (bare-min / few-shot / info prompt variants)
- Baselines/LLMs/Gemini/TabFact/code/*.py
EOF

# ---- 4b. FraudDetector (ECOD + Leave-One-Out) ----
"${CP[@]}" "$SRC/fraud_baseline/" "$DST/Baselines/FraudDetector/"

# ---- 4c. Random (guessing strategies: random / constant / probability) ----
mkdir -p "$DST/Baselines/Random/Adult" "$DST/Baselines/Random/TwitterBot"
"${CP[@]}" "$A_LLM/baseline_guessing_strategies.py" "$DST/Baselines/Random/Adult/" 2>/dev/null || true
"${CP[@]}" "$A_LLM/compare_baselines_vs_llms.py" "$DST/Baselines/Random/Adult/" 2>/dev/null || true
"${CP[@]}" "$A_LLM/BASELINES_COMPLETE_GUIDE.md" "$DST/Baselines/Random/Adult/" 2>/dev/null || true
"${CP[@]}" "$A_LLM/results/baselines/" "$DST/Baselines/Random/Adult/results/" 2>/dev/null || true
"${CP[@]}" "$TBII/baseline_guessing_strategies.py" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/baseline_guessing_full_range.py" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/compare_baselines_vs_llms.py" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/compare_llm_with_baselines.py" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/BASELINES_COMPLETE_GUIDE.md" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/BASELINE_STRATEGIES_SUMMARY.md" "$DST/Baselines/Random/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/outputs/baselines/" "$DST/Baselines/Random/TwitterBot/results/" 2>/dev/null || true
"${CP[@]}" "$SRC/BASELINE_EVALUATION_SUMMARY.md" "$DST/Baselines/" 2>/dev/null || true

echo "############################################"
echo "# 5. Results (curated, per-dataset comparison artifacts)"
echo "############################################"

# Adult
"${CP[@]}" "$SRC/adult_income_dataset/analysis_output/" "$DST/Results/Adult/llm_model_comparison/"
for f in FINAL_RESULTS_ALL_DATA_EVALUATION.md NORMAL_VS_SMART_SAMPLING_RESULTS.md HIERARCHICAL_CLUSTERING_FINDINGS.md; do
  "${CP[@]}" "$A_LLM/$f" "$DST/Results/Adult/" 2>/dev/null || true
done
"${CP[@]}" "$SRC/error_detection_system/src/attribution/clustering-organized/WHY_METHODS_WORK_ANALYSIS.md" "$DST/Results/Adult/" 2>/dev/null || true
"${CP[@]}" "$SRC/error_detection_system/src/attribution/declarative/results/summary_declarative.csv" "$DST/Results/Adult/" 2>/dev/null || true
"${CP[@]}" "$SRC/error_detection_system/src/attribution/declarative/results/RESULTS.md" "$DST/Results/Adult/declarative_RESULTS.md" 2>/dev/null || true
"${CP[@]}" "$SRC/error_detection_system/src/attribution/llm-based/TFM_LLM_ATTRIBUTION_RESULTS.md" "$DST/Results/Adult/" 2>/dev/null || true

# TwitterBot
"${CP[@]}" "$TBII/COMPLETE_SUMMARY.md" "$DST/Results/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/COMPLETE_EVALUATION_SUMMARY.md" "$DST/Results/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/WHY_BAREMIN_WINS_ANALYSIS.md" "$DST/Results/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/outputs/evaluation_summary.csv" "$DST/Results/TwitterBot/" 2>/dev/null || true
"${CP[@]}" "$TBII/outputs/EVALUATION_RESULTS.txt" "$DST/Results/TwitterBot/" 2>/dev/null || true

# TabFact
"${CP[@]}" "$SRC/tabfact/outputs/" "$DST/Results/TabFact/error_detection_outputs/"
"${CP[@]}" "$LFACT/FACTCHECK_LLM_ATTRIBUTION_RESULTS.md" "$DST/Results/TabFact/" 2>/dev/null || true
"${CP[@]}" "$LFACT/overall_metrics.csv" "$DST/Results/TabFact/" 2>/dev/null || true
"${CP[@]}" "$LFACT/per_column_metrics.csv" "$DST/Results/TabFact/" 2>/dev/null || true

echo "############################################"
echo "# Done copying. Sizes:"
echo "############################################"
du -sh "$DST"/* "$DST"/*/* 2>/dev/null
