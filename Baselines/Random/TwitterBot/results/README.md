# Baseline Guessing Strategies for Intent Attribution

## Overview

This directory contains baseline guessing strategies to evaluate the performance of LLM models on intent attribution tasks. These baselines provide a reference point to assess whether LLM models add value over simple statistical strategies.

## 📊 Strategies Implemented

### 1. Random Guessing (50/50)
- For each changed feature, randomly predict intentional (1) or unintentional (-1) with equal probability
- **Expected Accuracy**: ~50%
- **Actual Mean Accuracy**: 50.49% (across 5 runs)

### 2. Constant Guessing
Two variants:
- **Always Intentional (1)**: Predict all changes as intentional
  - **Accuracy**: 100% (since all ground truth is intentional)
- **Always Unintentional (-1)**: Predict all changes as unintentional
  - **Accuracy**: 0%

### 3. Random Guessing with Probability
Predict based on probability distribution:
- **P=0.6**: 60% chance of intentional → **Accuracy**: ~59%
- **P=0.7**: 70% chance of intentional → **Accuracy**: ~70% (mean: 69.55%)
- **P=0.8**: 80% chance of intentional → **Accuracy**: ~80% (mean: 79.47%)
- **P=0.9**: 90% chance of intentional → **Accuracy**: ~90% (mean: 89.76%)

## 🎯 Key Findings

### Baseline Performance Summary

| Strategy | Mean Accuracy | Std Dev |
|----------|---------------|---------|
| Constant Always Intentional | 100.00% | 0.00% |
| Probability 0.9 | 89.76% | 0.39% |
| Probability 0.8 | 79.47% | 0.57% |
| Probability 0.7 | 69.55% | 0.78% |
| Probability 0.6 | 59.04% | N/A |
| Random Guessing | 50.49% | N/A |
| Constant Always Unintentional | 0.00% | 0.00% |

### LLM vs Baseline Comparison

**Overall Statistics:**
- **Baseline Mean**: 64.21% ± 33.07%
- **LLM Mean**: 33.92% ± 20.19%

**LLM Performance Categories:**
- **Above 70% (Strong)**: 1/15 models (6.7%)
  - bare-min-qwen: 80.06%
- **50-70% (Moderate)**: 3/15 models (20.0%)
  - bare-min-llama, bare-min-gemini, bare-min-R1
- **Below 50% (Weak)**: 11/15 models (73.3%)

## 📈 Interpretation

### What This Means

1. **Random Guessing Baseline**: Any model below 50% accuracy is worse than random chance
2. **Probability Baselines**: Simple probability-based strategies can achieve 60-90% accuracy without any intelligence
3. **LLM Value**: Only 4 out of 15 LLM models outperform random guessing
4. **Top Performer**: `bare-min-qwen` (80.06%) is the only LLM that beats the 70% probability baseline

### Model Recommendations

- ✅ **Recommended**: Models with >70% accuracy demonstrate meaningful value
- ⚠️ **Questionable**: Models between 50-70% offer limited value over probabilistic guessing
- ❌ **Not Recommended**: Models below 50% perform worse than random chance

## 📁 Files Generated

### Summary Files
- `BASELINE_COMPARISON_SUMMARY.md` - Comparison of all baseline strategies
- `BASELINE_VS_LLM_ANALYSIS.md` - Detailed analysis comparing baselines and LLMs
- `baseline_comparison.csv` - Baseline results in CSV format

### Visualizations
- `baseline_vs_llm_comparison.png` - Bar chart comparing all models
- `baseline_vs_llm_distribution.png` - Box plot showing distributions

### Per-Strategy Results
Each baseline strategy has three files:
- `{strategy}_SUMMARY.md` - Human-readable summary
- `{strategy}_results.json` - Full metrics in JSON format
- `{strategy}_per_column.csv` - Per-feature accuracy breakdown

## 🔧 Running the Scripts

### Generate Baselines
```bash
python baseline_guessing_strategies.py
```

This will:
- Run random guessing 5 times (different seeds)
- Test constant strategies (always 1, always -1)
- Test probability strategies (P=0.6, 0.7, 0.8, 0.9) with 5 runs each
- Generate detailed reports and statistics

### Compare with LLMs
```bash
python compare_baselines_vs_llms.py
```

This will:
- Load baseline results
- Load LLM evaluation results
- Create comparison visualizations
- Generate statistical analysis

## 📊 Statistical Validation

All stochastic strategies (random and probability-based) were run 5 times with different random seeds to ensure stability:

- **Random Guessing**: 50.49% ± 0.78%
- **Probability 0.7**: 69.55% ± 0.78%
- **Probability 0.8**: 79.47% ± 0.57%
- **Probability 0.9**: 89.76% ± 0.39%

The low standard deviations confirm that results are stable and reproducible.

## 🎓 Practical Implications

### For Model Selection

1. **Minimum Threshold**: Models should exceed 50% to be considered useful
2. **Competitive Threshold**: Models should exceed 70% to outperform simple probability-based strategies
3. **Strong Performance**: Models above 80% demonstrate genuine understanding
4. **Perfect Performance**: 100% is theoretically possible (constant always intentional) but unrealistic for generalization

### For Future Work

Consider:
- Investigating why most LLM models underperform
- Analyzing prompt engineering impact (bare-min vs few-shots vs info)
- Understanding why `bare-min-qwen` significantly outperforms others
- Examining whether simpler prompts lead to better performance

## 📝 Notes

- All ground truth changes are **intentional** (bot-to-human evasion)
- Mask value `1` = changed (intentional), `0` = unchanged
- Prediction value `1` = intentional, `-1` = unintentional, `0` = unchanged
- Accuracy = correct predictions / total changed features
- Dataset: 4001 records × 18 features = 1931 total changes

---

**Generated**: December 8, 2025  
**Purpose**: Establish baseline performance for intent attribution evaluation
