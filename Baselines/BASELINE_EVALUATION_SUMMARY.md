# Baseline Guessing Strategies - Complete Summary

## 📋 Overview

Implemented and evaluated baseline guessing strategies for **two datasets**:
1. **Twitter Bot Detection** - Intent attribution for bot-to-human evasion
2. **Adult Income** - Intent attribution for socioeconomic data manipulation

## 🎯 Purpose

Establish minimum performance thresholds by testing:
- **Random Guessing (50/50)**: Equal chance of intentional/unintentional
- **Constant Guessing**: Always predict the same value
- **Probability Guessing**: Biased random predictions (40%, 50%, 60%, 70%, 80%, 90%)

## 📊 Results Comparison

### Twitter Bot Dataset

**Ground Truth**: ALL changes are intentional (100%)

| Strategy | Accuracy | Notes |
|----------|----------|-------|
| Constant Always Intentional | 100.00% | Theoretical maximum |
| Probability 0.9 | 89.76% | 90% bias toward intentional |
| Probability 0.8 | 79.47% | 80% bias toward intentional |
| **bare-min-qwen (LLM)** | **80.06%** | **BEST LLM** ⭐ |
| Probability 0.7 | 69.55% | 70% bias toward intentional |
| Probability 0.6 | 59.04% | 60% bias toward intentional |
| Random 50/50 | 50.49% | Pure random |

**LLM Performance**:
- Above random (50%): **4/15 models (26.7%)** ❌
- Mean accuracy: **33.92%** (below random!)
- Best LLM: **bare-min-qwen (80.06%)**
- Worst LLM: **few-shots-mixtral (4.92%)**

**Key Finding**: ⚠️ **Most LLMs fail** - only 1 model beats 70% baseline

---

### Adult Income Dataset

**Ground Truth**: Balanced (47% intentional, 53% unintentional)

| Strategy | Macro F1 | Notes |
|----------|----------|-------|
| **GEMINI Bare-Min (LLM)** | **0.8181** | **BEST OVERALL** ⭐ |
| **LLAMA Few-Shot (LLM)** | **0.8098** | **2nd BEST** |
| **QWEN Bare-Min (LLM)** | **0.7977** | **3rd BEST** |
| Random 50/50 | 0.5013 | As expected |
| Probability 0.5 | 0.4980 | Similar to random |
| Probability 0.4 | 0.4964 | Biased toward unintentional |
| Probability 0.6 | 0.4899 | Biased toward intentional |
| Probability 0.7 | 0.4724 | Too much bias |
| Constant Always Unintentional | 0.3462 | Poor due to imbalance |
| Constant Always Intentional | 0.3199 | Poor due to imbalance |

**LLM Performance**:
- Above random (0.5): **14/15 models (93.3%)** ✅
- Mean Macro F1: **0.6999**
- Best LLM: **GEMINI Bare-Min (0.8181)**
- Worst LLM: **MIXTRAL Bare-Min (0.3767)**

**Key Finding**: ✅ **Most LLMs succeed** - 14 models beat random baseline

---

## 🔍 Side-by-Side Comparison

| Metric | Twitter Bot | Adult Income |
|--------|-------------|--------------|
| **LLMs Above Random** | 26.7% (4/15) ❌ | 93.3% (14/15) ✅ |
| **Mean Performance** | 33.92% | 0.6999 F1 |
| **Best LLM** | bare-min-qwen (80%) | GEMINI (0.8181) |
| **Best Prompting** | Bare Minimum | Varies by model |
| **Ground Truth** | 100% intentional | 47% / 53% split |
| **Total Changes** | 1,931 | 28,256 |
| **Task Difficulty** | HARD 🔴 | MODERATE 🟡 |

## 💡 Key Insights

### 1. Dataset Characteristics Matter

**Twitter Bot** (Harder):
- Unbalanced ground truth (all intentional)
- Subtle behavioral changes
- Only 1,931 changes
- → **Most LLMs fail**

**Adult Income** (Easier):
- Balanced ground truth (47%/53%)
- Clear socioeconomic inconsistencies
- 28,256 changes
- → **Most LLMs succeed**

### 2. Prompt Engineering

**Twitter Bot**:
- ✅ Bare-min prompts work best
- ❌ Few-shots and info prompts hurt performance

**Adult Income**:
- ✅ Different models prefer different prompts
- ✅ GEMINI: Bare-min (0.8181)
- ✅ LLAMA: Few-shot (0.8098)

### 3. Baseline Validation

Both datasets show expected baseline behavior:
- ✅ Random 50/50 achieves ~50%
- ✅ Probability-based match their bias
- ✅ Constant strategies depend on ground truth distribution

### 4. Model Selection

**For Twitter Bot**:
- 🏆 **bare-min-qwen** (80.06%)
- Avoid: All few-shots models

**For Adult Income**:
- 🏆 **GEMINI Bare-Min** (0.8181)
- 🥈 **LLAMA Few-Shot** (0.8098)
- 🥉 **QWEN Bare-Min** (0.7977)

## 📁 File Locations

### Twitter Bot Dataset
```
/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/
├── baseline_guessing_strategies.py
├── compare_baselines_vs_llms.py
├── BASELINES_COMPLETE_GUIDE.md
└── outputs/baselines/
    ├── BASELINE_VS_LLM_ANALYSIS.md
    ├── baseline_comparison.csv
    └── baseline_vs_llm_comparison.png
```

### Adult Income Dataset
```
/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/
├── baseline_guessing_strategies.py
├── compare_baselines_vs_llms.py
├── BASELINES_COMPLETE_GUIDE.md
└── results/baselines/
    ├── BASELINE_VS_LLM_ANALYSIS.md
    ├── baseline_comparison.csv
    └── baseline_vs_llm_comparison.png
```

## 🎓 Practical Recommendations

### When to Use Each Model

**Twitter Bot Detection**:
- Use: **bare-min-qwen** (only reliable option)
- Most models underperform - be cautious

**Adult Income Manipulation**:
- Use: **GEMINI** or **LLAMA** with appropriate prompts
- Most models work well - more flexibility

### General Guidelines

1. **Always test against baselines** - validates model usefulness
2. **Random guessing (50%) is minimum threshold**
3. **Simple prompts often outperform complex ones**
4. **Task difficulty varies significantly by dataset**
5. **Ground truth distribution affects baseline performance**

## 📊 Statistical Validation

All stochastic strategies run 5 times with different seeds:

**Twitter Bot**:
- Random: 50.49% ± 0.78%
- P=0.7: 69.55% ± 0.78%
- P=0.8: 79.47% ± 0.57%
- P=0.9: 89.76% ± 0.39%

**Adult Income**:
- Random: 50.18% ± 0.39%
- P=0.4: 50.50% ± 0.09%
- P=0.5: 49.84% ± 0.13%
- P=0.6: 49.24% ± 0.04%
- P=0.7: 48.75% ± 0.16%

Low standard deviations confirm reproducibility.

## ✅ Final Conclusions

### Twitter Bot Dataset
- 🔴 **HARD TASK** - Most LLMs fail
- 🏆 **Winner**: bare-min-qwen (80.06%)
- ⚠️ **73% of models** worse than random
- 💡 **Lesson**: Simple baselines (80% probability) competitive with best LLM

### Adult Income Dataset
- 🟡 **MODERATE TASK** - Most LLMs succeed
- 🏆 **Winner**: GEMINI Bare-Min (0.8181 F1)
- ✅ **93% of models** beat random
- 💡 **Lesson**: Task characteristics matter - balanced data is easier

### Overall
- **Baselines are essential** for validating model performance
- **Dataset difficulty varies dramatically**
- **Prompt engineering can help or hurt**
- **No one-size-fits-all solution** - test multiple approaches

---

**Created**: December 8, 2025  
**Purpose**: Establish minimum performance thresholds for intent attribution tasks  
**Methodology**: Random, Constant, and Probability-based guessing strategies
