# Baseline Guessing Strategies - Adult Income Dataset

## 🎯 Summary

Successfully implemented and evaluated baseline guessing strategies for the Adult Income dataset's intent attribution task. The results show that **93.3% of LLM models outperform random guessing**, which is much better than the Twitter dataset (only 26.7%).

## 📊 Key Findings

### Baseline Performance
| Strategy | Macro F1 | Accuracy | Notes |
|----------|----------|----------|-------|
| Random (50/50) | 0.4999 | 50.18% | As expected |
| Probability 0.5 | 0.4980 | 49.84% | Similar to random |
| Probability 0.6 | 0.4899 | 49.24% | Biased toward intentional |
| Probability 0.4 | 0.4964 | 50.50% | Biased toward unintentional |
| Probability 0.7 | 0.4724 | 48.75% | More bias → worse due to imbalance |
| Constant Always Intentional | 0.3199 | 47.04% | High recall (1.0), zero precision for -1 |
| Constant Always Unintentional | 0.3462 | 52.96% | High recall (1.0), zero precision for 1 |

### Top LLM Performers
| Rank | Model | Trial | Macro F1 | vs Random |
|------|-------|-------|----------|-----------|
| 1 | GEMINI | Bare Minimum | 0.8181 | +31.81% |
| 2 | LLAMA | Info + Few-Shot | 0.8098 | +30.98% |
| 3 | QWEN | Bare Minimum | 0.7977 | +29.77% |
| 4 | QWEN | Info + Few-Shot | 0.7870 | +28.70% |
| 5 | QWEN | With Info | 0.7646 | +26.46% |

### LLM Performance Summary
- **Above Random (0.5)**: 14/15 models (93.3%) ✅
- **Mean Macro F1**: 0.6999
- **Best Model**: GEMINI (Bare Minimum) - 0.8181
- **Worst Model**: MIXTRAL (Bare Minimum) - 0.3767

## 🔍 Ground Truth Distribution

Unlike the Twitter dataset (all intentional), the Adult Income dataset has a more balanced distribution:
- **Intentional (1)**: 47.04% (13,291 changes)
- **Unintentional (-1)**: 52.96% (14,965 changes)
- **Total Changes**: 28,256 (from 19,539 records × 15 features)

This **balanced distribution** makes:
- Random guessing achieve ~50% accuracy (fair for both classes)
- Constant strategies perform poorly (biased toward one class)
- Probability-based strategies need to match ground truth distribution

## 💡 Key Insights

### 1. Much Better Than Twitter Dataset
- **Adult Income**: 93.3% of LLMs beat random
- **Twitter**: Only 26.7% of LLMs beat random
- **Reason**: Adult income has simpler, more detectable manipulation patterns

### 2. Bare Minimum Prompts Excel
- **GEMINI (Bare Minimum)**: 0.8181 (BEST)
- **QWEN (Bare Minimum)**: 0.7977 (3rd)
- **DEEPSEEK-R1 (Bare Minimum)**: 0.6673 (6th)
- **LLAMA (Bare Minimum)**: 0.6197 (9th)

### 3. Prompt Engineering Matters
Different trials show different patterns:
- **Best for GEMINI**: Bare Minimum (0.8181)
- **Best for LLAMA**: Info + Few-Shot (0.8098)
- **Best for QWEN**: Bare Minimum (0.7977)
- **Recommendation**: Test multiple prompting strategies per model

### 4. Baseline Strategies Show Expected Behavior
✅ **Random 50/50**: 0.4999 F1 (exactly as predicted)  
✅ **Probability-based**: Performance depends on match with ground truth distribution  
❌ **Constant strategies**: Poor due to class imbalance (0 precision for one class)

## 📁 Files Generated

### Documentation
```
tenth-trial/
├── baseline_guessing_strategies.py          # Main script
├── compare_baselines_vs_llms.py            # Comparison script
└── results/baselines/
    ├── BASELINE_COMPARISON_SUMMARY.md      # Baseline-only results
    ├── BASELINE_VS_LLM_ANALYSIS.md        # Full analysis
    ├── baseline_comparison.csv             # Results CSV
    ├── baseline_vs_llm_comparison.png      # Visualization
    └── [per-strategy detailed results]     # Individual reports
```

### Per-Strategy Files (7 strategies)
Each has:
- `{strategy}_SUMMARY.md` - Human-readable summary
- `{strategy}_results.json` - Full metrics

## 🚀 Usage

### Run Baseline Evaluation
```bash
cd /home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial
python baseline_guessing_strategies.py
```

### Compare with LLMs
```bash
python compare_baselines_vs_llms.py
```

### View Results
```bash
cat results/baselines/BASELINE_VS_LLM_ANALYSIS.md
```

## 📊 Statistical Details

- **Dataset**: 19,539 records, 15 features, 28,256 total changes
- **Ground Truth**: Mixed (47% intentional, 53% unintentional)
- **Stochastic Runs**: 5 runs per random/probability strategy with different seeds
- **Evaluation Metrics**: Macro F1, Macro Precision, Macro Recall, Accuracy

### Stability of Random Strategies
| Strategy | Mean F1 | Std Dev |
|----------|---------|---------|
| Random 50/50 | 0.5013 | ±0.0039 |
| Probability 0.4 | 0.4964 | ±0.0009 |
| Probability 0.5 | 0.4980 | ±0.0013 |
| Probability 0.6 | 0.4899 | ±0.0006 |
| Probability 0.7 | 0.4724 | ±0.0018 |

Low standard deviations confirm stable and reproducible results.

## 🎓 Comparison: Adult Income vs Twitter Bot

| Metric | Adult Income | Twitter Bot |
|--------|--------------|-------------|
| **LLMs Above Random** | 93.3% (14/15) | 26.7% (4/15) |
| **Mean LLM F1/Acc** | 0.6999 | 0.3392 |
| **Best LLM** | GEMINI: 0.8181 | bare-min-qwen: 0.8006 |
| **Ground Truth** | Balanced (47%/53%) | All intentional (100%) |
| **Task Difficulty** | Easier | Harder |
| **Best Prompting** | Varies by model | Bare Minimum |

### Why Adult Income is Easier?
1. **Balanced classes**: Prevents bias toward one prediction
2. **Simpler manipulations**: Income, age, etc. are easier to detect than bot behavior
3. **More changes**: 28,256 vs 1,931 provides more signal
4. **Clearer patterns**: Socioeconomic data has more obvious inconsistencies

## ✅ Conclusions

### For Adult Income Dataset
1. ✅ **Most LLMs are effective** (93.3% beat random)
2. ✅ **GEMINI (Bare Minimum)** achieves best performance (0.8181 F1)
3. ✅ **All LLMs except 1** outperform best baseline strategy
4. ⚠️ **MIXTRAL (Bare Minimum)** underperforms random guessing

### Recommendations
1. **Use GEMINI with Bare Minimum prompting** for this dataset
2. **Alternative**: LLAMA with Info + Few-Shot (0.8098 F1)
3. **Avoid**: MIXTRAL with Bare Minimum prompting
4. **Test multiple prompting strategies** - optimal choice varies by model

---

**Created**: December 8, 2025  
**Dataset**: Adult Income (tenth-trial)  
**Location**: `/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines/`
