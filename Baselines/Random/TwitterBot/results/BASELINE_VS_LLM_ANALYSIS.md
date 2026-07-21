# Baseline vs LLM Performance Analysis

## Summary Statistics

### Overall Performance

| Category | Count | Mean Accuracy | Std Dev | Min | Max |
|----------|-------|---------------|---------|-----|-----|
| Baseline | 7 | 0.6421 (64.21%) | 0.3307 | 0.0000 | 1.0000 |
| LLM | 15 | 0.3392 (33.92%) | 0.2019 | 0.0492 | 0.8006 |

### Top 5 Performers (All Categories)

| Rank | Model | Type | Accuracy |
|------|-------|------|----------|
| 1 | Constant_Guessing_Always_Intentional | Baseline | 1.0000 (100.00%) |
| 2 | Probability_0.9_Run1 | Baseline | 0.9001 (90.01%) |
| 3 | bare-min-qwen | LLM | 0.8006 (80.06%) |
| 4 | Probability_0.8_Run1 | Baseline | 0.7970 (79.70%) |
| 5 | Probability_0.7_Run1 | Baseline | 0.7022 (70.22%) |

### Top 5 LLM Models

| Rank | Model | Accuracy | vs Random | vs Best Baseline |
|------|-------|----------|-----------|------------------|
| 1 | bare-min-qwen | 0.8006 (80.06%) | +30.06% | -19.94% |
| 2 | bare-min-llama | 0.5640 (56.40%) | +6.40% | -43.60% |
| 3 | bare-min-gemini | 0.5634 (56.34%) | +6.34% | -43.66% |
| 4 | bare-min-R1 | 0.5329 (53.29%) | +3.29% | -46.71% |
| 5 | info-qwen | 0.3703 (37.03%) | +-12.97% | -62.97% |

### Bottom 5 Performers (All Categories)

| Rank | Model | Type | Accuracy |
|------|-------|------|----------|
| 1 | Constant_Guessing_Always_Unintentional | Baseline | 0.0000 (0.00%) |
| 2 | few-shots-mixtral | LLM | 0.0492 (4.92%) |
| 3 | few-shots-R1 | LLM | 0.0642 (6.42%) |
| 4 | few-shots-llama | LLM | 0.2009 (20.09%) |
| 5 | info-llama | LLM | 0.2191 (21.91%) |

## Key Insights

1. **Best LLM Performance**: `bare-min-qwen` achieved 80.06% accuracy
2. **Worst LLM Performance**: `few-shots-mixtral` achieved 4.92% accuracy
3. **LLMs Above Random Baseline**: 4/15 (26.7%)
4. **LLMs Above 70% Accuracy**: 1/15 (6.7%)
5. **Mean LLM Accuracy**: 33.92%

## Baseline Strategy Analysis

The baseline strategies validate expected behaviors:

- **Constant Always Intentional** achieves 100% (since all ground truth is intentional)
- **Random Guessing** achieves ~50% (as expected for 50/50 chance)
- **Probability-based** strategies achieve accuracy proportional to their bias

## Recommendations

1. Models performing below 50% are worse than random guessing
2. Models between 50-70% show limited value over probabilistic baselines
3. Models above 70% demonstrate meaningful intent attribution capability
4. Models above 80% show strong performance but still have room for improvement
