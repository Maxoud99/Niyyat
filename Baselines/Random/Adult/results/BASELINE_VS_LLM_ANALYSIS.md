# Baseline vs LLM Performance Analysis - Adult Income Dataset

## Summary Statistics

### Overall Performance

| Category | Count | Mean Macro F1 | Std Dev | Min | Max |
|----------|-------|---------------|---------|-----|-----|
| Baseline | 7 | 0.4465 | 0.0783 | 0.3199 | 0.4999 |
| LLM | 15 | 0.6999 | 0.1159 | 0.3767 | 0.8181 |

### Top 10 Performers (All Categories)

| Rank | Model | Trial | Type | Macro F1 |
|------|-------|-------|------|----------|
| 1 | GEMINI | Bare Minimum | LLM | 0.8181 |
| 2 | LLAMA | Info + Few-Shot | LLM | 0.8098 |
| 3 | QWEN | Bare Minimum | LLM | 0.7977 |
| 4 | QWEN | Info + Few-Shot | LLM | 0.7870 |
| 5 | QWEN | With Info | LLM | 0.7646 |
| 6 | GEMINI | Info + Few-Shot | LLM | 0.7639 |
| 7 | GEMINI | With Info | LLM | 0.7615 |
| 8 | DEEPSEEK-R1 | Info + Few-Shot | LLM | 0.7454 |
| 9 | LLAMA | With Info | LLM | 0.7201 |
| 10 | DEEPSEEK-R1 | Bare Minimum | LLM | 0.6673 |

### Top 5 LLM Models by Trial


#### Bare Minimum

| Rank | Model | Macro F1 | vs Random (0.5) |
|------|-------|----------|----------------|
| 1 | GEMINI | 0.8181 | +0.3181 |
| 2 | QWEN | 0.7977 | +0.2977 |
| 3 | DEEPSEEK-R1 | 0.6673 | +0.1673 |
| 4 | LLAMA | 0.6197 | +0.1197 |
| 5 | MIXTRAL | 0.3767 | -0.1233 |

#### With Info

| Rank | Model | Macro F1 | vs Random (0.5) |
|------|-------|----------|----------------|
| 1 | QWEN | 0.7646 | +0.2646 |
| 2 | GEMINI | 0.7615 | +0.2615 |
| 3 | LLAMA | 0.7201 | +0.2201 |
| 4 | MIXTRAL | 0.6260 | +0.1260 |
| 5 | DEEPSEEK-R1 | 0.6223 | +0.1223 |

#### Info + Few-Shot

| Rank | Model | Macro F1 | vs Random (0.5) |
|------|-------|----------|----------------|
| 1 | LLAMA | 0.8098 | +0.3098 |
| 2 | QWEN | 0.7870 | +0.2870 |
| 3 | GEMINI | 0.7639 | +0.2639 |
| 4 | DEEPSEEK-R1 | 0.7454 | +0.2454 |
| 5 | MIXTRAL | 0.6177 | +0.1177 |

## Key Insights

1. **Best LLM Performance**: `GEMINI (Bare Minimum)` achieved 0.8181 Macro F1
2. **Worst LLM Performance**: `MIXTRAL (Bare Minimum)` achieved 0.3767 Macro F1
3. **LLMs Above Random Baseline (0.5)**: 14/15 (93.3%)
4. **Mean LLM Macro F1**: 0.6999
5. **LLMs Above Best Baseline (0.4999)**: 14/15 (93.3%)

## Ground Truth Distribution

- **Intentional**: 47.04% (13,291 changes)
- **Unintentional**: 52.96% (14,965 changes)
- **Total Changes**: 28,256

## Baseline Strategy Analysis

The baseline strategies show interesting patterns:

- **Random Guessing (50/50)**: ~0.50 Macro F1 (as expected)
- **Constant strategies** perform poorly due to class imbalance
- **Probability-based strategies** are sensitive to ground truth distribution

## Recommendations

1. **Best prompting strategy**: Info + Few-Shot
2. Models should exceed 0.5 Macro F1 to be useful
3. The GEMINI model with Bare Minimum prompting shows best performance
