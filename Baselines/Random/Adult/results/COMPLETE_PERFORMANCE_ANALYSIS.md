# Complete Performance Analysis - Adult Income Dataset
## Baselines vs LLMs vs ML Classifier

**Generated:** 2025-12-08 17:29:45

---

## Summary Statistics

| Category | Count | Mean Macro F1 | Std Dev | Min | Max |
|----------|-------|---------------|---------|-----|-----|
| Baseline | 7 | 0.4465 | 0.0783 | 0.3199 | 0.4999 |
| LLM | 15 | 0.6999 | 0.1159 | 0.3767 | 0.8181 |
| **ML Classifier** | **1** | **0.9140** | **N/A** | **0.9140** | **0.9140** |

---

## Overall Performance Ranking (Top 15)

| Rank | Model/Strategy | Type | Trial | Macro F1 | vs Random | vs Best LLM |
|------|----------------|------|-------|----------|-----------|-------------|
| 🏆 1 | **Random Forest Classifier** | ML Classifier | Supervised ML | **0.9140** | +0.4140 | +0.0958 |
| 2 | GEMINI | LLM | Bare Minimum | 0.8181 | +0.3181 | +0.0000 |
| 3 | LLAMA | LLM | Info + Few-Shot | 0.8098 | +0.3098 | -0.0083 |
| 4 | QWEN | LLM | Bare Minimum | 0.7977 | +0.2977 | -0.0204 |
| 5 | QWEN | LLM | Info + Few-Shot | 0.7870 | +0.2870 | -0.0311 |
| 6 | QWEN | LLM | With Info | 0.7646 | +0.2646 | -0.0535 |
| 7 | GEMINI | LLM | Info + Few-Shot | 0.7639 | +0.2639 | -0.0543 |
| 8 | GEMINI | LLM | With Info | 0.7615 | +0.2615 | -0.0567 |
| 9 | DEEPSEEK-R1 | LLM | Info + Few-Shot | 0.7454 | +0.2454 | -0.0727 |
| 10 | LLAMA | LLM | With Info | 0.7201 | +0.2201 | -0.0980 |
| 11 | DEEPSEEK-R1 | LLM | Bare Minimum | 0.6673 | +0.1673 | -0.1508 |
| 12 | MIXTRAL | LLM | With Info | 0.6260 | +0.1260 | -0.1921 |
| 13 | DEEPSEEK-R1 | LLM | With Info | 0.6223 | +0.1223 | -0.1958 |
| 14 | LLAMA | LLM | Bare Minimum | 0.6197 | +0.1197 | -0.1984 |
| 15 | MIXTRAL | LLM | Info + Few-Shot | 0.6177 | +0.1177 | -0.2004 |

---

## ML Classifier Performance Details

**Model:** Random Forest (100 trees, max_depth=10)

### Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.9140 |
| Macro F1 | 0.9140 |
| Macro Precision | 0.9140 |
| Macro Recall | 0.9140 |

### Competitive Analysis

- **Baselines beaten:** 7/7 (100.0%)
- **LLMs beaten:** 15/15 (100.0%)
- **Margin over best LLM:** +0.0958 (+11.7%)
- **Margin over best baseline:** +0.4140 (+82.8%)

---

## Top 10 LLM Models (All Trials)

| Rank | Model | Trial | Macro F1 | vs Classifier |
|------|-------|-------|----------|---------------|
| 1 | GEMINI | Bare Minimum | 0.8181 | -0.0958 |
| 2 | LLAMA | Info + Few-Shot | 0.8098 | -0.1042 |
| 3 | QWEN | Bare Minimum | 0.7977 | -0.1163 |
| 4 | QWEN | Info + Few-Shot | 0.7870 | -0.1269 |
| 5 | QWEN | With Info | 0.7646 | -0.1493 |
| 6 | GEMINI | Info + Few-Shot | 0.7639 | -0.1501 |
| 7 | GEMINI | With Info | 0.7615 | -0.1525 |
| 8 | DEEPSEEK-R1 | Info + Few-Shot | 0.7454 | -0.1685 |
| 9 | LLAMA | With Info | 0.7201 | -0.1939 |
| 10 | DEEPSEEK-R1 | Bare Minimum | 0.6673 | -0.2467 |

---

## LLM Performance by Prompting Strategy


### Bare Minimum

- **Models:** 5
- **Mean F1:** 0.6559
- **Best F1:** 0.8181
- **Worst F1:** 0.3767
- **Gap to Classifier:** 0.2581

### With Info

- **Models:** 5
- **Mean F1:** 0.6989
- **Best F1:** 0.7646
- **Worst F1:** 0.6223
- **Gap to Classifier:** 0.2151

### Info + Few-Shot

- **Models:** 5
- **Mean F1:** 0.7448
- **Best F1:** 0.8098
- **Worst F1:** 0.6177
- **Gap to Classifier:** 0.1692

---

## Key Insights

1. **🏆 Best Overall:** Random Forest Classifier with 0.9140 Macro F1 (Rank #1)
2. **Best LLM:** GEMINI (Bare Minimum) with 0.8181 Macro F1 (Rank #2)
3. **Performance Gap:** ML Classifier outperforms best LLM by 0.0958 (11.7%)
4. **LLMs Above Random (0.5):** 14/15 (93.3%)
5. **Mean LLM Performance:** 0.6999

---

## Recommendations

1. **For production deployment:** Use Random Forest Classifier (91.4% accuracy, fastest inference)
2. **Best LLM approach:** GEMINI with Bare Minimum prompting
3. **Best prompting strategy:** Info + Few-Shot (highest mean performance across models)
4. **Minimum acceptable threshold:** Macro F1 > 0.5 (better than random)
5. **Cost consideration:** ML Classifier is free and local; LLMs require API costs

---

## Dataset Information

- **Dataset:** Adult Income Dataset (tenth-trial)
- **Total Changes:** 28,256
- **Intentional:** 13,291 (47.04%)
- **Unintentional:** 14,965 (52.96%)
- **Features:** 15 (age, workclass, education, etc.)
- **Records:** 19,539 manipulated records from 6,513 originals
