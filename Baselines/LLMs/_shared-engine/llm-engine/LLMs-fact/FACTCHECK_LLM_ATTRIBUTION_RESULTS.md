# TabFact Attribution Results: LLM vs. Scenario B / B+

Generated: April 27, 2026

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Dataset | TabFact (FactCheck error model) |
| Total records evaluated | **984** |
| Feature columns evaluated | **5** (subject_entity, subject_type, subject_location, value, claim_domain, metric — excluding identifier) |
| Total cells evaluated | **4,920** (984 × 5) |
| Error cells | **723** (14.7% of all cells) |
| Clean cells | **4,197** (85.3% of all cells) |
| Intentional errors (mask=+1) | 723 (100% of error cells) |
| Unintentional errors (mask=−1) | 0 (0% of error cells) |
| Chunks (10 records each) | 99 |

> **Important characteristic:** All errors in the TabFact dataset are intentional (mask=+1). There are no unintentional errors in the ground truth. This makes the task one of **intentional detection only** — a model that labels all errors as intentional achieves perfect F1 = 1.0.

---

## 2. Scenario B and B+ Results

Because every error in TabFact is intentional, the clustering-based attribution system (Scenario B / B+) trivially assigns the intentional label to all detected errors:

| System | F1 Intentional | Accuracy | Notes |
|---|---|---|---|
| **Scenario B** | **1.0000** | 1.0000 | All errors labeled intentional — correct |
| **Scenario B+** | **1.0000** | 1.0000 | All errors labeled intentional — correct |

This serves as the **theoretical ceiling** for this dataset. Any system that correctly identifies all errors as intentional achieves F1 = 1.0. The challenge is that LLMs must make this judgment from the record content alone, without knowing the label distribution.

---

## 3. LLM-Based Attribution Results

All five models were evaluated on **723 error cells** (984 records, 99 chunks of 10) across three prompt variants:
- **bareminimum** — no examples, no feature metadata (equivalent to zero_shot)
- **info** — column schema and feature descriptions provided
- **few-shots** — column schema + labeled examples per class

### 3.1 Metric Definitions

| Symbol | Meaning |
|---|---|
| **Accuracy** | Fraction of error cells correctly predicted as intentional (1) |
| **INT-F1** | F1 for the intentional class; Precision = 1.0 always (no unintentional cells in eval set), so F1 = 2R/(1+R) |
| **Correct** | Cells predicted as intentional (TP) |
| **Wrong** | Cells predicted as unintentional (FN) |
| **Missed** | Cells predicted as no-error (FN) — 0 for all models |

> No model ever predicted 0 (missed) on error cells — all wrong predictions were calls to −1 (unintentional).

### 3.2 Full Results (All 15 Variants, sorted by INT-F1)

| Rank | Variant | Accuracy | INT-Precision | INT-Recall | **INT-F1** | Correct | Wrong | Missed |
|---|---|---|---|---|---|---|---|---|
| 1 | info-gemini | 0.7981 | 1.0000 | 0.7981 | **0.8877** | 577 | 146 | 0 |
| 2 | baremin-qwen | 0.7870 | 1.0000 | 0.7870 | **0.8808** | 569 | 154 | 0 |
| 3 | few-shots-gemini | 0.7801 | 1.0000 | 0.7801 | **0.8765** | 564 | 159 | 0 |
| 4 | info-llama | 0.7759 | 1.0000 | 0.7759 | **0.8738** | 561 | 162 | 0 |
| 5 | info-qwen | 0.7704 | 1.0000 | 0.7704 | **0.8703** | 557 | 166 | 0 |
| 6 | few-shots-qwen | 0.7580 | 1.0000 | 0.7580 | **0.8623** | 548 | 175 | 0 |
| 7 | few-shots-llama | 0.7552 | 1.0000 | 0.7552 | **0.8605** | 546 | 177 | 0 |
| 8 | info-r1-qwen | 0.7497 | 1.0000 | 0.7497 | **0.8569** | 542 | 181 | 0 |
| 9 | info-mixtral | 0.7455 | 1.0000 | 0.7455 | **0.8542** | 539 | 184 | 0 |
| 10 | baremin-llama | 0.7261 | 1.0000 | 0.7261 | **0.8413** | 525 | 198 | 0 |
| 11 | bareminimum-gemini | 0.7109 | 1.0000 | 0.7109 | **0.8310** | 514 | 209 | 0 |
| 12 | baremin-mixtral | 0.7040 | 1.0000 | 0.7040 | **0.8263** | 509 | 214 | 0 |
| 13 | baremin-r1-qwen | 0.6888 | 1.0000 | 0.6888 | **0.8157** | 498 | 225 | 0 |
| 14 | few-shots-mixtral | 0.6888 | 1.0000 | 0.6888 | **0.8157** | 498 | 225 | 0 |
| 15 | few-shots-r1-qwen | 0.6874 | 1.0000 | 0.6874 | **0.8148** | 497 | 226 | 0 |

### 3.3 Per-Model Summary (Best Variant & Averages)

| Model | Best Variant | Best INT-F1 | Avg INT-F1 | Avg Accuracy |
|---|---|---|---|---|
| **Gemini-2.0-Flash** | info | **0.8877** | 0.8651 | 0.7630 |
| **Qwen2.5-32B** | bareminimum | **0.8808** | 0.8711 | 0.7718 |
| **LLaMA-3-70B** | info | **0.8738** | 0.8585 | 0.7524 |
| **DeepSeek-R1-Qwen-32B** | info | 0.8569 | 0.8291 | 0.7086 |
| **Mixtral-8x7B** | info | 0.8542 | 0.8321 | 0.7128 |

### 3.4 Per-Prompt-Strategy Summary (Avg across all models)

| Strategy | Avg INT-F1 | Avg Accuracy | Best Model |
|---|---|---|---|
| **info** | **0.8685** | 0.7679 | Gemini (0.8877) |
| **few-shots** | 0.8460 | 0.7339 | Gemini (0.8765) |
| **bareminimum** | 0.8390 | 0.7234 | Qwen (0.8808) |

---

## 4. Head-to-Head Comparison: LLM vs. Scenario B / B+

### 4.1 Gap to Perfect Score (F1 = 1.0)

| System | INT-F1 | Gap to 1.0 | Correct / 723 | Wrong |
|---|---|---|---|---|
| Scenario B | **1.0000** | 0.0000 | 723 / 723 | 0 |
| Scenario B+ | **1.0000** | 0.0000 | 723 / 723 | 0 |
| Gemini-2.0-Flash (info) | 0.8877 | −0.1123 | 577 / 723 | 146 |
| Qwen2.5-32B (bareminimum) | 0.8808 | −0.1192 | 569 / 723 | 154 |
| LLaMA-3-70B (info) | 0.8738 | −0.1262 | 561 / 723 | 162 |
| DeepSeek-R1-Qwen-32B (info) | 0.8569 | −0.1431 | 542 / 723 | 181 |
| Mixtral-8x7B (info) | 0.8542 | −0.1458 | 539 / 723 | 184 |
| Mixtral-8x7B (few-shots) | 0.8157 | −0.1843 | 498 / 723 | 225 |
| DeepSeek-R1-Qwen-32B (few-shots) | 0.8148 | −0.1852 | 497 / 723 | 226 |

### 4.2 All LLM Best Results vs. Scenario B / B+

| Rank | System | Best INT-F1 | vs. Scenario B (1.0) |
|---|---|---|---|
| — | **Scenario B** | **1.0000** | baseline |
| — | **Scenario B+** | **1.0000** | baseline |
| 1 | Gemini-2.0-Flash | 0.8877 | −0.1123 ❌ |
| 2 | Qwen2.5-32B | 0.8808 | −0.1192 ❌ |
| 3 | LLaMA-3-70B | 0.8738 | −0.1262 ❌ |
| 4 | DeepSeek-R1-Qwen-32B | 0.8569 | −0.1431 ❌ |
| 5 | Mixtral-8x7B | 0.8542 | −0.1458 ❌ |

**No LLM matches Scenario B / B+ on the TabFact dataset.** The gap ranges from −0.11 (Gemini) to −0.15 (Mixtral).

---

## 5. Key Findings

### 5.1 Why Scenario B / B+ Scores Perfectly
The TabFact dataset has a **degenerate label distribution** for the attribution task — all 723 error cells are intentional. The clustering-based system, by observing the cluster structure of the full dataset, correctly assigns the intentional label to all error cells, resulting in F1 = 1.0. This is a structural advantage that LLMs cannot exploit from per-record prompting alone.

### 5.2 LLM Performance Pattern
- **Gemini-2.0-Flash** achieves the best single-variant F1 (0.8877, `info`), correctly labeling 577/723 error cells.
- **Qwen2.5-32B** has the **highest average F1** (0.8711) across its 3 variants — the most consistent model.
- **Prompt strategy ranking:** `info` > `few-shots` > `bareminimum` — providing schema context consistently helps.
- Notably, Qwen's `bareminimum` (no context, 0.8808) beats all models' `few-shots` results, suggesting Qwen has strong prior knowledge about tabular data errors.
- **DeepSeek-R1** underperforms relative to its TFM results — chain-of-thought reasoning does not help on this single-class attribution task.

### 5.3 Prompt Sensitivity
| Model | Bareminimum F1 | Info F1 | Few-shots F1 | Max Δ |
|---|---|---|---|---|
| Gemini | 0.8310 | **0.8877** | 0.8765 | 0.0567 |
| Qwen | **0.8808** | 0.8703 | 0.8623 | 0.0185 |
| LLaMA | 0.8413 | **0.8738** | 0.8605 | 0.0325 |
| DeepSeek-R1 | 0.8157 | **0.8569** | 0.8148 | 0.0421 |
| Mixtral | 0.8263 | **0.8542** | 0.8157 | 0.0385 |

Qwen shows the smallest prompt sensitivity (Δ=0.0185), while Gemini benefits most from the `info` prompt (Δ=0.0567).

### 5.4 Contrast with TFM Results
Unlike the TFM dataset (mixed intentional/unintentional), TabFact presents a simpler attribution problem — but one where LLMs are still imperfect. The TFM task is harder (2 classes, imbalanced), while TabFact is structurally trivial for the clustering system (1 class) but still challenging for LLMs reasoning record-by-record.

| Aspect | TabFact | TFM-Inject |
|---|---|---|
| Error class balance | 100% intentional | 18.9% intentional / 81.1% unintentional |
| Scenario B / B+ F1 | **1.0000** | 0.9135–0.9395 |
| Best LLM F1 | 0.8877 (Gemini info) | 0.9201 (DeepSeek-R1 few_shot) |
| LLM gap to clustering | −0.11 to −0.15 | −0.02 to −0.29 |
| Best prompt strategy | info | few-shots (mostly) |

---

## 6. Run Statistics

| Variant | ok/total | API errors | Parse errors | Runtime |
|---|---|---|---|---|
| bareminimum-gemini | 94/99 | 1 | 4 | 33.4 min |
| info-gemini | 99/99 | 0 | 0 | 24.3 min |
| few-shots-gemini | 99/99 | 0 | 0 | 24.9 min |
| baremin-mixtral | 98/99 | 0 | 1 | 8.3 min |
| info-mixtral | 95/99 | 0 | 4 | 7.4 min |
| few-shots-mixtral | 98/99 | 0 | 1 | 8.0 min |
| baremin-llama | 99/99 | 0 | 0 | 9.4 min |
| info-llama | 99/99 | 0 | 0 | 10.4 min |
| few-shots-llama | 99/99 | 0 | 0 | 10.8 min |
| baremin-qwen | 99/99 | 0 | 0 | 5.7 min |
| info-qwen | 99/99 | 0 | 0 | 6.3 min |
| few-shots-qwen | 99/99 | 0 | 0 | 7.7 min |
| baremin-r1-qwen | 98/99 | 1 | 0 | 17.6 min |
| info-r1-qwen | 99/99 | 0 | 0 | 16.3 min |
| few-shots-r1-qwen | 95/99 | 1 | 3 | 17.7 min |

- **Fastest model:** Qwen (5.7–7.7 min/run)
- **Slowest model:** DeepSeek-R1 (16–18 min/run, due to `<think>` chain-of-thought)
- **Most API failures:** Gemini bareminimum (5 failed chunks)

---

## 7. Output File Locations

All results are under:
`error_detection_system/src/attribution/llm-based/LLMs-fact/`

```
outputs/
├── bareminimum-gemini/    intent_labels.csv, intent_explanations.csv, run_stats.json
├── info-gemini/
├── few-shots-gemini/
├── baremin-mixtral/
├── info-mixtral/
├── few-shots-mixtral/
├── baremin-llama/
├── info-llama/
├── few-shots-llama/
├── baremin-qwen/
├── info-qwen/
├── few-shots-qwen/
├── baremin-r1-qwen/
├── info-r1-qwen/
└── few-shots-r1-qwen/

evaluation_results/
├── overall_metrics.csv     (per-variant accuracy, P, R, F1, counts)
└── per_column_metrics.csv  (per-column breakdown for each variant)
```
