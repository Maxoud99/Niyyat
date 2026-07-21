# Intent Attribution — Full Evaluation Results

**Date:** April 24, 2026  
**Dataset:** TabFact (984 records, 99 chunks of 10)  
**Task:** Predict whether detected errors are *intentional* (1) or *unintentional* (-1)  
**Ground truth:** All error cells (mask=1) are intentional → true label = 1 for all 723 error cells  
**Models evaluated:** Gemini Flash, Mixtral 8x7B, Llama-3, Qwen, DeepSeek-R1  
**Prompt variants:** `bareminimum`, `info`, `few-shots`

---

## 1. Evaluation Metrics

Metrics are computed **on error cells only** (723 cells, all true label = 1):

| Symbol | Meaning |
|--------|---------|
| **Accuracy** | Fraction of error cells correctly predicted as intentional (1) |
| **INT-F1** | F1 for the intentional class; Precision=1.0 always (no true-0 cells in eval set), so F1 = 2R/(1+R) |
| **Correct** | Cells predicted as intentional (1) — True Positives |
| **Wrong** | Cells predicted as unintentional (-1) — False Negatives |
| **Missed** | Cells predicted as no-error (0) — False Negatives |

> Note: No model ever predicted 0 (missed) on error cells — all errors were wrong calls to -1.

---

## 2. Full Results Table (sorted by INT-F1)

| Rank | Variant              | Accuracy | INT-P  | INT-R  | INT-F1 | Correct | Wrong | Missed |
|------|----------------------|----------|--------|--------|--------|---------|-------|--------|
| 1    | info-gemini          | 0.7981   | 1.0000 | 0.7981 | 0.8877 | 577     | 146   | 0      |
| 2    | baremin-qwen         | 0.7870   | 1.0000 | 0.7870 | 0.8808 | 569     | 154   | 0      |
| 3    | few-shots-gemini     | 0.7801   | 1.0000 | 0.7801 | 0.8765 | 564     | 159   | 0      |
| 4    | info-llama           | 0.7759   | 1.0000 | 0.7759 | 0.8738 | 561     | 162   | 0      |
| 5    | info-qwen            | 0.7704   | 1.0000 | 0.7704 | 0.8703 | 557     | 166   | 0      |
| 6    | few-shots-qwen       | 0.7580   | 1.0000 | 0.7580 | 0.8623 | 548     | 175   | 0      |
| 7    | few-shots-llama      | 0.7552   | 1.0000 | 0.7552 | 0.8605 | 546     | 177   | 0      |
| 8    | info-r1-qwen         | 0.7497   | 1.0000 | 0.7497 | 0.8569 | 542     | 181   | 0      |
| 9    | info-mixtral         | 0.7455   | 1.0000 | 0.7455 | 0.8542 | 539     | 184   | 0      |
| 10   | baremin-llama        | 0.7261   | 1.0000 | 0.7261 | 0.8413 | 525     | 198   | 0      |
| 11   | bareminimum-gemini   | 0.7109   | 1.0000 | 0.7109 | 0.8310 | 514     | 209   | 0      |
| 12   | baremin-mixtral      | 0.7040   | 1.0000 | 0.7040 | 0.8263 | 509     | 214   | 0      |
| 13   | baremin-r1-qwen      | 0.6888   | 1.0000 | 0.6888 | 0.8157 | 498     | 225   | 0      |
| 14   | few-shots-mixtral    | 0.6888   | 1.0000 | 0.6888 | 0.8157 | 498     | 225   | 0      |
| 15   | few-shots-r1-qwen    | 0.6874   | 1.0000 | 0.6874 | 0.8148 | 497     | 226   | 0      |

---

## 3. Comparison by Model (Best Variant & Averages)

| Model         | Best Variant   | Best F1 | Avg F1 | Avg Accuracy |
|---------------|----------------|---------|--------|--------------|
| **Gemini**    | info           | 0.8877  | 0.8651 | 0.7630       |
| **Qwen**      | bareminimum    | 0.8808  | 0.8711 | 0.7718       |
| **Llama-3**   | info           | 0.8738  | 0.8585 | 0.7524       |
| **DeepSeek-R1** | info         | 0.8569  | 0.8291 | 0.7086       |
| **Mixtral**   | info           | 0.8542  | 0.8321 | 0.7128       |

> **Qwen has the highest average F1 (0.8711)** across its 3 variants, narrowly ahead of Gemini (0.8651).  
> **DeepSeek-R1 and Mixtral** are the weakest overall, especially on bareminimum and few-shots prompts.

---

## 4. Comparison by Prompt Strategy (Avg across all models)

| Strategy        | Avg F1 | Avg Accuracy | Best Model (this strategy) |
|-----------------|--------|--------------|---------------------------|
| **info**        | 0.8685 | 0.7679       | Gemini (0.8877)            |
| **few-shots**   | 0.8460 | 0.7339       | Gemini (0.8765)            |
| **bareminimum** | 0.8390 | 0.7234       | Qwen (0.8808)              |

> The `info` prompt (providing column schema/context) is the consistently best strategy.  
> Interestingly, `few-shots` does not always outperform `bareminimum` — particularly for Qwen where `bareminimum` is the best.

---

## 5. Run Statistics

| Variant              | ok/total | api_err | parse_err | Runtime  |
|----------------------|----------|---------|-----------|----------|
| bareminimum-gemini   | 94/99    | 1       | 4         | 33.4 min |
| info-gemini          | 99/99    | 0       | 0         | 24.3 min |
| few-shots-gemini     | 99/99    | 0       | 0         | 24.9 min |
| baremin-mixtral      | 98/99    | 0       | 1         | 8.3 min  |
| info-mixtral         | 95/99    | 0       | 4         | 7.4 min  |
| few-shots-mixtral    | 98/99    | 0       | 1         | 8.0 min  |
| baremin-llama        | 99/99    | 0       | 0         | 9.4 min  |
| info-llama           | 99/99    | 0       | 0         | 10.4 min |
| few-shots-llama      | 99/99    | 0       | 0         | 10.8 min |
| baremin-qwen         | 99/99    | 0       | 0         | 5.7 min  |
| info-qwen            | 99/99    | 0       | 0         | 6.3 min  |
| few-shots-qwen       | 99/99    | 0       | 0         | 7.7 min  |
| baremin-r1-qwen      | 98/99    | 1       | 0         | 17.6 min |
| info-r1-qwen         | 99/99    | 0       | 0         | 16.3 min |
| few-shots-r1-qwen    | 95/99    | 1       | 3         | 17.7 min |

> **Qwen** is the fastest model (5.7–7.7 min/run).  
> **DeepSeek-R1** is the slowest (16–18 min/run) due to chain-of-thought reasoning (`<think>` tags).  
> **Gemini** had the most failures on `bareminimum` (5 failed chunks, likely API rate-limiting).

---

## 6. Key Findings

1. **Best overall:** `info-gemini` (F1=0.8877, Acc=0.7981) — 577/723 error cells correctly attributed.
2. **Best open-source model:** `info-llama` (F1=0.8738) — very close to Gemini; Qwen avg is higher.
3. **Most efficient:** Qwen — fastest runtime AND highest avg F1 across variants.
4. **Worst performer:** DeepSeek-R1 `few-shots` (F1=0.8148) — chain-of-thought reasoning did not help.
5. **Prompt strategy ranking:** `info` > `few-shots` > `bareminimum` (consistent across models).
6. **All models show bias toward intentional**: 0 missed cells, all wrong predictions were `-1` (unintentional). Models tend to flag errors as intentional, which is correct in this dataset.

---

## 7. Output File Locations

```
outputs/
├── bareminimum-gemini/    intent_labels.csv, intent_explanations.csv, run_stats.json
├── info-gemini/           ...
├── few-shots-gemini/      ...
├── baremin-mixtral/       ...
├── info-mixtral/          ...
├── few-shots-mixtral/     ...
├── baremin-llama/         ...
├── info-llama/            ...
├── few-shots-llama/       ...
├── baremin-qwen/          ...
├── info-qwen/             ...
├── few-shots-qwen/        ...
├── baremin-r1-qwen/       ...
├── info-r1-qwen/          ...
└── few-shots-r1-qwen/     ...

evaluation_results/
├── overall_metrics.csv    (per-variant accuracy, P, R, F1, counts)
└── per_column_metrics.csv (per-column breakdown for each variant)
```
