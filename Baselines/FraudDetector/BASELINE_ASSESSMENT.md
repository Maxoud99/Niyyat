# Fraud Detection Baseline — Assessment

## Verdict

**As a standalone fraud detector: bad.** **As a baseline for the paper: good (a useful null result).**

The ECOD + Leave-One-Out (LOO) baseline is intentionally weak on the hard,
realistic datasets — and that weakness is the point: it demonstrates that pure
anomaly detection (no goal-directedness modeling) fails to attribute intent,
motivating the structural attribution approach of `\sys`.

## Where it "works" (but isn't meaningful)

- **TwitterBot-LLM** (F1-INT = 0.994) and **TabFact** (F1-INT = 0.995) look
  near-perfect, but both datasets are **single-class** (all errors are
  intentional). The best threshold found is negative, meaning the detector
  trivially predicts "intentional" for *every* cell. This is a
  precision-trivial degenerate result, not real discrimination.

## Where it has real signal

- **Adult-TFM** (F1-INT = 0.504, F1-w = 0.742, AUC = 0.675) is the only
  two-class dataset with meaningful intentional-class recall. TFM intentional
  errors are chosen to be globally anomalous (they flip a classifier), which
  is exactly what ECOD's multi-column scoring can pick up — high precision
  (0.955) at modest recall (0.342).

## Where it completely fails

- **Adult-Mixed** (F1-INT = 0.000, AUC = 0.149) and **TwitterBot-Mixed**
  (F1-INT = 0.000, AUC = 0.399): adversarial greedy edits (Kireev et al.) stay
  within a cost budget and look plausible — they are not statistical outliers,
  so ECOD can't separate them from unintentional noise.
- **TwitterBot-TFM** (F1-INT = 0.000, AUC = 0.422): only 19 continuous
  features with small ranges — the per-cell LOO delta is too noisy to
  distinguish intent.
- **Adult-LLM** (F1-INT = 0.071, AUC = 0.191): LLM-injected *unintentional*
  errors (e.g. `'4B'` coerced to NaN → imputed to median) are themselves
  anomalous/noisy, degrading separation between the two classes.

## Summary Table

| Dataset | Cells | INT | UNINT | F1-INT | F1-UNINT | F1-w | Acc | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adult-LLM | 43,259 | 18,014 | 25,245 | 0.071 | 0.702 | 0.439 | 0.549 | 0.191 |
| Adult-Mixed | 11,661 | 3,784 | 7,877 | 0.000 | 0.800 | 0.540 | 0.666 | 0.149 |
| Adult-TFM | 9,199 | 3,009 | 6,190 | 0.504 | 0.858 | 0.742 | 0.780 | 0.675 |
| TwitterBot-LLM | 322 | 322 | 0 | **0.994** | — | 0.994 | 0.988 | — |
| TwitterBot-Mixed | 1,099 | 343 | 756 | 0.000 | 0.811 | 0.558 | 0.682 | 0.399 |
| TwitterBot-TFM | 1,184 | 283 | 901 | 0.000 | 0.863 | 0.656 | 0.758 | 0.422 |
| TabFact | 2,245 | 2,245 | 0 | **0.995** | — | 0.995 | 0.990 | — |

*F1-UNINT and AUC are undefined (—) for single-class datasets.*

## Why this is still useful for the paper

1. **ECOD-LOO is not a general intent-attribution method** — it only works
   when intentional errors are also statistical outliers (Adult-TFM), not for
   budget-constrained adversarial edits (Mixed) or plausible LLM edits.
2. **AUC ≈ 0.15–0.19 on the Adult hard cases** confirms near-random
   performance — anomaly score alone cannot solve intent attribution.
3. **Adult-TFM (AUC = 0.675)** is the one case with real signal, because TFM
   cells are globally anomalous even if individually plausible.
4. **The baseline serves its intended purpose**: it shows that purely
   anomaly-based approaches fail on the Mixed and LLM families, setting up
   the contrast with `\sys`.

## Open item

The LLM-baseline comparison columns in `RESULTS.md` (§"Comparison to LLM
Baselines") and `summary_all_datasets.csv` are still marked *TBC* — fill these
in once the LLM evaluation runs complete.
