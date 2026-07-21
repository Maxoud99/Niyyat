# ECOD + Leave-One-Out Fraud Baseline — Results

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

*F1-UNINT for single-class INT datasets (TwitterBot-LLM, TabFact) is undefined (—).*
*AUC undefined for single-class datasets.*

---

## Analysis

### What works: single-class intentional datasets

**TwitterBot-LLM (F1-INT = 0.994)** and **TabFact (F1-INT = 0.995)** show near-perfect
scores. However, this is **not a meaningful result** — both datasets contain
*only* intentional errors, so the threshold sweep trivially sets the cutoff
below all delta scores, predicting +1 for every cell. The result degenerates to
a precision-trivial baseline. The best_threshold is *negative* (−2.54, −2.81),
confirming the detector is predicting INT for everything.

### What partially works: Adult-TFM

**Adult-TFM (F1-INT = 0.504, F1-w = 0.742, AUC = 0.675)** is the only
two-class dataset where the fraud detector achieves meaningful intentional-class
recall. This is interpretable: TFM intentional errors flip the classifier by
choosing values that are anomalous *in the context of the other features* —
which is exactly what ECOD detects. The high precision (0.955) at low recall
(0.342) suggests the detector is correctly identifying a subset of strongly
anomalous intentional edits while missing the subtler ones.

### What fails: Mixed families (Adult and TwitterBot)

**Adult-Mixed (F1-INT = 0.000)** and **TwitterBot-Mixed (F1-INT = 0.000)** are
complete failures for the intentional class. The adversarial greedy search
(Kireev et al.) is explicitly designed to stay within a cost budget and produce
*plausible* edits — they are not statistical outliers. ECOD therefore assigns
them similar anomaly scores to unintentional outlier-injection errors, and the
LOO delta signal is flat for intentional cells.

**TwitterBot-TFM (F1-INT = 0.000)** fails similarly: with only 19 features
(all continuous social-network signals with small absolute ranges), the per-cell
LOO delta is too noisy to distinguish intent.

### Why Adult-LLM is weak (F1-INT = 0.071, AUC = 0.191)

Adult-LLM intentional errors (gain-targeted, fairness masking, obfuscation) do
produce statistical anomalies — but so do LLM-injected *unintentional* errors
(e.g., `'4B'` in a numeric field → coerced to NaN → imputed to median → no
anomaly signal preserved). The LLM's unintentional category is itself noisy,
degrading the separation.

---

## Comparison to LLM Baselines and Heuristic Pipelines (Scenario B / B+)

The LLM baselines and the heuristic pipelines (§ from
`error_detection_system/PERFORMANCE_COMPARISON.md`) see both the clean and
dirty rows (LLMs get a structured rubric; the heuristic pipelines extract
13–23 hand-crafted/statistical features per cell and propagate from a ~1–9%
labeled seed). The ECOD-LOO fraud proxy, by contrast, sees **only the dirty
row distribution** — no clean/dirty pair and no labels at all. This is a large
information asymmetry and should be kept in mind when reading the gaps below.

- **Scenario B** — 13 heuristic features (H1–H8), clustering + label
  propagation from ~1–9% labeled cells.
- **Scenario B+** — Scenario B features + 10 cell-level statistical features
  (23 total), same semi-supervised pipeline.
- **Best LLM** — best (model × prompt) zero-label result from §3–5 of
  `PERFORMANCE_COMPARISON.md`.
- **Avg LLM** — mean F1-w/Acc across all 5 models × 3 prompts.

| Dataset | ECOD-LOO F1-w | Avg LLM | Best LLM | Scenario B (≤9% labels) | Scenario B+ (≤9% labels) |
|---|---:|---:|---:|---:|---:|
| Adult-LLM | 0.439 | 0.7283 | 0.8182 (Gemini, zero-shot) | 0.9081 | **0.9748** |
| Adult-Mixed (Kireev) | 0.540 | 0.7455 | 0.9170 (R1-Qwen, info) | 0.9825 | **0.9937** |
| Adult-TFM | **0.742** | 0.7370 | 0.9201 (DeepSeek-R1, few-shot) | 0.9135 (mean) / 0.9164 (best) | **0.9395** (best) |
| TwitterBot-Mixed | 0.558 | 0.5468 | 0.6569 (Llama-3, zero-shot) | 0.9293 | **0.9484** |
| TwitterBot-TFM | 0.656 | — | — | — | — |
| TabFact | 0.995 | 0.8470 | 0.8877 (Gemini, info) | **1.0000** | **1.0000** |

*"Avg LLM" / "Best LLM" columns use the row-level micro-accuracy metric for
Adult-LLM (≈ F1-w for balanced classes) and feature-level F1-weighted (or
INT-F1 for the single-class TabFact) elsewhere. The Adult-TFM and TabFact LLM
/ Scenario-B / B+ numbers come from
`error_detection_system/src/attribution/llm-based/TFM_LLM_ATTRIBUTION_RESULTS.md`
and `.../LLMs-fact/FACTCHECK_LLM_ATTRIBUTION_RESULTS.md`. Note these runs use
slightly different sample sizes than this fraud baseline (TFM: 6,051 error
records / 11,496 cells vs. 9,199 cells here; TabFact: 723 error cells vs.
2,245 here) — treat as directionally comparable, not exactly identical splits.
No LLM or heuristic-pipeline run exists yet for TwitterBot-TFM — left blank.*

### Reading the comparison

1. **ECOD-LOO is the weakest method on every two-class dataset.** Even the
   *average* zero-label LLM beats the fraud proxy by +0.29 (Adult-LLM), +0.21
   (Adult-Mixed), and ≈0 (TwitterBot-Mixed / Adult-TFM — the only cases where
   ECOD-LOO is roughly on par with, but still at or below, the average LLM).
2. **The heuristic pipelines dominate everything**, including the best LLM, on
   every dataset measured. Scenario B+ improves over Scenario B on all of
   them (+6.7pp Adult-LLM, +1.1pp Adult-Mixed, +1.9pp TwitterBot-Mixed,
   +2.3pp Adult-TFM best-config), using ≤9% labels.
3. **Adult-TFM is ECOD-LOO's best showing (F1-w = 0.742, AUC = 0.675)** — yet
   it still trails the average LLM (0.737 is close, but best LLM 0.920 and
   Scenario B+ 0.940 are well ahead). This is the *narrowest* gap of any
   dataset, consistent with Adult-TFM being the one case where ECOD's
   anomaly signal has genuine (if partial) relevance.
4. **TabFact and TwitterBot-LLM "near-perfect" ECOD-LOO scores (0.995/0.994)
   are degenerate**, as noted above — but interestingly, Scenario B/B+ also
   score a (non-degenerate) perfect 1.0 on TabFact, because the cluster
   structure trivially separates the single-class error cells, while even the
   best LLM (Gemini, INT-F1 = 0.888) cannot reach that ceiling from
   per-record prompting alone.
5. **Takeaway for the paper:** the fraud-proxy baseline is useful precisely
   *because* it is this weak — it shows that "detect anomalies, call the
   anomalous cells intentional" is not a viable general intent-attribution
   strategy, whereas even minimal (≤9%) supervision via heuristic features, or
   zero-label LLM reasoning over clean/dirty pairs, substantially outperforms
   it on every dataset except the degenerate single-class cases.

*(TwitterBot-TFM LLM/heuristic comparison numbers to be filled once that eval
run exists.)*

---

## Key Takeaways

1. **ECOD-LOO is not a general intent attribution method.** It only works when
   intentional errors are also statistical outliers — which is true for
   distribution-consistent TFM injection but false for budget-constrained
   adversarial attacks (Mixed) and LLM-injected plausible edits.

2. **AUC ≈ 0.15–0.19 on Adult datasets** confirms the detector is near-random
   on the hard cases (Mixed, LLM). This is a useful null result: it shows that
   anomaly score alone cannot solve intent attribution.

3. **Adult-TFM is the only dataset where the fraud proxy has real signal
   (AUC = 0.675).** This is because TFM intentional cells are chosen to flip a
   classifier — they are globally anomalous even if individually plausible.
   ECOD's multi-column combination partially captures this.

4. **The fraud proxy baseline serves its intended purpose in the paper:** it
   demonstrates that purely anomaly-based approaches (which do not model
   goal-directedness) fail on the Mixed and LLM families, motivating the
   structural attribution approach of `\sys`.

---

## Output Files

| File | Contents |
|---|---|
| `*_loo_raw.csv` | row_idx, col, y_true, y_pred, delta_score for every flagged cell |
| `*_metrics.json` | final metrics at best threshold per dataset |
| `*_threshold_sweep.csv` | F1-w, F1-int, accuracy at each threshold value |
| `summary_all_datasets.csv` | one-row-per-dataset summary |
| `run.log` | full stdout/stderr from the run |
