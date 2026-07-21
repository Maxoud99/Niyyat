# Credit Card Fraud (ULB) — NIYYAT Onboarding: Approach

## Motivation

Chile Customs (`NIYYAT/Datasets/ChileCustoms`) is NIYYAT's only real-world
dataset with an external, third-party ground-truth signal: `NU_REGR`, a
record of whether a declaration was genuinely selected for physical customs
inspection. That signal is real and unbiased by us, but it is a noisy,
**population-level** proxy: inspection-selection does not confirm that the
specific flagged cell is the reason a declaration was chosen, and 89% of
flagged cells on inspected declarations turn out to be structural artefacts
(truncation, typos) with no plausible link to intent at all. In short:
**"suspicious" (selected for inspection) does not mean "intentional"**.

The [Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark)
(Amazon Science) aggregates several public fraud datasets with standardised
schemas. Among them, the **ULB / Kaggle Credit Card Fraud** dataset carries
a categorically stronger label than Chile Customs: `Class` is a **confirmed**
fraud determination already made by the card issuer on this exact
transaction — not a suspicion flag, not a sampling-selection signal. There
is no "suspicious-but-unconfirmed" tier to confuse with intent. This makes
it the right dataset to test the "suspicious ≠ intentional" lesson learned
from Chile Customs: here, the row-level label genuinely *is* intent-grade.

## Dataset

- **Source**: ULB/Worldline, Sept 2013, 284,807 real European card
  transactions, downloaded from OpenML (id 1597, `creditcard`/`phpKo8OWT`),
  which mirrors the canonical Kaggle release used by the FDB benchmark.
- **Schema**: `Time` (elapsed seconds, sequence index), `V1`–`V28`
  (PCA-rotated, anonymised — no disclosed business meaning), `Amount`
  (EUR), `Class` (0=legitimate, 1=confirmed fraud — label only, never a
  feature).
- **Class balance**: 492 confirmed-fraud rows (0.173%) out of 284,807.

## Error detection (4-detector ensemble)

Unlike eBay/Chile Customs, this schema is pure-numeric and anonymised — no
categorical, free-text, or business-meaning fields exist for NADEEF-null,
AutoTest-format, or fuzzy-typo detectors to act on. The ensemble keeps only
the detector families that are well-defined on anonymised numeric data
(`credit_card_fraud/creditcard_error_detector.py`):

| # | Detector | Type | Threshold | Flags |
|---|---|---|---|---|
| D1 | `statistical_outlier_zscore` | behavioural | \|z\| > 4 | 36,781 |
| D2 | `statistical_outlier_iqr` | behavioural | 5.0× Tukey fence | 53,926 |
| D3 | `ml_outlier_isolation_forest` | behavioural | contamination=0.01, top-3 cell attribution | 8,547 |
| D4 | `duplicate_row` | **structural** | exact match on (V1-28, Amount) | 414,497 |

`Time` is excluded from cell-level flagging (a monotonic re-based index,
not a substantive value). The classic 1.5× Tukey IQR fence was tuned up to
5.0× after an initial run flagged 4.5% of all cells — these PCA components
are heavy-tailed, not normal, so the textbook multiplier over-fires badly;
5.0× brings the rate to ~0.65%, proportionate to eBay's ~3% raw behavioural
density.

**Structural vs. behavioural**, same principle as Chile Customs/eBay:
exact-duplicate rows are a data-export artefact, not a value any human
chose — excluded from intent labelling entirely (mask stays 0), same as
Chile's package-count/arithmetic checks and eBay's crawler-duplicate
detector.

## Intent labelling (direct, not a proxy)

For each **behavioural** flagged cell (z-score / IQR / Isolation-Forest):

```
Class == 1 (confirmed fraud)   -> +1  intentional
Class == 0 (confirmed legit)   -> -1  unintentional
```

`Class` is excluded from every feature/clean/dirty column — it is the
label, never an input, exactly like Chile Customs' `NU_REGR` exclusion.

Result: **65,434 labelled cells** — 3,503 intentional (5.35%), 61,931
unintentional (94.65%). 441 of 492 confirmed-fraud rows (89.6%) carry at
least one behavioural flag — substantially stronger detector-to-label
coupling than Chile Customs, where only 145 of 381 inspected declarations
(38%) contained a behavioural-anomaly cell at all. This is direct evidence
that fraud transactions in this dataset really do look statistically
anomalous in feature space, which is exactly the property that makes the
`Class`-implies-intent labelling defensible here in a way the
inspection-implies-intent labelling never fully was for Chile Customs.

## Validation sample (tractability)

29,428 of 284,807 rows carry at least one labelled cell. To keep the
clustering/RF pipelines tractable while preserving every labelled cell, we
keep all 29,428 + a `random_state=42` draw of 15,572 additional unflagged,
confirmed-legitimate rows (45,000 rows total — same order of magnitude as
Chile Customs' 49,689-row v1 sample), via
`credit_card_fraud/build_validation_sample.py`. All 65,434 labelled cells
are preserved exactly; only the unlabelled background population is
subsampled.

## Pseudo-clean reference

No oracle clean counterfactual exists (we don't know what a fraudulent
transaction's values "would have been" had it been legitimate). `clean` is
the global per-column median computed over the **confirmed-legitimate**
(Class==0) population, broadcast to every row
(`credit_card_fraud/build_pseudo_clean.py`). No categorical grouping key
exists in the anonymised PCA space, so this is the project's coarsest
fallback tier (cf. eBay/Chile's "global" last resort) — but, unlike
eBay/Chile, the **source population itself is oracle-correct** (genuinely
non-fraudulent transactions, not a constructed behavioural-tier proxy).
Only the row-personalisation is approximate.

## Declarative domain description (`info.txt`)

`error_detection_system/src/attribution/declarative/configs/credit_card_fraud.txt`
(copied to `NIYYAT/Datasets/CreditCardFraud/info.txt`) is written with an
explicit honesty constraint: V1–V28 have no disclosed business meaning, so
the description prohibits inventing per-column semantics or valid ranges.
The domain knowledge it *does* supply is general, well-documented
payment-fraud knowledge: the card-testing pattern (small Amount), the
cash-out pattern (large Amount), and — the strongest available signal —
multivariate co-occurrence (several Vn cells flagged simultaneously is more
fraud-consistent than one cell in isolation). Gemini 2.5 extracted 6
constraints from this description (`results/constraints/credit_card_fraud_constraints.json`),
covering exactly these three signals plus their negations (Time-only or
single-cell-only flags as accidental/noise).

## NIYYAT integration

Registered in the same three places every other real-data dataset is wired
into:
- `fraud_baseline/datasets.py` — `load_credit_card_fraud()`, key
  `credit_card_fraud` in `LOADERS`.
- `error_detection_system/src/attribution/heuristics/run_error_analysis.py`
  — `CCFRAUD_CFG`, added to `DATASETS` (Scenario B, DBSCAN clustering).
- `error_detection_system/src/attribution/declarative/run_all_declarative.py`
  and `.../no_clustering/run_all_no_clustering.py` — `DATASET_CONFIGS["credit_card_fraud"]`
  (Scenario C, B+C, and the no-clustering ablation).

Working files live under `credit_card_fraud/` (mirroring `chile_customs/`);
the dataset/mask copy for NIYYAT browsing lives under
`NIYYAT/Datasets/CreditCardFraud/`; results land under
`NIYYAT/Results/CreditCardFraud/` (mirroring `NIYYAT/Results/ChileCustoms/`).

LLM baselines (Llama/Qwen/Mixtral/R1-Qwen/Gemini zero-shot/few-shot/info)
were intentionally left for a later pass, per scope.
