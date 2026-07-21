# Heuristics Analysis — Error Fingerprint Pipeline

**Run:** `run_20260320_165116` (full-label evaluation, 5-fold CV)  
**Datasets:** LLM Adult Income (tenth-trial) · Kireev Mixed Error Pipeline  
**Pipeline:** H1–H8 → 13-feature matrix → RandomForest classifier

---

## Overall Pipeline Performance

| Metric | LLM Dataset | Kireev Dataset |
|---|---|---|
| Accuracy | 0.9205 ± 0.0014 | 0.9896 ± 0.0014 |
| Precision | 0.9128 ± 0.0027 | 0.9844 ± 0.0035 |
| Recall | 0.8945 ± 0.0023 | 0.9841 ± 0.0045 |
| **F1** | **0.9035 ± 0.0017** | **0.9842 ± 0.0022** |
| AUC | 0.9729 ± 0.0004 | 0.9984 ± 0.0004 |

The combined 13 features produce strong, consistent classification on both datasets.

---

## Feature Importance Rankings

Rankings come from the Random Forest's `feature_importances_` averaged across 5 folds on the last fold's model.

### Raw feature-level importance

| Rank (avg) | Feature | Heuristic | LLM | Kireev | **Average** |
|---|---|---|---|---|---|
| 1 | `h3_distribution_score` | H3 | 0.2587 | 0.1364 | **0.1976** |
| 2 | `h1_plausible` | H1 | 0.1800 | 0.1653 | **0.1727** |
| 3 | `h7_gain_direction` | H7 | 0.0550 | 0.2131 | **0.1341** |
| 4 | `h4_coherence_score` | H4 | 0.1830 | 0.0576 | **0.1203** |
| 5 | `h2_is_obfuscation` | H2 | 0.1382 | 0.0891 | **0.1137** |
| 6 | `h5_error_count` | H5 | 0.0384 | 0.1329 | **0.0856** |
| 7 | `h6_column_importance` | H6 | 0.0220 | 0.1283 | **0.0751** |
| 8 | `h2_min_edit_distance` | H2 | 0.0908 | 0.0252 | **0.0580** |
| 9 | `h7_comprehensibility` | H7 | 0.0135 | 0.0205 | **0.0170** |
| 10 | `h5_codependent_flag` | H5 | 0.0102 | 0.0193 | **0.0148** |
| 11 | `h8_is_sensitive` | H8 | 0.0038 | 0.0056 | **0.0047** |
| 12 | `h7_mutability` | H7 | 0.0038 | 0.0054 | **0.0046** |
| 13 | `h8_is_majority_value` | H8 | 0.0025 | 0.0012 | **0.0019** |

### Aggregated per-heuristic importance

| Rank | Heuristic | LLM total | Kireev total | **Avg total** |
|---|---|---|---|---|
| 1 | **H3** Distribution Position | 0.2587 | 0.1364 | **0.1976** |
| 2 | **H1** Value Plausibility | 0.1800 | 0.1653 | **0.1727** |
| 3 | **H2** String Anomaly | 0.2291 | 0.1143 | **0.1717** |
| 4 | **H7** User Incentive | 0.0724 | 0.2390 | **0.1557** |
| 5 | **H4** Row Coherence | 0.1830 | 0.0576 | **0.1203** |
| 6 | **H5** Error Pattern | 0.0486 | 0.1522 | **0.1004** |
| 7 | **H6** Column Importance | 0.0220 | 0.1283 | **0.0751** |
| 8 | **H8** Sensitivity Flag | 0.0063 | 0.0068 | **0.0065** |

---

## Heuristic-by-Heuristic Analysis

---

### 🥇 H3 — Distribution Position  `[avg: 0.198]`

**What it does:** Measures how *central vs. extreme* the erroneous value is within its column's statistical distribution. A value near the column mean/mode scores ~1.0 ("blends in" → intentional). A value in the distribution's tails scores ~0.0 (unusual → unintentional).

For numerical columns the score is derived from the percentile rank; for categorical columns from the value's frequency in the clean reference.

**Why it is the best single feature:**
- Intentional manipulators blend their changes into the "normal" range to avoid detection.
- Unintentional errors (OCR noise, random substitution, encoding corruption) land in the tails.
- This signal is **dataset-agnostic** — it works on any column type without needing ground-truth labels.
- It is the **#1 feature on the LLM dataset** (0.259) and **#3 on Kireev** (0.136).

**Output feature:** `h3_distribution_score` ∈ [0, 1] — continuous, high = intentional.

---

### 🥈 H1 — Value Plausibility  `[avg: 0.173]`

**What it does:** Checks whether the erroneous cell value belongs to the column's *known domain*, learned from the dirty dataset (errors are sparse enough not to corrupt domain estimates).

- **Categorical:** 1 if the dirty value is in the clean vocabulary, 0 otherwise.
- **Numerical:** 1 if the value is within [p5, p95] of the clean distribution, 0 otherwise.

**Why it ranks 2nd:**
- Out-of-vocabulary / out-of-range values are a very strong signal for unintentional errors (random noise, encoding corruption).
- Intentional manipulators almost always use plausible values — the whole point is to not look wrong.
- Consistent across both datasets (0.180 LLM / 0.165 Kireev): the most *stable* single feature.

**Known limitation:** ~29% of intentional errors in LLM data use obfuscation tokens (`nan`, `Unknown`, `—`) which are also out-of-vocabulary and will be scored 0 by H1. H2 is responsible for rescuing those cases.

**Output feature:** `h1_plausible` ∈ {0, 1} — binary.

---

### 🥉 H2 — String Anomaly  `[avg: 0.172]`  *(2 features)*

**What it does:** Characterises *how wrong* a categorical string is — distinguishing typos from deliberate obfuscation.

- `h2_min_edit_distance`: Levenshtein distance to the nearest clean-vocabulary entry. A typo has distance 1–2; an obfuscation token like `"Unknown"` is far from any real value.
- `h2_is_obfuscation`: 1 if the dirty value matches known placeholder patterns (`nan`, `Unknown`, `—`, suffix-mangled values like `"Private-DMV"`).

**Why it matters:**
- It directly distinguishes the two main *kinds* of intentional error (obfuscation tokens vs. deliberate vocab choices) from unintentional typos.
- Applies only to categorical columns; NaN is emitted for numerical columns.
- `h2_is_obfuscation` is the **4th most important feature on LLM** (0.138) — LLM-generated errors frequently use placeholder tokens.
- `h2_min_edit_distance` drops on Kireev (0.025) because Kireev errors are structural rather than string-substitution.

**Output features:** `h2_is_obfuscation` ∈ {0, 1, NaN}, `h2_min_edit_distance` ∈ [0, 10, NaN].

---

### H4 — Row Coherence  `[avg: 0.120]`

**What it does:** Trains one Random Forest predictor per column (using all other columns as features), then measures how well the dirty value *fits the row context*.

- Score ≈ 1.0 → value is exactly what the row predicts → **intentional** (the manipulator kept things coherent).
- Score ≈ 0.0 → value breaks the row-level correlation structure → **unintentional** (random noise).

**Why it matters:**
- Captures cross-column coherence — a signal that neither H1 (domain) nor H3 (distribution) can provide alone.
- **#2 on LLM dataset** (0.183): LLM-generated intentional errors carefully maintain row coherence.
- Drops to 0.058 on Kireev because Kireev errors have a different pattern (structural/row-level injections that sometimes break coherence by design).

**Output feature:** `h4_coherence_score` ∈ [0, 1] — continuous.

---

### H7 — User Incentive  `[avg: 0.156]`  *(3 features)*

**What it does:** Measures *behavioral motivation* — would a rational human actor want to change this column?

Three orthogonal sub-signals:
- `h7_gain_direction` (+1/0/-1): does changing this value move the outcome favourably?
- `h7_comprehensibility` (0/1): is this column something a typical person would understand?
- `h7_mutability` (0/1): can a human realistically change this value (vs. immutable facts like SSN)?

**Critical distinction from H6:** H7 is behavioral, H6 is statistical. The column `fnlwgt` has high MI with the target (H6 high) but nobody touches it because nobody understands it (H7 low). The column `race` has low MI (H6 low) but is frequently masked for privacy reasons (H7 high).

**Why it matters:**
- `h7_gain_direction` is the **#1 feature on Kireev** (0.213): Kireev's error injection is strongly outcome-directed, making "gain direction" extremely discriminative there.
- Drops on LLM (0.055) because LLM-generated errors follow a more diverse distribution of intent.

**Output features:** `h7_gain_direction`, `h7_comprehensibility`, `h7_mutability`.

---

### H5 — Error Pattern  `[avg: 0.100]`  *(2 features)*

**What it does:** Detects coordinated multi-cell edits — a hallmark of intentional manipulation where a person edits logically linked columns together.

- `h5_error_count`: how many erroneous cells are in the same row? High count → coordinated edit.
- `h5_codependent_flag`: does this cell's logically linked partner column (e.g. `education` ↔ `education-num`) also have an error in the same row?

**Why it matters:**
- Random noise hits columns independently; intentional manipulators edit coherent clusters.
- **#4 on Kireev** (0.133): Kireev errors are injected in coordinated multi-column bursts.
- Weaker on LLM (0.049): LLM-generated errors are often isolated single-cell modifications.

**Output features:** `h5_error_count` ∈ int≥1, `h5_codependent_flag` ∈ {0, 1}.

---

### H6 — Column Importance  `[avg: 0.075]`

**What it does:** Measures the *statistical importance* of the column (mutual information with the target outcome), normalised to [0, 1]. This is a per-column constant — every erroneous cell in the same column gets the same score.

**Important:** This is statistical importance, not human behavioral motivation. A high-MI column is an attractive attack target from a model-gaming perspective.

**Why it ranks 7th:**
- Per-column constants are less discriminative than per-cell signals — they can't distinguish two erroneous cells in the same column from each other.
- **#5 on Kireev** (0.128): Kireev errors are concentrated in high-MI columns, making this signal useful.
- Weak on LLM (0.022): LLM-generated errors are spread across columns of varying importance.

**Output feature:** `h6_column_importance` ∈ [0, 1] — per-column constant.

---

### H8 — Sensitivity Flag  `[avg: 0.007]`  ⚠️ Least Important

**What it does:** Detects privacy-motivated demographic masking — a user changes a minority demographic value to the majority class to "blend in".

Two binary signals:
- `h8_is_sensitive`: is this a sensitive demographic column (race, sex, age, etc.)?
- `h8_is_majority_value`: is the dirty value the majority class of that sensitive column?

**Why it is the weakest heuristic:**
- Both features are per-column constants (or near-constants for majority-class detection), providing almost no within-column discrimination.
- Sensitive columns (`race`, `sex`) are only 2 out of 14 columns — for the other 12, both features are always 0.
- The privacy-masking pattern it targets is real but **rare** relative to the full error corpus.
- `h8_is_majority_value` is the single **least important feature** across both datasets (0.0025 LLM / 0.0012 Kireev / 0.0019 avg).
- **Known limitation:** The majority-class signal is weak when the majority is not dominant (`sex` at 67% Male). Only works well for highly skewed sensitive columns (`race` at 86% White).

**Output features:** `h8_is_sensitive` ∈ {0, 1}, `h8_is_majority_value` ∈ {0, 1}.

---

## Summary Table

| Heuristic | Avg Importance | Rank | Verdict | Key Strength |
|---|---|---|---|---|
| **H3** Distribution Position | 0.198 | 🥇 1st | **Best overall** | Distribution-agnostic blending signal |
| **H1** Value Plausibility | 0.173 | 🥈 2nd | **Most stable** | Consistent across both datasets |
| **H2** String Anomaly | 0.172 | 🥉 3rd | **Best for LLM** | Distinguishes typos from obfuscation tokens |
| **H7** User Incentive | 0.156 | 4th | **Best for Kireev** | Behavioral motivation, outcome direction |
| **H4** Row Coherence | 0.120 | 5th | **Best for LLM (row-level)** | Cross-column coherence |
| **H5** Error Pattern | 0.100 | 6th | Good | Coordinated multi-cell edit detection |
| **H6** Column Importance | 0.075 | 7th | Moderate | Statistical importance of the target column |
| **H8** Sensitivity Flag | 0.007 | 8th | **Weakest** | Privacy masking (rare, per-column constant) |

---

## Key Observations

### 1. No single heuristic dominates both datasets
- H3 is #1 on LLM but only #3 on Kireev.
- H7 (`gain_direction`) is #1 on Kireev but only #6 on LLM.
- This cross-dataset complementarity is exactly why combining all 8 into a 13-feature vector works better than any individual signal.

### 2. LLM vs. Kireev emphasise different signals

| Signal type | LLM weight | Kireev weight | Winner |
|---|---|---|---|
| String/value-level (H1, H2, H3) | ~67% combined | ~35% combined | LLM |
| Behavioral/structural (H5, H6, H7) | ~13% combined | ~58% combined | Kireev |

LLM errors look "realistic" at the value level — they choose plausible-looking strings — so distribution and vocabulary checks are most useful. Kireev errors are more systematic and outcome-directed — they modify specific columns with clear gain direction — so behavioral and structural signals dominate.

### 3. Semi-supervised degradation is heuristic-driven
The label propagation methods (Scenario B) underperformed RF primarily because `h3_distribution_score` and `h4_coherence_score` are **continuous features** in a high-dimensional space. Label propagation via kNN struggles to propagate through continuous manifolds with only 1% labeled points. RF avoids this by learning non-linear splits.

### 4. H8 should not be removed (yet)
Despite its low importance score, H8 captures a real-world privacy motivation that other heuristics miss. It provides small but non-zero signal specifically for demographic-sensitive error injection scenarios. Removing it would shave 2 features at negligible accuracy cost (~0.006 F1 gain or loss).

---

## Feature Importance Chart (ASCII)

```
Feature                  | LLM    | Kireev | Avg    | Bar (avg)
-------------------------|--------|--------|--------|-------------------------
h3_distribution_score    | 0.259  | 0.136  | 0.198  | ████████████████████
h1_plausible             | 0.180  | 0.165  | 0.173  | █████████████████
h7_gain_direction        | 0.055  | 0.213  | 0.134  | █████████████
h4_coherence_score       | 0.183  | 0.058  | 0.120  | ████████████
h2_is_obfuscation        | 0.138  | 0.089  | 0.114  | ███████████
h5_error_count           | 0.038  | 0.133  | 0.086  | ████████
h6_column_importance     | 0.022  | 0.128  | 0.075  | ███████
h2_min_edit_distance     | 0.091  | 0.025  | 0.058  | █████
h7_comprehensibility     | 0.013  | 0.021  | 0.017  | █
h5_codependent_flag      | 0.010  | 0.019  | 0.015  | █
h8_is_sensitive          | 0.004  | 0.006  | 0.005  | ▌
h7_mutability            | 0.004  | 0.005  | 0.005  | ▌
h8_is_majority_value     | 0.003  | 0.001  | 0.002  | ▏
```

---

*Generated: 2026-03-20 | Run: run_20260320_165116 | 5-fold CV on LLM (43,259 errors) + Kireev (11,661 errors)*
