# Error Fingerprint: Intent Attribution Without Correct Values

## Core Idea

Classify erroneous cells as **intentional** vs **unintentional** using only:
- The **dirty dataset** (with errors already in it)
- A **blind binary mask** (which cells are erroneous — no intent labels)

**No clean/correct values. No reference dataset.**

## Key Insight

Intentional and unintentional errors leave fundamentally different **fingerprints**
in the dirty data itself:

| Property | Unintentional | Intentional |
|----------|--------------|-------------|
| Value validity | Often invalid (typos, noise) | Valid OR deliberate obfuscation |
| Row coherence | Breaks internal consistency | Maintains/improves consistency |
| Value frequency | Rare/unusual values | Common/majority values |
| Feature targeting | Random features | Strategic features |
| Co-occurrence | Random combinations | Correlated patterns |

---

## Heuristic Groups — Clean Decomposition

There are **8 heuristics**. They are **not parallel** — they have a logical ordering
from cheap/surface-level to expensive/deep. Each heuristic answers a *different*
question about the erroneous cell. No two heuristics measure the same thing.

### Ordering and Dependencies

```
LEVEL 1: Surface — what does the dirty value LOOK like?
  ├─ H1: Value Plausibility     (is it a valid domain value?)
  ├─ H2: String Anomaly         (does it have typo/obfuscation patterns?)
  └─ H3: Distribution Position  (where does it sit in the column's distribution?)

LEVEL 2: Context — how does it relate to its ROW and COLUMN?
  ├─ H4: Row Coherence          (does it fit the rest of the row?)
  └─ H5: Error Pattern          (which/how many cells are flagged in this row?)

LEVEL 3: Domain — what do we know about the COLUMN itself?
  ├─ H6: Column Strategic Importance  (does this column drive the outcome?)
  ├─ H7: User Incentive              (would a rational human change this column?)
  └─ H8: Sensitivity Flag            (is this a protected/sensitive attribute?)
```

Level 1 features are **per-cell, stateless** — they only look at the dirty value
and column statistics. Level 2 features need the **full row and mask**. Level 3
features are **per-column constants** — they don't depend on the specific dirty
value at all, only on *which column* was changed.

All levels run in **one pass** (no sequential dependency between levels). The
ordering is conceptual, not computational.

---

### H1: Value Plausibility (per erroneous cell)

**Question**: Is this dirty value a legitimate member of this column's domain?

- **Vocabulary membership** (categorical): is the dirty value one of the values
  that appear in clean cells of this column? Binary: 1 = in-vocab, 0 = novel.
- **In-range** (numerical): is the value within [p5, p95] of the column? Binary.
- **Numeric parsability** (categorical column): can the value be parsed as a
  number when the column is categorical? (e.g., `"0"` in the `education` column)

> **What it captures**: Unintentional errors often produce values that don't exist
> in the domain (`Bachleors`, `Whitw`, `09` for age). Intentional errors typically
> use real domain values (`Bachelors`, `White`, `42` for age).
>
> **Where it fails**: 29% of intentional errors are obfuscation (`nan`, `Unknown`,
> `—`) which are also novel → H1 says "not in vocab" → misclassified. That's
> why H2 exists. Also, 7.5% of unintentional errors happen to be valid values →
> H1 says "in vocab" → misclassified. That's why H4 exists.

---

### H2: String Anomaly (per erroneous cell, categorical only)

**Question**: Does this string *look like* a typo or *look like* deliberate obfuscation?

This is NOT the same as H1. H1 asks "is this value in the vocabulary?" (binary).
H2 asks "what *kind* of invalid value is this?" (characterization of the error type).

- **Min edit distance to nearest valid value**: Levenshtein distance. Low distance
  (1-2) = likely typo. High distance = likely random garbage or different value.
- **Typo pattern score**: are there transpositions (`aMle`), doubled letters
  (`Malee`), dropped letters (`Blac`), keyboard-adjacent substitutions?
- **Obfuscation detection**: does the value match known obfuscation patterns?
  - Exact: `nan`, `N/A`, `Unknown`, `?`, `—`, `None`, `0`, `-1`
  - Suffix: `-DMV`, `-obf`, `-high`, `-1` appended to a valid value
  - Placeholder: value is a single character, or all punctuation
- **Length anomaly**: |len(dirty) - median_len(column)| / median_len(column)

> **Why H2 is NOT H1**: Consider `"Bachleors"` and `"Unknown"`. H1 says both are
> "not in vocabulary" (identical score). H2 says `"Bachleors"` has edit distance 1
> to `"Bachelors"` (typo) while `"Unknown"` matches an obfuscation pattern
> (deliberate). This distinction is **critical** — it separates unintentional typos
> from intentional obfuscation, both of which fail H1 the same way.

---

### H3: Distribution Position (per erroneous cell)

**Question**: Where does this dirty value sit in the column's overall distribution?

- **Z-score** (numerical): (value - mean) / std
- **Percentile rank** (numerical): what fraction of the column is below this value?
- **Frequency rank** (categorical): how common is this value? Rank 1 = most common.
- **Is majority class** (categorical): binary — is this the single most frequent value?
- **Distance from median** (numerical): |value - median| / IQR

> **What it captures**: Intentional gain-targeted errors tend to use common,
> unremarkable values (blend in). Unintentional noise tends to produce
> extreme or rare values.
>
> **Where it fails**: For `capital-gain`, both types are 91-93% in-range.
> Distribution features are nearly useless there. For categorical columns with
> many categories (occupation: 15 values), frequency rank has some power.

---

### H4: Row Coherence (per erroneous cell) ← THE KEY HEURISTIC

**Question**: Does this dirty value make sense given the rest of the row?

Train a per-column predictor on the **entire dirty dataset** (clean + erroneous rows).
For each flagged cell, mask that cell and predict it from the remaining features.

- **Prediction match** (categorical): does the predictor's top-1 prediction match
  the dirty value? Binary.
- **Prediction probability** (categorical): classifier's confidence for the dirty
  value's class. Continuous [0,1].
- **Prediction residual** (numerical): |dirty_value - predicted_value| / column_std.
- **Rank of dirty value in prediction** (categorical): if the predictor ranks all
  possible values, where does the dirty value fall? Top-3 = coherent.

> **Why this is the most important heuristic**: This is the signal that separates
> the approach from "just a typo detector." When both types have valid values
> (the 7.5% unintentional overlap + all numerical errors), H1/H2/H3 are blind.
> H4 asks: "does `education=Doctorate` make sense for a 23-year-old part-time
> worker?" If no → likely unintentional (random valid substitution). If yes →
> likely intentional (chosen to be plausible).
>
> **Training on dirty data is OK**: The predictors learn the *majority pattern*
> (which is still the clean data — errors are sparse). A few percent of erroneous
> cells don't corrupt the learned column relationships.

---

### H5: Error Pattern (per erroneous row)

**Question**: What does the *pattern* of errors in this row tell us?

- **Error count**: how many cells are flagged in this row?
- **Error column set**: one-hot encoding of which columns are erroneous.
- **Multi-error indicator**: binary — more than one cell flagged in this row?
- **Co-dependent columns changed**: are semantically linked columns both flagged?
  (e.g., `education` AND `education-num` both erroneous → coordinated change)

> **What it captures**: Intentional manipulation often changes multiple related
> columns together (education + education-num, or multiple columns in a group
> attack). Unintentional errors are typically isolated typos in single cells.

---

### H6: Column Strategic Importance (per column, constant for all errors in that column)

**Question**: Does this column have high predictive power for the outcome?

A rational attacker targets features that *matter* for the decision. We measure
how much each column contributes to predicting the outcome — without a separate
downstream model, just from the dirty data itself.

- **Mutual information with target** (if target column exists): MI(column, target).
  Normalized to [0,1].
- **Max MI with any other column** (unsupervised fallback): for each column,
  compute MI with every other column. Take the max. High-MI columns carry
  more information → more attractive targets.
- **Variance-based importance**: for numerical columns, normalized variance.
  For categorical, number of unique values / total rows.

> **What it captures**: Errors in high-importance columns (education, occupation,
> hours-per-week) are more likely intentional than errors in low-importance columns
> (fnlwgt, native-country).
>
> **Limitation**: This is a per-column constant. It cannot distinguish two errors
> in the same column. It's a prior, not a discriminator.

---

### H7: User Incentive (per column, constant for all errors in that column)

**Question**: Would a *rational human* want to change this column?

This is **NOT** the same as H6. H6 is statistical (MI with outcome). H7 is
behavioral/economic (human motivation).

Example of the difference:
- `fnlwgt` might have high MI with income (H6 says important), but NO user even
  knows what fnlwgt means, let alone would manipulate it (H7 says low incentive).
- `education` has moderate MI (H6), but EVERY applicant understands that a higher
  degree helps their application (H7 says high incentive).
- `race` has low MI (H6), but minorities may change it to avoid discrimination
  (H7 says high incentive for fairness-triggered masking).

We operationalize this as:
- **Mutability score**: is this column user-declared (mutable) or system-controlled
  (immutable)? From the paper's Table 3 categorization:
  - Immutable (0.0): tax_id, credit_score, fnlwgt — system-generated
  - Softly immutable (0.5): race, sex, age — editable but constrained
  - Mutable (1.0): income, education, hours, occupation — user-declared
- **Gain direction**: for the target class, does increasing/decreasing this column
  correlate with a favorable outcome? Columns with clear gain direction have
  higher manipulation incentive.
- **User comprehensibility**: is this column's name/meaning understandable to a
  layperson? (Simple heuristic: column name length, presence in common vocabulary.)
  A column called `fnlwgt` is incomprehensible → low incentive. A column called
  `education` is obvious → high incentive.

> **Why this is separate from H6**: A reviewer could say "H6 already captures which
> columns matter." The answer: H6 captures what matters *to the model*. H7 captures
> what matters *to the user*. These diverge. `fnlwgt` matters to the model but not
> to the user. `race` matters to the user (fairness fear) but may have low MI.
> The *gap* between H6 and H7 is itself informative: high H6 + low H7 = unlikely
> intentional (the user wouldn't know to target this). Low H6 + high H7 = possible
> intentional (user attacked a field they *thought* mattered, but it doesn't).

---

### H8: Sensitivity Flag (per erroneous cell)

**Question**: Is this error in a protected/sensitive attribute?

- **Sensitive column flag**: binary — is this column in a known sensitive list?
  Auto-detected from column names matching: race, gender, sex, age, nationality,
  religion, disability, marital-status. Or user-supplied.
- **Majority-class value** (categorical): is the dirty value the majority class
  of this sensitive column? (e.g., `White` for race, `Male` for sex)

> **What it captures**: Fairness-triggered masking — users changing sensitive
> attributes to avoid discrimination. A specific sub-type of intentional error.
>
> **Limitation**: In the data, intentional sex errors are 54% Male / 46% Female
> — the majority-direction signal is weak for sex. Stronger for race (72% White).
> This is one feature among many, not a hard rule.

---

## Combination Strategy: How Heuristics Become a Prediction

### What each erroneous cell gets

8 heuristics. Each produces **one or two features** — the minimum needed to
capture the signal without redundancy. No bloat.

```
Cell e_i → [ H1:1 | H2:2 | H3:1 | H4:1 | H5:2 | H6:1 | H7:3 | H8:2 ] = 13 features
```

| H | Features | Count |
|---|----------|-------|
| H1 | `plausible` (binary: in-vocab or in-range, type-dependent) | 1 |
| H2 | `min_edit_distance` (continuous), `is_obfuscation` (binary) | 2 |
| H3 | `zscore_or_freq_rank` (continuous: z-score for numerical, frequency rank for categorical) | 1 |
| H4 | `coherence_score` (continuous: prediction probability for categorical, 1-residual for numerical) | 1 |
| H5 | `error_count` (integer), `codependent_flag` (binary) | 2 |
| H6 | `column_importance` (continuous: MI with target, or max MI) | 1 |
| H7 | `mutability` (continuous), `gain_direction` (continuous), `comprehensibility` (continuous) | 3 |
| H8 | `is_sensitive` (binary), `is_majority_value` (binary) | 2 |

**Total: 13 features per erroneous cell. Each one is non-redundant.**

Why 13, not fewer?
- H7 has 3 because mutability, gain direction, and comprehensibility are genuinely
  independent concepts (a column can be mutable but incomprehensible, like fnlwgt).
- H2 has 2 because edit distance (typo signal) and obfuscation detection (deliberate
  signal) point in *opposite directions* — they cannot be collapsed to one number.
- H5 has 2 because error count (how many) and codependency (which ones) are different.
- Everything else: 1 feature per heuristic. The minimal representation.

### Three options for combining them (and why we pick the hybrid)

**Option A: Raw features → RF directly.**
Feed all 13 features into a Random Forest. The RF learns optimal weighting.
✅ Best accuracy. ❌ No per-heuristic interpretability. ❌ "You just trained an RF."

**Option B: One score per heuristic → weighted sum → threshold.**
Collapse each H to a scalar h_i ∈ [0,1], then I(e) = Σ w_i · h_i.
✅ Interpretable. ❌ How to set weights without supervision? ❌ Loses nonlinear
interactions (e.g., "H1=valid AND H4=incoherent" is invisible in a sum).

**Option C (CHOSEN): Hybrid — RF on raw features + per-heuristic scores for explanation.**

```
                    ┌─────────────────────────────────────┐
                    │  Per erroneous cell e_i:            │
                    │                                     │
                    │  Raw features (13 dims)             │
                    │       │              │              │
                    │       ▼              ▼              │
                    │   [RF Classifier]  [Per-H Scores]   │
                    │       │              │              │
                    │       ▼              ▼              │
                    │  ŷ ∈ {-1, +1}    h₁...h₈ ∈ [0,1]  │
                    │  P(intent) ∈ [0,1]                  │
                    │                                     │
                    │  Output diagnostic tuple:           │
                    │  ⟨column, I(e_i), h₁...h₈, ŷ⟩     │
                    └─────────────────────────────────────┘
```

The RF does the heavy lifting (classification + I(e_i) via `predict_proba`).
The per-heuristic scores h₁...h₈ are for **interpretability and explanation** —
they tell the user *why* a cell was flagged as intentional.

### How per-heuristic scores are computed

Each heuristic's features are collapsed to a single scalar h_i ∈ [0,1] that
represents "how much does this heuristic point toward intentional?"

| Heuristic | Score formula | Intentional direction |
|-----------|--------------|----------------------|
| H1 | `plausible` (already binary) | High = intentional (valid value) |
| H2 | `is_obfuscation + max(0, 1 - min_edit_dist/5)` / 2 | High = intentional (obfuscation or close to valid) |
| H3 | `1 - abs(zscore)/max_z` or `1 - freq_rank/max_rank` | High = intentional (common value) |
| H4 | `coherence_score` (already [0,1]) | High = intentional (fits the row) |
| H5 | `(error_count > 1) * codependent_flag` | High = intentional (coordinated) |
| H6 | `column_importance` (already [0,1]) | High = intentional (strategic column) |
| H7 | `mean(mutability, gain_direction, comprehensibility)` | High = intentional (user would target) |
| H8 | `is_sensitive * is_majority_value` | High = intentional (fairness masking) |

These are simple, transparent formulas. They are NOT used for classification —
only for explanation. The RF uses the 13 raw features directly.

### Why not learn the weights?

The paper (§4.1.2) proposes adaptive weight tuning via feedback from model
correction. We cannot do this because we don't have a downstream model in the loop.
Instead:

- **Classification**: the RF implicitly learns the optimal nonlinear weighting
  from the labeled training sample over the 13 features. This is strictly better
  than any linear weighted sum.
- **Explanation**: the per-heuristic scores use fixed, domain-grounded formulas.
  No learned weights needed — the formulas encode the *direction* of each signal
  (what does "high" mean for each heuristic).
- **Feature importance**: the RF's feature importances tell us post-hoc which
  raw features (and implicitly which heuristics) contributed most. This is our
  empirical answer to "which heuristic matters."

### The weight question is actually the wrong question

A strict professor would say: "Why are you asking about weights? The Random Forest
IS the weighting mechanism. It learns a nonlinear, interaction-aware weighting
function from data. That's the whole point of using a tree ensemble instead of a
linear model."

The only weights we need to set are:
1. The RF hyperparameters (n_estimators, max_depth) → standard CV tuning.
2. The per-heuristic score formulas (for explanation) → fixed by design, not learned.

---

## Architecture

```
Dirty Dataset + Blind Mask
        │
        ├─→ [fit() phase]
        │    ├─ Learn column stats, vocabularies           (for H1, H2, H3)
        │    ├─ Train per-column predictors                (for H4)
        │    ├─ Compute MI, mutability, sensitivity flags  (for H6, H7, H8)
        │    └─ Detect co-dependent column pairs           (for H5)
        │
        ├─→ [extract() phase]  (H1-H8, single pass over erroneous cells)
        │    │
        │    │  LEVEL 1: Surface (per-cell, stateless)
        │    ├─ H1: Value Plausibility        (in-vocab? in-range?)
        │    ├─ H2: String Anomaly            (typo vs. obfuscation patterns)
        │    ├─ H3: Distribution Position     (z-score, frequency, percentile)
        │    │
        │    │  LEVEL 2: Context (per-row, needs mask + predictors)
        │    ├─ H4: Row Coherence             (predicted vs. actual dirty value)
        │    ├─ H5: Error Pattern             (multi-error, co-dependent columns)
        │    │
        │    │  LEVEL 3: Domain (per-column constants, already computed in fit)
        │    ├─ H6: Column Strategic Importance  (MI with outcome)
        │    ├─ H7: User Incentive              (mutability, gain direction)
        │    └─ H8: Sensitivity Flag            (protected attribute?)
        │         │
        │         ▼
        │   Feature Matrix: 13 raw features per erroneous cell
        │   + 8 heuristic scores (for explanation)
        │         │
        ├─→ [Clustering]  (cluster erroneous cells by raw feature fingerprint)
        │         │
        │         ▼
        │   [Sample ~1% from clusters proportionally]
        │         │
        │         ▼
        │   [Human labels on sample]  ← or ground-truth labels for evaluation
        │         │
        │         ▼
        └─→ [Train RF Classifier on 13 raw features]
                  │
                  ▼
            For each erroneous cell:
            ⟨column, ŷ, I(e_i) = P(intent), h₁...h₈⟩
```

## Non-Redundancy Proof

Every heuristic must answer a question that NO other heuristic answers.

| Heuristic | Question | Why no other H answers this |
|-----------|----------|----------------------------|
| H1 | Is this a valid domain value? | H2 characterizes *invalid* values but doesn't test membership. |
| H2 | What *kind* of invalid value? | H1 only says valid/invalid. H2 says typo vs. obfuscation. |
| H3 | Is this value common or rare? | H1 tests existence; H3 tests frequency/position. |
| H4 | Does this value fit the row? | H1-H3 are column-only. H4 uses cross-column prediction. |
| H5 | Is this error isolated or patterned? | H1-H4 are per-cell. H5 is per-row. |
| H6 | Is this column statistically important? | H7 measures *human* importance, not statistical. |
| H7 | Would a human want to change this? | H6 measures MI; H7 measures mutability/comprehensibility. |
| H8 | Is this a protected attribute? | A column can be high-MI (H6), mutable (H7), but not sensitive. |

## Deferred Heuristics (require downstream ML model)

- **Causal Impact** (paper §4.1.1): does changing `e_i` flip the model prediction?
- **Group Shift Potential**: does the error push across a decision boundary?
- **Effort-Based Perturbation Size**: minimal perturbation for outcome shift
- **Peer-Group Deviation** (paper §4.1.1 "Consistency"): deferred — expensive KNN
  on mixed-type data, advantage over H4 unproven empirically.

---

## Critical Self-Assessment

### Where Each Heuristic Fails (data-verified on Adult Income LLM dataset)

| Heuristic | Failure mode | Severity |
|-----------|-------------|----------|
| H1 | 29% of intentional are obfuscation (novel) → H1 says "invalid" | HIGH — rescued by H2 |
| H1 | 7.5% of unintentional are valid values → H1 says "valid" | MEDIUM — rescued by H4 |
| H2 | Only works for categorical columns | MEDIUM |
| H3 | capital-gain: both types 91-93% in-range → H3 blind | HIGH for that column |
| H4 | Predictor quality depends on feature correlations | MEDIUM |
| H5 | Co-dependent detection is dataset-specific | LOW |
| H6 | Per-column constant — no per-cell discrimination | BY DESIGN |
| H7 | Requires domain config or heuristic approximation | LOW |
| H8 | sex: 54%/46% majority direction — weak signal | MEDIUM |

### Power Ranking (honest estimate)

| Rank | Heuristic | Why |
|------|-----------|-----|
| 1 | H2: String Anomaly | Typo patterns are nearly perfect unintentional markers (92.5%) |
| 2 | H1: Value Plausibility | Strong binary separator, fails on obfuscation + numerical |
| 3 | H4: Row Coherence | **The contribution** — handles all ambiguous cases |
| 4 | H3: Distribution | Useful for categorical, weak for some numerical columns |
| 5 | H5: Error Pattern | Helps for coordinated attacks |
| 6 | H7: User Incentive | Behavioral prior — complements H6 |
| 7 | H6: Strategic Importance | Statistical prior — per-column constant |
| 8 | H8: Sensitivity | Weak standalone, strong when combined with H6+H7 |
