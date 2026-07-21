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
| Value validity | Often invalid (typos, noise) | Always valid (real domain values) |
| Row coherence | Breaks internal consistency | Maintains/improves consistency |
| Value frequency | Rare/unusual values | Common/majority values |
| Feature targeting | Random features | Strategic features (sensitive, high-importance) |
| Co-occurrence | Random combinations | Correlated patterns (e.g., education + education-num) |

## Feature Groups

### Group 1: Value Plausibility (per erroneous cell)
- Is the dirty value a member of the known vocabulary for this column?
- String similarity to nearest valid value (catches typos like "Bachleors")
- For numerical: is the value within [p5, p95] of the column?
- For categorical: frequency rank of this value in the column

### Group 2: Row-Level Coherence (per erroneous cell)
- Train a per-column predictor on the ENTIRE dirty dataset
- For each flagged cell: how well does the dirty value match what the
  rest of the row predicts?
- Intentional errors → HIGH coherence (value was chosen to be plausible)
- Unintentional errors → LOW coherence (value is random/broken)

### Group 3: Error Pattern (per erroneous record)
- Number of erroneous cells in this row
- Which features are erroneous (one-hot pattern)
- Are co-dependent features changed together? (e.g., education + education-num)

### Group 4: Value Distribution (per erroneous cell)
- Z-score of the dirty value within column
- Percentile rank
- Is this a majority-class value? (for categorical)
- For numerical: distance from column median

### Group 5: String-Level (per erroneous cell, categorical only)
- Edit distance from nearest valid vocabulary entry
- Contains typo-like patterns (transpositions, keyboard adjacency)
- Contains obfuscation tokens ("Unknown", "N/A", "—")
- String length anomaly

### Group 6: Column Strategic Importance (per erroneous cell) ← from paper §4.1.1
The paper argues that high-importance features are *more likely targets* for
intentional manipulation — a rational actor attacks features that matter.

We operationalize this **without** a downstream ML model:
- **Outcome correlation**: mutual information between the erroneous column and
  the outcome/target column (if present), or between all column pairs
  (unsupervised proxy: MI with the most correlated other column)
- **Variance-based importance**: columns with high variance and many unique
  values carry more information and are more attractive manipulation targets
- **Mutable vs. immutable flag**: user-supplied config list of columns known to
  be user-controlled (e.g., income, education, hours) vs. system-locked
  (e.g., tax-id, credit-score). Errors in mutable columns → higher intent prior.

> **Why it helps**: even without knowing the correct value, we know *which column
> was changed*. If that column is education, income, or race — all high-MI,
> user-declared fields — that is evidence of strategic choice, not accident.

### Group 7: Peer-Group Deviation (per erroneous cell) ← from paper §4.1.1 "Consistency"
Find the k nearest **clean** records (by all non-erroneous features in the same row)
and measure how much the erroneous value deviates from the peer group's value for
that same column.

- **Peer mean/median deviation**: |dirty_value − median(peer_values)| / std(peer_values)
- **Peer vocabulary match**: does the dirty value appear at all in the peer group's
  values for this column? (categorical)
- **Directional deviation**: is the dirty value *above* the peer median? (captures
  inflation patterns like income falsification)
- **Intra-group consistency**: if multiple records in the same row-cluster share the
  same erroneous column, do their dirty values shift in the same direction?
  (captures coordinated group-level manipulation from paper §3.2.2)

> **Why it helps**: row coherence (Group 2) asks "does this cell fit its own row?"
> Peer-group deviation asks "does this cell fit records that look like this row?"
> These are complementary signals. A gain-targeted falsification will *improve*
> row coherence (it was chosen to be plausible) but will *increase* peer deviation
> (it pushes above what similar people report).

> **Key distinction from Group 2**: Group 2 uses a per-column predictor trained
> on ALL rows. Group 7 uses only the k most similar clean rows. Group 2 captures
> global consistency; Group 7 captures local/contextual deviation.

### Group 8: Sensitivity Flag (per erroneous cell) ← from paper §4.1.1 "Minority-Sensitive"
The paper explicitly identifies errors in *sociologically sensitive attributes*
(race, sex, age, nationality, disability) as a distinct sub-type of intentional
error: **fairness-triggered masking** (a user hides a sensitive attribute to avoid
discrimination).

- **Sensitive column flag**: binary — is this column in a user-supplied or
  auto-detected sensitive list?
- **Auto-detection**: columns whose names match known sensitive keywords
  (race, gender, sex, age, nationality, religion, disability, marital-status)
- **Masking direction**: for sensitive categorical columns, does the dirty value
  move toward the majority class? (e.g., Female→Male, Black→White)
  This is the hallmark of fairness-triggered masking.
- **Obfuscation toward outlier**: does the dirty value move toward an extreme
  numerical outlier? (e.g., age=999, income=0) — privacy obfuscation pattern
  from paper §3.2.1

> **Why it's separate from importance**: a column can be sensitive (race) without
> being the highest-MI predictor, and can be high-MI without being sensitive (hours).
> Both signals are needed. Their *combination* is especially strong: an error in a
> column that is *both* sensitive and high-MI is a very strong intentional signal.

## Architecture

```
Dirty Dataset + Blind Mask
        │
        ├─→ [Feature Extraction]  (Groups 1-8)
        │    ├─ G1: Value Plausibility        (is the dirty value even valid?)
        │    ├─ G2: Row-Level Coherence       (does it fit its own row?)
        │    ├─ G3: Error Pattern             (which/how many cells changed?)
        │    ├─ G4: Value Distribution        (z-score, percentile, rarity)
        │    ├─ G5: String-Level              (typo patterns, obfuscation)
        │    ├─ G6: Column Strategic Importance  (MI with outcome, mutable flag)
        │    ├─ G7: Peer-Group Deviation      (vs. k-nearest clean neighbors)
        │    └─ G8: Sensitivity Flag          (sensitive column + masking direction)
        │         │
        │         ▼
        │   Feature Matrix (one row per erroneous cell)
        │         │
        ├─→ [Clustering]  (cluster erroneous cells by fingerprint)
        │         │
        │         ▼
        │   [Sample ~1% from clusters proportionally]
        │         │
        │         ▼
        │   [Human labels on sample]  ← or ground-truth labels for evaluation
        │         │
        │         ▼
        └─→ [Train RF Classifier]  → predict intent for ALL erroneous cells
```

## Why This Works

The classifier never sees the correct value. Instead it learns:

| Signal | What it catches |
|--------|----------------|
| G1: Value Plausibility | Typos/noise → invalid values → unintentional |
| G2: Row Coherence | Intentional → value chosen to fit the row → high coherence |
| G3: Error Pattern | Coordinated multi-cell changes → intentional |
| G4: Distribution | Rare extreme values → likely noise; common values → strategic choice |
| G5: String-Level | Keyboard-adjacent transpositions → typo → unintentional |
| G6: Strategic Importance | Error in high-MI column → rational target → intentional |
| G7: Peer Deviation | Error inflates value vs. similar records → gain-seeking → intentional |
| G8: Sensitivity | Error in race/sex toward majority class → fairness masking → intentional |

## Heuristics NOT included (require downstream ML model)

These heuristics from the paper are powerful but **require a trained decision model**
(e.g., a GBC trained on the dataset). They are deferred to future work:

- **Causal Impact** (§4.1.1): does changing `e_i` flip the model prediction?
  → Needs: `model.predict(row_with_dirty_value)` vs `model.predict(row_with_correction)`
- **Group Shift Potential** (§4.1.1): does the error push the record across a
  decision boundary?
  → Needs: model's decision boundary or confidence scores
- **Effort-Based Perturbation Size** (§4.1.1): minimal perturbation for outcome shift
  → Needs: counterfactual generation over the model

> These are the paper's **causal** heuristics. Our approach is the **structural** half
> of the same framework. Together they form the complete diagnostic tuple
> ⟨f_i, I(e_i), C(e_i), S(e_i)⟩ defined in the paper. We compute I(e_i) and S(e_i)
> (intent score + semantic class). C(e_i) (causal impact) is left for Stage 2.

---

## Critical Self-Assessment: Where This Approach Can Fail

Honest analysis of the assumptions vs. actual data (Adult Income LLM dataset).

### Assumption 1: "Intentional errors are always valid domain values"
**PARTIALLY TRUE.** 69.9% of intentional categorical errors are in the clean vocabulary.
But 29% are obfuscation tokens (`nan`, `Unknown`, `—`, `-DMV`, `-obf`). These are
a *sub-type* of intentional error: information obfuscation (paper §3.2.1). They are
**novel** values, not valid ones. G1 (plausibility) would classify them as
unintentional — **wrong**.

→ **Mitigation**: G5 (string features) catches obfuscation tokens explicitly.
The combination G1=invalid + G5=obfuscation_detected should route these correctly.
But G1 alone is not sufficient. **G5 is load-bearing for obfuscation attacks.**

### Assumption 2: "Unintentional errors are always broken/invalid"
**MOSTLY TRUE.** 92.5% of unintentional categorical errors are novel/broken values.
But **7.5% (547 cells)** are unintentional errors with perfectly valid values
(e.g., workclass: `Private`, sex: `Female`). These are the hard cases where a random
substitution happens to produce a valid value by accident.

→ **Impact**: These 547 cells will look intentional on G1. We need G2 (coherence)
and G7 (peer deviation) to rescue them. If the random valid substitution is
*incoherent with the row context*, G2 will flag it. This is the make-or-break
test for the approach.

### Assumption 3: "Numerical errors are separable by range"
**WEAK.** For capital-gain, 91% of intentional AND 93% of unintentional values
are in-range — essentially identical. G1/G4 features (z-score, in-range) are
**useless** for numerical capital-gain. For age and hours-per-week, the separation
is better (~93% vs ~46% in-range), but still imperfect.

→ **Impact**: For numerical columns, G1/G4 have limited discriminative power.
G2 (coherence) and G7 (peer deviation) must carry the load for numerical errors.

### Assumption 4 (G8): "Sensitive features → fairness masking toward majority"
**NUANCED.** In the sex column, intentional errors are split nearly equally:
548 Male, 482 Female. The majority class (Male) represents only ~54% of intentional
sex errors. This is NOT a clean "Female→Male" masking pattern. It includes
**individual discrimination** (changing someone's sex in either direction) and
obfuscation (`White`, `—`, `Unknown` appearing in the sex column).

→ **Impact**: G8's "masking direction toward majority" feature is a weak signal
for sex. It may be stronger for race (256 White out of 357 = 72% toward majority).
G8 should NOT be a hard classifier — it's one feature among many.

### The Real Discriminative Power Ranking (honest estimate)

| Group | Signal strength | Why |
|-------|----------------|-----|
| G5: String features | ★★★★★ | Typo patterns (`Bachleors`, `Whitw`) are nearly perfect unintentional markers |
| G1: Plausibility | ★★★★☆ | 93% separation for categorical, but fails for obfuscation and numerical |
| G2: Coherence | ★★★★☆ | The key signal for ambiguous cases — but depends on predictor quality |
| G7: Peer deviation | ★★★☆☆ | Promising for gain-targeted attacks, but expensive and noisy with mixed-type data |
| G3: Error pattern | ★★★☆☆ | Co-occurrence is useful (education+education-num) but dataset-specific |
| G4: Distribution | ★★☆☆☆ | Weak for numerical (both types overlap heavily in capital-gain) |
| G6: Strategic importance | ★★☆☆☆ | Static per-column — same value for ALL errors in that column |
| G8: Sensitivity | ★☆☆☆☆ | Majority-direction is unreliable; only helps if combined with others |

### Expected Failure Modes

1. **Obfuscation misclassified as unintentional**: `nan`, `Unknown`, `—` are novel
   values that look like noise on G1. Need G5 obfuscation detection to save these.
2. **Random valid substitutions misclassified as intentional**: 7.5% of unintentional
   errors happen to be valid. Need G2/G7 to detect row-level incoherence.
3. **Numerical errors in capital-gain**: Both types are in-range. Only G2/G7 can help.
4. **Individual discrimination in sensitive columns**: Intentional errors go in
   both directions (Male→Female and Female→Male), breaking the G8 assumption.
