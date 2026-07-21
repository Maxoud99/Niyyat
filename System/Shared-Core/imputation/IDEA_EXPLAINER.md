# The Idea — One Page

**Goal:** Given a cell we *know* is wrong, classify **why** it is wrong.
- Label **1** = *intentional* — someone deliberately changed this value (fraud, evasion, manipulation)
- Label **-1** = *unintentional* — a typo, OCR glitch, measurement error

We do not have access to the correct value at deployment time. We need to figure out the direction and magnitude of the corruption without it.

---

## The Core Idea in One Sentence

> Train a model on the clean rows to learn what a "normal" value looks like for each cell, then measure **how far and in which direction** the corrupted value deviates from that expectation — weighted by how confident we are in the prediction.

---

## The Three-Step Pipeline

```
STEP 1 — LEARN WHAT'S NORMAL
   Use clean rows (rows with no errors) to train one Random Forest
   per column: "given all other features of a row, predict column X."
   
   age_model   = RF trained on {workclass, education, occupation, ...} → age
   occup_model = RF trained on {age, workclass, education, ...}        → occupation
   ...

STEP 2 — IMPUTE THE EXPECTED VALUE
   For a flagged erroneous cell: mask the corrupted value, run the
   corresponding RF model. The output x̂* is "what this cell should
   probably be, given the rest of the row."
   
   If the row has multiple corrupted cells: use MICE — iteratively
   re-impute each corrupted column using the latest estimates for the
   other corrupted columns, until the estimates stop changing.

STEP 3 — COMPUTE DIAGNOSTIC FEATURES
   Now we have:
     x̃  = the corrupted (observed) value
     x̂* = the imputed (expected) value
     σ  = how much the 200 trees disagreed (confidence signal)

   Features:
     direction  = sign(x̃ - x̂*)          → did the corruption go UP or DOWN?
     magnitude  = |x̃ - x̂*|             → how far did it move?
     WRC        = |x̃ - x̂*| / (|x̂*| × (1 + σ))
                                         → relative change, dampened when uncertain
```

**These features feed a classifier that predicts intentional vs. unintentional.**

---

## End-to-End Example — Real Data (Adult Income Dataset)

### Setup

Row 3 in the dataset is an erroneous row with **3 corrupted cells** (all unintentional, label = −1).

| Column | Corrupted value (x̃) | Correct value (x*) |
|---|---|---|
| `age` | **81** | 18 |
| `workclass` | **"Privat."** | "Private" |
| `occupation` | **"5ales"** | "Sales" |

Three cells corrupted → MICE iterative imputation is triggered.

---

### Step 1 — Learn What's Normal (Training, done once)

The system trains RF models on the ~3,010 clean rows:

```
age_model      trained on: workclass, education, occupation, ...
workclass_model trained on: age, education, occupation, ...
occupation_model trained on: age, workclass, education, ...
```

---

### Step 2 — MICE Imputation (3 rounds until convergence)

Initialize: set all 3 corrupted cells to their column default (median/mode).

| Round | age estimate | workclass estimate | occupation estimate |
|---|---|---|---|
| Init | 36 (median) | "Private" (mode) | "Prof-specialty" (mode) |
| Round 1 | 51 | "Private" | "Other-service" |
| Round 2 | 57 | "Private" | "Other-service" |
| **Round 3** | **58** | **"Private"** | **"Other-service"** | ← converged |

Final imputed values: `age=58`, `workclass="Private"`, `occupation="Other-service"`

> Note: the correct value for `age` was 18, but the model predicts 58. The model is wrong on the exact number — but it doesn't need to be right. It just needs to flag that `age=81` is extreme.

---

### Step 3 — Compute Diagnostic Features

| Column | x̃ (corrupted) | x̂* (imputed) | σ_tree | Direction | Magnitude | WRC |
|---|---|---|---|---|---|---|
| `age` | 81 | 58 | 22.0 | +1 (UP) | 23 | **0.017** (dampened — σ is large) |
| `workclass` | "Privat." | "Private" | 0.30 | changed | 1 | 0.110 |
| `occupation` | "5ales" | "Other-service" | 0.79 | changed | 11 | 0.473 |

**WRC for `age` is 0.017** — small, because the imputer was uncertain (σ=22), so the WRC formula dampens it. The classifier interprets this as: "the imputed value is unreliable, don't read too much into the magnitude."

---

### Contrast: Intentional Example (Row 12, label = +1)

Row 12 has 3 intentionally corrupted cells:

| Column | x̃ (corrupted) | x* (correct) | x̂* (imputed) | WRC |
|---|---|---|---|---|
| `age` | 31 | 25 | 50 | **0.025** (small — subtle change) |
| `occupation` | "Sales" | "Adm-clerical" | "?" | **0.862** (high — suspicious swap) |
| `hours-per-week` | 55 | 48 | 40 | **0.025** (small — plausible value) |

The WRC values are small for `age` and `hours-per-week` — the corrupted values (31, 55) are plausible given the row context, so the imputer doesn't flag them strongly. The `occupation` swap stands out.

---

### What the Classifier Sees

| Feature | Row 3 (unintentional) | Row 12 (intentional) |
|---|---|---|
| `age` WRC | 0.017 | 0.025 |
| `occupation` WRC | 0.473 | **0.862** |
| `hours-per-week` WRC | — | 0.025 |
| `age` confidence | 0.043 (low) | 0.067 (low) |
| `workclass` confidence | 0.769 (high) | — |

The classifier combines all these features across all corrupted cells in the row and predicts intent. In practice (5-fold CV on 2,198 cells): **F1-macro = 0.834** vs. **0.674** without imputation.

---

## Why Not Just Look at the Raw Value?

A naive classifier sees `age=81` in Row 3. Is that intentional or unintentional?

- It *could* be a valid age (81-year-old worker).
- It *could* be a typo (81 instead of 18, digits transposed).
- It *could* be fraud (deliberately inflated).

Without the imputed reference point (`x̂* = 58`), you only know the value exists. With it, you know:
- The deviation is **+23 years** above the expected age for this row's profile
- The RF trees disagreed heavily (σ=22) → the row is unusual → dampen the signal
- The WRC is small → this looks like noise, not a deliberate manipulation

**That is the value of imputation for intent attribution.**

---

## One-Line Summary Per Component

| Component | What it does |
|---|---|
| RF per column | Learns the "normal" value for each feature from clean rows |
| MICE iteration | Handles rows where multiple cells are corrupted — they inform each other |
| σ_tree | Measures how much the 200 trees disagreed — row-level uncertainty |
| σ_OOB | Measures how predictable this column is in general — column-level uncertainty |
| WRC | Relative change, suppressed when uncertain — the key discriminative signal |
| Downstream RF | Classifies intent from the diagnostic features |
