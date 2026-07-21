# MICE Imputation — Brief Illustrated Guide

---

## What MICE Does in One Sentence

> Train one predictive model per column **as a target** (using all other columns as inputs) on clean rows, then for each corrupted cell, predict what its value **should have been** by feeding the row's other column values into that target column's model.

---

## Part 1 — Training the Column Models

We have 14 columns. We train **14 separate Random Forest models** — one per column — using only the **3,010 clean rows**.

```
Clean rows only (3,010 rows):

  age  | edu-num | hours | capital-gain | occupation | ...
  -----+---------+-------+--------------+------------+----
   34  |   13    |  40   |      0       |  Exec-mgr  | ...
   52  |   16    |  45   |    5000      |  Prof-spec | ...
   28  |    9    |  35   |      0       |  Sales     | ...
   ...

Train M_age:       (edu-num, hours, capital-gain, occupation, ...) → age
Train M_edu-num:   (age, hours, capital-gain, occupation, ...)     → edu-num
Train M_hours:     (age, edu-num, capital-gain, occupation, ...)   → hours
... (one model per column)
```

Each model learns: *"given everything else in the row, what should THIS column be?"*

The models are **never retrained**. They are fixed after this step.

---

## Part 2 — Single-Error Rows (Direct Prediction)

Row 312 has only `hours-per-week` corrupted. All other columns are clean and trustworthy.

```
Row 312 (observed):
  age=34 | edu-num=13 | occupation=Exec-mgr | hours=72* | capital-gain=0
                                                     ↑
                                               corrupted cell

Call M_hours with the clean columns as input:
  M_hours(age=34, edu-num=13, occupation=Exec-mgr, capital-gain=0)
  → x̂* = 38.1        (200 trees, mean prediction)
  → σ_tree = 4.2     (std across trees = uncertainty)
```

No iteration needed. One model call → one imputed value.

---

## Part 3 — Multi-Error Rows (The MICE Chain)

Row 871 has **two** corrupted cells: `age` and `edu-num`. You cannot call M_age using edu-num because edu-num is also corrupted — and vice versa. Classic chicken-and-egg.

**MICE breaks this with rounds:**

```
Row 871 (observed):
  age=19* | edu-num=14* | occupation=Prof-spec | hours=50 | capital-gain=5000
            ↑                   ↑
       corrupted             corrupted

── Round 0: Initialise corrupted cells with column mean/mode ──────────────

  working_row = { age=38.5,  edu-num=10.1,  occupation=Prof-spec, hours=50, capital-gain=5000 }
                  (col mean)   (col mean)         (untouched clean columns →)

── Round 1 ────────────────────────────────────────────────────────────────

  Step 1 — impute age:
    M_age( edu-num=10.1, occupation=Prof-spec, hours=50, capital-gain=5000 )
    → age = 41.2
    working_row = { age=41.2,  edu-num=10.1,  ... }

  Step 2 — impute edu-num:
    M_edu-num( age=41.2, occupation=Prof-spec, hours=50, capital-gain=5000 )
    → edu-num = 12.3                ↑ uses the UPDATED age from step 1
    working_row = { age=41.2,  edu-num=12.3,  ... }

── Round 2 ────────────────────────────────────────────────────────────────

  Step 1 — impute age:
    M_age( edu-num=12.3, ... )   ← now uses the updated edu-num from round 1
    → age = 42.0

  Step 2 — impute edu-num:
    M_edu-num( age=42.0, ... )
    → edu-num = 12.8

── Round 3 ────────────────────────────────────────────────────────────────

  Step 1 → age = 42.1
  Step 2 → edu-num = 12.9

  Convergence check:  |42.1 - 42.0| / 42.0 = 0.002  < tol(0.001)?  No → continue
                      |12.9 - 12.8| / 12.8 = 0.008  < tol? No...

── Round 4 ────────────────────────────────────────────────────────────────

  age = 42.1,  edu-num = 12.9   (no change → converged ✓)

Final imputed:  x̂*_age = 42.1,   x̂*_edu-num = 12.9
Ground truth:   x*_age = 43,      x*_edu-num = 13
```

Each round the models feed into each other, correcting their neighbours. It converges because the column models agree on a consistent interpretation of the row.

---

## Part 4 — Chaining the Models: How It Works in Code

After Phase 1 (fit), we have a dictionary of trained models:

```python
self.models_ = {
    "age":          RandomForestRegressor (fitted),
    "edu-num":      RandomForestRegressor (fitted),
    "hours":        RandomForestRegressor (fitted),
    "capital-gain": RandomForestRegressor (fitted),
    "occupation":   RandomForestClassifier (fitted),
    ...  # one per column
}
```

The chain in Phase 3 is simply this loop (from `_impute_multi_cell`):

```python
# working estimate of the full row (starts from col means)
encoded_row = { age: 38.5, edu-num: 10.1, occupation: ..., hours: 50, ... }

for round in range(max_rounds):
    for col in corrupted_cols:               # ["age", "edu-num"]

        # Build input = all columns EXCEPT the one we are predicting
        X = encoded_row.drop(col)            # shape (1, 13)

        # Call THAT column's model
        x_hat = self.models_[col].predict(X)

        # Update working row immediately → next column sees the fresh estimate
        encoded_row[col] = x_hat

    if converged: break
```

The key line is `encoded_row[col] = x_hat` **inside the inner loop**. This means:
- When we impute `edu-num` in step 2 of round 1, we already use the **round-1 estimate of `age`**, not the stale round-0 mean.
- Each update immediately propagates to the next column in the same round.

This is called **Gauss-Seidel** style updating (update in place) as opposed to **Jacobi** style (update all at once at the end of the round). Our implementation uses Gauss-Seidel, which converges faster.

---

## Part 5 — From Imputed Value to Intent Decision

The imputed value `x̂*` is **not the final answer**. It is the input to computing four numbers that describe the nature of the error:

```
For each corrupted cell (row i, column j):

  x̃  = observed (dirty) value    = 72    (hours)
  x̂* = imputed correct value     = 38.1

  direction  = sign(72 - 38.1)   = +1     ← error went UP
  magnitude  = |72 - 38.1|       = 33.9   ← error was large
  WRC        = 33.9 / (38.1 × (1 + 4.2)) = 0.171   ← relative to scale & uncertainty
  confidence = 1 / (1 + 4.2)    = 0.19   ← imputer was uncertain (trees disagreed)

  φ = [direction=+1, magnitude=33.9, WRC=0.171, confidence=0.19, x̃=72, x̂*=38.1, ...]
```

These four numbers become the **input feature vector φ** for the second Random Forest — the **intent classifier**:

```
RF #2 (intent classifier)
  Input:  φ_{ij}  (direction, magnitude, WRC, confidence, ...)
  Output: ℓ̂_{ij} ∈ { +1 (intentional), -1 (unintentional) }
```

Intuition behind each feature:

| Feature | Why it helps distinguish intent |
|---|---|
| **direction** | Intentional errors often go in a systematic direction (always inflate, always deflate) |
| **magnitude** | Intentional errors tend to be large and precise; accidents are small and noisy |
| **WRC** | Normalises magnitude by scale — a +10 change in `age` vs `capital-gain` means different things |
| **confidence** | Low-confidence imputations → unreliable signal; high-confidence → direction/magnitude are trustworthy |

The full pipeline in one diagram:

```
  Clean rows (3,010)
       │
       ▼
  ┌──────────────────────────────┐
  │  RF #1: 14 column models     │  ← trained on columns, applied to rows
  │  M_age, M_edu, M_hours, ...  │
  └──────────────────────────────┘
       │
       │  for each corrupted cell → x̂*
       ▼
  ┌──────────────────────────────┐
  │  Diagnostic features         │
  │  direction, magnitude,       │
  │  WRC, confidence             │
  └──────────────────────────────┘
       │
       │  φ_{ij} per cell
       ▼
  ┌──────────────────────────────┐
  │  RF #2: intent classifier    │  ← trained on cells, predicts intent
  │  5-fold CV, 2,198 cells      │
  └──────────────────────────────┘
       │
       ▼
  ℓ̂ ∈ { intentional, unintentional }
```

---

## Summary Table

| Question | Answer |
|---|---|
| How many models in RF #1? | 14 (one per column **as target**) |
| Trained on what? | 3,010 clean rows; each model uses the other 13 columns as `X`, one column as `y` |
| What does each model learn? | "Given other columns, predict this column" |
| Why chain them? | Multi-error rows: each column's estimate improves the others |
| How many rounds? | Up to 5; stops early when estimates stop changing |
| What does RF #2 do? | Takes the 4 diagnostic features and predicts +1/−1 |
| Are they trained together? | No — RF #1 trains first, RF #2 trains on RF #1's outputs |
