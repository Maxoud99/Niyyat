# Declarative User-Based Heuristics — Family C

## Concept

This module implements the third family of intent-attribution heuristics:
**declarative, user-defined integrity constraints**.

The three families in the full system are:

| Family | Scenario | Knowledge source | Nature |
|--------|----------|-----------------|--------|
| **B**  | 13 heuristic features | Dirty data + mask only | Inductive — error pattern signals |
| **B+** | 23 features (B + statistics) | Dirty data distribution | Inductive — statistical distribution |
| **C**  | Constraint features | User's NL description only | **Deductive — domain knowledge** |

Families B and B+ are fully automatic. Family C requires the user to write a
short natural language description of their dataset (what columns mean, valid
ranges, cross-column rules). The system converts that description into
**integrity constraints** via an LLM, then evaluates each constraint against
every erroneous cell to produce attribution features.

### Why deductive?

B and B+ learn from observed data — they are inductive. Family C is the
opposite: the user *declares* what they know about the domain, independent of
what the data shows. This captures semantic rules that no amount of statistical
analysis can recover:

- `"Husband" can only appear for Male sex` — not inferable from a corrupted dataset
- `education-num must exactly match education` — the mapping is domain knowledge
- `capital-gain >= 0` — a domain rule, not a statistical pattern

---

## Supported scenarios

| Scenario | Features used | Meaning |
|----------|---------------|---------|
| **C**        | Constraint features only | Declarative standalone |
| **B+C**      | B (13) + constraint features | General heuristics + domain knowledge |
| **(B+)+C**   | B+ (23) + constraint features | Full heuristic stack + domain knowledge |

---

## Files

| File | Purpose |
|------|---------|
| `extractor.py` | NL text → structured constraint list via Gemini (one-time per dataset) |
| `evaluator.py` | Constraint list + dirty df + mask → per-cell feature matrix |
| `pipeline.py`  | Clustering → 1% sampling → Random Forest (Scenario C / B+C / (B+)+C) |
| `run_declarative.py` | CLI entry point |
| `configs/adult_income.txt` | User description for Adult Income dataset |
| `configs/twibot20.txt` | User description for TwiBot-20 dataset |

---

## How it works

### Step 1 — User writes a description (once per dataset)

Plain text. Describe what each column means and what values are valid:

```
- age: age in years, between 17 and 90
- sex: must be Male or Female
- relationship: "Husband" only valid when sex is Male
- capital-gain: non-negative integer
```

### Step 2 — LLM extracts constraints (one-time, cached)

`extractor.py` sends the description to Gemini with a dataset-agnostic prompt.
Output is a JSON constraint spec, saved for reproducibility:

```json
{
  "constraints": [
    {
      "id": "C1",
      "description": "Age must be between 17 and 90",
      "columns": ["age"],
      "expression": "17 <= (lambda v: int(float(v)))(row['age']) <= 90"
    },
    {
      "id": "C2",
      "description": "Husband relationship implies Male sex",
      "columns": ["relationship", "sex"],
      "expression": "str(row['relationship']) != 'Husband' or str(row['sex']) == 'Male'"
    }
  ]
}
```

### Step 3 — Features per erroneous cell

For each erroneous cell `(row_i, col_c)`:

| Feature | Value |
|---------|-------|
| `violated_C1` | 1 if C1 is violated in row_i AND `col_c` ∈ C1.columns |
| `violated_C2` | 1 if C2 is violated in row_i AND `col_c` ∈ C2.columns |
| `n_col_violated` | count of violated constraints involving `col_c` |
| `col_violation_ratio` | `n_col_violated` / `n_applicable_constraints` |
| `n_applicable` | how many constraints list `col_c` |
| `row_total_violations` | total constraint violations anywhere in row_i |

### Step 4 — Attribution pipeline

Same pipeline as B / B+:
1. Cluster cells by feature vector (HDBSCAN or KMeans)
2. Sample 1% of cells proportionally from clusters
3. Train Random Forest on sampled cells with ground-truth labels
4. Predict intent on remaining cells
5. Evaluate (accuracy, F1 weighted, F1 per class)

---

## Quick start

```bash
cd error_detection_system/src/attribution/declarative

# Scenario C — declarative only
python run_declarative.py \
    --description configs/adult_income.txt \
    --dirty /path/to/dirty.csv \
    --mask  /path/to/mask.csv \
    --gt    /path/to/ground_truth.csv \
    --out   results/adult_income

# Scenario (B+)+C — full stack
python run_declarative.py \
    --description configs/adult_income.txt \
    --dirty /path/to/dirty.csv \
    --mask  /path/to/mask.csv \
    --gt    /path/to/ground_truth.csv \
    --external-features /path/to/Bplus_features.csv \
    --scenario BplusC \
    --out results/adult_income

# Force re-extraction (ignore cached constraints)
python run_declarative.py --description configs/adult_income.txt ... --force-reextract
```

---

## Outputs

```
results/<dataset>/<scenario>/
  ├── constraints.json        # extracted constraint spec (one-time, reused across scenarios)
  ├── features.csv            # per-cell feature matrix
  ├── feature_importance.csv  # RF feature importances
  └── metrics.json            # accuracy, F1 weighted, F1 macro, F1 per class
```

---

## Attribution principle

A cell `(row_i, col_c)` is tagged with a constraint violation if and only if:
1. The constraint is violated in `row_i`, **AND**
2. `col_c` appears in the constraint's `columns` list.

This ensures each cell is only held responsible for constraints it participates
in, not violations caused by other columns in the same row.
