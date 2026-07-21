# Declarative User-Based Heuristics for Intent Attribution

## Abstract

We introduce a third family of heuristics for erroneous-cell intent attribution — **declarative user-based heuristics** — that complements the two existing inductive families (structural and statistical). Rather than learning patterns from the observed data distribution, declarative heuristics are grounded in domain knowledge: the user describes their dataset in natural language, a large language model (LLM) converts that description into formal integrity constraints, and those constraints are evaluated against each erroneous cell to produce attribution features. Unlike LLM-based attribution, where the model is invoked once per cell or row at inference time, the LLM is called only *once per dataset* to extract constraints; inference is then fully deterministic and runs at native speed. The method is dataset-agnostic, requires no labelled examples, and produces features that are structurally complementary to the inductive families — capturing semantic invalidity that statistical analysis cannot recover.

---

## 1. Motivation

Intent attribution asks: given a detected erroneous cell, was it changed *deliberately* (intentional, label = 1) or *accidentally* (unintentional, label = −1)? The two existing families of heuristics address this from a purely data-driven perspective.

**Family B — Structural heuristics (H1–H8):** Signals derived exclusively from the dirty dataset and a binary error mask. They ask *how does this error look?* — is the value a typo, an outlier, in a strategically important column, consistent with the rest of the row? These features are entirely inductive and require no external knowledge.

**Family B+ — Statistical heuristics:** Augment Family B with ten cell-level statistical features (z-score from column mean, distance from median, percentile position, IsolationForest score, etc.) that ask *how far is this value from what is expected statistically?* These features implicitly model the correct-value distribution, but they do so from the dirty data itself and can be misled by errors that are statistically plausible (e.g., an age swapped from 35 to 45 — both within the normal distribution).

**Gap:** Neither family can represent semantic domain rules that a human expert holds in their head. Consider:

- A value of `relationship = "Husband"` in a row where `sex = "Female"` is *impossible* by domain logic, yet both values are statistically frequent and structurally valid.
- `education-num = 9` with `education = "Bachelors"` is logically inconsistent — the mapping is fixed and known — but both columns may independently look statistically normal.
- `capital-gain = −500` is semantically impossible (a financial constraint), but the sign flip may produce a z-score that does not flag it if the column is noisy.

These violations are only detectable if the system knows what the domain *says values should be* — not what the data *shows values look like*.

---

## 2. The Three-Family Framework

We organise all attribution heuristics into three epistemically distinct families:

| Family | Scenario | Knowledge source | Nature | LLM role |
|--------|----------|-----------------|--------|-----------|
| **B** — Structural | 13 features | Dirty data + mask | Inductive | None |
| **B+** — Statistical | 23 features (B + 10) | Dirty data distribution | Inductive | None |
| **C** — Declarative | *k* features (*k* = number of constraints) | User's NL description | **Deductive** | One-time extractor |

Families B and B+ are **inductive**: they generalise from observed error patterns and data statistics. Family C is **deductive**: the user declares semantic rules, and the system applies them top-down to each cell. The distinction matters because:

1. Inductive methods are bounded by what is observable in the (corrupted) data.
2. Declarative rules encode knowledge about what *cannot* be learned from data — logical invariants, cross-column dependencies, and hard domain constraints.
3. The two types of knowledge are complementary: their union is strictly more informative than either alone.

---

## 3. Family C: Declarative User-Based Heuristics

### 3.1 User Description

The user writes a short natural language description of their dataset — typically 10–30 lines. No special syntax or template is required. The description should state:

- What each column represents (semantic meaning)
- Valid ranges for numeric columns
- Allowed value sets for categorical columns
- Cross-column logical rules (if any)
- Any domain-specific constraints the data must satisfy

**Example (Adult Income, excerpt):**
> `education-num` is a numeric encoding of the education level, integer from 1 (Preschool) to 16 (Doctorate). It must exactly correspond to the `education` column. The value "Husband" in `relationship` can only appear when `sex` is Male. `capital-gain` is capital gains in USD and must be a non-negative integer.

This description is written once per dataset and reused across all experiments.

### 3.2 Constraint Extraction

The description is passed to an LLM (Gemini 2.5 Pro, temperature 0) via a **dataset-agnostic prompt** that instructs the model to:

1. Identify every integrity constraint, validity rule, or domain restriction in the text.
2. Express each as a Python boolean expression (evaluates to `True` when the constraint is *satisfied*).
3. List the column(s) each constraint involves.
4. Return the result as structured JSON.

The prompt contains no dataset-specific instructions, column names, or domain knowledge beyond what the user provided. The same prompt works for any dataset.

**Example output (abridged):**

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
      "id": "C7",
      "description": "education-num must match education exactly",
      "columns": ["education", "education-num"],
      "expression": "{'Preschool':1,'1st-4th':2,'5th-6th':3,'7th-8th':4,'9th':5,'10th':6,'11th':7,'12th':8,'HS-grad':9,'Some-college':10,'Assoc-voc':11,'Assoc-acdm':12,'Bachelors':13,'Masters':14,'Prof-school':15,'Doctorate':16}.get(str(row['education'])) == (lambda v: int(float(v)))(row['education-num'])"
    },
    {
      "id": "C9",
      "description": "Husband relationship implies Male sex",
      "columns": ["relationship", "sex"],
      "expression": "str(row['relationship']) != 'Husband' or str(row['sex']) == 'Male'"
    }
  ]
}
```

The constraint spec is saved as JSON and reused on every subsequent run — the LLM is not called again unless the user explicitly requests re-extraction (`--force-reextract`).

### 3.3 Feature Engineering

For each erroneous cell $(i, c)$ — row $i$, column $c$, masked as erroneous — the following features are computed:

**Per-constraint binary flags.** For each constraint $C_k$ with columns list $\mathcal{C}_k$:

$$\text{violated}_{C_k}(i, c) = \begin{cases} 1 & \text{if } C_k \text{ is violated in row } i \text{ and } c \in \mathcal{C}_k \\ 0 & \text{otherwise} \end{cases}$$

This is the **attribution principle**: cell $(i, c)$ is held responsible only for constraints that involve its column. A violated cross-column constraint (e.g., `education` ↔ `education-num`) is assigned to whichever column(s) in the constraint are erroneous according to the mask.

**Summary features.** Let $\mathcal{A}(c) = \{k : c \in \mathcal{C}_k\}$ be the set of constraints applicable to column $c$:

| Feature | Definition |
|---------|-----------|
| `n_col_violated` | $\sum_{k \in \mathcal{A}(c)} \mathbb{1}[C_k \text{ violated in row } i]$ |
| `col_violation_ratio` | `n_col_violated` / $|\mathcal{A}(c)|$, or 0 if $|\mathcal{A}(c)| = 0$ |
| `n_applicable` | $|\mathcal{A}(c)|$ — a constant per column, captures constraint coverage |
| `row_total_violations` | $\sum_k \mathbb{1}[C_k \text{ violated in row } i]$ — row-level signal |

The total feature vector for Scenario C has $k + 4$ dimensions, where $k$ is the number of extracted constraints (typically 10–20 per dataset).

### 3.4 Attribution Semantics

The feature vector encodes two complementary signals:

**Constraint violation as intentional evidence.** An intentional error is typically a *semantically valid but factually wrong* value — e.g., changing "Bachelors" to "HS-grad" is a valid categorical swap, but it decouples `education` from `education-num`. A cross-column constraint violation therefore strongly suggests intentional manipulation: the attacker changed one field but forgot to update the other, or deliberately chose a value that breaks a logical invariant.

**Constraint satisfaction as unintentional evidence.** An unintentional error (typo, ±1 drift, abbreviation) typically does not violate semantic constraints — a misspelling of "Bachelors" as "Bachleors" violates the enum constraint, correctly signalling its unintentional character.

**Cells with no applicable constraints.** If a column participates in no constraint (e.g., `fnlwgt`, the census weight), all constraint features are zero. The RF learns that zero constraint signal is uninformative and falls back to other feature families when combined (B+C, (B+)+C).

---

## 4. Combined Scenarios

Three combined scenarios extend the standalone families:

| Scenario | Feature families | Feature count | 1 % labels |
|----------|-----------------|---------------|-----------|
| **B** | Structural (H1–H8) | 13 | ✓ |
| **B+** | Structural + Statistical | 23 | ✓ |
| **C** | Declarative only | *k* + 4 | ✓ |
| **B + C** | Structural + Declarative | 13 + *k* + 4 | ✓ |
| **(B+) + C** | All three families | 23 + *k* + 4 | ✓ |

All scenarios use the same pipeline: cluster erroneous cells by feature vector (HDBSCAN or KMeans), sample 1 % of cells proportionally from clusters, train a Random Forest on the sampled labels, and predict on the remaining cells.

The combination scenarios test whether declarative features provide additive value beyond the existing inductive families:

- **B+C vs. B:** Can declarative features substitute for statistical features at lower engineering cost?
- **(B+)+C vs. B+:** Is there residual signal in constraints beyond what the statistical features already capture?

---

## 5. Comparison with LLM-Based Attribution

LLM-based attribution (Families LLM in the broader evaluation) also uses a language model, but in a fundamentally different role:

| Property | LLM-Based Attribution | Declarative (Family C) |
|----------|----------------------|----------------------|
| LLM call granularity | Once per chunk (10 cells) | Once per dataset |
| Inference cost | High — proportional to dataset size | Negligible — deterministic Python eval |
| Knowledge source | LLM's parametric knowledge | User's explicit text |
| Requires correct values | Yes (dirty + corrected pair) | No |
| Zero-label | Yes | Yes (C features) |
| Interpretable | Via LLM explanation | Direct constraint expression |
| Applicable to unseen domains | Depends on LLM training data | Yes — user description covers any domain |

The key distinction is **where the domain knowledge lives**: in LLM-based attribution it is implicit in model weights, applied per cell. In Family C it is explicit in the user's text, extracted once, and applied symbolically. This makes Family C considerably cheaper at inference time and fully auditable — the constraint expressions can be inspected, corrected, or supplemented by the user.

---

## 6. Implementation Notes

- **LLM for extraction:** Gemini 2.5 Pro, temperature 0, max output tokens 8 192. The extraction prompt is dataset-agnostic and identical across all datasets.
- **Expression evaluation:** Python `eval()` with a restricted namespace (`int`, `float`, `str`, `len`, `abs`, `min`, `max`). Each expression is wrapped in a `try/except` — failures return `None` (constraint not applicable), never crash the pipeline.
- **Caching:** Constraints are saved as JSON after extraction and reloaded on subsequent runs. The LLM is not called again unless `--force-reextract` is specified.
- **Syntax validation:** Each extracted expression is smoke-tested with a dummy row before acceptance. Expressions with `SyntaxError` are dropped and reported.
- **Clustering:** HDBSCAN (default) or KMeans. Same hyperparameters as Scenario B / B+ for fair comparison.
- **Classifier:** Random Forest, 200 trees, `class_weight="balanced"`, `max_depth=15`.

---

## 7. Limitations

1. **Constraint quality depends on description quality.** A vague user description produces weak or missing constraints. A detailed description (as provided in `configs/adult_income.txt`) produces strong, verifiable constraints.

2. **No applicable constraints = no signal.** Columns not mentioned in the description produce zero-valued constraint features. For such columns, Scenario C is equivalent to a constant-feature baseline. Combined scenarios (B+C) fall back to inductive features for unconstrained columns.

3. **Cross-column constraints require both columns to be observed correctly.** If both `education` and `education-num` are erroneous, the cross-column constraint may still evaluate as satisfied (both wrong in a consistent way). This is an inherent limitation of symbolic constraint checking without access to ground-truth values.

4. **LLM extraction can produce incorrect constraints.** The model may misinterpret ambiguous text or generate expressions with wrong value sets. All extracted constraints should be reviewed by the user via the saved JSON spec before running at scale. The `--force-reextract` flag allows iterative refinement.
