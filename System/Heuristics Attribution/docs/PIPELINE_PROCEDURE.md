# Attribution Pipeline — Full Procedure

> **What problem does this solve?**
> You have a dirty (corrupted) table and a mask that tells you *which cells are wrong*.
> You do **not** have the original clean values.
> The goal is to decide, for each erroneous cell:
>
> - **Intentional (1)** — someone deliberately changed this value (adversarial attack, data poisoning, fraud)
> - **Unintentional (0)** — a typo, OCR error, sensor noise, missing data

---

## 1. Big Picture

```
Dirty table  +  Error mask
       │
       ▼
  [Feature Extraction]   ← 8 heuristics, no clean values needed
       │
       ▼
  13 numbers per cell    ← the "error fingerprint"
       │
       ▼
  [Classifier]           ← Random Forest trained on labeled cells
       │
       ▼
  Label per cell:  0 = unintentional  /  1 = intentional
```

---

## 2. Inputs

| Input | Description |
|---|---|
| `dirty_df` | The corrupted DataFrame (N rows × M cols) |
| `mask_df` | Same shape as dirty_df. `1` = this cell is wrong, `0` = clean |
| `labels` *(optional)* | For each erroneous cell: `1` = intentional, `0` = unintentional |

Labels are only needed for **training/evaluation**. In production you train once and deploy.

---

## 3. Step-by-Step Procedure

### Step 1 — Fit the pipeline (unsupervised)

```python
pipe = AttributionPipeline(
    target_col="class",
    codependent_pairs=[("education", "education-num")],
    sensitive_cols=["race", "sex"],
)
pipe.fit(dirty_df, mask_df)
```

**No labels are used. No clean reference values are needed. This is a one-time step per dataset.**

The pipeline loops over all 8 heuristics and calls `heuristic.fit(dirty_df, mask_df)` on each one.
Each heuristic uses *only the clean cells* (where `mask_df == 0`) to build its internal statistics.
Here is exactly what each heuristic learns during `fit()`:

#### H1 — fit()
For each column it reads only the **clean cells** (mask = 0) and computes:
- **Column type detection**: if >90% of non-null clean values can be cast to `float` → numerical, otherwise categorical.
- **Categorical columns**: builds a `set` of all clean string values — the "known vocabulary".
- **Numerical columns**: computes the 5th and 95th percentile of the clean values — the "plausible range".

After fit, `col_stats_` maps each column to `{type, vocab}` or `{type, p5, p95}`.

#### H2 — fit()
For each **categorical** column it reads only the clean cells and:
- Auto-detects column type (same 90% rule as H1).
- Builds a vocabulary list of unique clean string values.
- If the vocabulary is very large (>200 entries), caps it to the 50 most frequent values to keep edit-distance computation tractable.
- Stores nothing for numerical columns (H2 outputs NaN for those).

After fit, `col_vocabs_` maps each categorical column to its trimmed vocabulary list.

#### H3 — fit()
For each column it reads only the clean cells and:
- **Categorical columns**: counts the frequency of every clean value → builds a `{value: rank}` lookup where rank 1 = most common. This lets the scoring function convert the dirty value's rank into a [0,1] score.
- **Numerical columns**: computes `clean_mean` and `clean_std`. The score will be `1 - min(1, |z-score| / 3)`.
- Edge case: if `clean_std == 0` (constant column), stores a flag so the score defaults to 1.0.

After fit, `col_stats_` maps each column to its distribution summary.

#### H4 — fit()
This is the most expensive heuristic to fit. For **every column** in the table it:
- Detects column type (same 90% rule).
- Trains a `RandomForestClassifier` (categorical targets) or `RandomForestRegressor` (numerical targets) using **all other columns as features** and this column as the target.
- Uses all rows (not just clean) for training — errors are sparse enough to be noise.
- Applies `OrdinalEncoder` to the feature matrix and `LabelEncoder` to categorical targets.
- Subsamples to 50,000 rows maximum on large datasets.
- Stores the fitted RF and the label/ordinal encoders so `compute()` can call `predict_proba()` later.

After fit, `predictors_` maps each column name to its trained RF model.

#### H5 — fit()
Discovers which pairs of columns are logically co-dependent by:
- **Name similarity**: strips digits/hyphens/common suffixes (e.g. `education-num` → `education`), then computes SequenceMatcher ratio between all column pairs. Pairs with ratio ≥ 0.8 are co-dependent.
- **Mutual information**: computes MI between all column pairs (on clean rows, capped at 20 columns), and adds pairs with MI ≥ 0.5.
- Merges auto-discovered pairs with any user-supplied `codependent_pairs`.

After fit, `codependent_pairs_` holds the full set of linked column pairs.

#### H6 — fit()
Estimates the statistical importance of each column relative to the target:
- **Supervised** (target_col provided): computes `mutual_info_classif` or `mutual_info_regression` between every feature column and `target_col`, using clean rows only. The target column itself gets the max MI of all other columns. Normalises all scores to [0,1].
- **Unsupervised** (no target_col): for each column, computes the max MI between it and up to 10 randomly sampled other columns. This approximates "how much does any other column depend on this one?". Normalises to [0,1].
- Uses `OrdinalEncoder` + `SimpleImputer` to handle mixed types.

After fit, `importance_scores_` maps each column to a float in [0,1].

#### H7 — fit()
Builds per-column behavioral profiles by:
- **Mutability score**: derived purely from column name keywords — e.g. `id`, `wgt`, `hash` → 0.0 (immutable); `race`, `sex` → 0.5 (sensitive, soft boundary); everything else → 1.0 (freely mutable). No data needed.
- **Gain direction**: for categorical columns with a known `target_col`, uses clean rows to compute: for each value `v` in the column, what is the mean target label (e.g. mean of `income == ">50K"`) for rows where this column = `v`? This builds a `{col: {value: mean_target}}` lookup. For numerical columns, uses Spearman correlation between the column and the target to determine the sign of the relationship.
- **Comprehensibility**: also keyword-based — names like `age`, `education`, `occupation` → 1.0 (obvious); names like `fnlwgt`, `capital-gain` → lower scores.

After fit, `col_profiles_` maps each column to `{mutability, gain_table, comprehensibility}`.

#### H8 — fit()
Very lightweight:
- Detects sensitive columns by checking if any keyword from `{race, gender, sex, age, nationality, marital, ethnic, ...}` appears in the column name (case-insensitive).
- Merges auto-detected columns with any user-supplied `sensitive_cols`.
- For each sensitive *categorical* column: uses clean rows to find the **majority class** (most frequent clean value). Stores it as the reference for the per-cell `h8_is_majority_value` score.
- For sensitive *numerical* columns (e.g. `age`): stores a flag indicating the majority-class signal is not applicable.

After fit, `sensitive_cols_` is the final set of sensitive column names and `majority_values_` maps each to its majority class.

---

### Step 2 — Extract the 13-feature matrix

```python
feat_df = pipe.compute_features(dirty_df, mask_df)
# shape: (N_errors, 13)
# index: MultiIndex (row_idx, col_name) — one row per erroneous cell
```

For **every erroneous cell** `(row i, col j)`, the 8 heuristics run in sequence and each contributes 1–2 numbers. The result is a matrix with one row per erroneous cell and 13 columns — the **error fingerprint**:

| # | Feature | Heuristic | What it measures |
|---|---|---|---|
| 1 | `h1_plausible` | H1 | 1 if the value is in the column's known vocabulary, 0 if not |
| 2 | `h2_min_edit_distance` | H2 | Minimum Levenshtein distance to the nearest known value (NaN for numerical cols) |
| 3 | `h3_distribution_score` | H3 | How central/common this value is in its column's distribution (1=central, 0=extreme) |
| 4 | `h4_coherence_score` | H4 | How well this value fits the rest of its row (1=fits perfectly, 0=incoherent) |
| 5 | `h5_error_count` | H5 | Number of erroneous cells in the same row |
| 6 | `h5_codependent_flag` | H5 | 1 if a logically linked partner column is also erroneous in this row |
| 7 | `h6_column_importance` | H6 | Statistical MI importance of this column for the target label (0–1) |
| 8 | `h6_importance_rank` | H6 | Rank of this column among all columns by importance (1=most important) |
| 9 | `h7_gain_direction` | H7 | Does the dirty value shift toward a favorable outcome for the subject? (0–1) |
| 10 | `h7_mutability` | H7 | Can a human realistically change this value? (0=immutable, 1=freely mutable) |
| 11 | `h7_comprehensibility` | H7 | Would a typical person understand what this column measures? (0–1) |
| 12 | `h8_is_sensitive` | H8 | 1 if this is a protected/demographic attribute |
| 13 | `h8_is_majority_value` | H8 | 1 if the dirty value equals the majority class of this sensitive column |

At this point you have **a number for every erroneous cell but no labels**. The next question is: how do you get labels to train the classifier?

---

### Step 3 — Getting Labels: Three Scenarios

This is the key design decision. You have three options depending on how many labels you can afford:

---

#### Scenario A — You already have full labels (evaluation / research)

This is what `test_adult_income.py` does. The injection process recorded ground truth:
- `masks.csv` encodes `-1=unintentional`, `0=clean`, `1=intentional`
- You convert this directly to binary labels and train on the full labeled set

```python
labels = pd.Series(...)   # ground truth from injection process
X = np.nan_to_num(feat_df.values, nan=-999.0)
y = labels.loc[feat_df.index].values

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
rf.fit(X, y)
```

Use 5-fold stratified CV to measure performance without overfitting.

---

#### Scenario B — You have NO labels at all (fully unsupervised)

When you have no labels, you use **clustering on the 13-feature matrix** to discover natural groups, then inspect a tiny sample from each cluster to assign labels. This is the semi-supervised path:

**Step B1 — Cluster the 13-feature matrix**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # or HDBSCAN for variable-density clusters

X = np.nan_to_num(feat_df.values, nan=-999.0)
X_scaled = StandardScaler().fit_transform(X)

# KMeans with k=2 (intentional vs unintentional)
km = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)

# Or HDBSCAN (no need to specify k, handles noise)
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, prediction_data=True)
cluster_labels = clusterer.fit_predict(X_scaled)
```

The hypothesis is that intentional errors form one natural cluster (coherent, plausible, gain-directed) and unintentional errors form another (incoherent, out-of-vocab, extreme).

**Step B2 — Sample ~1% from each cluster for human labeling**

```python
feat_df["cluster"] = cluster_labels

sample_per_cluster = []
for cid in feat_df["cluster"].unique():
    cluster_rows = feat_df[feat_df["cluster"] == cid]
    # Sample 1% or at least 30 rows, whichever is larger
    n_sample = max(30, int(len(cluster_rows) * 0.01))
    sampled = cluster_rows.sample(n=n_sample, random_state=42)
    sample_per_cluster.append(sampled)

to_label = pd.concat(sample_per_cluster)
to_label.to_csv("to_label.csv")   # send to human annotator
```

A human annotator looks at each sampled cell in `to_label.csv` and assigns:
- `1` = intentional (looks deliberate)
- `0` = unintentional (looks like noise/typo)

This typically requires labeling **1–2% of all errors** — e.g. for 43,000 errors, ~430–860 manual labels.

**Step B3 — Train RF on the labeled sample**

```python
# After human fills in labels column in to_label.csv
labeled = pd.read_csv("to_label_filled.csv", index_col=[0,1])
y_sample = labeled["label"].values
X_sample = np.nan_to_num(labeled.drop(columns=["cluster","label"]).values, nan=-999.0)

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_sample, y_sample)
```

**Step B4 — Apply to the remaining 99%**

```python
# The unlabeled errors
unlabeled_idx = feat_df.index.difference(to_label.index)
X_unlabeled = np.nan_to_num(feat_df.loc[unlabeled_idx].values, nan=-999.0)

predictions   = rf.predict(X_unlabeled)          # 0 or 1
probabilities = rf.predict_proba(X_unlabeled)[:, 1]  # P(intentional)

results = feat_df.loc[unlabeled_idx].copy()
results["predicted_intent"] = predictions
results["p_intentional"]    = probabilities
results.to_csv("attribution_results.csv")
```

---

#### Scenario C — You have a small existing labeled set from a previous dataset

If you previously labeled errors on a different dirty table with the same schema, you can transfer those labels:

```python
# Train on previously labeled data from another run
rf.fit(X_old_labeled, y_old_labeled)

# Apply directly to new dataset's feature matrix
X_new = np.nan_to_num(feat_new.values, nan=-999.0)
predictions = rf.predict(X_new)
```

This works because the 13 features are **schema-level signals** (vocabulary, distribution, coherence) that generalise across different versions of the same table.

---

### Step 4 — The Full Recommended Workflow (no labels available)

```
Dirty table + Error mask
        │
        ▼
[Step 1] pipe.fit()              ← learn column stats (unsupervised, ~seconds–minutes)
        │
        ▼
[Step 2] pipe.compute_features() ← extract 13-feature matrix for all N_errors cells
        │
        ▼
[Step 3B1] Cluster the matrix    ← KMeans(k=2) or HDBSCAN
        │
        │   Two clusters emerge:
        │   Cluster A: high h3, high h4, high h7 → likely intentional
        │   Cluster B: low h1, low h4, low h7    → likely unintentional
        │
        ▼
[Step 3B2] Sample 1% per cluster ← ~hundreds of cells to label manually
        │
        ▼
[Human] Label the sample         ← 0=unintentional / 1=intentional
        │
        ▼
[Step 3B3] Train RF on sample    ← RandomForest(class_weight="balanced")
        │
        ▼
[Step 3B4] Predict on remaining 99%
        │
        ▼
Attribution results:
  (row_idx, col_name) → 0 or 1 + P(intentional)
```

**Why cluster first instead of sampling randomly?**

If you sample 1% randomly, you might get 99% unintentional errors (the majority class) and almost no intentional ones — the classifier sees no positive examples. Clustering first **guarantees representation from both natural groups** in the small labeled sample, even if the classes are heavily imbalanced.

---

### Step 5 — Evaluate (when labels are available)

When you have ground-truth labels (research/evaluation setting), use stratified 5-fold CV:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

X = np.nan_to_num(feat_df.values, nan=-999.0)
y = labels.loc[feat_df.index].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    rf.fit(X[train_idx], y[train_idx])
    y_pred  = rf.predict(X[test_idx])
    y_proba = rf.predict_proba(X[test_idx])[:, 1]
    print(f"Fold {fold+1}  F1={f1_score(y[test_idx], y_pred):.4f}  AUC={roc_auc_score(y[test_idx], y_proba):.4f}")
```

Stratified CV ensures each fold has the same class ratio as the full dataset — critical when intentional errors are a minority.

---

## 4. Why Each Heuristic Works

### H1 — Value Plausibility
Intentional attackers often inject real-looking but wrong values (e.g. a valid occupation but the wrong one). Unintentional errors are more likely to produce garbage strings not in the vocabulary.
**Signal:** out-of-vocabulary → more likely unintentional.

### H2 — String Anomaly (Edit Distance)
A typo is 1–2 characters away from the correct value. A deliberate substitution is often semantically far but string-close (e.g. `"Male"` → `"Female"`).
**Signal:** very small edit distance → suspicious if the meaning flips.

### H3 — Distribution Position
Deliberate injections often use extreme values to shift model behavior. Unintentional noise is closer to the mean.
**Signal:** high Z-score / extreme percentile → more likely intentional.

### H4 — Row Coherence
Intentional errors often violate logical constraints within a row (e.g. `education=Masters` but `education-num=3`). Unintentional errors in one cell usually leave the rest of the row consistent.
**Signal:** low coherence with co-dependent columns → likely intentional.

### H5 — Error Pattern
Attackers tend to focus on a specific column (the one that matters for their goal). Unintentional errors are spread randomly across columns.
**Signal:** column is disproportionately targeted → errors in it are likely intentional.

### H6 — Column Importance
Rational attackers target columns that most influence the model's prediction. Random noise hits all columns equally.
**Signal:** error in a high-importance column → more likely intentional.

### H7 — User Incentive
The strongest signal. If changing this cell pushes the model's prediction toward a "better" outcome for the subject (e.g. `income < 50K` → `income > 50K`), there's a clear motive.
**Signal:** label flip or push in a beneficial direction → very likely intentional.

### H8 — Sensitivity Flag
Errors in protected attributes (`race`, `sex`) are often inserted to create or amplify bias. Random noise has no preference for sensitive columns.
**Signal:** error in a sensitive column with directional skew → likely intentional.

---

## 5. Two Operating Modes

| Mode | Labels needed? | Use case |
|---|---|---|
| **Unsupervised** | No | Explore: inspect the 13-feature matrix, cluster errors, find suspicious cells |
| **Supervised** | Yes (for training) | Production: train RF once on labeled data, deploy to classify new errors |

In the test script (`test_adult_income.py`), we simulate the supervised mode using cross-validation — we have ground-truth labels from the injection process.

---

## 6. Results on Adult Income Dataset

| Dataset | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| LLM (tenth-trial) | 0.9205 | 0.9128 | 0.8945 | 0.9035 | 0.9729 |
| Kireev (mixed_error_pipeline) | 0.9896 | 0.9844 | 0.9841 | 0.9842 | 0.9984 |

### Comparison to baselines

| Method | F1 (LLM dataset) |
|---|---|
| Random guess | ~0.50 |
| Best LLM prompt (Gemini) | 0.818 |
| Best clustering (HDBSCAN) | 0.904 |
| **This pipeline (H1–H8)** | **0.904** |

The pipeline matches the best unsupervised baseline without any LLM API calls, and far exceeds it on the Kireev dataset (0.984).

### Most important features (LLM dataset)

1. `h3_distribution_score` — 0.259
2. `h4_coherence_score` — 0.183
3. `h1_plausible` — 0.180

### Most important features (Kireev dataset)

1. `h7_gain_direction` — 0.213
2. `h1_plausible` — 0.165
3. `h3_distribution_score` — 0.136

---

## 7. Output Files (per run)

Each run of `test_adult_income.py` creates:

```
output/
└── run_YYYYMMDD_HHMMSS/
    ├── run_log.txt                  ← full console output
    ├── summary.csv                  ← one row per dataset, all metrics
    ├── llm_tenth-trial/
    │   ├── feature_matrix.csv       ← 13 features × N_errors rows
    │   ├── feature_distributions.csv← mean per feature, by intent class
    │   ├── cv_fold_metrics.csv      ← per-fold acc/prec/rec/F1/AUC
    │   ├── cv_summary.csv           ← mean ± std across folds
    │   ├── confusion_matrix.csv     ← TN/FP/FN/TP (aggregated)
    │   ├── feature_importances.csv  ← RF importance per feature, ranked
    │   └── classification_report.txt← sklearn report
    └── kireev_mixed_error_pipeline/
        └── (same files)
```

---

## 8. File / Code Map

```
heuristics/
├── pipeline.py              ← AttributionPipeline: orchestrates H1–H8
├── base.py                  ← BaseHeuristic abstract class
├── h1_value_plausibility.py ← vocabulary membership
├── h2_string_anomaly.py     ← edit distance to nearest known value
├── h3_distribution_position.py ← statistical outlier score
├── h4_row_coherence.py      ← within-row logical consistency
├── h5_error_pattern.py      ← column-level targeting signal
├── h6_column_importance.py  ← proxy model feature importance
├── h7_user_incentive.py     ← label-flip / prediction-push signal
├── h8_sensitivity_flag.py   ← protected attribute signal
├── test_adult_income.py     ← evaluation script (runs everything)
└── docs/
    ├── PIPELINE_PROCEDURE.md   ← this file
    └── ERROR_FINGERPRINT.md    ← academic-style writeup
```

---

## 9. How Each Heuristic Is Implemented

This section describes the internal mechanics of every heuristic — what it computes during `fit()`, what formula it applies in `compute()`, what the output numbers mean, and what the known limitations are.

---

### H1 — Value Plausibility (`h1_value_plausibility.py`)

**Output features:** `h1_plausible` (binary, always present)

**fit():**
- Reads only clean cells (mask = 0) per column.
- Auto-detects column type: >90% of non-null clean values parseable as float → **numerical**, otherwise → **categorical**.
- Categorical: builds a `set` of all clean string values (the vocabulary).
- Numerical: stores the 5th and 95th percentile of clean values as the plausible range `[p5, p95]`.

**compute() — formula:**
```
Categorical:  h1_plausible = 1  if dirty_value ∈ clean_vocab
                            0  otherwise

Numerical:    h1_plausible = 1  if p5 ≤ float(dirty_value) ≤ p95
                            0  otherwise (out of range or non-numeric)
```

**Known limitation:**
~29% of intentional errors in Adult Income use obfuscation tokens (`nan`, `Unknown`, `—`) which are also out-of-vocabulary. H1 scores these as 0 (same as unintentional typos). H2 is responsible for distinguishing them.

---

### H2 — String Anomaly (`h2_string_anomaly.py`)

**Output features:** `h2_min_edit_distance` (float, NaN for numerical columns)

**fit():**
- Categorical columns only. Builds a vocabulary list of unique clean string values.
- If vocab size > 200, trims to the 50 most frequent values to keep computation tractable.
- Numerical columns: stores nothing (H2 outputs NaN for them).

**compute() — formula:**
```
h2_min_edit_distance = min over all v in clean_vocab of:
    Levenshtein(dirty_value, v)

Capped at 10.  NaN for numerical columns.

Obfuscation check: if dirty_value ∈ OBFUSCATION_TOKENS
    (nan, none, n/a, unknown, ?, —, null, -1, 999, ...)
or if dirty_value = known_value + separator + obfuscation_suffix
    (e.g. "Private-DMV", "Bachelors-obf")
→ h2_is_obfuscation = 1 (separate internal flag, influences explanation score)
```

**Levenshtein** is implemented inline without external dependencies, using standard DP with an optimisation that skips vocabulary entries whose length differs from the dirty value by more than 5.

**Interpretation:**
- Small edit distance (1–2) + not obfuscation → likely **typo** (unintentional)
- Obfuscation token or suffix-mangled → likely **intentional**
- NaN → numerical column, H2 does not apply

---

### H3 — Distribution Position (`h3_distribution_position.py`)

**Output features:** `h3_distribution_score` (float in [0,1])

**fit():**
- Categorical: counts frequency of every clean value; builds `{value: rank}` where rank 1 = most common.
- Numerical: computes `clean_mean` and `clean_std` from clean rows.

**compute() — formula:**
```
Numerical:
    z = (dirty_value - clean_mean) / clean_std
    h3_distribution_score = 1 - min(1, |z| / 3.0)
    → 1.0 when z=0 (exactly at mean), 0.0 when |z| ≥ 3

Categorical:
    rank = position of dirty_value in frequency-sorted clean vocab
           (vocab_size + 1 if out-of-vocabulary → penalised as rarest)
    h3_distribution_score = 1 - (rank - 1) / max(1, vocab_size - 1)
    → 1.0 for most common value, 0.0 for rarest / OOV
```

**Interpretation:**
- High score (~1.0) = value is common / central → consistent with **intentional blending**
- Low score (~0.0) = value is extreme / rare / OOV → consistent with **unintentional noise**

**Known limitation:** For columns where both intentional and unintentional errors cluster similarly (e.g. `capital-gain` where 91–93% of both error kinds are in-range), H3 has low discrimination power. Other heuristics compensate.

---

### H4 — Row Coherence (`h4_row_coherence.py`)

**Output features:** `h4_coherence_score` (float in [0,1])

**fit():**
- For **every column** `c` in the table, trains a Random Forest using all other columns as features and column `c` as the target.
- Categorical target: `RandomForestClassifier` (50 trees) + `LabelEncoder` on target + `OrdinalEncoder` on features.
- Numerical target: `RandomForestRegressor` (50 trees).
- Training uses all rows (clean + dirty). Errors are sparse enough to be negligible noise.
- Subsamples to 50,000 rows on large datasets.
- Stores fitted RF, encoders, and `col_std` for each column.

**compute() — formula:**
```
Categorical target c:
    P = RF.predict_proba(row_without_c)   ← other columns as input
    h4_coherence_score = P[class == dirty_value]
    → 0.0 if dirty_value was unseen during fit

Numerical target c:
    predicted = RF.predict(row_without_c)
    h4_coherence_score = max(0, 1 - |predicted - dirty_value| / col_std)
    → 1.0 if dirty_value == predicted, 0.0 if deviation ≥ 1 std
```

**Interpretation:**
- High score (~1.0) = the dirty value is exactly what the row context predicts → **intentional** (the attacker preserved logical consistency)
- Low score (~0.0) = the dirty value is inconsistent with the row → **unintentional** (random noise breaks correlations)

---

### H5 — Error Pattern (`h5_error_pattern.py`)

**Output features:** `h5_error_count` (int), `h5_codependent_flag` (0/1)

**fit():**
- **Name similarity**: normalises column names (strips `num`, `no`, `id`, digits, hyphens), then computes SequenceMatcher ratio between all pairs. Pairs with ratio ≥ 0.8 are marked co-dependent.
- **Mutual information**: on clean rows (capped at 20 columns), computes pairwise MI using `mutual_info_classif`. Pairs with MI ≥ 0.5 are added.
- Merges auto-discovered pairs with user-supplied `codependent_pairs`.

**compute() — formula:**
```
h5_error_count       = number of erroneous cells in the same row
                       (from mask_df.loc[row_i].sum())

h5_codependent_flag  = 1  if any co-dependent partner of col_j is also
                          erroneous in row_i (mask_df.loc[row_i, partner] == 1)
                       0  otherwise
```

**Interpretation:**
- High `h5_error_count` = many cells edited in the same row → coordinated, **intentional** edit
- `h5_codependent_flag = 1` = logically linked column also altered → deliberate consistency maintenance → **intentional**
- Both = 0 → isolated single-cell error → more likely **unintentional**

---

### H6 — Column Importance (`h6_column_importance.py`)

**Output features:** `h6_column_importance` (float in [0,1]), `h6_importance_rank` (int)

**fit():**
- **Supervised** (target_col provided): computes `mutual_info_classif` or `mutual_info_regression` between every feature column and `target_col`, using clean rows only. The `target_col` itself gets the maximum MI of all feature columns. All scores divided by max to normalise to [0,1].
- **Unsupervised** (no target_col): for each column, samples up to 10 random other columns and takes the max pairwise MI. Approximates "how informationally central is this column?". Normalises to [0,1].
- Uses `OrdinalEncoder` + `SimpleImputer` to handle mixed types before MI computation.

**compute() — formula:**
```
h6_column_importance = importance_scores_[col_j]   ← constant per column

h6_importance_rank   = rank of col_j when all columns sorted
                       by importance_scores_ descending
                       (rank 1 = most important)
```

**Important distinction from H7:** H6 is *statistical* importance (MI with target). H7 is *behavioral* motivation (would a human want to change this?). They are not redundant — `fnlwgt` has high H6 (statistically predictive) but low H7 (nobody understands it). `race` has low H6 but high H7 (privacy motivation).

---

### H7 — User Incentive (`h7_user_incentive.py`)

**Output features:** `h7_mutability` (float), `h7_gain_direction` (float), `h7_comprehensibility` (float)

**fit():**

**Mutability** — keyword-based, no data needed:
```
Keywords → immutable (score 0.0): id, num, wgt, weight, score, code,
           key, index, ssn, hash, uuid, timestamp, date, time
Keywords → sensitive boundary (score 0.5): race, gender, sex,
           nationality, religion, disability, ethnicity
All other columns → freely mutable (score 1.0)
```

**Gain direction** — data-driven:
- For categorical columns with a `target_col`: groups clean rows by value in this column, computes the mean target label for each group. Builds a `{value: mean_target}` lookup table.
- For numerical columns: computes Spearman correlation between this column and the target on clean rows. Positive correlation → higher value is "better"; negative → lower value is "better".

**Comprehensibility** — keyword-based:
```
Obvious columns (score 1.0): age, education, occupation, sex, race,
                              hours, marital, relationship, income, salary
Opaque columns  (score 0.2): fnlwgt, capital, gain, loss, weight
Everything else: score 0.7
```

**compute() — formula:**
```
h7_mutability        = mutability score for col_j (constant per column)

h7_gain_direction    = Categorical:
                         mean_target for dirty_value in col_j's lookup
                         0.5 if dirty_value not seen during fit
                         0.5 if no target_col
                       Numerical:
                         0.8 if Spearman(col_j, target) > 0
                             AND dirty_value > clean_median   (going "up" is good)
                         0.2 if direction is opposite
                         0.5 if no target_col or correlation near zero

h7_comprehensibility = comprehensibility score for col_j (constant per column)
```

> **Note:** In the test scripts, the pipeline internally uses `h7_gain_direction` as one of the 13 features. The original design output 3 H7 features but the pipeline was simplified to expose only `h7_gain_direction` as the key signal. The other two (`h7_mutability`, `h7_comprehensibility`) are absorbed internally as weights in gain direction computation.

---

### H8 — Sensitivity Flag (`h8_sensitivity_flag.py`)

**Output features:** `h8_is_sensitive` (0/1), `h8_is_majority_value` (0/1)

**fit():**
- Auto-detects sensitive columns: checks if any keyword from `{race, gender, sex, age, nationality, religion, disability, marital, ethnic, origin, orientation}` appears in the column name (case-insensitive substring match).
- Merges auto-detected with user-supplied `sensitive_cols`.
- For each sensitive **categorical** column: finds the majority class (most frequent value in clean rows).
- For sensitive **numerical** columns (e.g. `age`): stores a flag that `h8_is_majority_value` is not applicable → outputs 0.

**compute() — formula:**
```
h8_is_sensitive      = 1  if col_j ∈ sensitive_cols_
                       0  otherwise  (constant per column)

h8_is_majority_value = 1  if col_j is sensitive AND categorical
                          AND dirty_value == majority_values_[col_j]
                       0  otherwise (non-sensitive, or numerical sensitive,
                          or dirty_value ≠ majority class)
```

**Interpretation:**
The pattern being detected is *privacy-motivated demographic masking*: a user changes a minority-class demographic value to the dominant class to "blend in" and be harder to profile. For example, in Adult Income:
- `race`: 86% White → changing any non-White value to "White" → `h8_is_majority_value = 1` → strong intentional signal
- `sex`: 67% Male → weaker signal (less dominant majority)

For non-sensitive columns, both features are 0 and add no signal.

---

### Summary Table — What Each Heuristic Computes During fit() and compute()

| Heuristic | fit() cost | What is stored | compute() operation | NaN possible? |
|---|---|---|---|---|
| H1 | Fast | vocab set or [p5,p95] per column | Set membership / range check | No |
| H2 | Fast | vocab list per categorical column | Levenshtein min-distance loop | Yes (numerical cols) |
| H3 | Fast | mean+std or freq-rank per column | Z-score formula or rank lookup | No |
| H4 | **Slow** | One RF model per column | `predict_proba` / `predict` per row | No |
| H5 | Medium | Co-dependent column pairs | Row mask sum + partner lookup | No |
| H6 | Medium | MI score per column | Direct lookup (constant per col) | No |
| H7 | Medium | Gain table + keyword scores | Gain table lookup or Spearman sign | No |
| H8 | Fast | Sensitive cols + majority values | Membership + majority check | No |

