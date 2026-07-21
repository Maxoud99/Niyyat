# No-Clustering Ablation — B, B+, C, B+C, (B+)+C

**Revision 2: re-run with intent-signal rules for Family C** (see
[`../../declarative/results/RESULTS.md`](../../declarative/results/RESULTS.md)
for what changed in the rule extraction and why). B and B+ are untouched by
that redesign — their numbers below are identical to Revision 1. Only C,
B+C, and (B+)+C changed, because they are the only scenarios that consume
the declarative constraint features.

This file is the no-clustering counterpart of `declarative/results/RESULTS.md`.
Every scenario (**B**, **B+**, **C**, **B+C**, **(B+)+C**) uses the *exact
same* feature engineering, the *exact same* Random Forest, and the *exact
same* 1% label budget (`n_seed = round(n_cells × 0.01)`, identical to the
clustering runs) — the **only** change is how the labeled seed set is
chosen:

| | With clustering (baseline) | Without clustering (this ablation) |
|---|---|---|
| Seed selection | DBSCAN clusters → proportional sampling per cluster | Plain uniform random sampling |
| Budget | `round(n_cells × 0.01)` | `round(n_cells × 0.01)` (identical) |
| Features | B (13) / B+ (23) / C (*k*+4) / combinations | Same |
| Classifier | RF, 200 trees, depth 15, `class_weight="balanced"` | Same |
| Evaluation | Held-out (non-seed) cells | Same |

---

## All-Dataset Summary — No Clustering (F1-weighted, 1% labels)

| Dataset | n_seed | B | B+ | C | B+C | (B+)+C |
|---|---:|---:|---:|---:|---:|---:|
| Adult-LLM | 433 | 0.888 | 0.906 | 0.776 | 0.886 | **0.905** |
| Adult-Mixed | 117 | 0.961 | 0.969 | 0.865 | **0.955** | 0.962 |
| Adult-TFM | 92 | **0.875** | 0.872 | 0.739 | 0.869 | 0.848 |
| TwitterBot-LLM V2 | 10 | 0.560 | 0.694 | **0.728** | 0.481 | 0.742 |
| TwitterBot-Mixed | 11 | 0.757 | 0.859 | 0.298 | 0.754 | **0.866** |
| TwitterBot-TFM | 12 | 0.892 | 0.745 | 0.090 | **0.918** | 0.750 |
| TabFact ‡ | 22 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## All-Dataset Δ — No-Clustering minus With-Clustering (F1-weighted, pp)

Both sides of this comparison now use Revision 2 intent-signal rules for
C / B+C / (B+)+C; B and B+ deltas are unchanged from Revision 1.

| Dataset | n_seed | B | B+ | C | B+C | (B+)+C |
|---|---:|---:|---:|---:|---:|---:|
| Adult-LLM | 433 | +0.3 | −3.7 | −1.6 | +2.6 | +0.8 |
| Adult-Mixed | 117 | −0.5 | −2.0 | **+5.2** | −0.3 | −1.3 |
| Adult-TFM | 92 | −4.2 | −5.9 | +1.1 | −5.2 | −5.3 |
| TwitterBot-LLM V2 | 10 | **−24.4** | +3.2 | −0.2 | **−25.2** | −8.2 |
| TwitterBot-Mixed | 11 | −9.1 | +11.0 | **−33.4** | −8.1 | +9.0 |
| TwitterBot-TFM | 12 | −5.4 | −10.7 | **−52.6** | +10.3 | +2.2 |
| TabFact ‡ | 22 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## What's different from Revision 1, and why

**The core finding survives the rule redesign: clustering's value is still
concentrated at extreme label scarcity, not at moderate budgets.** The three
Adult datasets (n_seed 92–433) again show small-to-moderate deltas, mostly
within ±5 pp; the three TwitterBot datasets (n_seed 10–12) again show large,
erratic swings in both directions. That qualitative pattern is a property of
the *sampling strategy*, not of the rule set, so it was expected to persist.

**TwitterBot-TFM's standalone C collapses even harder without clustering
under the new rules (−52.6 pp) than it did under the old ones.** With only
4 of 25 rules active on this dataset (per the declarative RESULTS.md's rule
inventory) and a 12-cell random draw, the no-clustering run again produced
the degenerate all-one-class outcome (F1-UNINT = 0.000) already seen in
Revision 1 — cluster-proportional sampling is what stops this specific
failure mode, and that conclusion is unchanged by which rule set produced
the underlying features.

**Adult-Mixed's standalone C is the one scenario where no-clustering now
*beats* clustering by a wide margin (+5.2 pp, reversing Revision 1's
roughly-flat +0.1 pp).** The new rule set gives C 31 dimensions instead of
18; at Adult-Mixed's relatively generous 117-seed budget, uniform random
sampling apparently covers this larger, richer feature space at least as
well as DBSCAN does, and the standalone-C number is high enough (0.865) that
small sampling-strategy differences move the needle more than they did when
C was weaker.

**TwitterBot-LLM V2's B+C also collapses further without clustering
(−25.2 pp, vs. Revision 1's −2.5 pp).** The broader, more principled rule
set (6 active rules vs. 2 before) increases B+C's feature dimensionality
from 13+19+4=36 to 13+25+4=42; combined with only 10 seeds, this is a harder
space for uniform random sampling to seed representatively than Revision
1's narrower one was — consistent with the same "more features cost more at
a fixed tiny budget" effect documented for the with-clustering pipeline.

**Net reading, updated:** the redesign changed *which* dataset shows the
biggest no-clustering penalty (TwitterBot-TFM's C, TwitterBot-LLM V2's B+C)
and added one case where no-clustering wins outright (Adult-Mixed's C), but
it did not change the underlying mechanism: clustering acts as a variance
reducer for the tiny seed sample, and that protection matters most exactly
when (a) the budget is tiny (≤12 seeds) and (b) the feature space is large
relative to the budget — both properties of the *feature count*, largely
independent of whether those features encode validity or intent.

---

## Per-Dataset No-Clustering Metrics (Revision 2)

### Adult-LLM (n_seed = 433)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.727 | 0.810 | 0.776 |
| B + C | 0.868 | 0.900 | 0.886 |
| (B+) + C | 0.889 | 0.917 | 0.905 |

### Adult-Mixed (n_seed = 117)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.796 | 0.898 | 0.865 |
| B + C | 0.929 | 0.967 | 0.955 |
| (B+) + C | 0.939 | 0.972 | 0.962 |

### Adult-TFM (n_seed = 92)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.609 | 0.803 | 0.739 |
| B + C | 0.774 | 0.915 | 0.869 |
| (B+) + C | 0.745 | 0.898 | 0.848 |

### TwitterBot-LLM V2 (n_seed = 10)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.668 | 0.769 | 0.728 |
| B + C | 0.109 | 0.732 | 0.481 |
| (B+) + C | 0.622 | 0.823 | 0.742 |

### TwitterBot-Mixed (n_seed = 11)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.433 | 0.237 | 0.298 |
| B + C | 0.627 | 0.811 | 0.754 |
| (B+) + C | 0.804 | 0.894 | **0.866** |

### TwitterBot-TFM (n_seed = 12)

| Scenario | F1-INT | F1-UNINT | F1-w |
|---|---:|---:|---:|
| C | 0.382 | 0.000 | 0.090 |
| B + C | 0.847 | 0.941 | **0.918** |
| (B+) + C | 0.599 | 0.797 | 0.750 |

### TabFact (n_seed = 22) ‡

All scenarios score F1-w = 1.000 (single-class, trivial regardless of
sampling strategy or rule revision).

---

## Reproducing

```bash
cd error_detection_system/src/attribution/no_clustering
python run_all_no_clustering.py                       # all 7 datasets, all 5 scenarios
python run_all_no_clustering.py --datasets adult_llm   # single dataset
python run_all_no_clustering.py --scenarios B Bplus    # single scenario subset
```

Constraints are loaded from the same cache as the clustering pipeline
(`../declarative/results/constraints/*.json`) — these now contain Revision 2
intent-signal rules; no LLM re-extraction is triggered by this module.

## Output Files

| Location | Contents |
|---|---|
| `results/<dataset_key>/<scenario>/features.csv` | Per-cell feature matrix + predicted_intent |
| `results/<dataset_key>/<scenario>/metrics.json` | Accuracy, F1-w, F1-INT, F1-UNINT, n_seed |
| `results/<dataset_key>/<scenario>/feature_importance.csv` | RF feature importances |
| `results/summary_no_clustering.csv` | All datasets × scenarios summary |
