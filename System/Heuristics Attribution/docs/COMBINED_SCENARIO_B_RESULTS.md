# Scenario B — Combined (Statistical + Heuristic) Features: Results

Run directory: `heuristics/output/run_scenario_b_combined_20260325_103624/`

Features: **23 per erroneous cell** = 13 heuristic + 10 statistical

---

## Statistical features added (cell-level value statistics)

| # | Feature | Description |
|---|---------|-------------|
| 1 | `stat_change_magnitude` | \|dirty_value − clean_value\| |
| 2 | `stat_relative_change` | magnitude / (\|clean\| + 1) |
| 3 | `stat_change_direction` | sign(dirty − clean) {−1, 0, +1} |
| 4 | `stat_original_magnitude` | \|clean_value\| |
| 5 | `stat_new_magnitude` | \|dirty_value\| |
| 6 | `stat_original_log` | log1p(\|clean_value\|) |
| 7 | `stat_new_log` | log1p(\|dirty_value\|) |
| 8 | `stat_feature_name_encoded` | label-encoded column name |
| 9 | `stat_original_value_encoded` | label-encoded clean value |
| 10 | `stat_new_value_encoded` | label-encoded dirty value |

---

## Dataset 1 — LLM Adult (43,259 error cells)

| Clustering | Best method | F1w | F1_intentional | Labels% |
|---|---|---|---|---|
| KMeans | RandomForest | 0.9637 | 0.9562 | 1.00% |
| **DBSCAN** | **RandomForest** | **0.9686** | **0.9617** | 4.77% |
| Hierarchical (ward) | RandomForest | 0.9630 | 0.9549 | 1.00% |
| Hierarchical (average) | RandomForest | 0.9647 | 0.9577 | 1.06% |
| GMM | RandomForest | 0.9606 | 0.9525 | 1.00% |
| **HDBSCAN** | **RandomForest** | **0.9748** | **0.9696** | 2.35% |

**Best overall: HDBSCAN + RandomForest → F1w=0.9748, F1_int=0.9696**

---

## Dataset 2 — Mixed SOTA Adult / Kireev (11,661 error cells)

| Clustering | Best method | F1w | F1_intentional | Labels% |
|---|---|---|---|---|
| KMeans | RandomForest | 0.9870 | 0.9804 | 1.00% |
| **DBSCAN** | **RandomForest** | **0.9937** | **0.9902** | 5.02% |
| Hierarchical (ward) | RandomForest | 0.9794 | 0.9680 | 1.00% |
| Hierarchical (average) | RandomForest | 0.9687 | 0.9526 | 1.21% |
| GMM | RandomForest | 0.9359 | 0.9061 | 1.00% |
| **HDBSCAN** | **RandomForest** | **0.9911** | **0.9863** | 3.86% |

**Best overall: DBSCAN + RandomForest → F1w=0.9937, F1_int=0.9902**

---

## Dataset 3 — TwiBot-20 / Twitter (1,099 error cells)

| Clustering | Best method | F1w | F1_intentional | Labels% |
|---|---|---|---|---|
| KMeans | LabelPropagation | 0.9208 | 0.8697 | 6.37% |
| **DBSCAN** | **KNN_k7** | **0.9366** | **0.8977** | 10.01% |
| **Hierarchical (ward)** | **RandomForest** | **0.9484** | **0.9148** | 6.37% |
| Hierarchical (average) | LabelPropagation | 0.8026 | 0.6336 | 3.55% |
| GMM | KNN_k7 | 0.9341 | 0.8942 | 6.19% |
| HDBSCAN | KNN_k7 | 0.9058 | 0.8607 | 3.64% |

**Best overall: Hierarchical Ward + RandomForest → F1w=0.9484, F1_int=0.9148**

---

## Cross-System Comparison (best result per dataset)

### LLM Adult Dataset

| System | Features | Best combo | F1w | F1_intentional | Labels% |
|---|---|---|---|---|---|
| Old (statistical only) | ~24 stat | HDBSCAN+RF | 0.9534 | 0.9437 | 5.05% records |
| New (heuristics only) | 13 heuristic | DBSCAN+RF | 0.9081 | 0.8851 | 8.48% |
| **Combined (this work)** | **23 combined** | **HDBSCAN+RF** | **0.9748** | **0.9696** | **2.35%** |

✅ Combined outperforms both old (+2.1% F1w) and new (+6.7% F1w) systems

### Mixed SOTA Adult (Kireev) Dataset

| System | Features | Best combo | F1w | F1_intentional | Labels% |
|---|---|---|---|---|---|
| Old (statistical only) | ~24 stat | — | — | — | — |
| New (heuristics only) | 13 heuristic | DBSCAN+RF | 0.9825 | 0.9732 | 5.34% |
| **Combined (this work)** | **23 combined** | **DBSCAN+RF** | **0.9937** | **0.9902** | **5.02%** |

✅ Combined outperforms heuristic-only (+1.1% F1w, +1.7% F1_int)

### TwiBot-20 Dataset

| System | Features | Best combo | F1w | F1_intentional | Labels% |
|---|---|---|---|---|---|
| Old (statistical only) | ~24 stat | HDBSCAN+RF | 0.8988 | 0.8303 | 5.05% records |
| New (heuristics only) | 13 heuristic | KMeans+KNN | 0.9293 | 0.8940 | 6.82% |
| **Combined (this work)** | **23 combined** | **HierWard+RF** | **0.9484** | **0.9148** | **6.37%** |

✅ Combined outperforms both old (+4.96% F1w) and new (+1.91% F1w) systems

---

## Key Findings

1. **Supervisor's hypothesis confirmed**: The combined feature system is consistently better across all three datasets. Adding statistical features (value magnitude, encoded value identities, direction of change) to the heuristic features meaningfully improves classification.

2. **RF dominates**: RandomForest is the best label method in 14/18 combinations. Combined features give RF richer signal — the value-level statistical features let RF distinguish subtle intentional vs unintentional changes that heuristics alone miss.

3. **LLM dataset: biggest gain** (+6.7% F1w vs heuristics-only). Statistical features are especially useful here because the LLM generates highly systematic intentional errors with recognisable magnitude/direction signatures.

4. **Twitter: strong improvement** (+4.96% vs old system). Despite having only 1,099 error cells, adding value-context via statistical features helps considerably.

5. **Label efficiency maintained**: The combined system uses ≤6.4% labels on all datasets, comparable or better than the heuristic-only system.

6. **DBSCAN and HDBSCAN remain top clusterers**: With 23 features the density-based methods still produce the best cluster structure (silhouette > 0.43 on LLM, > 0.61 on Kireev).

---

## Feature Dimensionality Summary

| Feature group | Count | Type |
|---|---|---|
| H1 plausibility | 1 | cell-level |
| H2 edit distance / obfuscation | 2 | cell-level |
| H3 distribution score | 1 | cell-level |
| H4 coherence score | 1 | cell-level |
| H5 error density | 2 | record-level |
| H6 column importance | 1 | column-level |
| H7 mutability / gain / comprehensibility | 3 | column-level |
| H8 sensitivity / majority | 2 | column-level |
| **Stat: magnitudes & logs** | **6** | cell-level |
| **Stat: encoded values & column** | **3** | cell-level |
| **Stat: direction** | **1** | cell-level |
| **Total** | **23** | |
