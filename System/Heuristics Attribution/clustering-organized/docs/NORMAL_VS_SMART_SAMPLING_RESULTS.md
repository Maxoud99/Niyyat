# Normal vs Smart Sampling: Comprehensive Comparison

## Executive Summary

**Result:** Normal sampling **significantly outperforms** Smart sampling across all 6 clustering algorithms.

- **Winner:** Normal Sampling (6/6 algorithms, 100% win rate)
- **Average F1:** Normal 87.74% vs Smart 85.33% (+2.41% improvement)
- **Best Overall:** HDBSCAN with Normal Sampling → **91.12% F1**

---

## Detailed Results

### 🔷 Normal Sampling Performance

| Algorithm            | F1 Score | F1 Intent | F1 Unintent | Variants | Samples | Clusters |
|---------------------|----------|-----------|-------------|----------|---------|----------|
| HDBSCAN             | **0.9112** | 0.9077    | 0.9143      | 452      | 773     | 452      |
| K-Means             | 0.8871   | 0.8863    | 0.8879      | 127      | 199     | 15       |
| DBSCAN              | 0.8760   | 0.8749    | 0.8770      | 174      | 353     | 174      |
| GMM                 | 0.8687   | 0.8632    | 0.8736      | 127      | 197     | 15       |
| Hierarchical-Ward   | 0.8624   | 0.8444    | 0.8783      | 127      | 198     | 15       |
| Hierarchical-Avg    | 0.8590   | 0.8453    | 0.8711      | 127      | 211     | 15       |

**Average:** 87.74% F1

### 🔶 Smart Sampling Performance

| Algorithm            | F1 Score | F1 Intent | F1 Unintent | Variants | Samples | Clusters |
|---------------------|----------|-----------|-------------|----------|---------|----------|
| HDBSCAN             | 0.9093   | 0.9055    | 0.9127      | 452      | 773     | 452      |
| Hierarchical-Ward   | 0.8479   | 0.8380    | 0.8567      | 127      | 197     | 15       |
| GMM                 | 0.8419   | 0.8342    | 0.8486      | 127      | 192     | 15       |
| DBSCAN              | 0.8410   | 0.8442    | 0.8382      | 174      | 353     | 174      |
| K-Means             | 0.8402   | 0.8303    | 0.8490      | 127      | 201     | 15       |
| Hierarchical-Avg    | 0.8392   | 0.8185    | 0.8574      | 127      | 191     | 15       |

**Average:** 85.33% F1

---

## Head-to-Head Comparison

| Algorithm            | Normal F1 | Smart F1 | Difference | % Change | Winner      |
|---------------------|-----------|----------|------------|----------|-------------|
| K-Means             | 0.8871    | 0.8402   | **+0.0469**| +5.58%   | ⚡ NORMAL   |
| DBSCAN              | 0.8760    | 0.8410   | **+0.0350**| +4.16%   | ⚡ NORMAL   |
| GMM                 | 0.8687    | 0.8419   | **+0.0269**| +3.19%   | ⚡ NORMAL   |
| Hierarchical-Avg    | 0.8590    | 0.8392   | **+0.0198**| +2.36%   | ⚡ NORMAL   |
| Hierarchical-Ward   | 0.8624    | 0.8479   | **+0.0145**| +1.71%   | ⚡ NORMAL   |
| HDBSCAN             | 0.9112    | 0.9093   | **+0.0019**| +0.21%   | ⚡ NORMAL   |

**Win/Loss Record:**
- Normal Sampling: **6 wins / 0 losses** (100%)
- Smart Sampling: **0 wins / 6 losses** (0%)

---

## Sampling Strategy Details

### Normal Sampling (Simple Proportional)
1. Cluster the data using selected algorithm
2. Allocate samples proportionally to cluster sizes
3. **Random selection** within each cluster
4. **No stratification** by intent labels

### Smart Sampling (User's Original Idea)
1. Cluster the data using selected algorithm
2. Proportional allocation with constraints (1-10 reps per cluster)
3. **Stratified sampling** within each cluster by intent label
4. Balance intentional/unintentional samples within clusters

---

## Key Insights

### 1. **Normal Sampling Dominates**
- Won **every single** algorithm comparison (6/6)
- Average improvement: **+2.41% F1** over Smart sampling
- Largest advantage: **K-Means** (+5.58%)

### 2. **Stratification May Hurt Performance**
Smart sampling's intent-based stratification appears to:
- Introduce sampling bias
- Overfit to training set distribution
- Miss naturally occurring patterns in clusters

### 3. **HDBSCAN is Best Regardless**
- **Normal:** 91.12% F1 (best overall)
- **Smart:** 90.93% F1 (still very good)
- Adaptive clustering + simple sampling = optimal

### 4. **K-Means Shows Biggest Gap**
- Normal: 88.71% F1
- Smart: 84.02% F1
- **5.58% difference** suggests K-Means particularly benefits from simple random sampling

### 5. **Sample Counts Similar**
Both strategies selected similar numbers of samples:
- Normal: ~199-773 samples (depending on clusters)
- Smart: ~191-773 samples (depending on clusters)

The difference is **how** samples are selected, not how many.

---

## Recommendations

### ✅ **Use Normal Sampling**
For production systems and best performance:
- **Algorithm:** HDBSCAN
- **Sampling:** Normal (simple proportional + random)
- **Expected F1:** ~91% on this dataset

### ⚠️ **Avoid Smart Sampling**
The stratification adds complexity without benefit:
- Lower F1 scores across all algorithms (-2.41% average)
- More complex implementation
- May introduce unintended bias

### 🎯 **Alternative: HDBSCAN + Normal Sampling**
Best configuration discovered:
```
Algorithm: HDBSCAN
Sampling: Normal (proportional + random)
Clusters: 452 (adaptive)
Samples: 773
Test Set: 8,441 records
F1 Score: 91.12%
```

---

## Technical Details

### Test Set Alignment
Both strategies tested on **identical** train/test splits:
- Train: 12,744 variants (19,815 samples)
- Test: 5,463 variants (8,441 samples)
- Split method: `sklearn.train_test_split(test_size=0.3, random_state=42)`

### Feature Engineering
All tests used the **optimized 16-feature aggregate**:
- 13 statistical features (counts, magnitudes, ratios)
- 3 feature-level indicators (diversity, most common, max change)
- Categorical encoding for feature names and values

### Random Forest Classifier
Consistent across all experiments:
- n_estimators: 200
- max_depth: 15
- class_weight: balanced
- random_state: 42

---

## Conclusion

**Simple is better.** Normal sampling (proportional allocation + random selection) consistently outperforms Smart sampling (stratified by intent) across all 6 clustering algorithms.

The stratification hypothesis—that balancing intentional/unintentional samples within clusters would improve performance—did not hold in practice. Random selection better preserves the natural distribution of the data.

**Best Practice:** Use HDBSCAN with Normal sampling for 91%+ F1 on intent classification tasks.

---

## Files Generated

- `results/clustering_comparison/algorithm_comparison.csv` - Full results
- `results/clustering_comparison/algorithm_comparison.png` - Visualizations
- `clustering_comparison_normal_vs_smart.log` - Execution log
- `NORMAL_VS_SMART_SAMPLING_RESULTS.md` - This summary

**Date:** December 10, 2025  
**Test Set Size:** 8,441 records  
**Algorithms Tested:** 6 (K-Means, DBSCAN, Hierarchical-Ward, Hierarchical-Average, GMM, HDBSCAN)  
**Sampling Strategies:** 2 (Normal, Smart)  
**Total Experiments:** 12
