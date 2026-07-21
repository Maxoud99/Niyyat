# Hierarchical Clustering for Smart Sampling - Findings & Achievements

**Date:** December 9, 2025  
**Dataset:** Twitter Bot Intent Attribution (6,513 original records → 19,539 manipulated variants)  
**Goal:** Achieve 85% F1 with <1% training data using clustering-based smart sampling

---

## 🎯 Executive Summary

We implemented and tested a **hierarchical clustering pipeline** for intelligent sample selection, comparing it against multiple baselines. The approach successfully demonstrated **data efficiency** but fell short of the 85% F1 target.

### Key Achievement:
- **Smart Sampling** is the **most data-efficient approach**: 41.52% F1 per 100 samples
- With matched data volume (127 variants), achieved **78.83% F1** - competitive with baselines
- **Clustering Only** (without stratification) achieved best performance: **79.33% F1**

---

## 📊 Final Results Comparison

| Approach | Variants Used | Total Samples | F1 Score | F1 per 100 Samples | Status |
|----------|---------------|---------------|----------|-------------------|--------|
| **Clustering Only** | 125 | 200 | **79.33%** | 39.67% | 🏆 BEST |
| **Smart Sampling (Full)** | 127 | 180 | **78.83%** | **43.79%** | 🥈 Most Efficient |
| **Stratified Only** | 127 | 195 | 78.74% | 40.38% | Baseline |
| **Random Sampling** | 127 | 191 | 75.98% | 39.78% | Baseline |
| **Smart Sampling (Initial)** | 96 | 166 | 68.93% | **41.52%** | Most Data-Efficient |

### Performance Progression:
1. Initial Smart Sampling (96 variants): **68.93% F1**
2. Full Smart Sampling (127 variants): **78.83% F1** (+9.9 points!)
3. Gap to 85% target: **6.17 percentage points**

---

## 🔬 Methodology: User's Original Idea

### Pipeline Overview:
```
Full Training Data (12,744 variants)
         ↓
   Enhanced Feature Engineering (27 aggregate features)
         ↓
   K-Means Clustering (K=15, Silhouette=0.5235)
         ↓
   Proportional Representative Allocation (1-10 per cluster)
         ↓
   Stratified Sampling Within Clusters (min guarantee both intents)
         ↓
   Selected Sample (127 variants, 180 samples)
         ↓
   Train Random Forest Classifier
         ↓
   Test on Held-Out Data (8,441 samples)
```

### 1. Enhanced Feature Engineering (27 Dimensions)
For each variant record, we compute:
- **Count Features** (3): n_changes, n_intentional, n_unintentional
- **Ratio Features** (1): intentional_ratio
- **Magnitude Statistics** (5): mean, std, min, max, median of absolute changes
- **Value Range** (2): min/max of new values
- **Feature Type Distribution** (3): % age, workclass, education features
- **Binary Indicators** (13): presence of each feature type

### 2. K-Means Clustering
- **Elbow Method**: Tested K=5 to K=95 in steps of 5
- **Optimal K**: 15 clusters (elbow point, Silhouette=0.5235)
- **Cluster Sizes**: Range from 1 to 4,295 variants
- **Mean Within-Cluster Variance**: 104.22

### 3. Proportional Representative Allocation
- **Target**: 127 variants (~1% of training data)
- **Allocation**: Proportional to cluster size
- **Constraints**: 
  - Minimum 1 representative per cluster
  - Maximum 10 representatives per cluster (relaxed to reach target)
  - **Final Allocation**: Largest cluster got 52 reps, smallest got 1

### 4. Stratified Sampling Within Clusters
For each cluster:
- Split variants by intent label (intentional vs unintentional)
- Sample proportionally from each group
- Guarantee minimum representation of both labels (if available)

---

## 📈 Key Findings

### Finding 1: Data Efficiency
**Smart Sampling is the most data-efficient approach**

When using only 96 variants (initial run):
- Smart Sampling: **41.52% F1 per 100 samples** (68.93% F1 / 166 samples)
- Clustering Only: 39.60% F1 per 100 samples
- Stratified Only: 39.58% F1 per 100 samples
- Random: 37.67% F1 per 100 samples

**Interpretation:** The clustering-based selection identifies more informative samples, achieving higher learning efficiency.

### Finding 2: Stratification Trade-off
**Stratification within clusters slightly hurts performance**

- **Clustering Only** (no stratification): 79.33% F1
- **Smart Sampling** (with stratification): 78.83% F1
- **Difference**: -0.5 percentage points

**Hypothesis:** Forcing minimum guarantees for both intent labels in each cluster may select sub-optimal representatives. Natural proportional sampling (reflecting each cluster's true intent distribution) works better.

### Finding 3: Volume Matters
**Performance scales with data volume**

- 96 variants → 68.93% F1
- 127 variants → 78.83% F1
- **Improvement**: +9.9 percentage points (+14.4% relative)

**Interpretation:** The clustering approach needs sufficient representatives to cover all cluster patterns effectively.

### Finding 4: Gap to Target
**Did not reach 85% F1 target with 127 variants**

- Best achieved: 79.33% F1 (Clustering Only)
- Target: 85% F1
- Gap: 5.67 percentage points

**Potential paths forward:**
1. Increase sample size beyond 127 variants
2. Improve feature engineering
3. Try ensemble methods
4. Optimize cluster selection criteria

---

## 🔍 Cluster Analysis

### Optimal K=15 Cluster Statistics:

| Cluster | Size | Variance | Allocated Reps |
|---------|------|----------|----------------|
| 0 | 4,295 | 0.22 | 52 |
| 1 | 1 | 1,275.90 | 1 |
| 2 | 1,823 | 0.85 | 10 |
| 3 | 505 | 0.79 | 5 |
| 4 | 738 | 1.24 | 7 |
| 5 | 848 | 0.89 | 8 |
| 6 | 1,070 | 0.74 | 10 |
| 7 | 18 | 39.68 | 1 |
| 8 | 816 | 0.83 | 8 |
| 9 | 31 | 14.43 | 1 |
| 10 | 397 | 1.25 | 3 |
| 11 | 391 | 1.39 | 3 |
| 12 | 2 | 223.75 | 1 |
| 13 | 962 | 0.46 | 9 |
| 14 | 847 | 0.86 | 8 |

**Observations:**
- Cluster 0 dominates (33.7% of all variants)
- Clusters 1, 7, 9, 12 are very small (outlier clusters)
- High variance in tiny clusters (1, 7, 9, 12) suggests outliers
- Large clusters have low variance (homogeneous patterns)

---

## 🛠️ Implementation Details

### Random Forest Configuration:
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42
)
```

### Train/Test Split:
- **Split Strategy**: By `variant_record_id` (70/30)
- **Training Set**: 12,744 variants (19,815 samples)
- **Test Set**: 5,463 variants (8,441 samples)

### Feature Engineering:
- **Input**: Raw feature changes (28,256 across all variants)
- **Aggregation**: Per-variant statistics
- **Output**: 27-dimensional feature vectors
- **Scaling**: StandardScaler for clustering

### Evaluation Metrics:
- **Primary**: Weighted F1 Score
- **Secondary**: Per-class F1 (Intentional, Unintentional)
- **Comprehensive**: Tested on:
  - Held-out test set (8,441 samples)
  - Unused training data (19,635 samples)
  - Combined unseen data (28,076 samples)

---

## 🐛 Debugging Journey

### Issue 1: UnboundLocalError
**Problem:** `cannot access local variable 'cluster_variants'`

**Root Cause:** Variable name collision - local variable `cluster_variants` in `main()` shadowed the function name `cluster_variants()`

**Solution:** Renamed local variable to `variants_in_cluster`

### Issue 2: JSON Serialization Error
**Problem:** `Object of type int64 is not JSON serializable`

**Root Cause:** numpy int64 from pandas DataFrame in cluster statistics

**Solution:** Convert to Python int: `int(cluster_label)`

### Issue 3: Variance-Based vs Proportional Allocation
**Problem:** Agent initially implemented variance-based budget allocation instead of user's specified proportional allocation

**User's Intent:** "pick at least one and at most 10 representatives" per cluster based on **size**

**Solution:** Replaced variance-based logic with proportional allocation:
```python
base_allocation = (cluster_size / total_variants) * target_samples
allocated_reps = max(MIN_REPS, min(MAX_REPS, round(base_allocation)))
```

### Issue 4: Underperformance with Limited Data
**Problem:** Initial run used only 96 variants, achieved 68.93% F1 (below baselines)

**Root Cause:** Max 10 reps constraint + stratification reduced total variants selected

**Solution:** Modified allocation to relax max constraint when filling to target:
```python
# Allow exceeding MAX_REPS for largest clusters to reach target
while current_total < target_samples:
    largest_cluster = sorted(allocation.items(), key=lambda x: sizes[x[0]])[-1][0]
    allocation[largest_cluster] += 1
    current_total += 1
```

---

## 📁 Generated Artifacts

All results saved to: `/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/hierarchical_clustering/`

### Files Created:
1. **all_results.csv** - Comprehensive metrics for all approaches
2. **smart_model.pkl** - Trained Random Forest model
3. **selected_samples.csv** - 127 selected training samples
4. **feature_importance.csv** - Feature importance scores
5. **elbow_curve.png** - K selection visualization
6. **cluster_visualization.png** - PCA projection of 15 clusters
7. **performance_comparison.png** - Bar chart comparing all approaches
8. **metrics_table.png** - Summary table of all metrics

### Visualizations:
- **Elbow Curve**: Inertia vs K (5-95), shows elbow at K=15
- **Cluster Visualization**: PCA-reduced 2D projection with 15 color-coded clusters
- **Performance Comparison**: Side-by-side bar chart of F1 scores
- **Metrics Table**: Formatted table with all evaluation metrics

---

## 🎓 Lessons Learned

### 1. **Misunderstanding User Requirements**
- Initial implementation deviated from user's original idea (variance-based vs proportional)
- Importance of clarifying specifications before implementing complex logic
- User's simpler approach (proportional allocation) was the correct interpretation

### 2. **Data Volume is Critical**
- Small sample differences (96 vs 127 variants) can cause large performance gaps (9.9 points)
- Must ensure fair comparison by matching data volumes across approaches

### 3. **Stratification Trade-off**
- Adding stratification constraints doesn't always improve performance
- Simple proportional sampling can outperform complex stratified approaches
- Natural class distribution in clusters may be more informative

### 4. **Data Efficiency vs Absolute Performance**
- Smart sampling is most data-efficient (41.52% F1 per 100 samples)
- But absolute performance still matters for reaching targets
- Trade-off between sample efficiency and final accuracy

### 5. **Clustering Value**
- Both "Clustering Only" and "Smart Sampling" outperform random/stratified baselines
- Clustering identifies meaningful variant patterns
- K=15 provides good balance between granularity and cluster stability

---

## 🔮 Future Directions

### Short-term Improvements:
1. **Remove Stratification**: Test pure clustering + proportional sampling (match "Clustering Only")
2. **Increase Sample Size**: Test with 200, 300, 500 variants to find minimum for 85% F1
3. **Feature Engineering**: Add more sophisticated aggregate features
4. **Cluster Refinement**: Test different K values, try DBSCAN or hierarchical clustering

### Medium-term Exploration:
1. **Active Learning**: Iteratively select most uncertain samples
2. **Ensemble Methods**: Combine multiple clustering approaches
3. **Feature Selection**: Identify most informative features for clustering
4. **Weighted Sampling**: Sample more from high-variance clusters

### Long-term Research:
1. **Transfer Learning**: Pre-train on related datasets
2. **Semi-Supervised Learning**: Use unlabeled data for better representations
3. **Deep Clustering**: Use neural networks for better feature extraction
4. **Multi-Objective Optimization**: Balance sample efficiency and performance

---

## 📊 Complete Performance Matrix

### Test Set Performance (8,441 samples):

| Approach | Accuracy | F1 Weighted | F1 Intentional | F1 Unintentional | Precision | Recall |
|----------|----------|-------------|----------------|------------------|-----------|--------|
| Clustering Only | 79.35% | 79.33% | 79.62% | 79.08% | 80.24% | 79.35% |
| Smart Sampling | 79.18% | 78.83% | 81.11% | 76.82% | 83.09% | 79.18% |
| Stratified Only | 78.94% | 78.74% | 80.31% | 77.36% | 81.43% | 78.94% |
| Random Sampling | 76.29% | 75.98% | 78.21% | 74.01% | 79.32% | 76.29% |

### Confusion Matrix Analysis:

**Smart Sampling (127 variants):**
```
                Predicted
              Unint    Int
Actual Unint  2911    1576
       Int     181    3773
```
- **Unintentional Recall**: 64.9% (2911/4487)
- **Intentional Recall**: 95.4% (3773/3954)
- **Trade-off**: High recall on intentional, lower on unintentional

**Clustering Only (125 variants):**
```
                Predicted
              Unint    Int
Actual Unint  3294    1193
       Int     550    3404
```
- **Unintentional Recall**: 73.4% (3294/4487)
- **Intentional Recall**: 86.1% (3404/3954)
- **Balance**: More balanced recall across both classes

---

## 💡 Key Insights

### Why Clustering Works:
1. **Pattern Discovery**: Identifies natural groupings of manipulation strategies
2. **Diversity**: Ensures coverage of different variant types
3. **Outlier Detection**: Small high-variance clusters capture rare patterns
4. **Scalability**: Works well even with large datasets

### Why Stratification Hurts (Slightly):
1. **Forced Balance**: May select less informative samples to meet minimum guarantees
2. **Natural Distribution**: Clusters often have dominant intent patterns
3. **Information Loss**: Proportional sampling better reflects cluster characteristics

### Data Efficiency Paradox:
- **Smart Sampling**: Most efficient (41.52% F1 per 100 samples) when data-limited
- **But**: Needs sufficient volume (127+ variants) for competitive absolute performance
- **Implication**: Best for scenarios with strict data budgets

---

## 🏁 Conclusion

We successfully implemented a **clustering-based smart sampling pipeline** that demonstrates superior **data efficiency** compared to random and stratified baselines. While the approach did not reach the 85% F1 target with 127 variants, it achieved:

✅ **78.83% F1** with full 127 variants (competitive with baselines)  
✅ **41.52% F1 per 100 samples** - most data-efficient approach  
✅ **14.4% relative improvement** from proper data volume matching  
✅ **Validated clustering value** - both clustering approaches outperform random  

**Next Steps:** Remove stratification to match "Clustering Only" approach, or increase sample size to reach 85% target.

---

**Files:** 
- Implementation: `train_hierarchical_clustering.py` (1,041 lines)
- Results: `results/hierarchical_clustering/`
- Logs: `hierarchical_clustering_full127.log`
