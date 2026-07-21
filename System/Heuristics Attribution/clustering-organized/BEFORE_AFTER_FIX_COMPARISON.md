# Before/After Fix Comparison

## FeatureLevelKNN: Before vs After Data Leakage Fix

| Aspect | BEFORE (with leakage) | AFTER (fixed) |
|--------|----------------------|---------------|
| **Features Used** | 21 features | 10 features |
| **Includes feat_* indicators** | ✅ YES (15 indicators) | ❌ NO (removed) |
| **F1 Score** | 91.72% | 91.03% |
| **vs Baseline** | -0.15% | -0.84% |
| **Data Leakage** | ❌ YES | ✅ NO |
| **Label Propagation** | ❌ NO (supervised learning) | ✅ YES (true propagation) |
| **Runtime** | 0.02s | 0.06s |

**Drop in performance after fix:** -0.69 percentage points (91.72% → 91.03%)

This confirms the feat_* indicators were providing unfair advantage by encoding feature-specific patterns.

---

## All Methods Comparison

### Training Data (ALL USE THE SAME)
```
Sampled: 177 variants (0.98% of 18,026)
Features: 242 (0.97% of 25,022)
  ├─ Intentional labels: 109 (45.0%)
  └─ Unintentional labels: 133 (55.0%)
```

### Results

| Method | F1 Score | F1 Int | F1 Unint | Runtime | vs Baseline | Status |
|--------|----------|--------|----------|---------|-------------|--------|
| **ClusterMajorityVote** | **95.48%** | **95.15%** | **95.78%** | **0.13s** | **+3.61%** | ✅ **BEST** |
| Baseline (HDBSCAN+RF) | 91.87% | 91.53% | 92.17% | ~0.3s | - | ✅ Reference |
| FeatureLevelKNN (BEFORE) | 91.72% | 91.32% | 92.06% | 0.02s | -0.15% | ❌ Data leakage |
| Random Forest | 91.10% | 90.43% | 91.70% | 0.24s | -0.77% | ✅ Valid |
| FeatureLevelKNN (AFTER) | 91.03% | 90.52% | 91.48% | 0.06s | -0.84% | ✅ Valid |
| Label Propagation | 53.05% | 52.72% | 53.34% | 0.42s | -38.82% | ❌ Fails |

---

## Sample Count Breakdown

### Dataset Statistics
```
Total Features: 25,022
  ├─ Intentional: 11,781 (47.1%)
  └─ Unintentional: 13,241 (52.9%)

Total Variants: 18,026

Sampling: 1%
  ├─ Sampled Variants: 177 (0.98%)
  └─ Sampled Features: 242 (0.97%)
```

### Training/Test Split
```
Training:
  Variants: 177 (0.98%)
  Features: 242 (0.97%)
    ├─ Intentional: 109 (45.0%)
    └─ Unintentional: 133 (55.0%)

Testing:
  Variants: 17,849 (99.02%)
  Features: 24,780 (99.03%)
```

---

## Method Details

### 1. ClusterMajorityVote (95.48%) ✅
```
Training: 177 variants, 242 features (109 int + 133 unint)
Method: K-Means clustering → majority vote per cluster
Features: 10 aggregate features (no intent labels)
Result: 95.48% F1 (+3.61% vs baseline)
Why: 98% cluster purity, variant-level prediction
```

### 2. Baseline - HDBSCAN+RF (91.87%) ✅
```
Training: 177 variants, 242 features (109 int + 133 unint)
Method: HDBSCAN clustering → Random Forest classifier
Features: Individual feature representations
Result: 91.87% F1 (reference)
```

### 3. Random Forest (91.10%) ✅
```
Training: 177 variants, 242 features (109 int + 133 unint)
Method: Random Forest on individual features
Features: Individual feature representations
Result: 91.10% F1 (-0.77% vs baseline)
```

### 4. FeatureLevelKNN - BEFORE FIX (91.72%) ❌
```
Training: 177 variants, 242 features (109 int + 133 unint)
Method: k-NN (k=7) on individual features
Features: 21 (10 generic + 11 feat_* indicators) ← DATA LEAKAGE
Result: 91.72% F1 (-0.15% vs baseline)
Issue: feat_* indicators encode feature-specific patterns
```

### 5. FeatureLevelKNN - AFTER FIX (91.03%) ✅
```
Training: 177 variants, 242 features (109 int + 133 unint)
Method: k-NN (k=7) on individual features
Features: 10 generic features (NO feat_* indicators)
Result: 91.03% F1 (-0.84% vs baseline)
Fixed: Removed data leakage, now legitimate
```

### 6. Label Propagation (53.05%) ❌
```
Training: 177 variants (0.98%)
Method: sklearn LabelPropagation (graph-based)
Features: Variant-level aggregates
Result: 53.05% F1 (-38.82% vs baseline)
Issue: 97.4% remain unlabeled (sparse labels problem)
```

---

## Key Findings

### 1. All Methods Use Identical Training Data
- **177 variants** (0.98% of dataset)
- **242 features** (0.97% of dataset)
- **109 intentional + 133 unintentional labels**

No method has an advantage in training data quantity.

### 2. Why ClusterMajorityVote Wins
- **98% cluster purity** - clusters naturally separate by intent
- **K-Means** creates balanced clusters (vs HDBSCAN with noise)
- **Variant-level** prediction more stable than feature-level
- **Simple** majority voting beats complex classifiers

### 3. Data Leakage Impact
- **Before fix**: 91.72% (with 11 feat_* indicators)
- **After fix**: 91.03% (without feat_* indicators)
- **Drop**: -0.69 percentage points
- **Conclusion**: feat_* indicators were providing unfair advantage

### 4. Performance Ranking
1. **ClusterMajorityVote**: 95.48% (+3.61% vs baseline) ← USE THIS
2. Baseline (HDBSCAN+RF): 91.87%
3. FeatureLevelKNN (leakage): 91.72% (-0.15%)
4. Random Forest: 91.10% (-0.77%)
5. FeatureLevelKNN (fixed): 91.03% (-0.84%)
6. Label Propagation: 53.05% (-38.82%)

---

## Recommendations

✅ **RECOMMENDED: ClusterMajorityVote**
- Best performance (95.48%)
- No data leakage
- Fast (0.13s)
- Simple and interpretable
- Uses same 177 variants as baseline

❌ **NOT RECOMMENDED: FeatureLevelKNN**
- Below baseline after fix (91.03%)
- Was only competitive due to data leakage
- No advantage over simpler methods

❌ **NOT RECOMMENDED: Label Propagation**
- Catastrophic failure (53.05%)
- 97.4% unlabeled
- Not suitable for sparse labeling

---

## Final Answer to User's Question

### "How many records used for sampling with how many labels?"

**All methods use IDENTICAL training data:**

```
Records (Variants): 177 (0.98% of 18,026 total)
Features: 242 (0.97% of 25,022 total)
Labels:
  ├─ Intentional: 109 (45.0% of training)
  └─ Unintentional: 133 (55.0% of training)
```

**No method has more training data than another.**

The winner (ClusterMajorityVote) succeeds because:
1. K-Means creates purer clusters (98% purity)
2. Variant-level prediction is more stable
3. Natural intent separation in aggregate features

Not because it uses more samples!
