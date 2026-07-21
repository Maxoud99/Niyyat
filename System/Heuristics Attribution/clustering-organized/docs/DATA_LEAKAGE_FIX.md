# Data Leakage Fix - Removed Intent Labels from Clustering Features

**Date:** December 10, 2025  
**Status:** ✅ IMPLEMENTED & TESTED  
**Impact:** CRITICAL - Fixes fundamental data leakage issue

---

## Problem Identified

The pipeline was using **intent labels** (intentional/unintentional) as features for clustering:
- `n_intentional` - Count of intentional changes
- `n_unintentional` - Count of unintentional changes  
- `intentional_ratio` - Proportion of intentional changes

**This is DATA LEAKAGE!**

Intent labels are what we're trying to **predict** in Phase 4 (classifier training). Using them to create clusters in Phase 2 means:
- The clustering already "knows" the answer
- The model is cheating by using information it shouldn't have
- Results would not generalize to real-world scenarios where intent is unknown

---

## Solution

**Removed 3 intent-based features from clustering:**

### Before (15 → 13 → 10 features):
```python
features = {
    'n_changes': n_changes,
    'n_intentional': n_intentional,          # ❌ DATA LEAKAGE!
    'n_unintentional': n_unintentional,      # ❌ DATA LEAKAGE!
    'intentional_ratio': intentional_ratio,   # ❌ DATA LEAKAGE!
    'mean_magnitude': mean_magnitude,
    'std_magnitude': std_magnitude,
    # ... 10 more features
}
```

### After (10 features):
```python
features = {
    'n_changes': n_changes,                   # ✅ OK - just counts changes
    'mean_magnitude': mean_magnitude,         # ✅ OK - magnitude patterns
    'std_magnitude': std_magnitude,
    'min_magnitude': min_magnitude,
    'max_magnitude': max_magnitude,
    'median_magnitude': median_magnitude,
    'min_new_value_encoded': min_new_value_encoded,
    'max_new_value_encoded': max_new_value_encoded,
    'mean_relative_change': mean_relative_change,
    'feature_with_max_change_encoded': feature_with_max_change,
}
```

---

## Changes Made

### 1. Code Changes (`scripts/compare_clustering_algorithms.py`)

**Function:** `create_aggregate_features()`

**Removed:**
```python
# Basic counts and intent statistics
n_intentional = (variant_data['intent_label'] == 1).sum()
n_unintentional = (variant_data['intent_label'] == -1).sum()
intentional_ratio = (variant_data['intent_label'] == 1).mean()
```

**Updated comment:**
```python
# Basic counts (NO INTENT LABELS - avoid data leakage!)
n_changes = len(variant_data)
```

**Updated docstring:**
```python
def create_aggregate_features(self, df):
    """
    Create aggregate features per variant for clustering.
    
    CRITICAL: Does NOT use intent labels to avoid data leakage!
    Intent labels are what we're trying to predict, so they shouldn't
    influence the clustering.
    
    Creates 10 features per variant:
    - 1 count feature (n_changes)
    - 5 magnitude statistics
    - 2 encoded value statistics  
    - 2 derived features
    """
```

### 2. Documentation Changes (`docs/COMPLETE_PIPELINE_EXPLANATION.md`)

**Updated sections:**
- Phase 1, Step 1.3: Feature creation code and example
- Phase 2: Aggregate Features section
- Pipeline diagram (15 → 13 → 10 features)

**Added warnings:**
```markdown
**CRITICAL - NO DATA LEAKAGE:** We do NOT include intent labels in these features!
Intent labels are what we're trying to predict, so using them for clustering
would be data leakage.
```

**Documented removals:**
```markdown
**REMOVED TO AVOID DATA LEAKAGE (December 10, 2025):**
- ~~`n_intentional`~~ - Using intent labels for clustering is data leakage!
- ~~`n_unintentional`~~ - Intent is what we're trying to predict
- ~~`intentional_ratio`~~ - Should not influence clustering
```

---

## Testing

**Test Command:**
```bash
cd /home/mohamed/error_injector/llms_baseline/clustering-organized
python3 scripts/compare_clustering_algorithms.py --target_samples 10
```

**Result:**
```
✓ Created 18207 aggregate feature vectors
  Feature dimension: 10
```

✅ **Pipeline runs successfully with 10 features**

---

## Timeline of Feature Reductions

| Date | Features | Removed | Reason |
|------|----------|---------|--------|
| Original | 15 | - | Initial implementation |
| Dec 10, 2025 (1st) | 13 | `feature_diversity`, `most_common_feature` | Redundant |
| Dec 10, 2025 (2nd) | **10** | `n_intentional`, `n_unintentional`, `intentional_ratio` | **Data leakage** |

---

## Impact on Results

**Expected Changes:**
- Clustering will now be based purely on **change patterns** (magnitude, diversity, encoded values)
- No longer influenced by intent labels
- Results will be more realistic and generalizable
- F1 scores may change (could increase or decrease)

**Why this is better:**
- Honest evaluation - no cheating
- Real-world applicable - intent is unknown at clustering time
- Prevents overfitting to training data patterns

---

## Related Documents

- `COMPLETE_PIPELINE_EXPLANATION.md` - Full pipeline documentation (updated)
- `REDUNDANT_FEATURES_REMOVAL.md` - First removal (feature_diversity, most_common_feature)
- This document - Second removal (intent labels)

---

## Verification Checklist

- [x] Removed intent labels from `create_aggregate_features()`
- [x] Updated function docstring
- [x] Updated code comments
- [x] Tested pipeline with 10 features
- [x] Updated COMPLETE_PIPELINE_EXPLANATION.md (Step 1.3)
- [x] Updated COMPLETE_PIPELINE_EXPLANATION.md (Phase 2)
- [x] Updated pipeline diagram
- [x] Documented removals with reasons
- [x] Created this summary document

---

## Conclusion

✅ **Data leakage issue resolved!**

The pipeline now uses only 10 legitimate features for clustering that don't leak information about the intent labels we're trying to predict. This makes the evaluation honest and the results more meaningful.
