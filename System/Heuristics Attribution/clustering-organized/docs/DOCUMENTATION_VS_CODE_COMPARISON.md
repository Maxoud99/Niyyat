# Documentation vs Code Comparison Report

**Generated:** December 10, 2025  
**Purpose:** Identify discrepancies between COMPLETE_PIPELINE_EXPLANATION.md and actual code

---

## ✅ ACCURATE SECTIONS

### 1. Overview & High-Level Flow
- ✅ 6 clustering algorithms listed correctly
- ✅ 2 sampling strategies (Normal & Smart) 
- ✅ 12 combinations total (6 × 2)
- ✅ 1% auto-calculation feature documented
- ✅ 15 detailed log files documented

### 2. Aggregate Features for Clustering
- ✅ **RECENTLY FIXED** - Now correctly documents 15 aggregate features per variant
- ✅ Lists all features accurately:
  - 4 count features (n_changes, n_intentional, n_unintentional, intentional_ratio)
  - 5 magnitude statistics
  - 2 value statistics  
  - 4 derived features

### 3. Auto-Calculated 1% Target Samples
- ✅ Formula correct: `max(1, int(total_variants * 0.01))`
- ✅ Examples accurate (18,207 → 182)
- ✅ Override capability documented

### 4. DBSCAN Fix
- ✅ Problem statement accurate (eps=0.0 error)
- ✅ Solution documented correctly (fallback to 0.5)
- ✅ Code snippets match actual implementation

### 5. Train/Test Split
- ✅ Correctly documents exclusion logic using `~` operator
- ✅ Percentages calculation accurate

---

## ⚠️ DISCREPANCIES FOUND

### **CRITICAL ISSUE #1: Classifier Training Features**

**Location:** Phase 4, Step 4.1 "Create Training Set"

**Documentation Says:**
```python
for _, change in variant_changes.iterrows():
    feature_vector = create_feature_vector(change)  # ❌ Vague
```

**Actual Code Does:**
```python
# From load_data() - Lines 329-345
self.df['feature_name_encoded'] = pd.Categorical(self.df['feature_name'].astype(str)).codes
self.df['original_value_encoded'] = pd.Categorical(self.df['original_value'].astype(str)).codes
self.df['new_value_encoded'] = pd.Categorical(self.df['new_value'].astype(str)).codes
self.df['relative_change'] = self.df['change_magnitude'] / (original_numeric.abs() + 1)
self.df['change_direction'] = np.sign(new_numeric - original_numeric)
self.df['original_log'] = np.log1p(original_numeric.abs())
self.df['new_log'] = np.log1p(new_numeric.abs())
self.df['original_magnitude'] = original_numeric.abs()
self.df['new_magnitude'] = new_numeric.abs()

# Add feature type indicators
for feat in self.df['feature_name'].unique():
    self.df[f'feat_{feat}'] = (self.df['feature_name'] == feat).astype(int)
```

**Features Used for Training (from train_and_evaluate):**
```python
feature_cols = [col for col in self.df.columns 
               if col not in ['variant_record_id', 'original_record_id', 
                             'feature_name', 'original_value', 'new_value', 
                             'intent_label']]
```

**Actual Training Features (~20+ features per sample):**
1. `change_magnitude` - Numeric change size
2. `variant_idx` - Which variant (0, 1, or 2)
3. `feature_name_encoded` - Which feature changed (encoded)
4. `original_value_encoded` - Original value (encoded)
5. `new_value_encoded` - New value (encoded)
6. `relative_change` - Relative magnitude
7. `change_direction` - Sign of change (+1, 0, -1)
8. `original_log` - Log-transformed original
9. `new_log` - Log-transformed new
10. `original_magnitude` - Absolute original value
11. `new_magnitude` - Absolute new value
12. `feat_age`, `feat_workclass`, `feat_education`, ... - **One-hot encoded feature indicators**

**Impact:** 🔴 **CRITICAL** - Documentation doesn't explain the actual feature engineering for classifier training

---

### **ISSUE #2: Two Different Feature Sets**

**Confusion:** The pipeline uses TWO completely different feature sets:

| Purpose | Feature Set | Dimensionality | Used For |
|---------|-------------|----------------|----------|
| **Clustering** | 15 aggregate features per variant | 15D | Grouping similar variants |
| **Classification** | Individual change features | ~20-30D | Predicting intent labels |

**Documentation Problem:**
- ✅ Correctly explains clustering features (15 aggregates)
- ❌ Doesn't clearly explain classification features
- ❌ Doesn't emphasize this is TWO SEPARATE transformations

---

### **ISSUE #3: Random Forest Hyperparameters**

**Documentation Says (Phase 4, Step 4.3):**
```python
clf = RandomForestClassifier(
    n_estimators=100,      # 100 decision trees ❌
    max_depth=None,        # Grow trees fully ❌
    min_samples_split=2,   # ❌
    min_samples_leaf=1,    # ❌
    random_state=42,
    n_jobs=-1
)
```

**Actual Code (Line 1050):**
```python
rf = RandomForestClassifier(
    n_estimators=200,      # 200 trees ✅
    max_depth=15,          # Limited depth ✅
    min_samples_split=5,   # Different ✅
    class_weight='balanced', # MISSING FROM DOCS ✅
    random_state=42,
    n_jobs=-1
)
```

**Impact:** 🟡 **MEDIUM** - Hyperparameters are significantly different

---

### **ISSUE #4: Number of Algorithms**

**Documentation Says:**
- "8 clustering algorithms" (appears in multiple places)
- Lists: K-Means, DBSCAN, Ward, Average, Complete, Spectral, GMM, HDBSCAN

**Actual Code Tests:**
- **6 algorithms**: K-Means, DBSCAN, Ward, Average, GMM, HDBSCAN
- **NOT IMPLEMENTED:** Complete Linkage, Spectral Clustering

**Impact:** 🟡 **MEDIUM** - Documentation claims more algorithms than actually tested

---

### **ISSUE #5: Sampling Constraints**

**Documentation Says (Normal Sampling):**
```python
samples_per_cluster[cluster_id] = max(1, min(10, round(proportion * target_samples)))
```

**Actual Code (Lines 748, 754):**
```python
# Normal sampling - NO MAX CONSTRAINT
allocation[label] = max(1, int(round(base_allocation)))  # Only minimum=1

# Smart sampling - HAS MAX=10 constraint
allocation[label] = max(MIN_REPS, min(MAX_REPS, round(base_allocation)))
```

**Impact:** 🟡 **MEDIUM** - Normal sampling has NO max constraint (documentation shows min(10, ...))

---

### **ISSUE #6: Feature Type Indicators**

**Missing from Documentation:**

The code creates **one-hot encoded feature type indicators**:

```python
# Add feature type indicators
for feat in self.df['feature_name'].unique():
    self.df[f'feat_{feat}'] = (self.df['feature_name'] == feat).astype(int)
```

This means if the dataset has 14 features (age, workclass, education, etc.), the classifier gets 14 additional binary features indicating WHICH feature was changed.

**Total Classifier Features:** Base ~11 + Feature indicators ~14 = **~25 features per sample**

**Impact:** 🟡 **MEDIUM** - Documentation missing important feature engineering step

---

## 📊 SEVERITY SUMMARY

| Issue | Severity | Section | Fix Required |
|-------|----------|---------|--------------|
| Classifier training features not documented | 🔴 CRITICAL | Phase 4 | Complete rewrite |
| Two feature sets not clearly separated | 🔴 CRITICAL | Phase 1 & 4 | Add clarity section |
| Random Forest hyperparameters wrong | 🟡 MEDIUM | Phase 4 | Update parameters |
| Algorithm count mismatch (8 vs 6) | 🟡 MEDIUM | Multiple | Update to 6 |
| Normal sampling max constraint wrong | 🟡 MEDIUM | Phase 3 | Remove min(10, ...) |
| Feature type indicators missing | 🟡 MEDIUM | Phase 1 | Add documentation |

---

## 🔧 RECOMMENDED FIXES

### Fix #1: Add Clear Section on Two Feature Transformations

**Insert after Step 1.3:**

```markdown
### Step 1.3b: Understanding the Two Feature Sets ⚠️ IMPORTANT

This pipeline uses TWO COMPLETELY DIFFERENT feature transformations:

**Feature Set #1: For Clustering (15 aggregate features per variant)**
- Purpose: Group similar variants together
- Granularity: One vector per variant
- Features: Statistics about all changes in a variant (mean, std, counts, etc.)
- Used by: K-Means, DBSCAN, Hierarchical, GMM, HDBSCAN

**Feature Set #2: For Classification (~25 features per individual change)**
- Purpose: Predict if a single feature change is intentional or unintentional
- Granularity: One vector per feature change
- Features: Details about the specific change (magnitude, direction, encoded values, etc.)
- Used by: Random Forest classifier

**Example:**
```
Variant #42 has 3 feature changes:
  - age: 39 → 42
  - hours: 40 → 45  
  - education: HS → Bach

For Clustering:
  → 1 vector with 15 aggregate stats

For Classification:
  → 3 vectors, each with ~25 features
```
```

### Fix #2: Document Actual Classifier Features

**Replace Phase 4, Step 4.1 with:**

```markdown
### Step 4.1: Create Training Set

After sampling variants via clustering, we extract **individual feature changes** for classifier training.

**Features for EACH change (~25 total):**

```python
# Base features (11):
- change_magnitude          # How much it changed
- variant_idx               # Which variant (0, 1, 2)
- feature_name_encoded      # Which feature (age=0, education=1, etc.)
- original_value_encoded    # Original value encoded
- new_value_encoded         # New value encoded
- relative_change           # change / (|original| + 1)
- change_direction          # +1, 0, or -1
- original_log              # log1p(|original|)
- new_log                   # log1p(|new|)
- original_magnitude        # |original|
- new_magnitude             # |new|

# Feature type indicators (one-hot encoded, ~14):
- feat_age                  # 1 if age changed, 0 otherwise
- feat_workclass            # 1 if workclass changed, 0 otherwise
- feat_education            # 1 if education changed, 0 otherwise
- ... (one per feature in dataset)
```

**Example:**
```
Variant #42, age changed 39 → 42:

Feature vector:
[3,           # change_magnitude
 0,           # variant_idx
 0,           # feature_name_encoded (age=0)
 39,          # original_value_encoded
 42,          # new_value_encoded
 0.077,       # relative_change (3/40)
 1,           # change_direction (positive)
 3.69,        # original_log
 3.76,        # new_log
 39,          # original_magnitude
 42,          # new_magnitude
 1,           # feat_age (this feature changed)
 0,           # feat_workclass (not changed)
 0,           # feat_education (not changed)
 ...]         # ... other feature indicators
```
```

### Fix #3: Correct Random Forest Parameters

**Replace Phase 4, Step 4.3:**

```python
rf = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees (not 100)
    max_depth=15,            # Limited depth to prevent overfitting
    min_samples_split=5,     # Minimum 5 samples to split
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1
)
```

### Fix #4: Update Algorithm Count

**Global search/replace:**
- "8 clustering algorithms" → "6 clustering algorithms"
- Remove references to Complete Linkage and Spectral from lists
- Update "16 total combinations" → "12 total combinations"

### Fix #5: Fix Normal Sampling Pseudocode

**Phase 3, Strategy 1:**

```python
# WRONG:
samples_per_cluster[cluster_id] = max(1, min(10, round(proportion * target_samples)))

# CORRECT:
samples_per_cluster[cluster_id] = max(1, round(proportion * target_samples))
# Only minimum=1 constraint, NO maximum
```

---

## ✅ VALIDATION CHECKLIST

- [ ] Two feature sets clearly distinguished
- [ ] Classifier features fully documented (~25 features)
- [ ] Random Forest hyperparameters match code
- [ ] Algorithm count is 6 (not 8)
- [ ] Normal sampling has no max constraint
- [ ] Feature type indicators documented
- [ ] All code snippets tested against actual code
- [ ] Examples use real data from user's runs

---

**Next Steps:** Apply these fixes to COMPLETE_PIPELINE_EXPLANATION.md
