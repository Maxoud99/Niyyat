# URGENT: Label Propagation Script Requires Complete Rewrite

**Date:** January 28, 2026  
**Status:** 🔴 CRITICAL ERROR - Script tests wrong problem

---

## 🚨 Problem Summary

The `compare_label_propagation.py` script is fundamentally broken because it solves **THE WRONG PROBLEM**.

### ❌ What the Script Currently Does (WRONG)
- **Task:** Binary classification of manipulated (1) vs correct (0) cells
- **Dataset:** All 60,000 cells (97.65% correct, 2.35% manipulated)
- **Labels:** 0 = correct, 1 = error/manipulated
- **Result:** F1 = 0.42 (meaningless - wrong task!)

### ✅ What It Should Do (CORRECT - Match Your Pipeline)
- **Task:** Binary classification of intentional (1) vs unintentional (-1) manipulations
- **Dataset:** Only ~28,256 manipulated features (excludes correct features)
- **Labels:** 1 = intentional, -1 = unintentional (intent classification)
- **Baseline:** HDBSCAN + RF = **91.87% F1** (91.53% intentional, 92.17% unintentional)

---

## 📊 Your Actual Results (From run_20251210_172614-tenth-1%-final)

```
Algorithm: HDBSCAN
F1 Score: 0.9187 (91.87%)
F1 Intentional: 0.9153 (91.53%)
F1 Unintentional: 0.9217 (92.17%)
Silhouette Score: 0.9295
Variants Selected: 585 / 18207 (3.21%)
Training Samples: 1022 (3.62%)
Runtime: 3.16s
```

This is **EXCELLENT** performance for intent classification!

---

## 🔧 Required Changes

### 1. Data Loading
**Current (WRONG):**
```python
# Loads combined_dataset_no_id_v2.csv + ground_truth_masks_v2.csv
# Includes ALL cells (correct + manipulated)
# Labels: 0=correct, 1=manipulated
```

**Required (CORRECT):**
```python
# Load correct_records.csv + manipulated_records.csv + masks.csv
# Filter to ONLY manipulated features (intent != 0)
# Labels: 1=intentional, -1=unintentional
# Data: tenth-trial/data/raw/run_20251031_211812/
```

### 2. Aggregation
**Current (WRONG):**
```python
# Aggregate per record (all 4000 records)
# Include statistics for all cells
```

**Required (CORRECT):**
```python
# Aggregate per variant (18,207 variants)
# Only consider manipulated features per variant
# Match your pipeline's 10 aggregate features
```

### 3. Target Labels
**Current (WRONG):**
```python
y = df['error_label']  # 0 or 1
```

**Required (CORRECT):**
```python
y = df['intent_label']  # 1 or -1 (convert -1 to 0 for sklearn)
```

### 4. Evaluation
**Current (WRONG):**
```python
# Evaluate on all 60,000 cells
# Metrics biased by 97.65% correct cells
```

**Required (CORRECT):**
```python
# Evaluate on ~27,000 unseen manipulated features
# Clean metrics for intent classification
```

---

## 📝 Action Plan

1. **Stop using current script** - results are meaningless
2. **Rewrite from scratch** to match your pipeline architecture
3. **Use correct data paths:**
   - `/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/`
   - `correct_records.csv`
   - `run_20251031_211812/manipulated_records.csv`
   - `run_20251031_211812/masks.csv`
4. **Match your pipeline's processing:**
   - Extract only manipulated features (intent != 0)
   - Create 10 aggregate features per variant
   - Use proportional sampling from clusters
   - Train/evaluate on intent labels (1 vs -1)
5. **Compare against your 91.87% baseline**

---

## 🎯 Expected Results After Fix

| Approach | F1 Intentional | F1 Unintentional | Weighted F1 |
|----------|---------------|------------------|-------------|
| **Random Forest (your pipeline)** | **91.53%** | **92.17%** | **91.87%** |
| Label Propagation (to test) | 85-90%? | 85-90%? | 85-90%? |
| LabelSpreading (to test) | 85-90%? | 85-90%? | 85-90%? |

**Hypothesis:** Label propagation will likely achieve 85-90% F1 (worse than RF but reasonable).

**Why:** With balanced classes (~47% intentional, ~53% unintentional) and focused task, label propagation should work reasonably well. But RF's ability to learn complex decision boundaries will likely still win.

---

## ✅ Correct Research Question

**Can label propagation replace Random Forest for classifying manipulations as intentional vs unintentional?**

Answer this by comparing against your **91.87% F1 baseline**, NOT the meaningless 42% result!

---

**Status:** 🔴 Requires complete rewrite  
**Priority:** HIGH - Current results are misleading  
**Next Step:** Rewrite script to match your successful clustering pipeline
