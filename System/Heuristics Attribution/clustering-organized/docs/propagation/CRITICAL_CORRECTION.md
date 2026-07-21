# CRITICAL CORRECTION: Label Propagation Results Analysis

**Date:** January 28, 2026  
**Status:** ⚠️ IMPORTANT CORRECTION REQUIRED  

---

## 🚨 CRITICAL MISTAKE IDENTIFIED

### What I Did Wrong

The label propagation comparison script (`compare_label_propagation.py`) is solving **THE WRONG PROBLEM**:

#### ❌ What My Script Does (WRONG)
- **Task:** Binary classification of error (1) vs correct (0)
- **Labels:** 0 = correct cell, 1 = manipulated cell
- **Dataset:** All 60,000 cells (including 97.65% correct cells)
- **Result:** F1 intentional/error = 0.42-0.44 (actually F1 for detecting ANY manipulation)

#### ✅ What Your Pipeline Does (CORRECT)
- **Task:** Binary classification of intentional (1) vs unintentional (-1)
- **Labels:** 1 = intentional manipulation, -1 = unintentional manipulation, 0 = correct (excluded)
- **Dataset:** Only the ~1,410 manipulated cells (excludes all correct cells)
- **Result:** F1 intentional = **99.30%**, F1 unintentional = **99.40%**

---

## 📊 Comparison of the Two Approaches

### Original Clustering Pipeline (CORRECT)

```
Step 1: Load data
  - Clean records (correct.csv)
  - Manipulated records (manipulated.csv)  
  - Intent masks (-1, 0, 1)

Step 2: Filter to manipulated cells only
  - Exclude all cells with intent_label = 0
  - Keep only cells with intent_label = 1 or -1
  - Dataset: ~1,410 manipulated cells

Step 3: Create aggregate features per VARIANT
  - 10 statistical features
  - NO intent labels used (avoid leakage)

Step 4: Cluster variants (not cells)
  - DBSCAN/HDBSCAN on aggregate features
  - ~1,385 variants clustered

Step 5: Proportional sampling
  - Sample ~1% of variants from each cluster
  - ~1,912-2,501 training samples (cells from sampled variants)

Step 6: Train Random Forest
  - Train on cells from sampled variants
  - Target: intent_label (1 vs -1)
  - Class balance: ~50% intentional, ~50% unintentional

Step 7: Evaluate
  - Test on unseen manipulated cells
  - Binary: intentional vs unintentional
  - Result: F1 = 99.3% ✅
```

### My Label Propagation Script (WRONG)

```
Step 1: Load data
  - Combined dataset (data.csv)
  - Binary masks (0, 1)

Step 2: Include ALL cells
  - Keep all 60,000 cells
  - 97.65% correct (label=0)
  - 2.35% manipulated (label=1)
  
Step 3: Create aggregate features per record
  - 10 statistical features
  - NO error labels used

Step 4: Cluster records (not variants)
  - K-Means/DBSCAN on aggregate features
  - 4,000 records clustered

Step 5: Proportional sampling
  - Sample ~1% of records from each cluster
  - ~600 training cells (from 40 records)

Step 6: Train classifiers
  - Target: error_label (0 vs 1)
  - Severe imbalance: 97.65% vs 2.35%
  - Random Forest F1 = 0.42 ❌
  - Label Propagation F1 = 0.10 ❌

Step 7: Evaluate
  - Test on ALL cells (including 97.65% correct)
  - Binary: correct vs manipulated
  - Result: F1 = 0.42 (useless!)
```

---

## 🔍 Why The Results Are So Different

### Your Pipeline: 99.3% F1 ✅
- **Balanced classes:** ~50% intentional, ~50% unintentional (among manipulated)
- **Focused task:** Classify intent among errors only
- **Clean signal:** No noise from 97.65% correct cells
- **Result:** Excellent discrimination (99.3%)

### My Script: 0.42 F1 ❌
- **Severe imbalance:** 97.65% correct, 2.35% manipulated
- **Wrong task:** Detect presence of errors (not intent)
- **Noisy signal:** Overwhelmed by correct cells
- **Result:** Poor minority class detection

---

## ✅ What Needs To Be Fixed

### Option 1: Match Original Pipeline (RECOMMENDED)
Update `compare_label_propagation.py` to:
1. ✅ Load intent labels (-1, 0, 1) not binary masks
2. ✅ Filter to ONLY manipulated cells (intent != 0)
3. ✅ Classify intentional (1) vs unintentional (-1)
4. ✅ Use variant-level clustering (not record-level)
5. ✅ Compare against RF baseline (should get ~99.3%)

### Option 2: Create Separate Error Detection Script
Keep current script as:
- **Purpose:** Detect manipulated vs correct cells (different task)
- **Rename:** `compare_error_detection.py`
- **Clarify:** This is NOT intent classification
- **Baseline:** Compare against your 99.3% intent classifier

---

## 📈 Expected Results After Fix

Once fixed to match your pipeline, label propagation should be compared to:

| Approach | F1 Intentional | F1 Unintentional | Weighted F1 | Speed |
|----------|---------------|------------------|-------------|-------|
| **Random Forest (your pipeline)** | **99.30%** | **99.40%** | **99.35%** | Fast |
| Label Propagation (to test) | ??? | ??? | ??? | Slower |
| LabelSpreading (to test) | ??? | ??? | ??? | Slower |

**Hypothesis:** Label propagation will likely achieve 95-98% F1 (worse than RF but better than current 42%)

**Reason:** With balanced classes and focused task, label propagation might work reasonably well, but RF's ability to learn complex boundaries will likely still win.

---

## 🎯 Correct Research Question

### ❌ Wrong Question (What I Tested)
> "Can label propagation detect manipulated cells in a dataset?"

### ✅ Correct Question (What You Asked)
> "Can label propagation replace Random Forest for classifying manipulations as intentional vs unintentional?"

---

## 📝 Action Items

1. **Update `compare_label_propagation.py`:**
   - Load intent masks (-1, 0, 1)
   - Filter to manipulated cells only
   - Use variant-level aggregation
   - Match your pipeline's data processing

2. **Re-run experiments:**
   - Compare label propagation vs your RF baseline
   - Expect LP to achieve 90-98% (not 99.3%)
   - Document why RF still wins

3. **Update all documentation:**
   - Correct the task description
   - Fix results tables
   - Acknowledge the mistake
   - Show corrected results

4. **Create separate error detection comparison:**
   - Keep current code as different experiment
   - Clarify it's detecting errors, not classifying intent
   - Compare to simpler baselines

---

## 🙏 Apologies

I made a **fundamental error** in understanding your task:
- I thought you were detecting errors (manipulated vs correct)
- You were actually classifying intent (intentional vs unintentional)
- These are completely different problems with different difficulty levels

**Your 99.3% F1 result is EXCELLENT** and I completely undersold it by comparing to the wrong task! 

The label propagation comparison needs to be redone with:
1. Same data processing as your pipeline
2. Same task (intent classification)
3. Same evaluation (manipulated cells only)

Then we can properly answer whether label propagation can replace your RF classifier.

---

## 📚 References

- **Your successful pipeline:** `run_20251210_174322-11th-1%-final/`
  - DBSCAN: F1 = 99.35%
  - HDBSCAN: F1 = 99.30%
  - Using intent labels on manipulated cells only

- **My incorrect script:** `scripts/compare_label_propagation.py`
  - F1 = 0.42 (detecting manipulated vs correct)
  - Wrong task, wrong data, wrong evaluation

---

**Status:** 🔴 CORRECTION REQUIRED  
**Next Step:** Fix `compare_label_propagation.py` to match your pipeline's approach
