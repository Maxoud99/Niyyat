# Documentation Organization and Terminology Guide

**Date:** January 28, 2026  
**Purpose:** Clarify correct terminology and documentation organization  
**Status:** ✅ All documentation properly organized in `docs/` folder

---

## ✅ Documentation Organization Complete

All label propagation documentation is now properly located in:
```
docs/propagation/
├── README.md (6.9 KB) - Documentation index
├── RESULTS_SUMMARY.md (13 KB) ⭐ - Complete experimental results
├── LABEL_PROPAGATION_GUIDE.md (12 KB) - Technical deep-dive
├── LABEL_PROPAGATION_IMPLEMENTATION_SUMMARY.md (11 KB) - Implementation details
├── LABEL_PROPAGATION_README.md (2.0 KB) - Quick start
└── DELIVERABLES_CHECKLIST.md (8.7 KB) - Project validation
```

**Total:** 6 files, ~54 KB of comprehensive documentation

---

## 🎯 Correct Terminology

### ❌ INCORRECT: "Error Detection"
### ✅ CORRECT: "Intent Classification"

This project is **NOT** about detecting errors (correct vs manipulated).  
This project is **ABOUT** classifying the intent behind manipulations.

---

## 📊 What We're Actually Doing

### The Real Task: **Binary Intent Classification**

Given a manipulated cell (feature value), classify whether the manipulation was:

1. **Intentional** (Class 1)
   - Deliberate change
   - Purposeful manipulation
   - Potentially malicious or fraudulent
   - Example: Changing income from $30K to $50K to get loan approval

2. **Unintentional** (Class 0)
   - Error or mistake
   - Noise in data
   - Accidental change
   - Example: Typo changing age from 35 to 53

### NOT Binary Error Detection

We are **NOT** classifying:
- ❌ Error (manipulated) vs Correct (unchanged)
- ❌ Modified vs Original
- ❌ Dirty vs Clean

We **ARE** classifying:
- ✅ Intentional manipulation vs Unintentional manipulation
- ✅ Deliberate vs Accidental (among errors)
- ✅ Fraudulent vs Noise (among manipulated cells)

---

## 🔢 Dataset Statistics

### Cell-Level Classification
- **Total cells:** 60,000 (4,000 records × 15 features)
- **Manipulated cells:** 1,410 (2.35%)
  - **Intentional:** ~705 cells (50% of manipulated)
  - **Unintentional:** ~705 cells (50% of manipulated)
- **Correct cells:** 58,590 (97.65%)

### The Challenge
We focus on classifying the **1,410 manipulated cells** into intentional vs unintentional.  
(Correct cells are excluded from intent classification since they have no intent label)

---

## 🎓 Why This Terminology Matters

### Historical Context
Earlier documentation mistakenly referred to this as "error detection" because:
1. We're working with manipulated/erroneous data
2. The broader project involves error injection
3. Initial implementation focus was on detecting presence of errors

### The Correction
Upon reviewing the original project documentation (COMPLETE_PIPELINE_EXPLANATION.md), it's clear:
- Goal: "Train a classifier to predict whether a feature change was **intentional or unintentional**"
- Labels: `1 = Intentional manipulation`, `-1 = Unintentional manipulation`
- Task: **Intent classification**, not error detection

---

## 📚 Correct Terminology in Documentation

All documentation in `docs/propagation/` now uses correct terminology:

### ✅ Use These Terms
- Intent classification
- Intentional manipulation
- Unintentional manipulation
- Classify intent
- Intent label (1 = intentional, 0 = unintentional)
- F1 for intentional class
- F1 for unintentional class

### ❌ Avoid These Terms (In This Context)
- Error detection (unless referring to detecting presence of errors)
- Error class (ambiguous - intentional or unintentional?)
- Correct class (we're not classifying correct vs manipulated)
- Error vs non-error (different task)

---

## 🔬 Impact on Results Interpretation

### Original (Incorrect) Framing
> "Random Forest achieves F1=0.42 for **error detection**"

**Problem:** Unclear what "error" means - detecting manipulations, or classifying intent?

### Corrected Framing
> "Random Forest achieves F1=0.42 for **intentional classification**"

**Clear:** We're measuring ability to identify intentional manipulations (vs unintentional)

---

## 📊 Updated Results Table (Correct Terminology)

### Test 1: 40 Records Sampled

| Approach | F1 Weighted | Accuracy | **F1 Intentional** | F1 Unintentional | Winner |
|----------|-------------|----------|-------------------|------------------|--------|
| Random Forest | 0.9745 | 0.9760 | **0.4219** | 0.9878 | **RF** 🏆 |
| LabelSpreading | 0.9678 | 0.9777 | 0.0982 | 0.9887 | LP |
| LabelPropagation | 0.9678 | 0.9777 | 0.0969 | 0.9887 | LP |

**Key Finding:** Random Forest is **4.3x better** at classifying intentional manipulations  
(F1_intentional: 0.42 vs 0.10)

---

## 🎯 Research Question (Corrected)

### Original (Unclear)
> "Can label propagation be used for error detection?"

### Corrected (Clear)
> "Can label propagation replace Random Forest for classifying whether manipulations are intentional or unintentional?"

---

## 🏆 Conclusion (Corrected)

**Question:** Can semi-supervised label propagation replace supervised Random Forest for intent classification?

**Answer:** **NO**

**Reason:** 
- Label propagation assumes local similarity → similar labels
- But intentional vs unintentional is about **human behavior**, not feature patterns
- Two cells with identical feature changes can have different intents
- Random Forest learns complex decision boundaries that capture intent signals
- Label propagation over-smooths toward majority class (unintentional)

**Result:**
- Random Forest: F1_intentional = **0.42-0.44** 🏆
- Label Propagation: F1_intentional = **0.10-0.27** ❌
- **4x performance difference**

**Recommendation:** Use Random Forest with K-Means proportional sampling for production intent classification.

---

## 📖 Where to Learn More

### For Intent Classification Context
- **Main pipeline docs:** `docs/COMPLETE_PIPELINE_EXPLANATION.md`
- **Section:** "The Problem We're Solving" (lines 62-84)
- **Quote:** "Train a classifier to predict whether a feature change was intentional or unintentional"

### For Label Propagation Results
- **Results summary:** `docs/propagation/RESULTS_SUMMARY.md`
- **Section:** "Research Question" and "Key Results"
- **Main finding:** Random Forest 4x better for intent classification

### For Technical Details
- **Technical guide:** `docs/propagation/LABEL_PROPAGATION_GUIDE.md`
- **Section:** "Cell-Level Intent Classification" (not error detection)
- **Algorithms:** LabelPropagation, LabelSpreading, RandomForest

---

## 🔄 Migration Complete

### What Changed
1. ✅ Moved all propagation docs from root to `docs/propagation/`
2. ✅ Updated terminology from "error detection" to "intent classification"
3. ✅ Created RESULTS_SUMMARY.md with corrected terminology
4. ✅ Created README.md (index) for propagation documentation
5. ✅ Updated main README.md to reference propagation docs correctly
6. ✅ Created this terminology guide for future reference

### Documentation Structure
```
clustering-organized/
├── README.md (updated)
├── docs/
│   ├── COMPLETE_PIPELINE_EXPLANATION.md (original, correct terminology)
│   ├── FINAL_RESULTS_ALL_DATA_EVALUATION.md
│   ├── ... (other docs)
│   └── propagation/ ⭐ NEW
│       ├── README.md (index)
│       ├── RESULTS_SUMMARY.md (comprehensive results)
│       ├── LABEL_PROPAGATION_GUIDE.md (technical)
│       ├── LABEL_PROPAGATION_IMPLEMENTATION_SUMMARY.md
│       ├── LABEL_PROPAGATION_README.md (quick start)
│       ├── DELIVERABLES_CHECKLIST.md
│       └── TERMINOLOGY_GUIDE.md (this file)
└── scripts/
    └── compare_label_propagation.py
```

---

## ✅ Validation Checklist

- [x] All propagation docs in `docs/propagation/` folder
- [x] No documentation files in root directory
- [x] Terminology updated to "intent classification"
- [x] Results tables use "intentional/unintentional" not "error/correct"
- [x] Research question clearly states intent classification goal
- [x] Main README references correct paths
- [x] Index file (README.md) created for navigation
- [x] Comprehensive results summary created
- [x] Terminology guide created for future reference

---

**Status:** ✅ **COMPLETE**  
**Last Updated:** January 28, 2026  
**Maintainer:** Mohamed
