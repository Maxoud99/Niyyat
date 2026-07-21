# Manipulation Analysis Summary - READ ME FIRST

## Overview

This directory contains a comprehensive analysis of manipulation patterns for each error type in the Adult Income dataset experiment. The analysis examines which manipulations **make sense** (are logical and follow specifications) versus which **don't make sense** (violate constraints or are implausible).

## ⚠️ IMPORTANT: This Analysis Uses Ground Truth

All statistics and findings in this analysis are based on **masks-blind.csv**, which is the authoritative source of truth for which columns were actually modified.

## 📁 Files in This Directory

### 1. **manipulation_analysis_report.md** ⭐ START HERE
Statistical analysis report containing:
- Distribution of error types (15% gain, 10% fairness, 5% obfuscation, 70% unintentional)
- Detailed statistics for each error type
- Column modification frequencies (CORRECT - from masks)
- Plausibility rates
- Constraint violation counts

### 2. **detailed_analysis.csv**
Raw data file with all analyzed records:
- Column-by-column change information (from masks)
- Metadata (manipulation_type, plausibility_score, effort_k, etc.)
- Constraint check results
- Can be loaded into Excel/pandas for custom analysis

### 3. Visualizations (PNG files)

#### **column_frequency_by_error_type.png**
4-panel bar chart showing which columns are most frequently modified for each error type.
- Color-coded: Red (immutable), Yellow (soft-immutable), Blue (mutable)

#### **changes_distribution.png**
Violin plot showing the distribution of number of changes per record.
- Red dashed line at 3 shows the maximum allowed
- Reveals that many error types exceed this limit

#### **plausibility_by_error_type.png**
Stacked bar chart showing plausible vs implausible records.
- Green = plausible
- Red = implausible
- Shows obfuscation-DMV has highest plausibility (83.1%)

#### **heatmap_modifications.png**
Heatmap showing modification percentage for each column across error types.
- Darker colors = more frequent modifications

---

## 🔑 Key Findings at a Glance

### Top Modified Columns (Actual Truth from Masks):

1. **education**: 30.5% of all records
2. **education-num**: 25.2% of all records
3. **capital-gain**: 20.1% of all records
4. **hours-per-week**: 13.1% of all records
5. **native-country**: 12.1% of all records

### Critical Issues Found:

1. **Many records exceed max 3 changes** 🚨
   - Gain-targeted: 26.0% violate (762 records with 4+ changes)
   - **Fix**: Stricter validation before accepting LLM output

2. **Immutable fields being modified** 🚨
   - relationship: 703 violations (3.6% of all records)
   - marital-status: 59 violations (0.3%)
   - fnlwgt: 33 violations (0.2%)
   - **Fix**: Pre-validation before allowing changes

3. **Too many changes in gain-targeted** 🚨
   - Average 3.07 changes (slightly above ideal ≤3)
   - 52.1% have exactly 3 changes (good)
   - 26.0% have 4+ changes (bad)

### What's Working Well:

✅ **Unintentional errors** - Mostly 1 change per record (84.4%)
✅ **Fairness-masking** - Lightweight (83.2% have 0-1 changes)
✅ **Obfuscation-DMV** - 71.2% have exactly 3 changes (target achieved)
✅ **Class label** - Never modified (0 violations) ✓

---

## 📊 Statistics Summary

| Error Type | Count | % | Plausibility | Avg Changes | Avg Effort |
|------------|-------|---|--------------|-------------|------------|
| **Gain-Targeted** | 2,931 | 15% | 72.1% | 3.07 | 3.07 |
| **Fairness-Masking** | 1,954 | 10% | 79.2% | 0.86 | 0.86 |
| **Obfuscation-DMV** | 977 | 5% | 83.1% | 2.67 | 2.67 |
| **Unintentional** | 13,677 | 70% | 79.9% | 1.09 | 1.09 |
| **TOTAL** | **19,539** | **100%** | **78.8%** | **1.45** | **1.45** |

---

## 🎯 Recommendations for Script Improvements

### Fix Priority: 🔴 HIGH

1. **Enforce max 3 column changes**
   - Add strict validation
   - Retry LLM call if >3 changes returned
   - Currently 762 gain-targeted records violate this

2. **Protect immutable fields**
   - Pre-check before accepting LLM output
   - Never allow: fnlwgt, relationship, marital-status, class
   - Currently 795 total violations

3. **Fix numeric error generation for unintentional**
   - Only single-digit operations (swap/delete/insert)
   - Reject values outside valid ranges immediately

### Fix Priority: 🟡 MEDIUM

4. **Validate field types**
   - Prevent incorrect value assignments
   - Field-specific value checks

---

## 💻 How to Use These Files

### For Understanding the Data:
1. Read **manipulation_analysis_report.md** for detailed statistics
2. Check visualizations (PNG files) for patterns
3. Load **detailed_analysis.csv** for custom queries

### For Debugging the Generation Script:
1. Focus on issues documented in this README
2. Re-run analysis after fixes to verify improvements
3. Compare before/after plausibility rates

---

## 📈 Next Steps

1. **Fix the generation script** based on recommendations
2. **Re-run the manipulation generation** with fixes applied
3. **Re-run this analysis** to verify improvements

---

## 🔧 Reproducing This Analysis

To re-run this analysis:

```bash
python /home/mohamed/error_injector/llms_baseline/adult_income_dataset/generate_complete_analysis.py
```

This script will automatically:
- Load data from masks-blind.csv (ground truth)
- Generate all reports and visualizations
- Save to `analysis_output/` directory

---

**Dataset**: Adult Income  
**Run**: tenth-trial/run_20251031_211812  
**Total Records**: 19,539 manipulated (from 6,513 clean)  
**Analysis Date**: 2025-11-19  
**Data Source**: masks-blind.csv (ground truth)

