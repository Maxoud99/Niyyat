# Manipulation Analysis Summary - READ ME FIRST

## Overview

This directory contains a comprehensive analysis of manipulation patterns for each error type in the Adult Income dataset experiment. The analysis examines which manipulations **make sense** (are logical and follow specifications) versus which **don't make sense** (violate constraints or are implausible).

## 📁 Files in This Directory

### 1. **MANIPULATION_INSIGHTS.md** ⭐ START HERE
**The most important document** - A comprehensive guide explaining:
- ✅ What makes sense for each error type
- ❌ What doesn't make sense
- 💡 Recommendations for fixing issues
- Summary table of all problems
- Priority fixes needed

**Read this first** to understand the overall findings.

### 2. **manipulation_analysis_report.md**
Statistical analysis report containing:
- Distribution of error types (15% gain, 10% fairness, 5% obfuscation, 70% unintentional)
- Detailed statistics for each error type
- Column modification frequencies
- Plausibility rates
- Constraint violation counts

### 3. **EXAMPLES_ANALYSIS.md**
Concrete examples showing:
- 5 good examples per error type (what works well)
- 5 bad examples per error type (violations and issues)
- Side-by-side comparisons of clean vs manipulated records
- Specific rationales for each manipulation

**Great for understanding** what the manipulations actually look like in practice.

### 4. **detailed_analysis.csv**
Raw data file with all analyzed records:
- Column-by-column change information
- Metadata (manipulation_type, plausibility_score, effort_k, etc.)
- Constraint check results
- Can be loaded into Excel/pandas for custom analysis

### 5. Visualizations (PNG files)

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
- Reveals patterns like capital-loss being modified 100% of the time

---

## 🔑 Key Findings at a Glance

### Critical Issues Found (All Error Types):

1. **Capital-loss modified in 100% of all records** 🚨
   - Present in ALL error types
   - Usually with delta ≈ 0
   - **Root cause**: Likely a bug in generation script
   - **Fix**: Remove from default modification list

2. **Many records exceed max 3 changes** 🚨
   - Gain-targeted: 77% violate
   - Obfuscation-DMV: 71% violate
   - **Fix**: Stricter validation before accepting LLM output

3. **Immutable fields being modified** 🚨
   - relationship: 762 total violations
   - fnlwgt: Modified in unintentional errors
   - **Fix**: Pre-validation before allowing changes

### What's Working Well:

✅ **Unintentional errors** - Realistic typos and keyboard slips  
✅ **Fairness-masking direction** - Mostly masking to majority groups  
✅ **Obfuscation placeholders** - Using proper DMV indicators (nan, —, Unknown)  
✅ **Gain-targeted education upgrades** - Following logical progression  

### What Needs Fixing:

❌ **Gain-targeted**: Too many simultaneous changes (4-5 instead of max 3)  
❌ **Fairness-masking**: Bi-directional sex changes (should only be Female→Male)  
❌ **Obfuscation-DMV**: Modifying immutable 'relationship' field (14% of records)  
❌ **Unintentional**: Extreme numeric errors (age +727 years, hours +4455)  

---

## 📊 Statistics Summary

| Error Type | Count | % | Plausibility | Avg Changes | Avg Effort |
|------------|-------|---|--------------|-------------|------------|
| **Gain-Targeted** | 2,931 | 15% | 72.1% | 4.06 | 3.07 |
| **Fairness-Masking** | 1,954 | 10% | 79.2% | 1.86 | 0.86 |
| **Obfuscation-DMV** | 977 | 5% | 83.1% | 3.67 | 2.67 |
| **Unintentional** | 13,677 | 70% | 79.9% | 2.00 | 1.09 |
| **TOTAL** | **19,539** | **100%** | **78.3%** | **2.53** | **1.51** |

---

## 🎯 Recommendations for Script Improvements

### Fix Priority: 🔴 HIGH

1. **Remove capital-loss from automatic modifications**
   - It's being added to every record without reason
   - Adds noise without meaningful manipulation

2. **Enforce max 3 column changes**
   - Add strict validation
   - Retry LLM call if >3 changes returned

3. **Protect immutable fields**
   - Pre-check before accepting LLM output
   - Never allow: fnlwgt, relationship, marital-status, class

4. **Fix numeric error generation for unintentional**
   - Only single-digit operations (swap/delete/insert)
   - Reject values outside valid ranges immediately

### Fix Priority: 🟡 MEDIUM

5. **Fix fairness-masking directionality**
   - Only minority → majority
   - Female → Male (not Male → Female)
   - Non-White → White (not White → Unknown)

6. **Obfuscation should only use placeholders**
   - Not semantic changes (e.g., Private → Self-emp-inc)
   - Only "Unknown", "—", "N/A"

7. **Validate field types**
   - Prevent sex becoming "White"
   - Field-specific value checks

---

## 💻 How to Use These Files

### For Understanding the Data:
1. Read **MANIPULATION_INSIGHTS.md** for the big picture
2. Look at **EXAMPLES_ANALYSIS.md** for concrete cases
3. Check visualizations (PNG files) for patterns

### For Statistical Analysis:
1. Open **manipulation_analysis_report.md** for detailed stats
2. Load **detailed_analysis.csv** into pandas/Excel for custom queries

### For Debugging the Generation Script:
1. Focus on issues in **MANIPULATION_INSIGHTS.md**
2. Use examples from **EXAMPLES_ANALYSIS.md** to test fixes
3. Re-run analysis after fixes to verify improvements

---

## 📈 Next Steps

1. **Fix the generation script** based on recommendations
2. **Re-run the manipulation generation** with fixes applied
3. **Re-run this analysis** to verify improvements
4. **Compare before/after** plausibility rates and violation counts

---

## 🔧 Reproducing This Analysis

To re-run this analysis on new data:

```bash
# Main analysis script
python /home/mohamed/error_injector/llms_baseline/adult_income_dataset/analysis_manipulation_patterns.py

# Examples report
python /home/mohamed/error_injector/llms_baseline/adult_income_dataset/generate_examples_report.py
```

Both scripts will automatically:
- Load data from the correct paths
- Generate all reports and visualizations
- Save to `analysis_output/` directory

---

**Dataset**: Adult Income  
**Run**: tenth-trial/run_20251031_211812  
**Total Records**: 19,539 manipulated (from 6,513 clean)  
**Analysis Date**: 2025-11-19  
**Generated By**: analysis_manipulation_patterns.py
