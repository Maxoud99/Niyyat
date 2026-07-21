# What Makes Sense for Each Manipulation Type: Analysis & Insights

## Overview

This document provides an analysis of which manipulation patterns **make sense** (are logical and plausible) versus which patterns **don't make sense** (are violations or inconsistencies) for each error type in the Adult Income dataset manipulation experiment.

---

## 1. GAIN-TARGETED Errors

### ✅ What Makes Sense

**Purpose**: Simulate intentional upward manipulation to improve economic prospects.

**Appropriate Manipulations**:
1. **Education Upgrades** (75.1% modified)
   - HS-grad → Some-college (701 occurrences)
   - Some-college → Bachelors (397 occurrences)
   - Bachelors → Masters (285 occurrences)
   - Masters → Doctorate (114 occurrences)
   - ✓ **This makes perfect sense** - represents plausible educational progression

2. **Occupation Upgrades** (36.3% modified)
   - Sales → Exec-managerial (131 times)
   - Prof-specialty → Exec-managerial (120 times)
   - Adm-clerical → Exec-managerial (90 times)
   - ✓ **Makes sense** - upward career mobility

3. **Education-Num Increases** (82.6% modified)
   - Mean delta: +2.41 years
   - Median delta: +3 years
   - Range: +1 to +3 (mostly within spec)
   - ✓ **Makes sense** - aligns with education field changes

4. **Capital-Gain Increases** (87.7% modified)
   - Mean delta: +$1,587
   - Median delta: +$1,000
   - Range: mostly +$1,000 to +$3,000
   - ✓ **Makes sense** - modest financial gain increases

5. **Hours-Per-Week Increases** (20.4% modified)
   - Mean delta: +6.52 hours
   - Median delta: +5 hours
   - Range: typically +4 to +12
   - ✓ **Makes sense** - working more to increase income

### ❌ What Doesn't Make Sense

**Issues Identified**:

1. **Excessive Column Modifications**
   - 77.2% of records modify >3 columns (VIOLATION)
   - 51.7% modify exactly 4 columns
   - 22.5% modify 5 columns
   - ❌ **Problem**: Spec allows max 3 changes

2. **Capital-Loss Modified** (100% modified)
   - Should NOT be manipulated for gain-targeted errors
   - Capital-loss is a negative financial indicator
   - Mean delta: -$13.88 (mostly zero changes)
   - ❌ **Problem**: Manipulating capital-loss doesn't align with "gain" motivation

3. **Low Plausibility Rate** (72.1%)
   - Lower than other intentional types
   - ❌ **Problem**: Many combinations are implausible

4. **High Effort Values**
   - 52.1% have effort_k = 3
   - 25.9% have effort_k ≥ 4
   - ❌ **Problem**: Supposed to be "small plausible" changes

### 💡 Recommendations

- **Focus on 2-3 key attributes**: education + occupation OR education + capital-gain
- **Remove capital-loss from modifications**
- **Ensure education and education-num are synchronized**
- **Stick to conservative numeric deltas** (within specified ranges)
- **Limit to max 3 column changes**

---

## 2. FAIRNESS-MASKING Errors

### ✅ What Makes Sense

**Purpose**: Mask protected demographic attributes to majority categories (neutral rationale).

**Appropriate Manipulations**:
1. **Sex Masking** (55.3% modified)
   - Female → Male (542 times)
   - ✓ **Makes perfect sense** - masking to majority sex (Male)

2. **Race Masking** (17.0% modified)
   - Black → White (176 times)
   - Asian-Pac-Islander → White (50 times)
   - Other → White (15 times)
   - ✓ **Makes sense** - masking to majority race (White)

3. **Native-Country Masking** (9.2% modified)
   - Mexico → United-States (31 times)
   - Germany → United-States (8 times)
   - Canada → United-States (7 times)
   - ✓ **Makes sense** - masking to majority country

4. **Low Number of Changes**
   - 32.8% modify only 1 column
   - 50.5% modify 2 columns
   - 14.9% modify 3 columns
   - ✓ **Excellent** - focused, targeted changes

5. **Low Effort** (effort_k mean = 0.86)
   - 32.8% have effort_k = 0
   - 50.4% have effort_k = 1
   - ✓ **Makes sense** - simple category substitutions

6. **High Plausibility** (79.2%)
   - ✓ **Good** - most changes are plausible

### ❌ What Doesn't Make Sense

**Issues Identified**:

1. **Capital-Loss Modified** (100% modified)
   - Not a protected demographic attribute
   - Mean delta ≈ 0
   - ❌ **Problem**: Irrelevant to fairness masking

2. **Bi-directional Sex Changes**
   - Female → Male (542 times)
   - Male → Female (481 times) ← **This is backwards!**
   - ❌ **Problem**: Should only mask Female→Male, not Male→Female

3. **Relationship Field Violations** (6 instances)
   - Wife → Husband (3 times)
   - Immutable field modified
   - ❌ **Problem**: Violates immutability constraint

4. **Confusing Sex Changes**
   - Male → White (41 times) ← **Wrong column!**
   - Male → nan (10 times)
   - ❌ **Problem**: Mixed up with race field

5. **White → Unknown** (30 times)
   - ❌ **Problem**: Should mask TO majority, not FROM majority

### 💡 Recommendations

- **Remove capital-loss modifications**
- **Only mask minority → majority** (Female→Male, not Male→Female)
- **Never touch immutable fields** (relationship)
- **Fix field confusion** (sex vs race)
- **Only mask protected attributes**: sex, race, native-country

---

## 3. OBFUSCATION-DMV Errors

### ✅ What Makes Sense

**Purpose**: Replace categorical values with disguised missing value placeholders.

**Appropriate Manipulations**:
1. **Occupation Obfuscation** (81.6% modified)
   - Exec-managerial → nan (50 times)
   - Sales → nan (49 times)
   - Prof-specialty → nan (43 times)
   - ✓ **Makes perfect sense** - replacing with missing value indicators

2. **Workclass Obfuscation** (74.1% modified)
   - Private → nan (170 times)
   - Private → — (114 times)
   - ? → — (33 times)
   - ✓ **Makes sense** - DMV placeholders

3. **Native-Country Obfuscation** (65.7% modified)
   - United-States → nan (458 times)
   - United-States → — (38 times)
   - United-States → Unknown (34 times)
   - ✓ **Makes sense** - categorical obfuscation

4. **Education Obfuscation** (15.0% modified)
   - HS-grad → nan (42 times)
   - Bachelors → nan (20 times)
   - ✓ **Makes sense** - categorical field

5. **Race/Sex Obfuscation** (11.3% / 4.7%)
   - White → nan (41 times)
   - Male → — (24 times)
   - ✓ **Makes sense** - categorical fields

6. **High Plausibility** (83.1%)
   - ✓ **Excellent** - highest plausibility rate

### ❌ What Doesn't Make Sense

**Issues Identified**:

1. **Capital-Loss Modified** (100% modified)
   - Numeric field, not categorical
   - Mean delta ≈ 0
   - ❌ **Problem**: Spec says "ONLY categorical fields"

2. **Excessive Changes** (71.3% have 4 changes)
   - ❌ **Problem**: Violates max 3 changes constraint

3. **Relationship Field Violations** (138 instances, 14.1%)
   - Husband → nan (44 times)
   - Husband → — (18 times)
   - ❌ **Problem**: IMMUTABLE field modified extensively

4. **Workclass Semantic Changes**
   - Private → Self-emp-inc (143 times)
   - ❌ **Problem**: This is NOT obfuscation, it's a semantic change

5. **Numeric Field Obfuscation** (rare but present)
   - education-num modified (2 times)
   - capital-gain modified (1 time)
   - ❌ **Problem**: Spec explicitly forbids numeric obfuscation

### 💡 Recommendations

- **ONLY modify categorical fields**
- **Remove all numeric field modifications** (capital-loss, education-num, capital-gain)
- **Never touch immutable fields** (relationship)
- **Use ONLY placeholders**: "Unknown", "—", "N/A" (not actual values like Self-emp-inc)
- **Limit to 2-3 categorical fields per record**

---

## 4. UNINTENTIONAL Errors

### ✅ What Makes Sense

**Purpose**: Simulate benign human mistakes (typos, digit errors).

**Appropriate Manipulations**:
1. **Categorical Typos** (very realistic!)
   - **Education**:
     - HS-grad → HS-gard (664 times) - letter swap
     - Bachelors → Bachleors (476 times) - letter swap
     - Some-college → Some-colleage (424 times) - typo
   - ✓ **Perfect** - realistic keyboard errors

   - **Native-Country**:
     - United-States → United-Statrs (434 times)
     - United-States → United-Statets (251 times)
     - United-States → United-Staes (214 times)
   - ✓ **Perfect** - common typos

   - **Race**:
     - White → Whitw (346 times) - keyboard slip
     - White → Whit (48 times) - deletion
   - ✓ **Perfect** - typos

2. **Low Number of Changes**
   - 81.1% modify exactly 2 columns
   - 10.2% modify 1 column
   - 7.1% modify 3 columns
   - ✓ **Excellent** - sparse, realistic errors

3. **Low Effort** (mean = 1.09)
   - 84.4% have effort_k = 1
   - ✓ **Makes sense** - simple mistakes

4. **High Plausibility** (79.9%)
   - ✓ **Good** - most are plausible typos

### ❌ What Doesn't Make Sense

**Issues Identified**:

1. **Capital-Loss Modified** (100% modified)
   - Often with extreme/invalid values
   - Mean/median = nan (many invalid conversions)
   - ❌ **Problem**: Why is this field modified in every record?

2. **Extreme Numeric Errors**
   - **Age**:
     - Mean delta: +37.41 (too large)
     - Max delta: +727 years (impossible!)
     - Min delta: -81 years
   - ❌ **Problem**: Should be small digit swaps, not huge errors

   - **Education-Num**:
     - Mean delta: +32.90
     - Max delta: +1,488 (impossible!)
   - ❌ **Problem**: Out of valid range [1-16]

   - **Hours-Per-Week**:
     - Mean delta: +128.51
     - Max delta: +4,455 hours (impossible!)
   - ❌ **Problem**: Out of valid range [1-99]

   - **fnlwgt**:
     - Modified in some records (IMMUTABLE!)
     - Max delta: +7.7M
   - ❌ **Problem**: fnlwgt should NEVER change

3. **Relationship Field Violations** (559 instances, 4.1%)
   - Husband → Husbamd (65 times)
   - ❌ **Problem**: IMMUTABLE field

4. **Semantic Categorical Changes**
   - Male → Female (46 times)
   - Female → Male (44 times)
   - occupation: ? → Adm-clerical (40 times)
   - ❌ **Problem**: These are semantic changes, not typos

5. **Out-of-Range Values Kept**
   - Many numeric errors exceed valid ranges
   - plausibility_score = 0 for these
   - ❌ **Problem**: 20.1% are implausible

### 💡 Recommendations

- **Remove capital-loss from default modifications**
- **For numeric fields**:
  - Only swap/delete/insert **single digits**
  - Example: 45 → 54 (swap), 45 → 4 (delete), 45 → 455 (insert)
  - **Reject values outside valid ranges** or mark clearly as invalid
- **Never touch immutable fields** (fnlwgt, relationship, marital-status)
- **Categorical typos only** - no semantic changes (Male→Female is not a typo!)
- **Validate plausibility** before accepting changes

---

## Summary Table: What Makes Sense

| Error Type | ✅ Makes Sense | ❌ Doesn't Make Sense | Fix Priority |
|------------|----------------|----------------------|--------------|
| **Gain-Targeted** | Education upgrades, occupation mobility, capital-gain increases, education-num sync | >3 changes (77%), capital-loss modified (100%), low plausibility (28% implausible) | 🔴 HIGH |
| **Fairness-Masking** | Sex/race/country to majority, focused changes, low effort | Bi-directional sex masking, capital-loss modified (100%), immutable violations | 🟡 MEDIUM |
| **Obfuscation-DMV** | Categorical → placeholders, high plausibility | Numeric fields modified, relationship violations (14%), >3 changes (71%) | 🔴 HIGH |
| **Unintentional** | Realistic typos, low changes, low effort | Extreme numeric errors, immutable violations, capital-loss (100%), semantic changes | 🔴 HIGH |

---

## Key Cross-Cutting Issues

### 🚨 Critical Problems (All Error Types)

1. **Capital-Loss Modified in 100% of Records**
   - Present in ALL error types
   - Mean delta ≈ 0 in most cases
   - **Root cause**: Likely a bug in the generation script
   - **Impact**: Adds noise without meaningful manipulation
   - **Fix**: Remove capital-loss from default modification list

2. **Immutable Field Violations**
   - **relationship**: 762 total violations (0.3% fairness, 14.1% obfuscation, 4.1% unintentional)
   - **fnlwgt**: Modified in unintentional errors
   - **Fix**: Enforce immutability checks before accepting LLM output

3. **Exceeding Max Changes (>3 columns)**
   - Gain-targeted: 77.2% violate
   - Obfuscation-DMV: 71.3% violate
   - **Fix**: Stricter validation and retry logic

4. **LLM Confusion Between Fields**
   - Sex changes to "White" (fairness-masking)
   - **Fix**: Better prompt engineering and field validation

---

## Recommendations for Script Improvements

### High Priority
1. ✅ Remove capital-loss from automatic modifications
2. ✅ Enforce max 3 column changes (reject and retry)
3. ✅ Never modify immutable fields (pre-validation)
4. ✅ Validate field-specific constraints (e.g., sex ∈ {Male, Female})

### Medium Priority
5. ✅ Improve numeric error generation (single-digit operations only)
6. ✅ Fix fairness-masking direction (only minority → majority)
7. ✅ Obfuscation should only use DMV placeholders, not semantic values
8. ✅ Validate range constraints before accepting changes

### Low Priority
9. ✅ Better prompt engineering to reduce field confusion
10. ✅ Add more diverse error patterns for unintentional errors

---

**Generated**: 2025-11-19  
**Dataset**: Adult Income  
**Run**: tenth-trial/run_20251031_211812  
**Total Records Analyzed**: 19,539 manipulated records from 6,513 clean records
