# Manipulation Pattern Analysis Report

## Executive Summary

- **Total manipulated records**: 19539
- **Error types analyzed**: gain_targeted, fairness_masking, obfuscation_dmv, unintentional

## Distribution of Error Types

| Error Type | Count | Percentage | Plausible | Plausibility Rate |
|------------|-------|------------|-----------|-------------------|
| Gain Targeted | 2931 | 15.0% | 2114 | 72.1% |
| Fairness Masking | 1954 | 10.0% | 1547 | 79.2% |
| Obfuscation Dmv | 977 | 5.0% | 812 | 83.1% |
| Unintentional | 13677 | 70.0% | 10930 | 79.9% |

## Detailed Analysis by Error Type

### Gain Targeted

**Statistics:**
- Total records: 2931
- Plausible records: 2114 (72.1%)
- Average changes per record: 4.06
- Average effort (k): 3.07

**Most frequently modified columns:**

| Column | Modifications | Percentage | Type | Mutability |
|--------|---------------|------------|------|------------|
| capital-loss | 2931 | 100.0% | Numeric | Mutable |
| capital-gain | 2571 | 87.7% | Numeric | Mutable |
| education-num | 2421 | 82.6% | Numeric | Mutable |
| education | 2202 | 75.1% | Categorical | Mutable |
| occupation | 1065 | 36.3% | Categorical | Mutable |
| hours-per-week | 598 | 20.4% | Numeric | Mutable |
| workclass | 62 | 2.1% | Categorical | Mutable |
| age | 30 | 1.0% | Numeric | Soft-Immutable |
| native-country | 5 | 0.2% | Categorical | Soft-Immutable |
| race | 1 | 0.0% | Categorical | Soft-Immutable |

**Distribution of changes per record:**

| Number of Changes | Count | Percentage | Valid |
|-------------------|-------|------------|-------|
| 2 | 24 | 0.8% | ✓ |
| 3 | 642 | 21.9% | ✓ |
| 4 | 1515 | 51.7% | ✗ |
| 5 | 660 | 22.5% | ✗ |
| 6 | 81 | 2.8% | ✗ |
| 7 | 6 | 0.2% | ✗ |
| 8 | 3 | 0.1% | ✗ |

### Fairness Masking

**Statistics:**
- Total records: 1954
- Plausible records: 1547 (79.2%)
- Average changes per record: 1.86
- Average effort (k): 0.86

**Most frequently modified columns:**

| Column | Modifications | Percentage | Type | Mutability |
|--------|---------------|------------|------|------------|
| capital-loss | 1954 | 100.0% | Numeric | Mutable |
| sex | 1080 | 55.3% | Categorical | Soft-Immutable |
| race | 332 | 17.0% | Categorical | Soft-Immutable |
| native-country | 179 | 9.2% | Categorical | Soft-Immutable |
| workclass | 47 | 2.4% | Categorical | Mutable |
| occupation | 34 | 1.7% | Categorical | Mutable |
| relationship | 6 | 0.3% | Categorical | Immutable |
| capital-gain | 1 | 0.1% | Numeric | Mutable |

**Distribution of changes per record:**

| Number of Changes | Count | Percentage | Valid |
|-------------------|-------|------------|-------|
| 1 | 640 | 32.8% | ✓ |
| 2 | 986 | 50.5% | ✓ |
| 3 | 291 | 14.9% | ✓ |
| 4 | 37 | 1.9% | ✗ |

### Obfuscation Dmv

**Statistics:**
- Total records: 977
- Plausible records: 812 (83.1%)
- Average changes per record: 3.67
- Average effort (k): 2.67

**Most frequently modified columns:**

| Column | Modifications | Percentage | Type | Mutability |
|--------|---------------|------------|------|------------|
| capital-loss | 977 | 100.0% | Numeric | Mutable |
| occupation | 797 | 81.6% | Categorical | Mutable |
| workclass | 724 | 74.1% | Categorical | Mutable |
| native-country | 642 | 65.7% | Categorical | Soft-Immutable |
| education | 147 | 15.0% | Categorical | Mutable |
| relationship | 138 | 14.1% | Categorical | Immutable |
| race | 110 | 11.3% | Categorical | Soft-Immutable |
| sex | 46 | 4.7% | Categorical | Soft-Immutable |
| education-num | 2 | 0.2% | Numeric | Mutable |
| capital-gain | 1 | 0.1% | Numeric | Mutable |

**Distribution of changes per record:**

| Number of Changes | Count | Percentage | Valid |
|-------------------|-------|------------|-------|
| 1 | 3 | 0.3% | ✓ |
| 2 | 37 | 3.8% | ✓ |
| 3 | 240 | 24.6% | ✓ |
| 4 | 697 | 71.3% | ✗ |

### Unintentional

**Statistics:**
- Total records: 13677
- Plausible records: 10930 (79.9%)
- Average changes per record: 2.00
- Average effort (k): 1.09

**Most frequently modified columns:**

| Column | Modifications | Percentage | Type | Mutability |
|--------|---------------|------------|------|------------|
| capital-loss | 13677 | 100.0% | Numeric | Mutable |
| education | 3606 | 26.4% | Categorical | Mutable |
| education-num | 2493 | 18.2% | Numeric | Mutable |
| hours-per-week | 1954 | 14.3% | Numeric | Mutable |
| native-country | 1541 | 11.3% | Categorical | Soft-Immutable |
| capital-gain | 1355 | 9.9% | Numeric | Mutable |
| race | 615 | 4.5% | Categorical | Soft-Immutable |
| workclass | 590 | 4.3% | Categorical | Mutable |
| relationship | 559 | 4.1% | Categorical | Immutable |
| age | 526 | 3.8% | Numeric | Soft-Immutable |

**Distribution of changes per record:**

| Number of Changes | Count | Percentage | Valid |
|-------------------|-------|------------|-------|
| 1 | 1393 | 10.2% | ✓ |
| 2 | 11098 | 81.1% | ✓ |
| 3 | 977 | 7.1% | ✓ |
| 4 | 205 | 1.5% | ✗ |
| 5 | 4 | 0.0% | ✗ |


## Key Findings

### What Makes Sense for Each Error Type

#### 1. **Gain-Targeted Errors**
- **Expected**: Upward mobility in education, occupation, workclass
- **Expected**: Increases in capital-gain, education-num, hours-per-week
- **Expected**: Age adjustments (younger if <40, older if >40)
- **Actual top modifications**: capital-loss, capital-gain, education-num, education, occupation

#### 2. **Fairness-Masking Errors**
- **Expected**: race → White, sex → Male, native-country → United-States
- **Expected**: Protected attributes masking
- **Actual top modifications**: capital-loss, sex, race, native-country, workclass

#### 3. **Obfuscation (DMV) Errors**
- **Expected**: Categorical fields replaced with 'Unknown', '—', 'N/A'
- **Expected**: No numeric field obfuscation
- **Actual top modifications**: capital-loss, occupation, workclass, native-country, education

#### 4. **Unintentional Errors**
- **Expected**: Typos, transpositions, keyboard slips in categorical fields
- **Expected**: Digit swaps/removal/insertion in numeric fields
- **Expected**: May violate range constraints
- **Actual top modifications**: capital-loss, education, education-num, hours-per-week, native-country


## Violations and Issues

### Immutable Field Violations

- **Fairness Masking**: 6 violations
- **Obfuscation Dmv**: 138 violations
- **Unintentional**: 618 violations

### Records with >3 Changes

- **Gain Targeted**: 2265 records
- **Fairness Masking**: 37 records
- **Obfuscation Dmv**: 697 records
- **Unintentional**: 209 records

---

*Report generated on: 2025-11-19 16:32:35*
