# Manipulation Pattern Analysis Report

## Analysis Overview

This report provides a comprehensive statistical analysis of manipulation patterns across all error types in the Adult Income dataset experiment.

**Data Source**: masks-blind.csv (ground truth)  
**Total Records**: 19,539  
**Clean Records**: 6,513  
**Variants per Clean Record**: 3

---

## Global Column Modification Frequencies

The following table shows how frequently each column was modified across ALL records, based on the ground truth masks file:

| Rank | Column | Changes | Percentage | Type | Notes |
|------|--------|---------|------------|------|-------|
| 1 | education | 5,955 | 30.5% | Categorical |  |
| 2 | education-num | 4,916 | 25.2% | Numeric |  |
| 3 | capital-gain | 3,928 | 20.1% | Numeric |  |
| 4 | hours-per-week | 2,553 | 13.1% | Numeric |  |
| 5 | native-country | 2,367 | 12.1% | Categorical | (soft-immutable) |
| 6 | occupation | 2,100 | 10.7% | Categorical |  |
| 7 | workclass | 1,423 | 7.3% | Categorical |  |
| 8 | capital-loss | 1,331 | 6.8% | Numeric |  |
| 9 | sex | 1,274 | 6.5% | Categorical | (soft-immutable) |
| 10 | race | 1,058 | 5.4% | Categorical | (soft-immutable) |
| 11 | relationship | 703 | 3.6% | Categorical | ⚠️ IMMUTABLE VIOLATION |
| 12 | age | 556 | 2.8% | Numeric | (soft-immutable) |
| 13 | marital-status | 59 | 0.3% | Categorical | ⚠️ IMMUTABLE VIOLATION |
| 14 | fnlwgt | 33 | 0.2% | Numeric | ⚠️ IMMUTABLE VIOLATION |
| 15 | class | 0 | 0.0% | Categorical | ✓ Correctly immutable |


---

## Distribution of Error Types

| Error Type | Count | Percentage | Plausible | Implausible | Plausibility Rate |
|------------|-------|------------|-----------|-------------|-------------------|
| Gain Targeted | 2,931 | 15.0% | 2,114 | 817 | 72.1% |
| Fairness Masking | 1,954 | 10.0% | 1,547 | 407 | 79.2% |
| Obfuscation Dmv | 977 | 5.0% | 812 | 165 | 83.1% |
| Unintentional | 13,677 | 70.0% | 10,930 | 2,747 | 79.9% |

| **TOTAL** | **19,539** | **100%** | **15,403** | **4,136** | **78.8%** |

---

## Gain Targeted

**Total Records**: 2,931  
**Plausible**: 2,114 (72.1%)  
**Average Changes**: 3.07  
**Average Effort**: 3.07  

### Most Frequently Modified Columns

| Rank | Column | Frequency | Percentage | Type | Notes |
|------|--------|-----------|------------|------|-------|
| 1 | capital-gain | 2571 | 87.7% | Numeric |  |
| 2 | education-num | 2421 | 82.6% | Numeric |  |
| 3 | education | 2202 | 75.1% | Categorical |  |
| 4 | occupation | 1065 | 36.3% | Categorical |  |
| 5 | hours-per-week | 598 | 20.4% | Numeric |  |
| 6 | workclass | 62 | 2.1% | Categorical |  |
| 7 | capital-loss | 45 | 1.5% | Numeric |  |
| 8 | age | 30 | 1.0% | Numeric | (soft-immutable) |
| 9 | native-country | 5 | 0.2% | Categorical | (soft-immutable) |
| 10 | race | 1 | 0.0% | Categorical | (soft-immutable) |

### Changes per Record Distribution

| Changes | Count | Percentage | Status |
|---------|-------|------------|--------|
| 1 | 16 | 0.5% | ✓ Within limit |
| 2 | 627 | 21.4% | ✓ Within limit |
| 3 | 1,526 | 52.1% | ✓ Within limit |
| 4 | 670 | 22.9% | ✗ Exceeds limit (>3) |
| 5 | 83 | 2.8% | ✗ Exceeds limit (>3) |
| 6 | 6 | 0.2% | ✗ Exceeds limit (>3) |
| 7 | 3 | 0.1% | ✗ Exceeds limit (>3) |

---

## Fairness Masking

**Total Records**: 1,954  
**Plausible**: 1,547 (79.2%)  
**Average Changes**: 0.86  
**Average Effort**: 0.86  

### Most Frequently Modified Columns

| Rank | Column | Frequency | Percentage | Type | Notes |
|------|--------|-----------|------------|------|-------|
| 1 | sex | 1080 | 55.3% | Categorical | (soft-immutable) |
| 2 | race | 332 | 17.0% | Categorical | (soft-immutable) |
| 3 | native-country | 179 | 9.2% | Categorical | (soft-immutable) |
| 4 | workclass | 47 | 2.4% | Categorical |  |
| 5 | occupation | 34 | 1.7% | Categorical |  |
| 6 | relationship | 6 | 0.3% | Categorical | ⚠️ IMMUTABLE |
| 7 | capital-loss | 3 | 0.2% | Numeric |  |
| 8 | capital-gain | 1 | 0.1% | Numeric |  |

### Changes per Record Distribution

| Changes | Count | Percentage | Status |
|---------|-------|------------|--------|
| 0 | 640 | 32.8% | ✓ Within limit |
| 1 | 985 | 50.4% | ✓ Within limit |
| 2 | 290 | 14.8% | ✓ Within limit |
| 3 | 39 | 2.0% | ✓ Within limit |

---

## Obfuscation Dmv

**Total Records**: 977  
**Plausible**: 812 (83.1%)  
**Average Changes**: 2.67  
**Average Effort**: 2.67  

### Most Frequently Modified Columns

| Rank | Column | Frequency | Percentage | Type | Notes |
|------|--------|-----------|------------|------|-------|
| 1 | occupation | 797 | 81.6% | Categorical |  |
| 2 | workclass | 724 | 74.1% | Categorical |  |
| 3 | native-country | 642 | 65.7% | Categorical | (soft-immutable) |
| 4 | education | 147 | 15.0% | Categorical |  |
| 5 | relationship | 138 | 14.1% | Categorical | ⚠️ IMMUTABLE |
| 6 | race | 110 | 11.3% | Categorical | (soft-immutable) |
| 7 | sex | 46 | 4.7% | Categorical | (soft-immutable) |
| 8 | education-num | 2 | 0.2% | Numeric |  |
| 9 | capital-loss | 1 | 0.1% | Numeric |  |
| 10 | capital-gain | 1 | 0.1% | Numeric |  |

### Changes per Record Distribution

| Changes | Count | Percentage | Status |
|---------|-------|------------|--------|
| 0 | 3 | 0.3% | ✓ Within limit |
| 1 | 37 | 3.8% | ✓ Within limit |
| 2 | 240 | 24.6% | ✓ Within limit |
| 3 | 696 | 71.2% | ✓ Within limit |
| 4 | 1 | 0.1% | ✗ Exceeds limit (>3) |

---

## Unintentional

**Total Records**: 13,677  
**Plausible**: 10,930 (79.9%)  
**Average Changes**: 1.09  
**Average Effort**: 1.09  

### Most Frequently Modified Columns

| Rank | Column | Frequency | Percentage | Type | Notes |
|------|--------|-----------|------------|------|-------|
| 1 | education | 3606 | 26.4% | Categorical |  |
| 2 | education-num | 2493 | 18.2% | Numeric |  |
| 3 | hours-per-week | 1954 | 14.3% | Numeric |  |
| 4 | native-country | 1541 | 11.3% | Categorical | (soft-immutable) |
| 5 | capital-gain | 1355 | 9.9% | Numeric |  |
| 6 | capital-loss | 1282 | 9.4% | Numeric |  |
| 7 | race | 615 | 4.5% | Categorical | (soft-immutable) |
| 8 | workclass | 590 | 4.3% | Categorical |  |
| 9 | relationship | 559 | 4.1% | Categorical | ⚠️ IMMUTABLE |
| 10 | age | 526 | 3.8% | Numeric | (soft-immutable) |

### Changes per Record Distribution

| Changes | Count | Percentage | Status |
|---------|-------|------------|--------|
| 0 | 689 | 5.0% | ✓ Within limit |
| 1 | 11,544 | 84.4% | ✓ Within limit |
| 2 | 917 | 6.7% | ✓ Within limit |
| 3 | 521 | 3.8% | ✓ Within limit |
| 4 | 6 | 0.0% | ✗ Exceeds limit (>3) |

---


## Key Insights

### Immutable Field Violations

- **fnlwgt**: 33 violations (0.2% of all records)
- **class**: ✓ No violations (correctly protected)
- **marital-status**: 59 violations (0.3% of all records)
- **relationship**: 703 violations (3.6% of all records)


### Records Exceeding Maximum Changes (>3)

- **Gain Targeted**: 762 records (26.0%)
- **Obfuscation Dmv**: 1 records (0.1%)
- **Unintentional**: 6 records (0.0%)


---

*Report generated on: 2025-11-19 17:22:38*  
*Data source: masks-blind.csv (ground truth)*
