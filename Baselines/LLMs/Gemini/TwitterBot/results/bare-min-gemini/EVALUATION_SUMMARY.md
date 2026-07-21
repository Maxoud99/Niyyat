# Twitter Bot Intent Attribution - Evaluation Results Summary

**Model:** bare-min-gemini (Gemini 2.5 Pro)  
**Dataset:** 4,000 records, 1,931 changed features  
**Date:** December 4, 2025

---

## 📊 Overall Performance Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 56.34% | 1,088 out of 1,931 intentional changes correctly identified |
| **Precision** | 100.00% | All predicted intentional changes were correct |
| **Recall** | 56.34% | Detected 56.34% of all intentional changes |
| **F1-Score** | 72.08% | Harmonic mean of precision and recall |

---

## 🎯 Confusion Matrix

| Category | Count | Description |
|----------|-------|-------------|
| **True Positives (TP)** | 1,088 | ✅ Correctly identified as intentional |
| **False Negatives (FN)** | 843 | ❌ Missed (labeled as unintentional) |
| **False Positives (FP)** | 0 | ✅ No false alarms (excellent!) |
| **True Negatives (TN)** | N/A | Not applicable - all changes are intentional |

---

## 📈 Prediction Distribution

| Prediction | Count | Percentage |
|------------|-------|------------|
| Predicted Intentional (1) | 1,088 | 56.3% |
| Predicted Unintentional (-1) | 843 | 43.7% |
| Predicted Unchanged (0) | 0 | 0.0% |

---

## 📋 Per-Feature Performance

Sorted by number of changes:

| Feature | Changes | True Positives | False Negatives | Accuracy | Status |
|---------|---------|----------------|-----------------|----------|--------|
| **name_length** | 804 | 416 | 388 | 51.74% | ⚠️ Needs Improvement |
| **description_length** | 712 | 467 | 245 | 65.59% | ✅ Good |
| **friends_count** | 273 | 114 | 159 | 41.76% | ⚠️ Needs Improvement |
| **listed_count** | 121 | 86 | 35 | 71.07% | ✅ Good |
| **has_created_date** | 8 | 2 | 6 | 25.00% | ❌ Poor |
| **has_location** | 7 | 1 | 6 | 14.29% | ❌ Poor |
| **screen_name_length** | 4 | 1 | 3 | 25.00% | ❌ Poor |
| **favourites_count** | 1 | 1 | 0 | 100.00% | ✅ Perfect |
| **has_description** | 1 | 0 | 1 | 0.00% | ❌ Poor |

---

## 🔍 Key Insights

### ✅ Strengths

- **Perfect Precision (100%)**: No false positives - when the model says a change is intentional, it's always correct
- **Strong Performance on Specific Features**:
  - `listed_count`: 71.07% accuracy
  - `description_length`: 65.59% accuracy
  - `favourites_count`: 100% accuracy (though only 1 change)
- **Conservative Approach**: Only labels clear intentional cases, avoiding false alarms

### ⚠️ Weaknesses

- **Low Recall (56.34%)**: Misses many intentional bot evasion changes
- **Poor Performance on Boolean Features**:
  - `has_location`: 14.29% accuracy
  - `has_description`: 0% accuracy
  - `has_created_date`: 25% accuracy
- **Struggles with Numeric Features**:
  - `friends_count`: 41.76% accuracy
  - `name_length`: 51.74% accuracy

### 💡 Interpretation

The **bare-minimum model** is very conservative, only flagging obvious intentional changes. While it has **perfect precision** (no false alarms), it **misses approximately 43.7%** of bot evasion attempts.

This suggests the model needs more context or examples to:
1. Better identify subtle coordinated changes across multiple features
2. Understand that small changes in boolean/text fields can be intentional when part of a broader evasion strategy
3. Recognize patterns in follower/friend count manipulations

---

## 📁 Files Generated

- `evaluation_results.json` - Detailed JSON results
- `per_column_metrics.csv` - Feature-level performance metrics
- `intent_labels.csv` - LLM predictions for all features
- `intent_explanations.csv` - Detailed explanations for each decision
- `run_stats.json` - Runtime statistics

---

## 🎯 Recommendations for Improvement

1. **Add Few-Shot Examples**: Provide concrete examples of intentional bot evasion to improve recall
2. **Include Contextual Information**: Show relationships between features (e.g., coordinated changes)
3. **Feature-Specific Guidance**: Add specific rubrics for boolean and numeric features
4. **Consider Info-Enhanced Version**: Use DIFFS to make changes more explicit

The **info** and **few-shots** versions should address these limitations.
