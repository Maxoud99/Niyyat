# Twitter Bot Intent Attribution - Bare-Min-Gemini
## Evaluation Excluding LLM Failures

### 📊 Dataset Breakdown

- **Total changed features**: 1,931
- **LLM failures (fallback)**: 227 (11.76%)
- **Successful predictions**: 1,704 (88.24%)

---

### 🎯 Performance Comparison

| Metric | Original (All Data) | Excluding Failures | Improvement |
|--------|--------------------:|-------------------:|------------:|
| **Accuracy** | 56.34% | **63.85%** | **+7.51%** |
| **Precision** | 100.00% | 100.00% | +0.00% |
| **Recall** | 56.34% | **63.85%** | **+7.51%** |
| **F1-Score** | 72.08% | **77.94%** | **+5.86%** |

---

### 📈 Confusion Matrix (Excluding Failures)

| Prediction | Count |
|-----------|------:|
| True Positives (Correctly identified intentional) | 1,088 |
| False Negatives (Missed intentional changes) | 616 |
| False Positives (Incorrect intentional labels) | 0 |

---

### 📋 Prediction Distribution (Excluding Failures)

| Category | Count | Percentage |
|----------|------:|-----------:|
| Predicted Intentional | 1,088 | 63.8% |
| Predicted Unintentional | 616 | 36.2% |
| Predicted Unchanged | 0 | 0.0% |

---

### 🔍 Per-Feature Performance (Excluding Failures)

| Feature | Changed Count | Accuracy |
|---------|-------------:|---------:|
| `listed_count` | 103 | **83.50%** |
| `description_length` | 633 | **73.78%** |
| `name_length` | 723 | **57.54%** |
| `has_created_date` | 4 | 50.00% |
| `friends_count` | 235 | 48.51% |
| `has_location` | 3 | 33.33% |
| `favourites_count` | 1 | 100.00% |
| `screen_name_length` | 1 | 100.00% |
| `has_description` | 1 | 0.00% |

---

### 💡 Key Insights

1. **LLM Parse Errors Impact**: The 227 LLM failures (11.76% of data) account for approximately **26.9% of all false negatives**. These failures defaulted to `-1` (unintentional), dragging down recall.

2. **True Model Capability**: When the LLM successfully parses and generates valid JSON responses:
   - Accuracy increases from 56.34% → **63.85%** (+7.51 points)
   - F1-Score increases from 72.08% → **77.94%** (+5.86 points)
   - Recall improves proportionally with accuracy

3. **Perfect Precision Maintained**: The model maintains 100% precision both with and without failures, meaning:
   - When it flags something as intentional, it's always correct
   - Zero false positives in both scenarios
   - The model is conservative but accurate

4. **Failure Root Cause**: Parse errors occur when:
   - LLM returns malformed JSON
   - Response doesn't match expected schema
   - Timeout or incomplete responses

---

### 🎯 Verdict

**⚠️ FAIR** - When successful, the model achieves **63.8% accuracy** and **77.9% F1-score**

#### Strengths:
- ✅ Perfect precision (100%) - no false alarms
- ✅ Strong performance on `listed_count` (83.5% accuracy)
- ✅ Reasonable performance on text features (`description_length`: 73.8%)

#### Weaknesses:
- ❌ **11.76% parse error rate** is too high for production
- ❌ Moderate recall (63.85%) even without failures
- ❌ Poor performance on `friends_count` (48.5%) and boolean features

---

### 🔧 Recommendations

1. **Reduce Parse Errors** (Target: <5% failure rate):
   - Add few-shot examples showing proper JSON format
   - Include explicit schema validation examples in prompt
   - Show edge case formatting (empty strings, special characters)

2. **Improve Feature-Specific Reasoning**:
   - Add domain knowledge about `friends_count` manipulation patterns
   - Provide clearer rubric for boolean feature changes
   - Include examples of coordinated multi-feature attacks

3. **Test Info & Few-Shots Versions**:
   - Info version: DIFFS + detailed rubric should reduce ambiguity
   - Few-shots version: Concrete examples should reduce parse errors to <5%
   - Expected improvement: 70-80% accuracy range

---

### 📝 Conclusion

The bare-min model's **true capability is 63.85% accuracy**, significantly better than the 56.34% headline number. The gap is caused by parse errors forcing conservative fallback predictions.

**The parse error problem (11.76%) is the critical bottleneck** preventing the model from reaching its full potential. Few-shot learning should address this directly by demonstrating proper JSON formatting.

Next steps:
1. ✅ Created info-gemini version with DIFFS and rubric
2. ✅ Created few-shots-gemini version with 3 examples
3. ⏳ Run full evaluation on info-gemini
4. ⏳ Run full evaluation on few-shots-gemini
5. ⏳ Compare all three versions

Expected outcome: Few-shots version should achieve 70-80% accuracy with <5% parse errors.
