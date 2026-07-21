# Twitter Bot Intent Attribution - Info-Gemini
## ⚠️ POOR PERFORMANCE - Overly Conservative Model

### 📊 Overall Performance

| Metric | Value |
|--------|------:|
| **Accuracy** | **21.18%** |
| **Precision** | **100.00%** |
| **Recall** | **21.18%** |
| **F1-Score** | **34.96%** |

---

### 📈 Confusion Matrix

| Category | Count | Description |
|----------|------:|-------------|
| True Positives | 409 | Correctly identified intentional changes |
| False Negatives | 1,522 | Missed intentional changes (too conservative) |
| **False Positives** | **0** | **Perfect - no hallucinations** |

---

### ✅ IMPROVEMENT: Zero Parse Errors

The info-gemini model achieved **0% parse errors** (vs 13.25% for bare-min), meaning:
- All 400 chunks returned valid, parseable JSON
- No fallback to conservative -1 defaults
- The DIFFS and rubric helped with response formatting

---

### 📋 Prediction Distribution

For the 1,931 actually changed features:

| Prediction | Count | Percentage |
|-----------|------:|-----------:|
| Predicted Intentional (1) | 409 | 21.2% |
| Predicted Unintentional (-1) | 1,522 | 78.8% |

**Problem**: The model is **extremely conservative**, labeling 78.8% of real intentional changes as "unintentional".

---

### 🔍 Per-Feature Performance

| Feature | Changes | Accuracy | Insight |
|---------|--------:|---------:|---------|
| **`friends_count`** | 273 | **100.00%** | ✅ Perfect detection |
| **`listed_count`** | 121 | **99.17%** | ✅ Near-perfect |
| `name_length` | 804 | 0.25% | ❌ Almost all missed |
| `description_length` | 712 | 1.97% | ❌ 98% missed |
| `has_created_date` | 8 | 0.00% | ❌ All missed |
| `has_location` | 7 | 0.00% | ❌ All missed |
| `screen_name_length` | 4 | 0.00% | ❌ All missed |
| `favourites_count` | 1 | 0.00% | ❌ All missed |
| `has_description` | 1 | 0.00% | ❌ All missed |

**Key Pattern**: The model performs **perfectly on numeric features** (`friends_count`, `listed_count`) but **completely fails on text-length features** (`name_length`, `description_length`).

---

### ⚠️  Parse Errors

- **Parse errors**: 0 out of 400 chunks (0%)
- **Success rate**: 100%

✅ **Major improvement** over bare-min's 13.25% error rate! The DIFFS and rubric completely eliminated parsing issues.

---

### 📊 Comparison with Bare-Min-Gemini

| Metric | Bare-Min | Info-Gemini | Change |
|--------|----------|------------:|-------:|
| Accuracy | 56.34% | **21.18%** | **-35.16%** ❌ |
| Precision | 100.00% | **100.00%** | **+0.00%** ✅ |
| Recall | 56.34% | **21.18%** | **-35.16%** ❌ |
| F1-Score | 72.08% | **34.96%** | **-37.12%** ❌ |
| **Parse Errors** | **13.25%** | **0.00%** | **-13.25%** ✅ |

---

### 🔴 ROOT CAUSE ANALYSIS

#### Why Info Version Failed (Despite Fixing Parse Errors)

The info-enhanced version added:
1. **DIFFS**: Explicit before/after values for changed features
2. **Rubric**: Detailed guidance: "Small text length variations are normal profile updates, not intentional evasion"
3. **Feature-level context**: More information per feature

**What Went Wrong**: The rubric's guidance about "small text variations" being "normal" caused the LLM to:
- ✅ Correctly identify **large numeric changes** (friends_count, listed_count) as intentional
- ❌ Incorrectly dismiss **all text-length changes** as "normal variations"

The rubric statement: *"Small text length variation, considered a normal profile update"* appears 1,516 times in the explanations, accounting for 99% of false negatives.

---

### 💡 Key Findings

1. **✅ Parse Errors Eliminated**: 13.25% → 0% (DIFFS + rubric fixed formatting)
2. **❌ Overly Conservative**: 78.8% of intentional changes labeled as "normal variations"
3. **✅ Perfect Precision**: 100% - no false positives (trustworthy when it flags something)
4. **❌ Terrible Recall**: 21.18% - misses 4 out of 5 intentional changes
5. **🎯 Feature-Specific Bias**: Perfect on numeric features, 0% on text features
6. **📉 Net Regression**: -35% accuracy despite eliminating parse errors

---

### 🎯 Verdict

**⚠️ POOR - Overly Conservative with Feature-Specific Bias**

#### Why This Is Problematic:

1. **Massive False Negatives**: Missing 1,522 out of 1,931 intentional changes (78.8%)
2. **Rubric Backfire**: The guidance to ignore "small text variations" was taken too literally
3. **Feature Bias**: Perfect on `friends_count`/`listed_count`, 0% on text lengths
4. **Lower Overall Value**: F1-score of 34.96% vs 72.08% for bare-min
5. **Trade-off Failure**: Eliminated parse errors but made decision quality worse

#### What Worked:

- ✅ **Zero parse errors** - formatting guidance was effective
- ✅ **Perfect precision** - maintains trust (no false alarms)
- ✅ **Perfect numeric detection** - 100% on friends_count, 99% on listed_count

#### What Failed:

- ❌ **Rubric too prescriptive** - LLM followed "ignore small text changes" too strictly
- ❌ **Lost recall** - from 56% → 21% 
- ❌ **Text feature blindness** - dismissed ALL name/description length changes

---

### 🔧 Recommendations

1. **❌ DO NOT USE AS-IS**: 21.18% recall is too low for production
2. **Revise Rubric**:
   - Remove or soften "small text variations are normal" guidance
   - Emphasize that ALL changes in bot-to-human context are suspicious
   - Clarify that text length changes can be strategic (name shortening, bio filling)
3. **Leverage What Works**:
   - Keep DIFFS format (0% parse errors!)
   - Keep focus on numeric features (100% accuracy there)
   - Add specific examples of text-length evasion tactics
4. **Test Few-Shots Version**:
   - Should combine formatting success with better decision-making
   - Include examples of text-length changes being intentional
5. **Consider Hybrid Approach**:
   - Use info-gemini for `friends_count`/`listed_count` (perfect accuracy)
   - Use bare-min or few-shots for text features

---

### 📝 Conclusion

The info-gemini version demonstrates a **classic precision-recall trade-off gone wrong**. While it successfully:
- ✅ Eliminated all parse errors (0% vs 13.25%)
- ✅ Maintained perfect precision (100%)  
- ✅ Achieved perfect accuracy on numeric features

It **catastrophically failed** on the majority of features:
- ❌ 78.8% false negative rate
- ❌ 0% accuracy on all text-length features
- ❌ Overall F1-score dropped from 72% → 35%

**Root Cause**: The rubric's well-intentioned guidance about "normal text variations" was interpreted too literally by the LLM, causing it to dismiss genuine evasion tactics as benign profile updates.

**Key Insight**: Adding domain knowledge via rubrics can backfire if the guidance conflicts with the ground truth pattern. The rubric said "small text changes are normal," but in the bot-to-human evasion context, these changes ARE intentional.

**Status**: ⚠️ POOR - Do not use without significant rubric revision.

**Next**: Evaluate few-shots-gemini to see if concrete examples can provide better guidance than abstract rubrics.
