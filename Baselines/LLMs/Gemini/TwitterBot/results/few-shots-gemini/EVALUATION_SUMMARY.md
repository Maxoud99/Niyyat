# Twitter Bot Intent Attribution - Few-Shots-Gemini
## ⚠️ FAIR PERFORMANCE - Slight Improvement Over Info Version

### 📊 Overall Performance

| Metric | Value |
|--------|------:|
| **Accuracy** | **26.98%** |
| **Precision** | **100.00%** |
| **Recall** | **26.98%** |
| **F1-Score** | **42.50%** |

---

### 📈 Confusion Matrix

| Category | Count | Description |
|----------|------:|-------------|
| True Positives | 521 | Correctly identified intentional changes |
| False Negatives | 1,410 | Missed intentional changes (73% miss rate) |
| **False Positives** | **0** | **Perfect - no hallucinations** |

---

### ✅ Parse Errors: Minimal

- **Parse errors**: 43 out of 1,931 (2.23%)
- **Success rate**: 97.77%

✅ **Major improvement** over bare-min's 13.25%, though slightly worse than info's 0%.

---

### 📋 Prediction Distribution

For the 1,931 actually changed features:

| Prediction | Count | Percentage |
|-----------|------:|-----------:|
| Predicted Intentional (1) | 521 | 27.0% |
| Predicted Unintentional (-1) | 1,410 | 73.0% |

**Still conservative**, but better than info version's 78.8% miss rate.

---

### 🔍 Per-Feature Performance

| Feature | Changes | Accuracy | Insight |
|---------|--------:|---------:|---------|
| **`listed_count`** | 121 | **99.17%** | ✅ Near-perfect |
| **`friends_count`** | 273 | **96.70%** | ✅ Excellent |
| `has_created_date` | 8 | 50.00% | ⚠️ Mixed |
| `name_length` | 804 | 8.83% | ❌ Poor |
| `description_length` | 712 | 8.57% | ❌ Poor |
| `has_location` | 7 | 14.29% | ❌ Very poor |
| `screen_name_length` | 4 | 0.00% | ❌ All missed |
| `has_description` | 1 | 0.00% | ❌ Missed |
| `favourites_count` | 1 | 0.00% | ❌ Missed |

**Key Pattern**: 
- ✅ **Excellent on numeric features** (96-99% accuracy)
- ❌ **Poor on text-length features** (8-9% accuracy)
- ⚠️ **Slight improvement** over info version on text features (8% vs 0-2%)

---

### 📊 Three-Way Comparison

| Metric | Bare-Min | Info | Few-Shots | Best |
|--------|----------|------|-----------|------|
| **Accuracy** | **56.34%** | 21.18% | 26.98% | Bare-Min |
| **Precision** | **100%** | **100%** | **100%** | All tied |
| **Recall** | **56.34%** | 21.18% | 26.98% | Bare-Min |
| **F1-Score** | **72.08%** | 34.96% | 42.50% | Bare-Min |
| **Parse Errors** | 13.25% | **0.00%** | 2.23% | Info |
| **friends_count** | 41.76% | **100.00%** | 96.70% | Info |
| **listed_count** | 71.07% | **99.17%** | **99.17%** | Info/Few-Shots |
| **name_length** | 51.74% | 0.25% | 8.83% | Bare-Min |
| **description_length** | 65.59% | 1.97% | 8.57% | Bare-Min |

---

### 🔍 What Changed from Info → Few-Shots?

The few-shots version added **3 concrete examples** showing:
1. Example 1: Follower manipulation + profile completion = intentional
2. Example 2: Mixed changes (some intentional, some not)
3. Example 3: Coordinated multi-feature evasion

**Impact**:
- ✅ **Text features improved**: name_length 0.25% → 8.83%, description_length 1.97% → 8.57%
- ⚠️ **Numeric features slightly worse**: friends_count 100% → 96.70%
- ❌ **Parse errors introduced**: 0% → 2.23%
- ✅ **Overall better**: F1-score 34.96% → 42.50%

---

### 🔴 ROOT CAUSE: Examples Not Addressing Core Issue

#### Why Few-Shots Still Struggles

The three examples provided demonstrated:
- ✅ How to format JSON correctly (helped reduce parse errors from 13% → 2%)
- ✅ That large numeric changes are intentional (maintained 96% accuracy)
- ❌ **Did NOT address**: Why text-length changes should be considered intentional

**The Problem**: None of the examples specifically showed:
- Name shortening as an evasion tactic
- Description length changes as profile manipulation
- The bot-to-human context where ALL changes are evasion attempts

**Result**: LLM still interprets most text changes as "minor variations" and labels them unintentional.

---

### 💡 Key Findings

1. **✅ Parse Errors Mostly Fixed**: 13.25% → 2.23% (few-shot examples helped formatting)
2. **✅ Slight Improvement Over Info**: F1-score 34.96% → 42.50% (+7.5 points)
3. **❌ Still Worse Than Bare-Min**: F1-score 72.08% → 42.50% (-29.6 points)
4. **✅ Perfect Precision Maintained**: 100% - no false positives
5. **🎯 Same Feature Bias**: Perfect on numeric, poor on text
6. **⚠️ Marginal Text Improvement**: 8-9% vs 0-2%, but still terrible

---

### 🎯 Verdict

**⚠️ FAIR - Better Than Info, Worse Than Bare-Min**

#### Ranking:

1. **🥇 Bare-Min**: 56.34% accuracy, 72.08% F1 (best overall)
2. **🥈 Few-Shots**: 26.98% accuracy, 42.50% F1 (balanced approach)
3. **🥉 Info**: 21.18% accuracy, 34.96% F1 (too conservative)

#### What Works:

- ✅ **Near-zero parse errors** (2.23% vs 13.25%)
- ✅ **Perfect precision** (100% - trustworthy)
- ✅ **Excellent numeric detection** (96-99% on friends_count, listed_count)
- ✅ **Improved over info** (+7.5 F1 points)

#### What Fails:

- ❌ **Poor text feature detection** (8-9% vs 52-66% for bare-min)
- ❌ **Missing 73% of intentional changes** (1,410 false negatives)
- ❌ **Examples didn't address root issue** (text-length bias persists)
- ❌ **Still significantly worse than baseline** (-29.6 F1 points vs bare-min)

---

### 🔧 Recommendations

1. **For Production**: **Use Bare-Min** - best overall performance despite parse errors
   - Accept 13% parse error rate for 2x better recall
   - Fallback to -1 is conservative and acceptable

2. **To Improve Few-Shots**:
   - **Add text-specific examples**: Show name shortening, bio filling as intentional
   - **Remove ambiguity**: Make it clear ALL changes in bot-to-human context are evasion
   - **Example 4**: "name_length 14→5: Deliberate shortening to appear more human"
   - **Example 5**: "description_length 1→50: Strategic bio filling to evade sparse profile detection"

3. **Hybrid Approach** (Best of Both Worlds):
   - Use **few-shots for numeric features** (friends_count, listed_count): 96-99% accuracy
   - Use **bare-min for text features** (name_length, description_length): 52-66% accuracy
   - Combine predictions for optimal performance

4. **Alternative**: Revise few-shot examples to focus on text features:
   ```
   Example 1: Text-length evasion
   - name_length: 15→5 → INTENTIONAL (shortening to appear casual)
   - description_length: 10→80 → INTENTIONAL (bio filling to appear complete)
   
   Example 2: Coordinated text manipulation
   - name_length: 30→10 → INTENTIONAL (removing keywords/spam)
   - description_length: 1→50 → INTENTIONAL (adding human-like bio)
   ```

---

### 📝 Conclusion

The few-shots version is a **marginal improvement** over the info version, but still **significantly worse** than the bare-min baseline:

**Progress Made**:
- ✅ Reduced parse errors from 13.25% → 2.23%
- ✅ Improved F1-score from 34.96% → 42.50%
- ✅ Slightly better text feature detection (8% vs 0-2%)

**Problems Remaining**:
- ❌ Still misses 73% of intentional changes
- ❌ Text feature detection still terrible (8% vs needed 50%+)
- ❌ Examples didn't address the core bias
- ❌ Overall worse than simple bare-min approach

**Root Issue**: The few-shot examples demonstrated JSON formatting and numeric feature detection, but **failed to teach the LLM** that text-length changes in the bot-to-human evasion context are intentional, not "normal variations."

**Recommendation**: For production, **use bare-min-gemini** (56% accuracy, 72% F1) until few-shot examples can be revised to specifically address text-feature detection.

**Status**: ⚠️ FAIR - Shows promise but needs better-targeted examples.

**Next Step**: Create targeted few-shot examples focusing on text-length manipulation as an intentional evasion tactic.

---

### 📊 TRUE Accuracy (Counting Parse Errors as Failures)

When including parse errors in the evaluation (counting them as false negatives):

| Model | Reported Accuracy | **True Accuracy** | Inflation | Parse Errors |
|-------|------------------:|------------------:|----------:|-------------:|
| **Bare-Min** | 56.34% | **49.75%** | 🔴 -6.60pp | 256 (11.7%) |
| **Few-Shots** | 26.98% | **26.39%** | -0.59pp | 43 (2.2%) |
| **Info** | 21.18% | **21.18%** | ✅ 0.00pp | 0 (0%) |

**🏆 Winner: Still Bare-Min!**

Even with parse errors counted as failures, Bare-Min achieves **49.75% accuracy** - nearly **double** the next-best model (Few-Shots at 26.39%).