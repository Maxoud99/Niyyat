# Full Range Baseline Guessing for Intent Attribution

## Overview

This directory contains the results of comprehensive baseline guessing strategies for intent attribution on the **Twitter Bot Detection dataset**. We evaluated random guessing performance across the **full probability range from 0% to 100% in 10% increments**.

## Dataset Information

- **Dataset**: Twitter Bot Detection (bot-to-human evasion)
- **Total Records**: 4,000
- **Total Changes Evaluated**: 1,931 intentional changes
- **Ground Truth**: All changes are INTENTIONAL (bots trying to evade detection)

## Methodology

For each probability value P (from 0% to 100% in 10% steps):
1. **Random Guessing Strategy**: For each changed feature, predict "intentional" with probability P
2. **Multiple Runs**: Each probability was tested 10 times with different random seeds
3. **Statistical Analysis**: Calculate mean, standard deviation, min, and max accuracy

### Evaluation Metrics

- **Accuracy**: Percentage of changes correctly identified as intentional
- Since all ground truth is intentional (1), accuracy = (correct predictions) / (total changes)

## Key Results

### Summary Table

| Probability | Mean Accuracy | Std Dev | Expected | Difference |
|-------------|---------------|---------|----------|------------|
| 0%          | 0.0000        | 0.0000  | 0.00     | 0.0000     |
| 10%         | 0.0983        | 0.0042  | 0.10     | 0.0017     |
| 20%         | 0.2005        | 0.0072  | 0.20     | 0.0005     |
| 30%         | 0.3075        | 0.0115  | 0.30     | 0.0075     |
| 40%         | 0.4015        | 0.0109  | 0.40     | 0.0015     |
| **50%**     | **0.4996**    | 0.0146  | **0.50** | **0.0004** |
| 60%         | 0.5961        | 0.0057  | 0.60     | 0.0039     |
| 70%         | 0.6998        | 0.0107  | 0.70     | 0.0002     |
| 80%         | 0.8017        | 0.0113  | 0.80     | 0.0017     |
| 90%         | 0.8966        | 0.0045  | 0.90     | 0.0034     |
| 100%        | 1.0000        | 0.0000  | 1.00     | 0.0000     |

### Key Findings

1. **Excellent Alignment**: Mean accuracies closely match theoretical probabilities (max difference: 0.0075)
2. **Random Guessing Baseline**: P=50% achieves ~50% accuracy (0.4996)
3. **Edge Cases Validated**:
   - P=0%: Always wrong (0% accuracy)
   - P=100%: Always correct (100% accuracy)
4. **Low Variance**: Standard deviations are small (0.0042 to 0.0146), showing consistent behavior

## Files in This Directory

### Main Results
- **`FULL_RANGE_SUMMARY.md`**: Human-readable summary report
- **`full_range_summary.csv`**: Summary statistics for all probabilities
- **`full_range_all_results.json`**: Complete results including all 10 runs per probability

### Visualizations
- **`full_range_visualization.png`**: Comprehensive 4-panel visualization showing:
  1. Accuracy vs Probability with error bars
  2. Deviation from theoretical accuracy
  3. Standard deviation across runs
  4. Min-Max accuracy range
- **`full_range_simple.png`**: Simple line plot with theoretical comparison

### Detailed Results (Per Probability)
For each probability P (000, 010, 020, ..., 100):
- **`probability_XXX_results.json`**: Overall and per-column metrics
- **`probability_XXX_per_column.csv`**: Accuracy breakdown by feature

## Comparison with LLM Models

### Baseline Expectations

Any LLM model for intent attribution should:
- ✅ **Significantly exceed 50%**: Beat random guessing
- ✅ **Approach 70-90%**: Match or exceed informed probability baselines
- ✅ **Target 100%**: Strive for perfect classification (since all changes are intentional)

### Gemini Results (for reference)

The Gemini models achieved:
- **Bare-minimum**: ~83% accuracy
- **Info-enhanced**: ~85% accuracy  
- **Few-shots**: ~84% accuracy

These results significantly outperform random guessing (50%) and approach high-probability baselines (80-90%).

## How to Use These Baselines

### Compare Your Model
```python
import pandas as pd

# Load baseline summary
baselines = pd.read_csv('full_range_summary.csv')

# Your model's accuracy
your_accuracy = 0.85  # Example: 85%

# Compare
random_baseline = baselines[baselines['probability_percent'] == 50]['mean_accuracy'].values[0]
print(f"Your model: {your_accuracy:.4f}")
print(f"Random baseline (50%): {random_baseline:.4f}")
print(f"Improvement: {(your_accuracy - random_baseline) / random_baseline * 100:.1f}%")
```

### Reproduce Results
```bash
# Run the full range evaluation
python baseline_guessing_full_range.py --num-runs 10 --step 10

# Custom configuration
python baseline_guessing_full_range.py --num-runs 20 --step 5  # More runs, finer granularity
```

## Statistical Validation

The low standard deviations and small differences from theoretical values confirm:
1. ✅ **Implementation Correctness**: Random guessing behaves as expected
2. ✅ **Statistical Reliability**: Results are consistent across runs
3. ✅ **Baseline Validity**: These can be trusted as performance benchmarks

## Interpretation Guide

### For Researchers
- Use **P=50%** as the minimum acceptable performance
- Models should achieve **≥70%** to demonstrate meaningful learning
- Compare against **P=80-90%** for strong performance claims

### For Practitioners
- If your model performs **<50%**: Check for bugs or data issues
- If your model performs **~50%**: Model is essentially guessing randomly
- If your model performs **>70%**: Model has learned meaningful patterns
- If your model performs **>90%**: Excellent performance, approaching optimal

## Citation

If you use these baselines in your research, please reference:
- **Dataset**: Twitter Bot Detection (bot-to-human evasion)
- **Method**: Random guessing with probability distributions
- **Date**: December 29, 2025

## Related Files

### Other Baselines
- Parent directory: `../` contains other baseline strategies (constant guessing, etc.)
- Comparison: `../BASELINE_COMPARISON_SUMMARY.md`

### LLM Results
- Gemini results: `../../bare-min-gemini/`, `../../info-gemini/`, `../../few-shots-gemini/`
- Other models: `../../bare-min-*/`, `../../info-*/`, `../../few-shots-*/`

## Contact

For questions or issues with these baselines, please check:
1. The main script: `../../baseline_guessing_full_range.py`
2. Original baseline script: `../../baseline_guessing_strategies.py`
