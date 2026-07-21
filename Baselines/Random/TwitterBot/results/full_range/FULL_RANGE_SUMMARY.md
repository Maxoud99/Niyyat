# Full Range Baseline Guessing - Summary

**Generated:** 2025-12-29 13:15:07

## Overview

This report presents baseline guessing strategies across the full probability range (0% to 100% in 10% increments). Each probability was tested with multiple runs to establish statistical confidence.

## Results Summary

| Probability | Mean Accuracy | Std Dev | Min | Max | Runs |
|-------------|---------------|---------|-----|-----|------|
|   0% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 10 |
|  10% | 0.0983 | 0.0042 | 0.0917 | 0.1041 | 10 |
|  20% | 0.2005 | 0.0072 | 0.1890 | 0.2128 | 10 |
|  30% | 0.3075 | 0.0115 | 0.2931 | 0.3340 | 10 |
|  40% | 0.4015 | 0.0109 | 0.3817 | 0.4210 | 10 |
|  50% | 0.4996 | 0.0146 | 0.4744 | 0.5158 | 10 |
|  60% | 0.5961 | 0.0057 | 0.5847 | 0.6033 | 10 |
|  70% | 0.6998 | 0.0107 | 0.6888 | 0.7183 | 10 |
|  80% | 0.8017 | 0.0113 | 0.7778 | 0.8203 | 10 |
|  90% | 0.8966 | 0.0045 | 0.8897 | 0.9021 | 10 |
| 100% | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 10 |

## Key Observations

1. **Theoretical vs Actual**: The mean accuracy should closely match the probability
2. **Edge Cases**:
   - P=0% should achieve ~0% accuracy (always predict unintentional)
   - P=50% should achieve ~50% accuracy (random guessing)
   - P=100% should achieve ~100% accuracy (always predict intentional)
3. **Variance**: Higher variance indicates more random behavior

## Purpose

These baselines establish performance expectations across all probability ranges. Any LLM model should significantly outperform random guessing (50%) and ideally approach or exceed high-probability baselines (80-90%).
