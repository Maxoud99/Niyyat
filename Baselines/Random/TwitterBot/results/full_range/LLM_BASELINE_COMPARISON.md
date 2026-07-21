# LLM Intent Attribution vs Random Guessing Baselines

**Generated:** 2025-12-29 13:18:10

## Overview

This report compares LLM intent attribution performance against random guessing baselines across the full probability range (0% to 100%).

## Performance Summary

| Model | Accuracy | Closest Baseline | Baseline Acc | Difference | Beats Random | Rating |
|-------|----------|------------------|--------------|------------|--------------|--------|
| Qwen Bare-min | 0.8006 | 80% | 0.8017 | -0.0011 | Yes ✓ | Very Good |
| Llama Bare-min | 0.5640 | 60% | 0.5961 | -0.0322 | Yes ✓ | Poor |
| Gemini Bare-min | 0.5634 | 60% | 0.5961 | -0.0327 | Yes ✓ | Poor |
| R1 Bare-min | 0.5329 | 50% | 0.4996 | +0.0333 | Yes ✓ | Poor |
| Qwen Info | 0.3703 | 40% | 0.4015 | -0.0312 | No ✗ | Poor |
| Gemini Info | 0.3309 | 30% | 0.3075 | +0.0234 | No ✗ | Poor |
| Mixtral Bare-min | 0.3221 | 30% | 0.3075 | +0.0146 | No ✗ | Poor |
| Qwen Few-shots | 0.3159 | 30% | 0.3075 | +0.0084 | No ✗ | Poor |
| Gemini Few-shots | 0.2796 | 30% | 0.3075 | -0.0279 | No ✗ | Poor |
| R1 Info | 0.2527 | 20% | 0.2005 | +0.0523 | No ✗ | Poor |
| Mixtral Info | 0.2227 | 20% | 0.2005 | +0.0222 | No ✗ | Poor |
| Llama Info | 0.2191 | 20% | 0.2005 | +0.0186 | No ✗ | Poor |
| Llama Few-shots | 0.2009 | 20% | 0.2005 | +0.0005 | No ✗ | Poor |
| R1 Few-shots | 0.0642 | 10% | 0.0983 | -0.0341 | No ✗ | Poor |
| Mixtral Few-shots | 0.0492 | 10% | 0.0983 | -0.0491 | No ✗ | Poor |

## Key Findings

1. **Best Performing Model**: Qwen Bare-min (0.8006)
2. **Weakest Model**: Mixtral Few-shots (0.0492)
3. **Average Accuracy**: 0.3392
4. **Models Beating Random (50%)**: 4/15

## Interpretation

- **< 50%**: Worse than random guessing (likely implementation issue)
- **50-60%**: Marginally better than random
- **60-70%**: Demonstrates learning
- **70-80%**: Good performance
- **80-90%**: Very good performance
- **> 90%**: Excellent performance

## Baseline Reference

Random guessing with different probabilities:
| Probability | Mean Accuracy | Std Dev |
|-------------|---------------|---------|
| 0% | 0.0000 | 0.0000 |
| 10% | 0.0983 | 0.0042 |
| 20% | 0.2005 | 0.0072 |
| 30% | 0.3075 | 0.0115 |
| 40% | 0.4015 | 0.0109 |
| 50% | 0.4996 | 0.0146 |
| 60% | 0.5961 | 0.0057 |
| 70% | 0.6998 | 0.0107 |
| 80% | 0.8017 | 0.0113 |
| 90% | 0.8966 | 0.0045 |
| 100% | 1.0000 | 0.0000 |
