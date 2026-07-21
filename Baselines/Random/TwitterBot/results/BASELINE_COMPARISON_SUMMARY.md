# Baseline Guessing Strategies - Comparison Summary

**Generated:** 2025-12-08 15:46:56

## Overview

This report compares the performance of different baseline guessing strategies for intent attribution. All changes in the ground truth are INTENTIONAL, so accuracy represents the percentage of changes correctly identified as intentional.

## Strategy Comparison

| Strategy | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| Random_Guessing_Run1 | 0.5049 (50.49%) | 975 | 1931 |
| Constant_Guessing_Always_Intentional | 1.0000 (100.00%) | 1931 | 1931 |
| Constant_Guessing_Always_Unintentional | 0.0000 (0.00%) | 0 | 1931 |
| Probability_0.6_Run1 | 0.5904 (59.04%) | 1140 | 1931 |
| Probability_0.7_Run1 | 0.7022 (70.22%) | 1356 | 1931 |
| Probability_0.8_Run1 | 0.7970 (79.70%) | 1539 | 1931 |
| Probability_0.9_Run1 | 0.9001 (90.01%) | 1738 | 1931 |

## Expected Results

- **Random Guessing (50/50)**: Should achieve ~50% accuracy
- **Constant Always Intentional (1)**: Should achieve 100% accuracy (all ground truth is intentional)
- **Constant Always Unintentional (-1)**: Should achieve 0% accuracy
- **Probability Guessing P=0.7**: Should achieve ~70% accuracy
- **Probability Guessing P=0.8**: Should achieve ~80% accuracy
- **Probability Guessing P=0.9**: Should achieve ~90% accuracy

## Purpose

These baselines establish the minimum expected performance. Any LLM model should significantly outperform random guessing (50%) to demonstrate value. The constant strategy provides the theoretical maximum (100%).
