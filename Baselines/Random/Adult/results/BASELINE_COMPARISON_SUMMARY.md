# Baseline Guessing Strategies - Comparison Summary

**Generated:** 2025-12-08 16:28:37

## Overview

This report compares the performance of different baseline guessing strategies for intent attribution on the Adult Income dataset.

## Strategy Comparison

| Strategy | Accuracy | Macro F1 | Macro Prec | Macro Rec |
|----------|----------|----------|------------|----------|
| Random_Guessing_Run1 | 0.4990 | 0.4984 | 0.4986 | 0.4986 |
| Constant_Guessing_Always_Intentional | 0.4704 | 0.3199 | 0.2352 | 0.5000 |
| Constant_Guessing_Always_Unintentional | 0.5296 | 0.3462 | 0.2648 | 0.5000 |
| Probability_0.4_Run1 | 0.5058 | 0.4964 | 0.4994 | 0.4994 |
| Probability_0.5_Run1 | 0.5006 | 0.4999 | 0.5002 | 0.5002 |
| Probability_0.6_Run1 | 0.4926 | 0.4905 | 0.4980 | 0.4981 |
| Probability_0.7_Run1 | 0.4884 | 0.4742 | 0.4999 | 0.4999 |

## Expected Results

- **Random Guessing (50/50)**: Should achieve ~50% accuracy, ~50% F1
- **Constant Strategies**: Biased toward one class (high precision, low recall or vice versa)
- **Probability Guessing**: Performance proportional to ground truth distribution

## Purpose

These baselines establish the minimum expected performance. Any LLM model should significantly outperform random guessing to demonstrate value.
