#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Guessing Strategies for Intent Attribution - Adult Income Dataset
---------------------------------------------------------------------------
Tests the performance of simple baseline strategies:
1. Random Guessing: 50/50 chance of 1 or -1 for changed features
2. Constant Guessing: Always predict same value (1 or -1)
3. Random Guessing with Probability: Use custom probability distributions

Ground Truth:
- mask=1 means the feature was changed intentionally
- mask=-1 means the feature was changed unintentionally  
- mask=0 means the feature was unchanged

Baseline Predictions:
- 1 = predicted intentional
- -1 = predicted unintentional
- 0 = no change (copy from ground truth mask)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from datetime import datetime

# File paths
GROUND_TRUTH_MASK = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_v2_20260213_141240/masks.csv"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/baselines_2"


def normalize_value(val):
    """Normalize mask value to integer."""
    val_str = str(val).strip()
    if val_str in ("1", "1.0"):
        return 1
    elif val_str in ("-1", "-1.0"):
        return -1
    else:
        return 0


def create_random_guessing_mask(gt_mask: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Strategy 1: Random Guessing (50/50)
    For each changed feature, randomly predict 1 or -1 with equal probability.
    """
    np.random.seed(seed)
    pred_mask = gt_mask.copy().astype(float)
    
    # Vectorized approach: create random choices for all cells
    random_choices = np.random.choice([1, -1], size=pred_mask.shape)
    
    # Apply only where mask is not 0
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, random_choices)
    
    return pred_mask.astype(int)


def create_constant_guessing_mask(gt_mask: pd.DataFrame, constant_value: int = 1) -> pd.DataFrame:
    """
    Strategy 2: Constant Guessing
    For each changed feature, always predict the same value.
    
    Args:
        constant_value: Either 1 (always intentional) or -1 (always unintentional)
    """
    pred_mask = gt_mask.copy().astype(float)
    
    # Vectorized: set all non-zero values to constant_value
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, constant_value)
    
    return pred_mask.astype(int)


def create_probability_guessing_mask(gt_mask: pd.DataFrame, prob_intentional: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """
    Strategy 3: Random Guessing with Probability
    For each changed feature, predict 1 with probability p, -1 with probability (1-p).
    
    Args:
        prob_intentional: Probability of predicting 1 (intentional), e.g., 0.7 = 70% chance
    """
    np.random.seed(seed)
    pred_mask = gt_mask.copy().astype(float)
    
    # Generate random probabilities for all cells
    random_probs = np.random.random(size=pred_mask.shape)
    
    # Create prediction matrix: 1 if prob < threshold, else -1
    predictions = np.where(random_probs < prob_intentional, 1, -1)
    
    # Apply only where mask is not 0
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, predictions)
    
    return pred_mask.astype(int)


def evaluate_predictions(gt_mask: pd.DataFrame, pred_mask: pd.DataFrame, strategy_name: str):
    """
    Evaluate baseline strategy performance.
    
    Returns dict with metrics matching the LLM evaluation format.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION: {strategy_name}")
    print(f"{'='*70}")
    
    # Convert to numpy for faster processing
    gt_array = gt_mask.values.astype(float)
    pred_array = pred_mask.values.astype(float)
    
    # Flatten and filter only changed features (mask != 0)
    mask_nonzero = (gt_array != 0)
    all_gt = gt_array[mask_nonzero].flatten()
    all_pred = pred_array[mask_nonzero].flatten()
    
    # Calculate metrics for both classes
    intentional_gt = (all_gt == 1).astype(int)
    intentional_pred = (all_pred == 1).astype(int)
    
    unintentional_gt = (all_gt == -1).astype(int)
    unintentional_pred = (all_pred == -1).astype(int)
    
    # Metrics for intentional (1)
    prec_1 = precision_score(intentional_gt, intentional_pred, zero_division=0)
    rec_1 = recall_score(intentional_gt, intentional_pred, zero_division=0)
    f1_1 = f1_score(intentional_gt, intentional_pred, zero_division=0)
    
    # Metrics for unintentional (-1)
    prec_neg1 = precision_score(unintentional_gt, unintentional_pred, zero_division=0)
    rec_neg1 = recall_score(unintentional_gt, unintentional_pred, zero_division=0)
    f1_neg1 = f1_score(unintentional_gt, unintentional_pred, zero_division=0)
    
    # Macro averages
    macro_prec = (prec_1 + prec_neg1) / 2
    macro_rec = (rec_1 + rec_neg1) / 2
    macro_f1 = (f1_1 + f1_neg1) / 2
    
    # Overall accuracy
    accuracy = accuracy_score(all_gt, all_pred)
    
    # Count ground truth distribution
    intentional_count = int(intentional_gt.sum())
    unintentional_count = int(unintentional_gt.sum())
    total = len(all_gt)
    
    print(f"\nGround Truth Distribution:")
    print(f"  Intentional (1): {intentional_count} ({intentional_count/total*100:.2f}%)")
    print(f"  Unintentional (-1): {unintentional_count} ({unintentional_count/total*100:.2f}%)")
    print(f"  Total: {total}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Macro Precision: {macro_prec:.4f}")
    print(f"  Macro Recall: {macro_rec:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Intentional (1):")
    print(f"    Precision: {prec_1:.4f}")
    print(f"    Recall: {rec_1:.4f}")
    print(f"    F1: {f1_1:.4f}")
    print(f"  Unintentional (-1):")
    print(f"    Precision: {prec_neg1:.4f}")
    print(f"    Recall: {rec_neg1:.4f}")
    print(f"    F1: {f1_neg1:.4f}")
    
    return {
        'strategy': strategy_name,
        'overall': {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'macro_precision': macro_prec,
            'macro_recall': macro_rec,
            'total_changes': total,
            'intentional_count': intentional_count,
            'unintentional_count': unintentional_count
        },
        'intentional': {
            'precision': prec_1,
            'recall': rec_1,
            'f1': f1_1
        },
        'unintentional': {
            'precision': prec_neg1,
            'recall': rec_neg1,
            'f1': f1_neg1
        }
    }


def save_results(results: dict, strategy_name: str, output_dir: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{strategy_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save summary markdown
    md_path = os.path.join(output_dir, f"{strategy_name}_SUMMARY.md")
    with open(md_path, 'w') as f:
        f.write(f"# {results['strategy']} - Evaluation Summary\n\n")
        f.write(f"**Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Performance\n\n")
        f.write(f"- **Accuracy**: {results['overall']['accuracy']:.4f} ({results['overall']['accuracy']*100:.2f}%)\n")
        f.write(f"- **Macro F1**: {results['overall']['macro_f1']:.4f}\n")
        f.write(f"- **Macro Precision**: {results['overall']['macro_precision']:.4f}\n")
        f.write(f"- **Macro Recall**: {results['overall']['macro_recall']:.4f}\n\n")
        
        f.write(f"## Ground Truth Distribution\n\n")
        f.write(f"- **Total Changes**: {results['overall']['total_changes']}\n")
        f.write(f"- **Intentional (1)**: {results['overall']['intentional_count']} ")
        f.write(f"({results['overall']['intentional_count']/results['overall']['total_changes']*100:.2f}%)\n")
        f.write(f"- **Unintentional (-1)**: {results['overall']['unintentional_count']} ")
        f.write(f"({results['overall']['unintentional_count']/results['overall']['total_changes']*100:.2f}%)\n\n")
        
        f.write(f"## Per-Class Metrics\n\n")
        f.write(f"### Intentional (1)\n")
        f.write(f"- Precision: {results['intentional']['precision']:.4f}\n")
        f.write(f"- Recall: {results['intentional']['recall']:.4f}\n")
        f.write(f"- F1: {results['intentional']['f1']:.4f}\n\n")
        
        f.write(f"### Unintentional (-1)\n")
        f.write(f"- Precision: {results['unintentional']['precision']:.4f}\n")
        f.write(f"- Recall: {results['unintentional']['recall']:.4f}\n")
        f.write(f"- F1: {results['unintentional']['f1']:.4f}\n")
    
    print(f"✓ Summary saved to: {md_path}")


def run_all_baselines(num_runs: int = 5):
    """Run all baseline strategies multiple times and report statistics."""
    print("Loading ground truth mask...")
    gt_mask = pd.read_csv(GROUND_TRUTH_MASK, dtype=str, keep_default_na=False)
    print(f"Loaded {len(gt_mask)} records with {len(gt_mask.columns)} columns")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    # Strategy 1: Random Guessing (run multiple times)
    print(f"\n{'#'*70}")
    print(f"Strategy 1: Random Guessing (50/50) - {num_runs} runs")
    print(f"{'#'*70}")
    
    random_f1s = []
    random_accuracies = []
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        pred_mask = create_random_guessing_mask(gt_mask, seed=42+i)
        results = evaluate_predictions(gt_mask, pred_mask, f"Random_Guessing_Run{i+1}")
        random_f1s.append(results['overall']['macro_f1'])
        random_accuracies.append(results['overall']['accuracy'])
        
        if i == 0:  # Save detailed results for first run
            save_results(results, "random_guessing", OUTPUT_DIR)
            all_results.append(results)
    
    print(f"\n{'='*70}")
    print(f"Random Guessing Statistics ({num_runs} runs):")
    print(f"  Mean Accuracy: {np.mean(random_accuracies):.4f} ± {np.std(random_accuracies):.4f}")
    print(f"  Mean Macro F1: {np.mean(random_f1s):.4f} ± {np.std(random_f1s):.4f}")
    print(f"{'='*70}")
    
    # Strategy 2a: Constant Guessing - Always Intentional (1)
    print(f"\n{'#'*70}")
    print(f"Strategy 2a: Constant Guessing - Always Intentional (1)")
    print(f"{'#'*70}")
    
    pred_mask = create_constant_guessing_mask(gt_mask, constant_value=1)
    results = evaluate_predictions(gt_mask, pred_mask, "Constant_Guessing_Always_Intentional")
    save_results(results, "constant_always_intentional", OUTPUT_DIR)
    all_results.append(results)
    
    # Strategy 2b: Constant Guessing - Always Unintentional (-1)
    print(f"\n{'#'*70}")
    print(f"Strategy 2b: Constant Guessing - Always Unintentional (-1)")
    print(f"{'#'*70}")
    
    pred_mask = create_constant_guessing_mask(gt_mask, constant_value=-1)
    results = evaluate_predictions(gt_mask, pred_mask, "Constant_Guessing_Always_Unintentional")
    save_results(results, "constant_always_unintentional", OUTPUT_DIR)
    all_results.append(results)
    
    # Strategy 3: Random with Probability (multiple probability values)
    probabilities = [0.4, 0.5, 0.6, 0.7]
    
    for prob in probabilities:
        print(f"\n{'#'*70}")
        print(f"Strategy 3: Random Guessing with P(intentional)={prob}")
        print(f"{'#'*70}")
        
        prob_f1s = []
        prob_accuracies = []
        for i in range(num_runs):
            print(f"\n--- Run {i+1}/{num_runs} ---")
            pred_mask = create_probability_guessing_mask(gt_mask, prob_intentional=prob, seed=42+i)
            results = evaluate_predictions(gt_mask, pred_mask, f"Probability_{prob}_Run{i+1}")
            prob_f1s.append(results['overall']['macro_f1'])
            prob_accuracies.append(results['overall']['accuracy'])
            
            if i == 0:  # Save detailed results for first run
                save_results(results, f"probability_{int(prob*100)}", OUTPUT_DIR)
                all_results.append(results)
        
        print(f"\n{'='*70}")
        print(f"Probability Guessing P={prob} Statistics ({num_runs} runs):")
        print(f"  Mean Accuracy: {np.mean(prob_accuracies):.4f} ± {np.std(prob_accuracies):.4f}")
        print(f"  Mean Macro F1: {np.mean(prob_f1s):.4f} ± {np.std(prob_f1s):.4f}")
        print(f"{'='*70}")
    
    # Create comparison summary
    create_comparison_summary(all_results, OUTPUT_DIR)
    
    print(f"\n{'='*70}")
    print(f"ALL BASELINE EVALUATIONS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")


def create_comparison_summary(results_list: list, output_dir: str):
    """Create a summary comparing all baseline strategies."""
    summary_path = os.path.join(output_dir, "BASELINE_COMPARISON_SUMMARY.md")
    
    with open(summary_path, 'w') as f:
        f.write("# Baseline Guessing Strategies - Comparison Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This report compares the performance of different baseline guessing strategies ")
        f.write("for intent attribution on the Adult Income dataset.\n\n")
        
        f.write("## Strategy Comparison\n\n")
        f.write("| Strategy | Accuracy | Macro F1 | Macro Prec | Macro Rec |\n")
        f.write("|----------|----------|----------|------------|----------|\n")
        
        for result in results_list:
            strategy = result['strategy']
            acc = result['overall']['accuracy']
            f1 = result['overall']['macro_f1']
            prec = result['overall']['macro_precision']
            rec = result['overall']['macro_recall']
            f.write(f"| {strategy} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |\n")
        
        f.write("\n## Expected Results\n\n")
        f.write("- **Random Guessing (50/50)**: Should achieve ~50% accuracy, ~50% F1\n")
        f.write("- **Constant Strategies**: Biased toward one class (high precision, low recall or vice versa)\n")
        f.write("- **Probability Guessing**: Performance proportional to ground truth distribution\n\n")
        
        f.write("## Purpose\n\n")
        f.write("These baselines establish the minimum expected performance. Any LLM model should ")
        f.write("significantly outperform random guessing to demonstrate value.\n")
    
    print(f"\n✓ Comparison summary saved to: {summary_path}")
    
    # Also save as CSV
    comparison_df = pd.DataFrame([
        {
            'Strategy': result['strategy'],
            'Accuracy': result['overall']['accuracy'],
            'Macro_F1': result['overall']['macro_f1'],
            'Macro_Precision': result['overall']['macro_precision'],
            'Macro_Recall': result['overall']['macro_recall']
        }
        for result in results_list
    ])
    comparison_df = comparison_df.sort_values('Macro_F1', ascending=False)
    csv_path = os.path.join(output_dir, "baseline_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved to: {csv_path}")


if __name__ == "__main__":
    print("="*70)
    print("BASELINE GUESSING STRATEGIES EVALUATION")
    print("Adult Income Dataset - Intent Attribution")
    print("="*70)
    print("\nTesting three baseline strategies:")
    print("  1. Random Guessing (50/50)")
    print("  2. Constant Guessing (always same value)")
    print("  3. Random Guessing with Probability (custom distribution)")
    print("="*70)
    
    # Run all baselines with 5 runs for stochastic strategies
    run_all_baselines(num_runs=5)
