#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Guessing Strategies for Intent Attribution
----------------------------------------------------
Tests the performance of simple baseline strategies:
1. Random Guessing: 50/50 chance of 1 or -1 for changed features
2. Constant Guessing: Always predict same value (1 or -1)
3. Random Guessing with Probability: Use custom probability distributions

Purpose: Establish baseline performance to compare against LLM results.

Ground Truth:
- All changes in the dataset are INTENTIONAL (bot-to-human evasion)
- mask=1 means the feature was changed (ground truth: intentional)
- mask=0 means the feature was unchanged (ground truth: N/A)

Baseline Predictions:
- 1 = predicted intentional
- -1 = predicted unintentional
- 0 = no change (copy from ground truth mask)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import json
import os
from datetime import datetime

# File paths
GROUND_TRUTH_MASK = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_mask.csv"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/baselines"

# Columns to exclude
EXCLUDE_COLS = ["user_id"]


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
    For each changed feature (mask=1), randomly predict 1 or -1 with equal probability.
    """
    np.random.seed(seed)
    pred_mask = gt_mask.copy()
    
    for col in pred_mask.columns:
        if col in EXCLUDE_COLS:
            continue
        
        for idx in range(len(pred_mask)):
            val = normalize_value(pred_mask.iloc[idx][col])
            if val == 1:  # Changed feature
                # Randomly choose 1 (intentional) or -1 (unintentional)
                pred_mask.iloc[idx, pred_mask.columns.get_loc(col)] = np.random.choice([1, -1])
            # else: keep as 0 (unchanged)
    
    return pred_mask


def create_constant_guessing_mask(gt_mask: pd.DataFrame, constant_value: int = 1) -> pd.DataFrame:
    """
    Strategy 2: Constant Guessing
    For each changed feature (mask=1), always predict the same value.
    
    Args:
        constant_value: Either 1 (always intentional) or -1 (always unintentional)
    """
    pred_mask = gt_mask.copy()
    
    for col in pred_mask.columns:
        if col in EXCLUDE_COLS:
            continue
        
        for idx in range(len(pred_mask)):
            val = normalize_value(pred_mask.iloc[idx][col])
            if val == 1:  # Changed feature
                pred_mask.iloc[idx, pred_mask.columns.get_loc(col)] = constant_value
            # else: keep as 0 (unchanged)
    
    return pred_mask


def create_probability_guessing_mask(gt_mask: pd.DataFrame, prob_intentional: float = 0.7, seed: int = 42) -> pd.DataFrame:
    """
    Strategy 3: Random Guessing with Probability
    For each changed feature (mask=1), predict 1 with probability p, -1 with probability (1-p).
    
    Args:
        prob_intentional: Probability of predicting 1 (intentional), e.g., 0.7 = 70% chance
    """
    np.random.seed(seed)
    pred_mask = gt_mask.copy()
    
    for col in pred_mask.columns:
        if col in EXCLUDE_COLS:
            continue
        
        for idx in range(len(pred_mask)):
            val = normalize_value(pred_mask.iloc[idx][col])
            if val == 1:  # Changed feature
                # Predict based on probability
                if np.random.random() < prob_intentional:
                    pred_mask.iloc[idx, pred_mask.columns.get_loc(col)] = 1
                else:
                    pred_mask.iloc[idx, pred_mask.columns.get_loc(col)] = -1
            # else: keep as 0 (unchanged)
    
    return pred_mask


def evaluate_predictions(gt_mask: pd.DataFrame, pred_mask: pd.DataFrame, strategy_name: str):
    """
    Evaluate baseline strategy performance.
    
    Returns dict with metrics.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATION: {strategy_name}")
    print(f"{'='*70}")
    
    # Get columns to evaluate
    eval_cols = [col for col in gt_mask.columns if col not in EXCLUDE_COLS]
    
    # Flatten all values for overall metrics
    all_gt = []
    all_pred = []
    
    # Per-column metrics
    column_metrics = {}
    
    for col in eval_cols:
        col_gt = []
        col_pred = []
        
        for idx in range(len(gt_mask)):
            gt_val = normalize_value(gt_mask.iloc[idx][col])
            pred_val = normalize_value(pred_mask.iloc[idx][col])
            
            # Only evaluate changed features (mask=1)
            if gt_val == 1:
                # Ground truth is always 1 (intentional)
                # Prediction is 1 (correct) or -1 (incorrect)
                col_gt.append(1)
                col_pred.append(1 if pred_val == 1 else 0)  # Convert -1 to 0 for binary classification
                
                all_gt.append(1)
                all_pred.append(1 if pred_val == 1 else 0)
        
        # Calculate per-column metrics if there are any changes
        if len(col_gt) > 0:
            accuracy = accuracy_score(col_gt, col_pred)
            # Since all ground truth is 1, precision/recall are simple
            tp = sum(col_pred)
            fn = len(col_pred) - tp
            
            column_metrics[col] = {
                'total_changes': len(col_gt),
                'correct_predictions': tp,
                'incorrect_predictions': fn,
                'accuracy': accuracy
            }
    
    # Overall metrics
    if len(all_gt) > 0:
        overall_accuracy = accuracy_score(all_gt, all_pred)
        tp = sum(all_pred)
        fn = len(all_pred) - tp
        total = len(all_gt)
        
        print(f"\nOverall Performance:")
        print(f"  Total changed features: {total}")
        print(f"  Correct predictions (TP): {tp}")
        print(f"  Incorrect predictions (FN): {fn}")
        print(f"  Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        
        # Additional insights
        print(f"\nInterpretation:")
        print(f"  - The model correctly identified {tp}/{total} intentional changes")
        print(f"  - The model missed {fn}/{total} intentional changes (labeled as unintentional)")
    else:
        overall_accuracy = 0
        tp = fn = total = 0
        print("No changes found in dataset!")
    
    return {
        'strategy': strategy_name,
        'overall': {
            'total_changes': total,
            'correct_predictions': tp,
            'incorrect_predictions': fn,
            'accuracy': overall_accuracy
        },
        'per_column': column_metrics
    }


def save_results(results: dict, strategy_name: str, output_dir: str):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{strategy_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Save per-column metrics as CSV
    if results['per_column']:
        df_cols = pd.DataFrame.from_dict(results['per_column'], orient='index')
        df_cols = df_cols.sort_values('accuracy', ascending=False)
        csv_path = os.path.join(output_dir, f"{strategy_name}_per_column.csv")
        df_cols.to_csv(csv_path)
        print(f"✓ Per-column metrics saved to: {csv_path}")
    
    # Save summary markdown
    md_path = os.path.join(output_dir, f"{strategy_name}_SUMMARY.md")
    with open(md_path, 'w') as f:
        f.write(f"# {results['strategy']} - Evaluation Summary\n\n")
        f.write(f"**Evaluated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Overall Performance\n\n")
        f.write(f"- **Total Changed Features**: {results['overall']['total_changes']}\n")
        f.write(f"- **Correct Predictions**: {results['overall']['correct_predictions']}\n")
        f.write(f"- **Incorrect Predictions**: {results['overall']['incorrect_predictions']}\n")
        f.write(f"- **Accuracy**: {results['overall']['accuracy']:.4f} ({results['overall']['accuracy']*100:.2f}%)\n\n")
        
        f.write(f"## Interpretation\n\n")
        f.write(f"This baseline strategy correctly identified **{results['overall']['correct_predictions']}** ")
        f.write(f"out of **{results['overall']['total_changes']}** intentional changes.\n\n")
        
        if results['per_column']:
            f.write(f"## Top 10 Features by Accuracy\n\n")
            f.write(f"| Feature | Accuracy | Correct | Total Changes |\n")
            f.write(f"|---------|----------|---------|---------------|\n")
            
            sorted_cols = sorted(results['per_column'].items(), 
                               key=lambda x: x[1]['accuracy'], reverse=True)[:10]
            for col, metrics in sorted_cols:
                f.write(f"| {col} | {metrics['accuracy']:.4f} | ")
                f.write(f"{metrics['correct_predictions']}/{metrics['total_changes']} | ")
                f.write(f"{metrics['total_changes']} |\n")
    
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
    
    random_accuracies = []
    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        pred_mask = create_random_guessing_mask(gt_mask, seed=42+i)
        results = evaluate_predictions(gt_mask, pred_mask, f"Random_Guessing_Run{i+1}")
        random_accuracies.append(results['overall']['accuracy'])
        
        if i == 0:  # Save detailed results for first run
            save_results(results, "random_guessing", OUTPUT_DIR)
            all_results.append(results)
    
    print(f"\n{'='*70}")
    print(f"Random Guessing Statistics ({num_runs} runs):")
    print(f"  Mean Accuracy: {np.mean(random_accuracies):.4f} ± {np.std(random_accuracies):.4f}")
    print(f"  Min Accuracy: {np.min(random_accuracies):.4f}")
    print(f"  Max Accuracy: {np.max(random_accuracies):.4f}")
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
    probabilities = [0.6, 0.7, 0.8, 0.9]
    
    for prob in probabilities:
        print(f"\n{'#'*70}")
        print(f"Strategy 3: Random Guessing with P(intentional)={prob}")
        print(f"{'#'*70}")
        
        prob_accuracies = []
        for i in range(num_runs):
            print(f"\n--- Run {i+1}/{num_runs} ---")
            pred_mask = create_probability_guessing_mask(gt_mask, prob_intentional=prob, seed=42+i)
            results = evaluate_predictions(gt_mask, pred_mask, f"Probability_{prob}_Run{i+1}")
            prob_accuracies.append(results['overall']['accuracy'])
            
            if i == 0:  # Save detailed results for first run
                save_results(results, f"probability_{int(prob*100)}", OUTPUT_DIR)
                all_results.append(results)
        
        print(f"\n{'='*70}")
        print(f"Probability Guessing P={prob} Statistics ({num_runs} runs):")
        print(f"  Mean Accuracy: {np.mean(prob_accuracies):.4f} ± {np.std(prob_accuracies):.4f}")
        print(f"  Min Accuracy: {np.min(prob_accuracies):.4f}")
        print(f"  Max Accuracy: {np.max(prob_accuracies):.4f}")
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
        f.write("for intent attribution. All changes in the ground truth are INTENTIONAL, so ")
        f.write("accuracy represents the percentage of changes correctly identified as intentional.\n\n")
        
        f.write("## Strategy Comparison\n\n")
        f.write("| Strategy | Accuracy | Correct | Total |\n")
        f.write("|----------|----------|---------|-------|\n")
        
        for result in results_list:
            strategy = result['strategy']
            acc = result['overall']['accuracy']
            correct = result['overall']['correct_predictions']
            total = result['overall']['total_changes']
            f.write(f"| {strategy} | {acc:.4f} ({acc*100:.2f}%) | {correct} | {total} |\n")
        
        f.write("\n## Expected Results\n\n")
        f.write("- **Random Guessing (50/50)**: Should achieve ~50% accuracy\n")
        f.write("- **Constant Always Intentional (1)**: Should achieve 100% accuracy (all ground truth is intentional)\n")
        f.write("- **Constant Always Unintentional (-1)**: Should achieve 0% accuracy\n")
        f.write("- **Probability Guessing P=0.7**: Should achieve ~70% accuracy\n")
        f.write("- **Probability Guessing P=0.8**: Should achieve ~80% accuracy\n")
        f.write("- **Probability Guessing P=0.9**: Should achieve ~90% accuracy\n\n")
        
        f.write("## Purpose\n\n")
        f.write("These baselines establish the minimum expected performance. Any LLM model should ")
        f.write("significantly outperform random guessing (50%) to demonstrate value. The constant ")
        f.write("strategy provides the theoretical maximum (100%).\n")
    
    print(f"\n✓ Comparison summary saved to: {summary_path}")
    
    # Also save as CSV
    comparison_df = pd.DataFrame([
        {
            'Strategy': result['strategy'],
            'Accuracy': result['overall']['accuracy'],
            'Correct_Predictions': result['overall']['correct_predictions'],
            'Total_Changes': result['overall']['total_changes']
        }
        for result in results_list
    ])
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    csv_path = os.path.join(output_dir, "baseline_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Comparison CSV saved to: {csv_path}")


if __name__ == "__main__":
    print("="*70)
    print("BASELINE GUESSING STRATEGIES EVALUATION")
    print("="*70)
    print("\nTesting three baseline strategies:")
    print("  1. Random Guessing (50/50)")
    print("  2. Constant Guessing (always same value)")
    print("  3. Random Guessing with Probability (custom distribution)")
    print("="*70)
    
    # Run all baselines with 5 runs for stochastic strategies
    run_all_baselines(num_runs=5)
