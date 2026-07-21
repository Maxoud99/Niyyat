#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extended Baseline Guessing Strategies for Intent Attribution
-------------------------------------------------------------
Tests the performance of random guessing strategies across full probability range:
- Random guessing with probabilities from 0% to 100% in 10% increments
- Multiple runs for each probability to get statistical confidence

Purpose: Establish comprehensive baseline performance across all probability ranges.

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
from sklearn.metrics import accuracy_score
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
GROUND_TRUTH_MASK = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/combined_mask.csv"
OUTPUT_DIR = "/home/mohamed/error_injector/llms_baseline/klim-kireev/datasets/twitter-bot/intent-attribution/outputs/baselines/full_range"

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


def create_probability_guessing_mask(gt_mask: pd.DataFrame, prob_intentional: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """
    Random Guessing with Probability
    For each changed feature (mask=1), predict 1 with probability p, -1 with probability (1-p).
    
    Args:
        gt_mask: Ground truth mask dataframe
        prob_intentional: Probability of predicting 1 (intentional), range [0, 1]
        seed: Random seed for reproducibility
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


def evaluate_predictions(gt_mask: pd.DataFrame, pred_mask: pd.DataFrame) -> dict:
    """
    Evaluate baseline strategy performance.
    
    Returns dict with metrics.
    """
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
    else:
        overall_accuracy = 0
        tp = fn = total = 0
    
    return {
        'overall': {
            'total_changes': total,
            'correct_predictions': tp,
            'incorrect_predictions': fn,
            'accuracy': overall_accuracy
        },
        'per_column': column_metrics
    }


def run_full_range_baselines(num_runs: int = 10, probability_step: int = 10):
    """
    Run baseline guessing strategies across full probability range.
    
    Args:
        num_runs: Number of runs per probability for statistical confidence
        probability_step: Step size for probabilities (e.g., 10 for 0%, 10%, 20%, ..., 100%)
    """
    print("="*80)
    print("FULL RANGE BASELINE GUESSING EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Probability range: 0% to 100% in {probability_step}% steps")
    print(f"  - Runs per probability: {num_runs}")
    print("="*80)
    
    # Load ground truth
    print("\nLoading ground truth mask...")
    gt_mask = pd.read_csv(GROUND_TRUTH_MASK, dtype=str, keep_default_na=False)
    print(f"✓ Loaded {len(gt_mask)} records with {len(gt_mask.columns)} columns")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate probability values
    probabilities = [p / 100.0 for p in range(0, 101, probability_step)]
    
    # Store results
    all_results = []
    summary_stats = []
    
    # Run for each probability
    for prob in probabilities:
        prob_pct = int(prob * 100)
        print(f"\n{'#'*80}")
        print(f"Probability: {prob_pct}% (P(intentional) = {prob:.2f})")
        print(f"{'#'*80}")
        
        accuracies = []
        detailed_results = []
        
        for run_idx in range(num_runs):
            # Create prediction mask
            pred_mask = create_probability_guessing_mask(gt_mask, prob_intentional=prob, seed=42 + prob_pct + run_idx)
            
            # Evaluate
            results = evaluate_predictions(gt_mask, pred_mask)
            accuracies.append(results['overall']['accuracy'])
            detailed_results.append(results)
            
            print(f"  Run {run_idx+1:2d}/{num_runs}: Accuracy = {results['overall']['accuracy']:.4f} "
                  f"({results['overall']['correct_predictions']}/{results['overall']['total_changes']})")
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        print(f"\n  Summary Statistics:")
        print(f"    Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    Range: [{min_acc:.4f}, {max_acc:.4f}]")
        print(f"    Expected: ~{prob:.2f} (theoretical)")
        print(f"    Difference: {abs(mean_acc - prob):.4f}")
        
        # Store summary
        summary_stats.append({
            'probability_percent': prob_pct,
            'probability': prob,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'min_accuracy': min_acc,
            'max_accuracy': max_acc,
            'num_runs': num_runs,
            'total_changes': detailed_results[0]['overall']['total_changes']
        })
        
        # Save first run's detailed results
        strategy_name = f"probability_{prob_pct:03d}"
        save_detailed_results(detailed_results[0], strategy_name, prob, OUTPUT_DIR)
        
        all_results.append({
            'probability': prob,
            'probability_percent': prob_pct,
            'runs': detailed_results,
            'statistics': {
                'mean': mean_acc,
                'std': std_acc,
                'min': min_acc,
                'max': max_acc
            }
        })
    
    # Save summary statistics
    save_summary_statistics(summary_stats, OUTPUT_DIR)
    
    # Create visualizations
    create_visualizations(summary_stats, OUTPUT_DIR)
    
    # Save all results
    save_all_results(all_results, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")
    
    return summary_stats


def save_detailed_results(results: dict, strategy_name: str, probability: float, output_dir: str):
    """Save detailed results for a single run."""
    # Save JSON
    json_path = os.path.join(output_dir, f"{strategy_name}_results.json")
    results_with_meta = {
        'strategy': strategy_name,
        'probability': probability,
        'probability_percent': int(probability * 100),
        **results
    }
    with open(json_path, 'w') as f:
        json.dump(results_with_meta, f, indent=2)
    
    # Save per-column metrics as CSV
    if results['per_column']:
        df_cols = pd.DataFrame.from_dict(results['per_column'], orient='index')
        df_cols = df_cols.sort_values('accuracy', ascending=False)
        csv_path = os.path.join(output_dir, f"{strategy_name}_per_column.csv")
        df_cols.to_csv(csv_path)


def save_summary_statistics(summary_stats: list, output_dir: str):
    """Save summary statistics across all probabilities."""
    # Save as CSV
    df = pd.DataFrame(summary_stats)
    csv_path = os.path.join(output_dir, "full_range_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary statistics saved to: {csv_path}")
    
    # Save as Markdown
    md_path = os.path.join(output_dir, "FULL_RANGE_SUMMARY.md")
    with open(md_path, 'w') as f:
        f.write("# Full Range Baseline Guessing - Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This report presents baseline guessing strategies across the full probability range ")
        f.write("(0% to 100% in 10% increments). Each probability was tested with multiple runs to ")
        f.write("establish statistical confidence.\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Probability | Mean Accuracy | Std Dev | Min | Max | Runs |\n")
        f.write("|-------------|---------------|---------|-----|-----|------|\n")
        
        for stat in summary_stats:
            f.write(f"| {stat['probability_percent']:3d}% | ")
            f.write(f"{stat['mean_accuracy']:.4f} | ")
            f.write(f"{stat['std_accuracy']:.4f} | ")
            f.write(f"{stat['min_accuracy']:.4f} | ")
            f.write(f"{stat['max_accuracy']:.4f} | ")
            f.write(f"{stat['num_runs']} |\n")
        
        f.write("\n## Key Observations\n\n")
        f.write("1. **Theoretical vs Actual**: The mean accuracy should closely match the probability\n")
        f.write("2. **Edge Cases**:\n")
        f.write("   - P=0% should achieve ~0% accuracy (always predict unintentional)\n")
        f.write("   - P=50% should achieve ~50% accuracy (random guessing)\n")
        f.write("   - P=100% should achieve ~100% accuracy (always predict intentional)\n")
        f.write("3. **Variance**: Higher variance indicates more random behavior\n\n")
        
        f.write("## Purpose\n\n")
        f.write("These baselines establish performance expectations across all probability ranges. ")
        f.write("Any LLM model should significantly outperform random guessing (50%) and ideally ")
        f.write("approach or exceed high-probability baselines (80-90%).\n")
    
    print(f"✓ Summary markdown saved to: {md_path}")


def save_all_results(all_results: list, output_dir: str):
    """Save complete results including all runs."""
    json_path = os.path.join(output_dir, "full_range_all_results.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Complete results saved to: {json_path}")


def create_visualizations(summary_stats: list, output_dir: str):
    """Create visualizations of the results."""
    df = pd.DataFrame(summary_stats)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy vs Probability with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(df['probability_percent'], df['mean_accuracy'], 
                 yerr=df['std_accuracy'], fmt='o-', capsize=5, 
                 markersize=8, linewidth=2, color='#2E86AB', ecolor='#A23B72')
    ax1.plot([0, 100], [0, 1], 'r--', label='Theoretical (y=x)', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Probability of Predicting Intentional (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Accuracy vs Probability', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Difference from theoretical
    ax2 = axes[0, 1]
    differences = [abs(row['mean_accuracy'] - row['probability']) for _, row in df.iterrows()]
    ax2.bar(df['probability_percent'], differences, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('|Mean Accuracy - Theoretical|', fontsize=12, fontweight='bold')
    ax2.set_title('Deviation from Theoretical Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Standard Deviation
    ax3 = axes[1, 0]
    ax3.plot(df['probability_percent'], df['std_accuracy'], 'o-', 
             markersize=8, linewidth=2, color='#6A4C93')
    ax3.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
    ax3.set_title('Accuracy Standard Deviation Across Runs', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-5, 105)
    
    # Plot 4: Min-Max Range
    ax4 = axes[1, 1]
    ax4.fill_between(df['probability_percent'], df['min_accuracy'], df['max_accuracy'], 
                     alpha=0.3, color='#1B998B', label='Min-Max Range')
    ax4.plot(df['probability_percent'], df['mean_accuracy'], 'o-', 
             markersize=8, linewidth=2, color='#1B998B', label='Mean')
    ax4.plot([0, 100], [0, 1], 'r--', label='Theoretical', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Probability (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax4.set_title('Accuracy Range (Min, Mean, Max)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-5, 105)
    ax4.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, "full_range_visualization.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {fig_path}")
    
    plt.close()
    
    # Create a simple line plot as well
    plt.figure(figsize=(12, 7))
    plt.plot(df['probability_percent'], df['mean_accuracy'], 'o-', 
             markersize=10, linewidth=3, color='#2E86AB', label='Empirical Mean')
    plt.plot([0, 100], [0, 1], 'r--', label='Theoretical', linewidth=2, alpha=0.7)
    plt.fill_between(df['probability_percent'], 
                     df['mean_accuracy'] - df['std_accuracy'],
                     df['mean_accuracy'] + df['std_accuracy'],
                     alpha=0.2, color='#2E86AB', label='±1 Std Dev')
    plt.xlabel('Probability of Predicting Intentional (%)', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Baseline Guessing Performance: Full Probability Range', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 105)
    plt.ylim(-0.05, 1.05)
    
    simple_fig_path = os.path.join(output_dir, "full_range_simple.png")
    plt.savefig(simple_fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Simple visualization saved to: {simple_fig_path}")
    
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run full range baseline guessing evaluation')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of runs per probability (default: 10)')
    parser.add_argument('--step', type=int, default=10,
                       help='Probability step size in percent (default: 10)')
    
    args = parser.parse_args()
    
    # Run evaluation
    summary_stats = run_full_range_baselines(num_runs=args.num_runs, probability_step=args.step)
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Prob %':>7} | {'Mean Acc':>9} | {'Std Dev':>8} | {'Min':>7} | {'Max':>7} | {'Diff from Theory':>16}")
    print("-" * 80)
    for stat in summary_stats:
        diff = abs(stat['mean_accuracy'] - stat['probability'])
        print(f"{stat['probability_percent']:6d}% | {stat['mean_accuracy']:9.4f} | "
              f"{stat['std_accuracy']:8.4f} | {stat['min_accuracy']:7.4f} | "
              f"{stat['max_accuracy']:7.4f} | {diff:16.4f}")
    print("="*80)
