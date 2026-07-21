#!/usr/bin/env python3
"""
Optimal Threshold Finder for Stage 2 Cell-Level Detection
----------------------------------------------------------
Analyzes suspicion scores to find threshold that maximizes F1 score.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
# Use the latest run with 0 failed chunks
SUSPICIONS_PATH = PROJECT_ROOT / "outputs/stage2/LLMs/twitterBot-dataset/run_20251218_125345/cell_suspicions.csv"
MASKS_PATH = PROJECT_ROOT / "datasets/twitter_bot/ground_truth_masks.csv"
DATA_PATH = PROJECT_ROOT / "datasets/twitter_bot/labeled_dataset.csv"

print("="*80)
print("OPTIMAL THRESHOLD ANALYSIS FOR STAGE 2")
print("="*80)

# Load data
print("\n📂 Loading data...")
suspicions = pd.read_csv(SUSPICIONS_PATH)
gt_masks_full = pd.read_csv(MASKS_PATH)
df = pd.read_csv(DATA_PATH)

print(f"   Suspicions shape: {suspicions.shape}")
print(f"   Ground truth (full) shape: {gt_masks_full.shape}")

# Get indices of erroneous records (same logic as run_stage2_llm_twitter.py)
erroneous_original_indices = df[df['is_erroneous'] == 1].index.tolist()
print(f"   Total erroneous records: {len(erroneous_original_indices)}")

# Filter ground truth to only erroneous records
gt_masks = gt_masks_full.iloc[erroneous_original_indices].reset_index(drop=True)
print(f"   Ground truth (erroneous only) shape: {gt_masks.shape}")

# Flatten to cell-level
y_true = gt_masks.values.flatten()
y_scores = suspicions.values.flatten()

# Remove any NaN values
valid_mask = ~(np.isnan(y_true) | np.isnan(y_scores))
y_true = y_true[valid_mask]
y_scores = y_scores[valid_mask]

print(f"\n📊 Dataset statistics:")
print(f"   Total cells: {len(y_true)}")
print(f"   True error cells: {y_true.sum()} ({100*y_true.sum()/len(y_true):.2f}%)")
print(f"   Clean cells: {(y_true == 0).sum()} ({100*(y_true == 0).sum()/len(y_true):.2f}%)")

print(f"\n📈 Suspicion score distribution:")
print(f"   Min: {y_scores.min():.3f}")
print(f"   25th percentile: {np.percentile(y_scores, 25):.3f}")
print(f"   Median: {np.median(y_scores):.3f}")
print(f"   75th percentile: {np.percentile(y_scores, 75):.3f}")
print(f"   Max: {y_scores.max():.3f}")
print(f"   Mean: {y_scores.mean():.3f}")
print(f"   Std: {y_scores.std():.3f}")

# Score distribution by class
scores_error = y_scores[y_true == 1]
scores_clean = y_scores[y_true == 0]

print(f"\n📊 Scores for ERROR cells:")
print(f"   Mean: {scores_error.mean():.3f}")
print(f"   Median: {np.median(scores_error):.3f}")
print(f"   Std: {scores_error.std():.3f}")

print(f"\n📊 Scores for CLEAN cells:")
print(f"   Mean: {scores_clean.mean():.3f}")
print(f"   Median: {np.median(scores_clean):.3f}")
print(f"   Std: {scores_clean.std():.3f}")

# Test different thresholds
print("\n" + "="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

thresholds = np.arange(0.0, 1.01, 0.05)
results = []

print(f"\n{'Threshold':<12} {'Flagged':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8} {'TN':<8}")
print("-"*110)

best_f1 = 0
best_threshold = 0
best_metrics = None

for threshold in thresholds:
    y_pred = (y_scores >= threshold).astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    flagged = y_pred.sum()
    flagged_pct = 100 * flagged / len(y_pred)
    
    results.append({
        'threshold': threshold,
        'flagged': flagged,
        'flagged_pct': flagged_pct,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    })
    
    marker = ""
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = results[-1]
        marker = "⭐"
    
    print(f"{threshold:<12.2f} {flagged:>6} ({flagged_pct:>4.1f}%) {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} "
          f"{tp:>7} {fp:>7} {fn:>7} {tn:>7} {marker}")

# Summary
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

# Best F1
print(f"\n🎯 BEST F1-SCORE: {best_f1:.3f}")
print(f"   Threshold: {best_threshold:.2f}")
print(f"   Precision: {best_metrics['precision']:.3f}")
print(f"   Recall: {best_metrics['recall']:.3f}")
print(f"   Flagged: {best_metrics['flagged']} cells ({best_metrics['flagged_pct']:.2f}%)")
print(f"   TP: {best_metrics['tp']:>6}  FP: {best_metrics['fp']:>6}")
print(f"   FN: {best_metrics['fn']:>6}  TN: {best_metrics['tn']:>6}")

# Current threshold (0.5)
current = [r for r in results if abs(r['threshold'] - 0.5) < 0.01][0]
print(f"\n📊 CURRENT (threshold=0.50):")
print(f"   F1: {current['f1']:.3f}")
print(f"   Precision: {current['precision']:.3f}")
print(f"   Recall: {current['recall']:.3f}")
print(f"   Flagged: {current['flagged']} cells ({current['flagged_pct']:.2f}%)")
print(f"   TP: {current['tp']:>6}  FP: {current['fp']:>6}")
print(f"   FN: {current['fn']:>6}  TN: {current['tn']:>6}")

# Improvement
improvement_f1 = ((best_f1 - current['f1']) / current['f1']) * 100 if current['f1'] > 0 else 0
improvement_fp = current['fp'] - best_metrics['fp']
improvement_fn = current['fn'] - best_metrics['fn']

print(f"\n📈 IMPROVEMENT:")
print(f"   F1 change: {best_f1 - current['f1']:+.3f} ({improvement_f1:+.1f}%)")
print(f"   FP change: {improvement_fp:+d} ({'↓' if improvement_fp > 0 else '↑'} {abs(improvement_fp)} fewer FPs)" if improvement_fp != 0 else "   FP change: 0")
print(f"   FN change: {improvement_fn:+d} ({'↓' if improvement_fn > 0 else '↑'} {abs(improvement_fn)} fewer FNs)" if improvement_fn != 0 else "   FN change: 0")

# Other interesting thresholds
print(f"\n💡 OTHER NOTABLE THRESHOLDS:")

# Best precision
best_prec = max(results, key=lambda x: x['precision'])
if best_prec['recall'] > 0.20:  # Only if recall not too low
    print(f"\n   Best Precision: {best_prec['precision']:.3f} @ threshold={best_prec['threshold']:.2f}")
    print(f"     Recall: {best_prec['recall']:.3f}, F1: {best_prec['f1']:.3f}")
    print(f"     FP: {best_prec['fp']}, FN: {best_prec['fn']}")

# Best recall
best_rec = max(results, key=lambda x: x['recall'])
if best_rec['precision'] > 0.20:  # Only if precision not too low
    print(f"\n   Best Recall: {best_rec['recall']:.3f} @ threshold={best_rec['threshold']:.2f}")
    print(f"     Precision: {best_rec['precision']:.3f}, F1: {best_rec['f1']:.3f}")
    print(f"     FP: {best_rec['fp']}, FN: {best_rec['fn']}")

# Balanced (closest to equal FP and FN)
balanced = min(results, key=lambda x: abs(x['fp'] - x['fn']))
print(f"\n   Balanced FP/FN: @ threshold={balanced['threshold']:.2f}")
print(f"     Precision: {balanced['precision']:.3f}, Recall: {balanced['recall']:.3f}, F1: {balanced['f1']:.3f}")
print(f"     FP: {balanced['fp']}, FN: {balanced['fn']} (difference: {abs(balanced['fp'] - balanced['fn'])})")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"\n💡 Recommendation: Set threshold to {best_threshold:.2f} in your config")
print(f"   Expected F1 improvement: {best_f1:.3f} → {current['f1']:.3f} ({improvement_f1:+.1f}%)")

