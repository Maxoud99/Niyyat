#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline Guessing Strategies for Intent Attribution - eBay Listings
---------------------------------------------------------------------------
Same methodology as adult_income_dataset/tenth-trial/baseline_guessing_strategies.py
and klim-kireev's twitter-bot variant, applied to the eBay mask
(wdc_product_analysis/intentionality/data/masks.csv, built by
build_dataset_mask.py: 0=clean, 1=intentional, -1=unintentional).

Tests the performance of simple baseline strategies:
1. Random Guessing: 50/50 chance of 1 or -1 for changed features
2. Constant Guessing: Always predict same value (1 or -1)
3. Random Guessing with Probability: Use custom probability distributions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import os
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_MASK = os.path.join(HERE, "data", "masks.csv")
OUTPUT_DIR = os.path.join(HERE, "results", "baselines")


def create_random_guessing_mask(gt_mask: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    pred_mask = gt_mask.copy().astype(float)
    random_choices = np.random.choice([1, -1], size=pred_mask.shape)
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, random_choices)
    return pred_mask.astype(int)


def create_constant_guessing_mask(gt_mask: pd.DataFrame, constant_value: int = 1) -> pd.DataFrame:
    pred_mask = gt_mask.copy().astype(float)
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, constant_value)
    return pred_mask.astype(int)


def create_probability_guessing_mask(gt_mask: pd.DataFrame, prob_intentional: float = 0.7, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    pred_mask = gt_mask.copy().astype(float)
    random_probs = np.random.random(size=pred_mask.shape)
    predictions = np.where(random_probs < prob_intentional, 1, -1)
    mask_nonzero = (pred_mask != 0)
    pred_mask = pred_mask.where(~mask_nonzero, predictions)
    return pred_mask.astype(int)


def evaluate_predictions(gt_mask: pd.DataFrame, pred_mask: pd.DataFrame, strategy_name: str):
    gt_array = gt_mask.values.astype(float)
    pred_array = pred_mask.values.astype(float)

    mask_nonzero = (gt_array != 0)
    all_gt = gt_array[mask_nonzero].flatten()
    all_pred = pred_array[mask_nonzero].flatten()

    intentional_gt = (all_gt == 1).astype(int)
    intentional_pred = (all_pred == 1).astype(int)
    unintentional_gt = (all_gt == -1).astype(int)
    unintentional_pred = (all_pred == -1).astype(int)

    prec_1 = precision_score(intentional_gt, intentional_pred, zero_division=0)
    rec_1 = recall_score(intentional_gt, intentional_pred, zero_division=0)
    f1_1 = f1_score(intentional_gt, intentional_pred, zero_division=0)

    prec_neg1 = precision_score(unintentional_gt, unintentional_pred, zero_division=0)
    rec_neg1 = recall_score(unintentional_gt, unintentional_pred, zero_division=0)
    f1_neg1 = f1_score(unintentional_gt, unintentional_pred, zero_division=0)

    macro_prec = (prec_1 + prec_neg1) / 2
    macro_rec = (rec_1 + rec_neg1) / 2
    macro_f1 = (f1_1 + f1_neg1) / 2
    accuracy = accuracy_score(all_gt, all_pred)

    intentional_count = int(intentional_gt.sum())
    unintentional_count = int(unintentional_gt.sum())
    total = len(all_gt)

    print(f"[{strategy_name}] acc={accuracy:.4f} macro_f1={macro_f1:.4f} "
          f"(int={intentional_count}, unint={unintentional_count}, total={total})")

    return {
        'strategy': strategy_name,
        'overall': {
            'accuracy': accuracy, 'macro_f1': macro_f1,
            'macro_precision': macro_prec, 'macro_recall': macro_rec,
            'total_changes': total,
            'intentional_count': intentional_count,
            'unintentional_count': unintentional_count,
        },
        'intentional': {'precision': prec_1, 'recall': rec_1, 'f1': f1_1},
        'unintentional': {'precision': prec_neg1, 'recall': rec_neg1, 'f1': f1_neg1},
    }


def save_results(results: dict, strategy_name: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{strategy_name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)


def create_comparison_summary(results_list: list, output_dir: str):
    summary_path = os.path.join(output_dir, "BASELINE_COMPARISON_SUMMARY.md")
    with open(summary_path, 'w') as f:
        f.write("# Baseline Guessing Strategies - eBay - Comparison Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Strategy | Accuracy | Macro F1 | Macro Prec | Macro Rec |\n")
        f.write("|----------|----------|----------|------------|----------|\n")
        for result in results_list:
            o = result['overall']
            f.write(f"| {result['strategy']} | {o['accuracy']:.4f} | {o['macro_f1']:.4f} | "
                    f"{o['macro_precision']:.4f} | {o['macro_recall']:.4f} |\n")

    comparison_df = pd.DataFrame([
        {'Strategy': r['strategy'], 'Accuracy': r['overall']['accuracy'],
         'Macro_F1': r['overall']['macro_f1'],
         'Macro_Precision': r['overall']['macro_precision'],
         'Macro_Recall': r['overall']['macro_recall']}
        for r in results_list
    ]).sort_values('Macro_F1', ascending=False)
    comparison_df.to_csv(os.path.join(output_dir, "baseline_comparison.csv"), index=False)
    print(f"Comparison saved -> {output_dir}")


def run_all_baselines(num_runs: int = 5):
    gt_mask = pd.read_csv(GROUND_TRUTH_MASK, dtype=str, keep_default_na=False)
    gt_mask = gt_mask.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    print(f"Loaded {len(gt_mask)} records x {len(gt_mask.columns)} columns")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = []

    random_f1s, random_accuracies = [], []
    for i in range(num_runs):
        pred_mask = create_random_guessing_mask(gt_mask, seed=42 + i)
        results = evaluate_predictions(gt_mask, pred_mask, f"Random_Guessing_Run{i+1}")
        random_f1s.append(results['overall']['macro_f1'])
        random_accuracies.append(results['overall']['accuracy'])
        if i == 0:
            save_results(results, "random_guessing", OUTPUT_DIR)
            all_results.append(results)
    print(f"Random Guessing: acc={np.mean(random_accuracies):.4f}+-{np.std(random_accuracies):.4f} "
          f"f1={np.mean(random_f1s):.4f}+-{np.std(random_f1s):.4f}")

    for const_val, label in [(1, "constant_always_intentional"), (-1, "constant_always_unintentional")]:
        pred_mask = create_constant_guessing_mask(gt_mask, constant_value=const_val)
        results = evaluate_predictions(gt_mask, pred_mask, label)
        save_results(results, label, OUTPUT_DIR)
        all_results.append(results)

    for prob in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        prob_f1s, prob_accuracies = [], []
        for i in range(num_runs):
            pred_mask = create_probability_guessing_mask(gt_mask, prob_intentional=prob, seed=42 + i)
            results = evaluate_predictions(gt_mask, pred_mask, f"Probability_{prob}_Run{i+1}")
            prob_f1s.append(results['overall']['macro_f1'])
            prob_accuracies.append(results['overall']['accuracy'])
            if i == 0:
                save_results(results, f"probability_{int(prob*100)}", OUTPUT_DIR)
                all_results.append(results)
        print(f"P={prob}: acc={np.mean(prob_accuracies):.4f}+-{np.std(prob_accuracies):.4f} "
              f"f1={np.mean(prob_f1s):.4f}+-{np.std(prob_f1s):.4f}")

    create_comparison_summary(all_results, OUTPUT_DIR)
    print(f"ALL BASELINE EVALUATIONS COMPLETE -> {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_baselines(num_runs=5)
