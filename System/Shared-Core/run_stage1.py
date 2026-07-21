#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: Record-Level Detection - Main Script
----------------------------------------------
Train and evaluate Stage 1 ensemble on Twitter bot dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from detectors.stage1.ensemble import Stage1Ensemble
from evaluation.metrics import evaluate_stage1
from preprocessing.intelligent_preprocessor import IntelligentPreprocessor
import yaml


def run_multi_seed_voting(X, y_true, config_path, num_runs=5, base_seed=42, voting_threshold=0.6, verbose=True):
    """
    Run Stage 1 detection multiple times with different seeds and aggregate via majority voting.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature data
    y_true : np.ndarray
        Ground truth labels
    config_path : Path
        Path to configuration file
    num_runs : int, default=5
        Number of runs with different seeds
    base_seed : int, default=42
        Base seed (will increment for each run)
    voting_threshold : float, default=0.6
        Percentage of runs that must agree (0.6 = 60%)
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    y_pred_final : np.ndarray
        Final predictions after multi-seed voting
    all_predictions : list
        List of prediction arrays from each run
    all_details : list
        List of detail DataFrames from each run
    """
    if verbose:
        print("\n" + "="*80)
        print("MULTI-SEED STABILITY VOTING")
        print("="*80)
        print(f"\nRunning {num_runs} times with different seeds...")
        print(f"Seeds: {[base_seed + i for i in range(num_runs)]}")
        print(f"Voting threshold: {voting_threshold*100:.0f}% (≥{int(np.ceil(voting_threshold * num_runs))}/{num_runs} runs must agree)")
    
    all_predictions = []
    all_details = []
    
    for run_idx in range(num_runs):
        seed = base_seed + run_idx
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"RUN {run_idx + 1}/{num_runs} (seed={seed})")
            print(f"{'─'*80}")
        
        # Create ensemble with this seed
        ensemble = Stage1Ensemble(
            config_path=config_path,
            contamination=0.24,  # Read from config
            voting_threshold=2,  # Read from config
            verbose=False  # Suppress individual run details
        )
        
        # Update random states in detectors
        if 'isolation_forest' in ensemble.detectors:
            ensemble.detectors['isolation_forest'].random_state = seed
            ensemble.detectors['isolation_forest'].model.random_state = seed
        
        # Train and predict
        y_pred, details = ensemble.fit_predict(X, return_details=True)
        
        all_predictions.append(y_pred)
        all_details.append(details)
        
        if verbose:
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  Flagged: {y_pred.sum():4d} | Precision: {precision:.3f} | Recall: {recall:.3f}")
    
    # Stack all predictions (n_samples, n_runs)
    prediction_matrix = np.stack(all_predictions, axis=1)
    
    # Count how many runs flagged each record
    vote_counts = prediction_matrix.sum(axis=1)
    
    # Apply voting threshold
    min_votes_required = int(np.ceil(voting_threshold * num_runs))
    y_pred_final = (vote_counts >= min_votes_required).astype(int)
    
    if verbose:
        print(f"\n{'='*80}")
        print("MULTI-SEED AGGREGATION")
        print(f"{'='*80}")
        print(f"\nVote distribution:")
        for votes in range(num_runs + 1):
            count = (vote_counts == votes).sum()
            flagged = "✓ FLAGGED" if votes >= min_votes_required else ""
            print(f"  {votes}/{num_runs} votes: {count:4d} records {flagged}")
        print(f"\nFinal flagged: {y_pred_final.sum()} records ({100*y_pred_final.sum()/len(y_pred_final):.2f}%)")
    
    return y_pred_final, all_predictions, all_details


def main(dataset_path=None, multi_seed_override=None):
    """Run Stage 1 detection pipeline.
    
    Parameters
    ----------
    dataset_path : str or Path, optional
        Path to labeled dataset CSV. If None, uses default twitter_bot dataset.
    multi_seed_override : bool, optional
        Override config file multi_seed.enabled setting
    """
    print("="*80)
    print("STAGE 1: RECORD-LEVEL DETECTION")
    print("="*80)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent  # error_detection_system/
    
    # Use provided dataset path or default
    if dataset_path is not None:
        DATASET_PATH = Path(dataset_path)
    else:
        DATASET_PATH = BASE_DIR / 'datasets' / 'twitter_bot' / 'labeled_dataset.csv'
    
    CONFIG_PATH = BASE_DIR / 'config' / 'stage1_config.yaml'
    
    # Create timestamped output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = BASE_DIR / 'outputs' / 'stage1' / f'run_{timestamp}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Data: {DATASET_PATH}")
    print(f"⚙️  Config: {CONFIG_PATH}")
    print(f"📊 Output: {OUTPUT_DIR}")
    
    # Verify dataset exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(DATASET_PATH)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Verify required column exists
    if 'is_erroneous' not in df.columns:
        raise ValueError(f"Dataset must contain 'is_erroneous' column. Found columns: {df.columns.tolist()}")
    
    # Intelligent preprocessing (auto-detects and handles categorical features)
    preprocessor = IntelligentPreprocessor(
        target_col='is_erroneous',
        pca_threshold=50,
        pca_variance=0.95,
        verbose=True
    )
    
    X, y_true, preprocessing_info = preprocessor.fit_transform(df)
    
    # Save preprocessing info to output directory
    preprocessing_info_path = OUTPUT_DIR / 'preprocessing_info.txt'
    preprocessor.save_info(preprocessing_info_path)
    
    print(f"\n✓ Preprocessing info saved: {preprocessing_info_path}")
    
    print(f"\nLabel distribution:")
    print(f"  Clean (0): {(y_true == 0).sum()} ({100*(y_true == 0).sum()/len(y_true):.2f}%)")
    print(f"  Erroneous (1): {(y_true == 1).sum()} ({100*(y_true == 1).sum()/len(y_true):.2f}%)")
    
    # Load config to check if multi-seed is enabled
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    multi_seed_config = config['stage1'].get('multi_seed', {})
    multi_seed_enabled = multi_seed_config.get('enabled', False)
    
    # Allow command-line override
    if multi_seed_override is not None:
        multi_seed_enabled = multi_seed_override
        print(f"\n⚠️  Multi-seed voting: {'ENABLED' if multi_seed_enabled else 'DISABLED'} (command-line override)")
    
    if multi_seed_enabled:
        # Multi-seed voting mode
        num_runs = multi_seed_config.get('num_runs', 5)
        base_seed = multi_seed_config.get('base_seed', 42)
        voting_threshold = multi_seed_config.get('voting_threshold', 0.6)
        
        y_pred, all_predictions, all_details = run_multi_seed_voting(
            X=X,
            y_true=y_true,
            config_path=CONFIG_PATH,
            num_runs=num_runs,
            base_seed=base_seed,
            voting_threshold=voting_threshold,
            verbose=True
        )
        
        # Use the first run's details for now (could aggregate these too)
        details = all_details[0]
        details['final_prediction'] = y_pred
        
        # Store config info for report (ensemble object not available in multi-seed mode)
        num_detectors = 4  # We know we have 4 detectors
        ensemble_voting_threshold = int(np.ceil(config['stage1'].get('voting', {}).get('threshold_percentage', 0.5) * num_detectors))
        contamination = config['stage1'].get('isolation_forest', {}).get('contamination', 0.24)
        
    else:
        # Single run mode (original behavior)
        # Initialize ensemble
        print("\n" + "="*80)
        print("INITIALIZING ENSEMBLE")
        print("="*80)
        
        ensemble = Stage1Ensemble(
            config_path=CONFIG_PATH,
            contamination=0.02,
            voting_threshold=3,
            verbose=True
        )
        
        # Train and predict
        print("\n" + "="*80)
        print("TRAINING ENSEMBLE")
        print("="*80)
        
        y_pred, details = ensemble.fit_predict(X, return_details=True)
        
        # Store config info for report
        num_detectors = len(ensemble.detectors)
        ensemble_voting_threshold = ensemble.voting_threshold
        contamination = ensemble.contamination
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    metrics = evaluate_stage1(y_true, y_pred, details)
    
    print("\n📊 Overall Metrics:")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  False Positive Rate: {metrics['fpr']:.3f}")
    print(f"  True Negative Rate: {metrics['tnr']:.3f}")
    
    print("\n🔢 Confusion Matrix:")
    print(f"  True Negatives:  {metrics['tn']:4d}")
    print(f"  False Positives: {metrics['fp']:4d}")
    print(f"  False Negatives: {metrics['fn']:4d}")
    print(f"  True Positives:  {metrics['tp']:4d}")
    
    print("\n🎯 Per-Detector Performance:")
    for detector_name, detector_metrics in metrics['per_detector'].items():
        print(f"\n  {detector_name}:")
        print(f"    Precision: {detector_metrics['precision']:.3f}")
        print(f"    Recall: {detector_metrics['recall']:.3f}")
        print(f"    F1-Score: {detector_metrics['f1']:.3f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # 1. Predictions with details
    results_path = OUTPUT_DIR / 'detections.csv'
    details['ground_truth'] = y_true
    details.to_csv(results_path, index=False)
    print(f"\n✓ Detections saved: {results_path}")
    
    # 2. Metrics
    metrics_path = OUTPUT_DIR / 'metrics.json'
    # Convert numpy types to Python types for JSON
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            metrics_json[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                for k, v in value.items() if not isinstance(v, dict)}
        elif isinstance(value, (np.integer, np.floating)):
            metrics_json[key] = float(value)
        else:
            metrics_json[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # 3. Summary report
    report_path = OUTPUT_DIR / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 1: RECORD-LEVEL DETECTION - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET\n")
        f.write("-"*80 + "\n")
        f.write(f"Total records: {len(y_true)}\n")
        f.write(f"Clean records: {(y_true == 0).sum()} ({100*(y_true == 0).sum()/len(y_true):.2f}%)\n")
        f.write(f"Erroneous records: {(y_true == 1).sum()} ({100*(y_true == 1).sum()/len(y_true):.2f}%)\n\n")
        
        f.write("PREPROCESSING\n")
        f.write("-"*80 + "\n")
        f.write(f"Original features: {preprocessing_info['original_shape'][1] - 1}\n")  # -1 for target
        f.write(f"Final features: {preprocessing_info['final_shape'][1]}\n")
        f.write(f"Categorical columns: {len(preprocessing_info['categorical_cols'])}\n")
        f.write(f"One-hot encoding: {'Applied' if preprocessing_info['applied_onehot'] else 'Not needed'}\n")
        f.write(f"PCA reduction: {'Applied' if preprocessing_info['applied_pca'] else 'Not needed'}\n")
        if preprocessing_info['applied_pca']:
            f.write(f"  Components: {preprocessing_info['n_components']}\n")
            f.write(f"  Variance explained: {preprocessing_info['variance_explained']:.3f}\n")
        f.write("\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Enabled detectors: {num_detectors}\n")
        f.write(f"Voting threshold: ≥{ensemble_voting_threshold}/{num_detectors}\n")
        f.write(f"Contamination: {contamination}\n")
        if multi_seed_enabled:
            f.write(f"Multi-seed: {num_runs} runs, {voting_threshold*100:.0f}% threshold\n")
        f.write("\n")
        
        f.write("RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Flagged records: {y_pred.sum()} ({100*y_pred.sum()/len(y_pred):.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.3f}\n")
        f.write(f"Recall: {metrics['recall']:.3f}\n")
        f.write(f"F1-Score: {metrics['f1']:.3f}\n")
        f.write(f"False Positive Rate: {metrics['fpr']:.3f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(f"True Negatives:  {metrics['tn']:4d}\n")
        f.write(f"False Positives: {metrics['fp']:4d}\n")
        f.write(f"False Negatives: {metrics['fn']:4d}\n")
        f.write(f"True Positives:  {metrics['tp']:4d}\n\n")
        
        f.write("PER-DETECTOR PERFORMANCE\n")
        f.write("-"*80 + "\n")
        for detector_name, detector_metrics in metrics['per_detector'].items():
            f.write(f"\n{detector_name}:\n")
            f.write(f"  Precision: {detector_metrics['precision']:.3f}\n")
            f.write(f"  Recall: {detector_metrics['recall']:.3f}\n")
            f.write(f"  F1-Score: {detector_metrics['f1']:.3f}\n")
    
    print(f"✓ Summary report saved: {report_path}")
    
    print("\n" + "="*80)
    print("✅ STAGE 1 COMPLETE!")
    print("="*80)
    print(f"\n📂 Results saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Review metrics and confusion matrix")
    print("  2. Analyze per-detector contributions")
    print("  3. Tune parameters if needed (config/stage1_config.yaml)")
    print("  4. Proceed to Stage 2 (cell-level attribution)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Stage 1 Record-Level Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default twitter_bot dataset
  python run_stage1.py
  
  # Run with adult income dataset
  python run_stage1.py --dataset ../datasets/adult_income/labeled_dataset.csv
  
  # Run with custom dataset and enable multi-seed voting
  python run_stage1.py --dataset /path/to/dataset.csv --multi-seed
        """
    )
    
    parser.add_argument('--dataset', '-d', type=str, default=None,
                       help='Path to labeled dataset CSV (must contain "is_erroneous" column). '
                            'Default: datasets/twitter_bot/labeled_dataset.csv')
    parser.add_argument('--multi-seed', action='store_true', 
                       help='Enable multi-seed voting (overrides config)')
    parser.add_argument('--no-multi-seed', action='store_true',
                       help='Disable multi-seed voting (overrides config)')
    
    args = parser.parse_args()
    
    # Determine multi-seed override
    multi_seed_override = None
    if args.multi_seed:
        multi_seed_override = True
    elif args.no_multi_seed:
        multi_seed_override = False
    
    main(dataset_path=args.dataset, multi_seed_override=multi_seed_override)
