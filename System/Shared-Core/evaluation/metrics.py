#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Stage 1
-------------------------------
Compute precision, recall, F1, confusion matrix, etc.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from typing import Dict, Optional


def evaluate_stage1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    details: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Evaluate Stage 1 detection performance.
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground truth labels (0=clean, 1=erroneous)
    y_pred : np.ndarray, shape (n_samples,)
        Predicted labels (0=clean, 1=erroneous)
    details : pd.DataFrame, optional
        Detailed predictions with per-detector flags
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - precision, recall, f1, accuracy
        - tn, fp, fn, tp (confusion matrix)
        - fpr, tnr (false positive rate, true negative rate)
        - per_detector: metrics per detector (if details provided)
    """
    # Overall metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate (specificity)
    tpr = recall  # True positive rate = recall
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'fpr': fpr,
        'tnr': tnr,
        'tpr': tpr
    }
    
    # Per-detector metrics (if details provided)
    if details is not None:
        per_detector = {}
        flag_cols = [col for col in details.columns if col.endswith('_flag')]
        
        for col in flag_cols:
            detector_name = col.replace('_flag', '')
            detector_preds = details[col].values
            
            det_precision = precision_score(y_true, detector_preds, zero_division=0)
            det_recall = recall_score(y_true, detector_preds, zero_division=0)
            det_f1 = f1_score(y_true, detector_preds, zero_division=0)
            det_accuracy = accuracy_score(y_true, detector_preds)
            
            # Confusion matrix
            det_tn, det_fp, det_fn, det_tp = confusion_matrix(y_true, detector_preds).ravel()
            det_fpr = det_fp / (det_fp + det_tn) if (det_fp + det_tn) > 0 else 0
            
            per_detector[detector_name] = {
                'precision': det_precision,
                'recall': det_recall,
                'f1': det_f1,
                'accuracy': det_accuracy,
                'tn': int(det_tn),
                'fp': int(det_fp),
                'fn': int(det_fn),
                'tp': int(det_tp),
                'fpr': det_fpr
            }
        
        metrics['per_detector'] = per_detector
    
    return metrics


def print_evaluation_report(metrics: Dict, title: str = "EVALUATION REPORT"):
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary from evaluate_stage1()
    title : str
        Report title
    """
    print("="*80)
    print(title)
    print("="*80)
    
    print("\n📊 Overall Metrics:")
    print(f"  Precision:         {metrics['precision']:.3f}")
    print(f"  Recall:            {metrics['recall']:.3f}")
    print(f"  F1-Score:          {metrics['f1']:.3f}")
    print(f"  Accuracy:          {metrics['accuracy']:.3f}")
    print(f"  False Positive Rate: {metrics['fpr']:.3f}")
    print(f"  True Negative Rate:  {metrics['tnr']:.3f}")
    
    print("\n🔢 Confusion Matrix:")
    print(f"  True Negatives:  {metrics['tn']:4d}")
    print(f"  False Positives: {metrics['fp']:4d}")
    print(f"  False Negatives: {metrics['fn']:4d}")
    print(f"  True Positives:  {metrics['tp']:4d}")
    
    if 'per_detector' in metrics:
        print("\n🎯 Per-Detector Performance:")
        for detector_name, det_metrics in metrics['per_detector'].items():
            print(f"\n  {detector_name}:")
            print(f"    Precision: {det_metrics['precision']:.3f}")
            print(f"    Recall:    {det_metrics['recall']:.3f}")
            print(f"    F1-Score:  {det_metrics['f1']:.3f}")
            print(f"    FPR:       {det_metrics['fpr']:.3f}")
    
    print("\n" + "="*80)
