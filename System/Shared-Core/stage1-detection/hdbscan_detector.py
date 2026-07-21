#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDBSCAN Detector
----------------
Hierarchical Density-Based Spatial Clustering with Noise.

HDBSCAN is a clustering algorithm that identifies clusters of varying
densities and labels points that don't belong to any cluster as noise/outliers.
"""

import numpy as np
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    
from typing import Optional
from .base_detector import BaseDetector


class HDBSCANDetector(BaseDetector):
    """
    HDBSCAN anomaly detector.
    
    Uses HDBSCAN clustering to identify outliers. Points labeled as noise
    (cluster=-1) or with low membership probabilities are flagged as anomalies.
    
    Note: Requires hdbscan package: pip install hdbscan
    
    Parameters
    ----------
    min_cluster_size : int, default=30
        Minimum size of clusters
    min_samples : int or None, default=None
        Number of samples in neighborhood for core point
        If None, defaults to min_cluster_size
    contamination : float, default=0.02
        Expected proportion of outliers (used for threshold tuning)
    metric : str, default='euclidean'
        Distance metric: euclidean, manhattan, cosine, etc.
    cluster_selection_method : str, default='eom'
        How to select clusters: 'eom' (excess of mass) or 'leaf'
    alpha : float, default=1.0
        Distance scaling parameter (larger = more conservative)
    decision_threshold_percentile : float or None, default=None
        Custom decision threshold as percentile of outlier scores (0-100)
        If set, uses outlier scores instead of just cluster=-1
        Higher values = stricter threshold = fewer FPs
    """
    
    def __init__(
        self,
        min_cluster_size: int = 30,
        min_samples: Optional[int] = None,
        contamination: float = 0.02,
        metric: str = 'euclidean',
        cluster_selection_method: str = 'eom',
        alpha: float = 1.0,
        decision_threshold_percentile: Optional[float] = None
    ):
        super().__init__(name='HDBSCAN')
        
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "hdbscan package is required for HDBSCANDetector. "
                "Install it with: pip install hdbscan"
            )
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.contamination = contamination
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha
        self.decision_threshold_percentile = decision_threshold_percentile
        
        self.model = None
        self.labels_ = None
        self.outlier_scores_ = None
        self.threshold_ = None
        self.decision_threshold_ = None
        
    def fit(self, X: np.ndarray) -> 'HDBSCANDetector':
        """
        Fit HDBSCAN on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : HDBSCANDetector
            Fitted detector
        """
        # Initialize HDBSCAN model
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            alpha=self.alpha,
            prediction_data=True  # Enable approximate prediction
        )
        
        # Fit and get labels
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.outlier_scores_ = self.model.outlier_scores_
        
        # Compute contamination-based threshold
        # Use outlier scores - higher score = more likely to be outlier
        self.threshold_ = np.percentile(self.outlier_scores_, 
                                       100 * (1 - self.contamination))
        
        # Mark as fitted BEFORE computing custom threshold
        self.is_fitted = True
        
        # Compute custom threshold if percentile is specified
        if self.decision_threshold_percentile is not None:
            scores = self.decision_function(X)
            self.decision_threshold_ = np.percentile(scores, self.decision_threshold_percentile)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        If decision_threshold_percentile is set, uses outlier scores with custom threshold.
        Otherwise, flags points with cluster=-1 (noise) or high outlier scores.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        scores = self.decision_function(X)
        
        # Use custom threshold if specified
        if self.decision_threshold_percentile is not None:
            predictions = (scores >= self.decision_threshold_).astype(int)
        else:
            # Default: use contamination-based threshold on outlier scores
            predictions = (scores >= self.threshold_).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (outlier scores from HDBSCAN).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
            GLOSH outlier scores: measure how much of an outlier each point is
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Use GLOSH outlier scores
        # Higher score = more likely to be outlier
        scores = self.outlier_scores_
        
        return scores
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'contamination': self.contamination,
            'metric': self.metric,
            'cluster_selection_method': self.cluster_selection_method,
            'alpha': self.alpha,
            'decision_threshold_percentile': self.decision_threshold_percentile,
            'decision_threshold_value': self.decision_threshold_ if self.is_fitted else None,
            'contamination_threshold_value': self.threshold_ if self.is_fitted else None,
            'n_clusters': len(np.unique(self.labels_[self.labels_ != -1])) if self.is_fitted else None,
            'n_noise_points': np.sum(self.labels_ == -1) if self.is_fitted else None
        })
        return params
