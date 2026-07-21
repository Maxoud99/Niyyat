#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Outlier Factor Detector
------------------------------
Density-based local anomaly detector.

Detects local outliers by comparing local density with neighbors' densities.
Good for contextual anomalies that global methods might miss.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF
from typing import Optional
from .base_detector import BaseDetector


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor (LOF) anomaly detector.
    
    Detects outliers based on local density deviation.
    Good for contextual/local anomalies.
    
    Parameters
    ----------
    contamination : float, default=0.02
        Expected proportion of outliers (0.0 to 0.5)
        Used for sklearn's internal threshold (if decision_threshold_percentile is None)
    n_neighbors : int or str, default='sqrt_n'
        Number of neighbors to use
        If 'sqrt_n', will use sqrt(n_samples), capped at 200
    metric : str, default='manhattan'
        Distance metric: euclidean, manhattan, chebyshev, etc.
    algorithm : str, default='auto'
        Algorithm to compute neighbors: auto, ball_tree, kd_tree, brute
    leaf_size : int, default=30
        Leaf size for tree-based algorithms
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all CPUs)
    decision_threshold_percentile : float or None, default=None
        Custom decision threshold as percentile of LOF scores (0-100)
        If set, overrides contamination-based threshold for predictions
        Higher values = stricter threshold = fewer FPs, more FNs
        Example: 95 means flag top 5% highest scores
        If None, uses sklearn's contamination-based threshold
    """
    
    def __init__(
        self,
        contamination: float = 0.02,
        n_neighbors: str = 'sqrt_n',
        metric: str = 'manhattan',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        n_jobs: int = -1,
        decision_threshold_percentile: Optional[float] = None
    ):
        super().__init__(name='LocalOutlierFactor')
        
        self.contamination = contamination
        self.n_neighbors_config = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        self.decision_threshold_percentile = decision_threshold_percentile
        
        self.model = None
        self.n_neighbors_actual = None
        self.decision_threshold_ = None  # Computed threshold for custom percentile
        
    def _compute_n_neighbors(self, n_samples: int) -> int:
        """
        Compute actual number of neighbors based on config.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in data
            
        Returns
        -------
        n_neighbors : int
            Actual number of neighbors to use
        """
        if self.n_neighbors_config == 'sqrt_n':
            # sqrt(n), capped at 200, min 5
            n = max(5, min(200, int(np.sqrt(n_samples))))
        else:
            n = int(self.n_neighbors_config)
            
        # Ensure n_neighbors < n_samples
        n = min(n, n_samples - 1)
        
        return n
    
    def fit(self, X: np.ndarray) -> 'LOFDetector':
        """
        Fit LOF on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : LOFDetector
            Fitted detector
        """
        n_samples = X.shape[0]
        self.n_neighbors_actual = self._compute_n_neighbors(n_samples)
        
        # Initialize LOF model
        self.model = SklearnLOF(
            n_neighbors=self.n_neighbors_actual,
            contamination=self.contamination,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
            novelty=False  # Fit and predict on same data
        )
        
        # Fit the model
        self.model.fit(X)
        
        # Mark as fitted BEFORE computing threshold (decision_function needs this)
        self.is_fitted = True
        
        # Compute custom threshold if percentile is specified
        if self.decision_threshold_percentile is not None:
            scores = self.decision_function(X)
            self.decision_threshold_ = np.percentile(scores, self.decision_threshold_percentile)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Note: LOF with novelty=False can only predict on training data.
        For new data, set novelty=True and use predict().
        
        If decision_threshold_percentile is set, uses custom threshold.
        Otherwise, uses sklearn's contamination-based threshold.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict (must be same as training data)
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        # Use custom threshold if specified
        if self.decision_threshold_percentile is not None:
            scores = self.decision_function(X)
            predictions = (scores >= self.decision_threshold_).astype(int)
        else:
            # sklearn returns -1 for anomalies, 1 for normal
            # Convert to 0=normal, 1=anomaly
            sklearn_preds = self.model.fit_predict(X)
            predictions = (sklearn_preds == -1).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (LOF values).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
            LOF > 1 indicates outlier (lower local density than neighbors)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Negative outlier factor (sklearn convention)
        # More negative = more anomalous
        # Negate to make higher = more anomalous
        scores = -self.model.negative_outlier_factor_
        
        return scores
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'contamination': self.contamination,
            'n_neighbors_config': self.n_neighbors_config,
            'n_neighbors_actual': self.n_neighbors_actual,
            'metric': self.metric,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'n_jobs': self.n_jobs,
            'decision_threshold_percentile': self.decision_threshold_percentile,
            'decision_threshold_value': self.decision_threshold_ if self.is_fitted else None
        })
        return params
