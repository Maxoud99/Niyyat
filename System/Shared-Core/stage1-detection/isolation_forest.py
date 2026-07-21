#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolation Forest Detector
--------------------------
Tree-based ensemble anomaly detector for global outlier detection.

Uses sklearn's IsolationForest implementation.
Anomalies are isolated with fewer tree splits (shorter path lengths).
"""

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF
from typing import Optional
from .base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector.
    
    Fast, scalable, good for global outliers.
    Works by isolating anomalies using random binary trees.
    
    Parameters
    ----------
    contamination : float, default=0.02
        Expected proportion of outliers (0.0 to 0.5)
    n_estimators : int, default=1000
        Number of trees in the forest
    max_samples : str or int, default='auto'
        Number of samples per tree ('auto' = 256)
    max_features : float, default=1.0
        Proportion of features per split (1.0 = all features)
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all CPUs)
    """
    
    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 1000,
        max_samples: str = 'auto',
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        super().__init__(name='IsolationForest')
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Initialize sklearn model
        self.model = SklearnIF(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            bootstrap=False,
            verbose=0
        )
        
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """
        Fit Isolation Forest on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : IsolationForestDetector
            Fitted detector
        """
        self.model.fit(X)
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
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
        
        # sklearn returns -1 for anomalies, 1 for normal
        # Convert to 0=normal, 1=anomaly
        sklearn_preds = self.model.predict(X)
        predictions = (sklearn_preds == -1).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
            sklearn returns negative scores for anomalies, so we negate
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # sklearn: negative scores = anomalies, positive = normal
        # Negate to make higher scores = more anomalous
        scores = -self.model.decision_function(X)
        
        return scores
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        })
        return params
