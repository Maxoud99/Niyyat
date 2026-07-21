#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaussian Mixture Model (GMM) Detector
--------------------------------------
Probabilistic anomaly detector using Gaussian Mixture Models.

Uses the negative log-likelihood as anomaly score - records that don't
fit well into any Gaussian component are considered anomalies.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional
from .base_detector import BaseDetector


class GMMDetector(BaseDetector):
    """
    Gaussian Mixture Model (GMM) anomaly detector.
    
    Models data as mixture of Gaussian distributions and flags records
    with low likelihood as anomalies.
    
    Parameters
    ----------
    n_components : int or str, default='auto'
        Number of Gaussian components
        If 'auto', uses sqrt(n_samples), capped between 5 and 50
    contamination : float, default=0.02
        Expected proportion of outliers (0.0 to 0.5)
        Used to determine threshold from score distribution
    covariance_type : str, default='full'
        Type of covariance: 'full', 'tied', 'diag', 'spherical'
        'full': each component has its own general covariance matrix
        'tied': all components share same covariance matrix
        'diag': diagonal covariance matrix (independent features)
        'spherical': single variance per component
    max_iter : int, default=100
        Maximum EM iterations
    n_init : int, default=10
        Number of initializations (best one kept)
    random_state : int, default=42
        Random seed for reproducibility
    decision_threshold_percentile : float or None, default=None
        Custom decision threshold as percentile of anomaly scores (0-100)
        If set, overrides contamination-based threshold
        Higher values = stricter threshold = fewer FPs
    """
    
    def __init__(
        self,
        n_components: str = 'auto',
        contamination: float = 0.02,
        covariance_type: str = 'full',
        max_iter: int = 100,
        n_init: int = 10,
        random_state: int = 42,
        decision_threshold_percentile: Optional[float] = None
    ):
        super().__init__(name='GaussianMixtureModel')
        
        self.n_components_config = n_components
        self.contamination = contamination
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.decision_threshold_percentile = decision_threshold_percentile
        
        self.model = None
        self.n_components_actual = None
        self.threshold_ = None
        self.decision_threshold_ = None
        
    def _compute_n_components(self, n_samples: int) -> int:
        """
        Compute actual number of components based on config.
        
        Parameters
        ----------
        n_samples : int
            Number of samples in data
            
        Returns
        -------
        n_components : int
            Actual number of components to use
        """
        if self.n_components_config == 'auto':
            # sqrt(n), capped between 5 and 50
            n = max(5, min(50, int(np.sqrt(n_samples))))
        else:
            n = int(self.n_components_config)
            
        # Ensure reasonable bounds
        n = max(2, min(n, n_samples // 10))
        
        return n
    
    def fit(self, X: np.ndarray) -> 'GMMDetector':
        """
        Fit GMM on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : GMMDetector
            Fitted detector
        """
        n_samples = X.shape[0]
        self.n_components_actual = self._compute_n_components(n_samples)
        
        # Initialize GMM model
        self.model = GaussianMixture(
            n_components=self.n_components_actual,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state
        )
        
        # Fit the model
        self.model.fit(X)
        
        # Compute contamination-based threshold
        scores = -self.model.score_samples(X)  # Negative log-likelihood
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        
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
        
        If decision_threshold_percentile is set, uses custom threshold.
        Otherwise, uses contamination-based threshold.
        
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
            predictions = (scores >= self.threshold_).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (negative log-likelihood).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
            Negative log-likelihood: low probability = high score
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Negative log-likelihood as anomaly score
        # Lower probability (log-likelihood) = higher anomaly score
        scores = -self.model.score_samples(X)
        
        return scores
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'n_components_config': self.n_components_config,
            'n_components_actual': self.n_components_actual,
            'contamination': self.contamination,
            'covariance_type': self.covariance_type,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'random_state': self.random_state,
            'decision_threshold_percentile': self.decision_threshold_percentile,
            'decision_threshold_value': self.decision_threshold_ if self.is_fitted else None,
            'contamination_threshold_value': self.threshold_ if self.is_fitted else None
        })
        return params
