#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Ensemble Detector
------------------------------
Ensemble of three statistical outlier detection methods:
1. Mahalanobis Distance
2. IQR-Based Outliers
3. Z-Score Method

Flags a record if ≥2 of 3 methods agree (2/3 voting).
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import mahalanobis
from typing import Optional, Tuple
from .base_detector import BaseDetector


class StatisticalEnsembleDetector(BaseDetector):
    """
    Statistical ensemble anomaly detector.
    
    Combines three statistical methods with voting:
    - Mahalanobis distance (considers correlations)
    - IQR-based outliers (robust to skewness)
    - Z-score method (standard deviation-based)
    
    Parameters
    ----------
    mahalanobis_percentile : float, default=99
        Threshold percentile for Mahalanobis distance
    iqr_multiplier : float, default=1.5
        IQR multiplier (1.5 = standard boxplot rule)
    min_outlier_features : int, default=3
        Minimum features that must be IQR outliers
    zscore_threshold : float, default=3.0
        Z-score threshold (|z| > threshold = outlier)
    voting_rule : str, default='2_of_3'
        Voting rule: '2_of_3' (≥2 methods must agree)
    """
    
    def __init__(
        self,
        mahalanobis_percentile: float = 99,
        iqr_multiplier: float = 1.5,
        min_outlier_features: int = 3,
        zscore_threshold: float = 3.0,
        voting_rule: str = '1_of_3'
    ):
        super().__init__(name='StatisticalEnsemble')
        
        self.mahalanobis_percentile = mahalanobis_percentile
        self.iqr_multiplier = iqr_multiplier
        self.min_outlier_features = min_outlier_features
        self.zscore_threshold = zscore_threshold
        self.voting_rule = voting_rule
        
        # Fitted parameters
        self.mean_ = None
        self.cov_ = None
        self.cov_inv_ = None
        self.feature_q1_ = None
        self.feature_q3_ = None
        self.feature_iqr_ = None
        self.feature_mean_ = None
        self.feature_std_ = None
        self.mahalanobis_threshold_ = None
        
    def fit(self, X: np.ndarray) -> 'StatisticalEnsembleDetector':
        """
        Fit statistical models on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : StatisticalEnsembleDetector
            Fitted detector
        """
        # 1. Mahalanobis: compute mean and covariance
        self.mean_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        
        # Add regularization to avoid singular matrix
        reg = 1e-6
        self.cov_ += np.eye(self.cov_.shape[0]) * reg
        self.cov_inv_ = np.linalg.inv(self.cov_)
        
        # Compute Mahalanobis distances for training data to set threshold
        mahal_distances = np.array([
            mahalanobis(x, self.mean_, self.cov_inv_) for x in X
        ])
        self.mahalanobis_threshold_ = np.percentile(
            mahal_distances, self.mahalanobis_percentile
        )
        
        # 2. IQR: compute quartiles for each feature
        self.feature_q1_ = np.percentile(X, 25, axis=0)
        self.feature_q3_ = np.percentile(X, 75, axis=0)
        self.feature_iqr_ = self.feature_q3_ - self.feature_q1_
        
        # 3. Z-score: compute mean and std for each feature
        self.feature_mean_ = np.mean(X, axis=0)
        self.feature_std_ = np.std(X, axis=0) + 1e-8  # Avoid division by zero
        
        self.is_fitted = True
        return self
    
    def _detect_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Mahalanobis distance.
        
        Returns binary flags (1=outlier, 0=normal).
        """
        distances = np.array([
            mahalanobis(x, self.mean_, self.cov_inv_) for x in X
        ])
        return (distances > self.mahalanobis_threshold_).astype(int)
    
    def _detect_iqr(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers using IQR method (per feature).
        
        Record is flagged if ≥ min_outlier_features are outliers.
        Returns binary flags (1=outlier, 0=normal).
        """
        # Compute bounds for each feature
        lower_bounds = self.feature_q1_ - self.iqr_multiplier * self.feature_iqr_
        upper_bounds = self.feature_q3_ + self.iqr_multiplier * self.feature_iqr_
        
        # Check which features are outliers (outside bounds)
        outlier_mask = (X < lower_bounds) | (X > upper_bounds)
        
        # Count outlier features per record
        outlier_counts = outlier_mask.sum(axis=1)
        
        # Flag if ≥ min_outlier_features are outliers
        return (outlier_counts >= self.min_outlier_features).astype(int)
    
    def _detect_zscore(self, X: np.ndarray) -> np.ndarray:
        """
        Detect outliers using Z-score method.
        
        Record is flagged if ANY feature has |z| > threshold.
        Returns binary flags (1=outlier, 0=normal).
        """
        # Compute Z-scores for all features
        z_scores = (X - self.feature_mean_) / self.feature_std_
        
        # Check if any feature exceeds threshold
        extreme_mask = np.abs(z_scores) > self.zscore_threshold
        
        # Flag if ANY feature is extreme
        return (extreme_mask.any(axis=1)).astype(int)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using ensemble voting.
        
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
        
        # Get predictions from each method
        mahal_flags = self._detect_mahalanobis(X)
        iqr_flags = self._detect_iqr(X)
        zscore_flags = self._detect_zscore(X)
        
        # Voting: flag if ≥2 of 3 methods agree
        votes = mahal_flags + iqr_flags + zscore_flags
        predictions = (votes >= 2).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (vote counts / 3).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (0 to 1, higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Get predictions from each method
        mahal_flags = self._detect_mahalanobis(X)
        iqr_flags = self._detect_iqr(X)
        zscore_flags = self._detect_zscore(X)
        
        # Score = proportion of methods that flag
        votes = mahal_flags + iqr_flags + zscore_flags
        scores = votes / 3.0
        
        return scores
    
    def get_method_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get individual method predictions for analysis.
        
        Returns
        -------
        mahal_flags, iqr_flags, zscore_flags : tuple of np.ndarray
            Binary flags from each method
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before prediction")
        
        return (
            self._detect_mahalanobis(X),
            self._detect_iqr(X),
            self._detect_zscore(X)
        )
    
    def get_params(self):
        """Get detector parameters."""
        params = super().get_params()
        params.update({
            'mahalanobis_percentile': self.mahalanobis_percentile,
            'iqr_multiplier': self.iqr_multiplier,
            'min_outlier_features': self.min_outlier_features,
            'zscore_threshold': self.zscore_threshold,
            'voting_rule': self.voting_rule
        })
        return params
