#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-k LOF Ensemble Detector
------------------------------
Ensemble of LOF detectors with different k values to capture
both local and global anomalies.

Theory: Different k values capture anomalies at different scales:
  - Small k (20-40): Local anomalies (single field errors)
  - Medium k (60-90): Moderate anomalies (few field errors)
  - Large k (100-150): Global anomalies (entire record unusual)

Aggregation via voting or score averaging.
"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF
from typing import Optional, List, Union
from .base_detector import BaseDetector


class MultiKLOFDetector(BaseDetector):
    """
    Multi-k LOF Ensemble Detector.
    
    Runs LOF with multiple k values and aggregates predictions via voting
    or score averaging. Captures both local and global anomalies.
    
    Parameters
    ----------
    contamination : float, default=0.02
        Expected proportion of outliers (0.0 to 0.5)
    k_values : list of int, default=[30, 60, 90, 120, 150]
        List of k values to use for LOF ensemble
        Small k: local anomalies, Large k: global anomalies
    voting_threshold : float, default=0.6
        Percentage of k values that must agree (0.0 to 1.0)
        0.6 = 60% (e.g., 3 out of 5 must agree)
    aggregation : str, default='vote'
        Aggregation method: 'vote' or 'average_scores'
    metric : str, default='manhattan'
        Distance metric: euclidean, manhattan, chebyshev, etc.
    algorithm : str, default='auto'
        Algorithm to compute neighbors: auto, ball_tree, kd_tree, brute
    leaf_size : int, default=30
        Leaf size for tree-based algorithms
    n_jobs : int, default=-1
        Number of parallel jobs (-1 = use all CPUs)
    """
    
    def __init__(
        self,
        contamination: float = 0.02,
        k_values: List[int] = [30, 60, 90, 120, 150],
        voting_threshold: float = 0.6,
        aggregation: str = 'vote',
        metric: str = 'manhattan',
        algorithm: str = 'auto',
        leaf_size: int = 30,
        n_jobs: int = -1
    ):
        super().__init__(name='MultiKLOF')
        
        self.contamination = contamination
        self.k_values = sorted(k_values)  # Sort for consistency
        self.voting_threshold = voting_threshold
        self.aggregation = aggregation
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs
        
        self.models = {}  # Dictionary of LOF models per k
        self.thresholds = {}  # Thresholds per k for score aggregation
        
    def fit(self, X: np.ndarray) -> 'MultiKLOFDetector':
        """
        Fit Multi-k LOF ensemble on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data (normalized)
            
        Returns
        -------
        self : MultiKLOFDetector
            Fitted detector
        """
        n_samples = X.shape[0]
        
        # Validate k values
        max_k = max(self.k_values)
        if max_k >= n_samples:
            raise ValueError(f"Largest k value ({max_k}) must be < n_samples ({n_samples})")
        
        # Train LOF for each k value
        for k in self.k_values:
            # Ensure k < n_samples
            k_actual = min(k, n_samples - 1)
            
            # Initialize LOF model
            model = SklearnLOF(
                n_neighbors=k_actual,
                contamination=self.contamination,
                metric=self.metric,
                algorithm=self.algorithm,
                leaf_size=self.leaf_size,
                n_jobs=self.n_jobs,
                novelty=False
            )
            
            # Fit the model
            model.fit(X)
            
            # Store model
            self.models[k] = model
            
            # Compute threshold from negative outlier factor
            # sklearn's negative_outlier_factor_: lower = more anomalous
            scores = -model.negative_outlier_factor_  # Flip so higher = more anomalous
            self.thresholds[k] = np.percentile(scores, 100 * (1 - self.contamination))
        
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using multi-k ensemble.
        
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
        
        if self.aggregation == 'vote':
            return self._predict_by_voting(X)
        elif self.aggregation == 'average_scores':
            return self._predict_by_averaging(X)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _predict_by_voting(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by majority voting across k values.
        
        Parameters
        ----------
        X : np.ndarray
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray
            Binary predictions based on voting
        """
        n_samples = X.shape[0]
        n_k_values = len(self.k_values)
        
        # Get predictions from each k value
        vote_matrix = np.zeros((n_samples, n_k_values), dtype=int)
        
        for idx, k in enumerate(self.k_values):
            # Get predictions for this k
            # sklearn returns -1 for anomalies, 1 for normal
            sklearn_preds = self.models[k].fit_predict(X)
            predictions_k = (sklearn_preds == -1).astype(int)
            vote_matrix[:, idx] = predictions_k
        
        # Count votes
        vote_counts = vote_matrix.sum(axis=1)
        
        # Apply voting threshold
        min_votes_required = int(np.ceil(self.voting_threshold * n_k_values))
        final_predictions = (vote_counts >= min_votes_required).astype(int)
        
        return final_predictions
    
    def _predict_by_averaging(self, X: np.ndarray) -> np.ndarray:
        """
        Predict by averaging LOF scores across k values.
        
        Parameters
        ----------
        X : np.ndarray
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray
            Binary predictions based on averaged scores
        """
        # Get averaged scores
        avg_scores = self.decision_function(X)
        
        # Use average threshold
        avg_threshold = np.mean(list(self.thresholds.values()))
        
        # Apply threshold
        predictions = (avg_scores > avg_threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores (averaged across k values).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns
        -------
        scores : np.ndarray, shape (n_samples,)
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before scoring")
        
        n_samples = X.shape[0]
        n_k_values = len(self.k_values)
        
        # Get scores from each k value
        score_matrix = np.zeros((n_samples, n_k_values))
        
        for idx, k in enumerate(self.k_values):
            # Get LOF scores for this k
            # sklearn's negative_outlier_factor_: lower = more anomalous
            # We flip it so higher = more anomalous
            self.models[k].fit(X)
            scores_k = -self.models[k].negative_outlier_factor_
            score_matrix[:, idx] = scores_k
        
        # Average scores across k values
        avg_scores = score_matrix.mean(axis=1)
        
        return avg_scores
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit detector and return predictions.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        self.fit(X)
        return self.predict(X)
    
    def get_params(self) -> dict:
        """Get detector parameters."""
        return {
            'name': self.name,
            'contamination': self.contamination,
            'k_values': self.k_values,
            'voting_threshold': self.voting_threshold,
            'aggregation': self.aggregation,
            'metric': self.metric,
            'algorithm': self.algorithm,
            'n_models': len(self.models)
        }
    
    def get_vote_details(self, X: np.ndarray) -> np.ndarray:
        """
        Get vote counts for each sample (for analysis).
        
        Parameters
        ----------
        X : np.ndarray
            Data to analyze
            
        Returns
        -------
        vote_counts : np.ndarray, shape (n_samples,)
            Number of k values that flagged each sample
        """
        if not self.is_fitted:
            raise RuntimeError("Detector must be fitted before getting vote details")
        
        n_samples = X.shape[0]
        n_k_values = len(self.k_values)
        
        # Get predictions from each k value
        vote_matrix = np.zeros((n_samples, n_k_values), dtype=int)
        
        for idx, k in enumerate(self.k_values):
            sklearn_preds = self.models[k].fit_predict(X)
            predictions_k = (sklearn_preds == -1).astype(int)
            vote_matrix[:, idx] = predictions_k
        
        # Count votes
        vote_counts = vote_matrix.sum(axis=1)
        
        return vote_counts
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"MultiKLOFDetector(contamination={self.contamination}, "
                f"k_values={self.k_values}, voting_threshold={self.voting_threshold}, "
                f"aggregation='{self.aggregation}', metric='{self.metric}')")
