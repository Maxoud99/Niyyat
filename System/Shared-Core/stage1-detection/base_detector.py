#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base Detector Class
-------------------
Abstract base class for all Stage 1 detectors.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Dict, Any


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    
    All detectors must implement:
    - fit(X): Train the detector on data
    - predict(X): Return binary predictions (0=normal, 1=anomaly)
    - decision_function(X): Return anomaly scores
    """
    
    def __init__(self, name: str):
        """
        Initialize base detector.
        
        Parameters
        ----------
        name : str
            Detector name for logging and identification
        """
        self.name = name
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseDetector':
        """
        Fit the detector on training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : BaseDetector
            Fitted detector instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (binary labels).
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        pass
    
    @abstractmethod
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
        """
        pass
    
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
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get detector parameters.
        
        Returns
        -------
        params : dict
            Detector parameters
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted
        }
