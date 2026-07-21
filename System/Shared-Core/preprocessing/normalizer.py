#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Normalizer
---------------
Feature normalization utilities for Stage 1 preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Union, Tuple, Optional


class DataNormalizer:
    """
    Data normalization for anomaly detection.
    
    Supports multiple scaling methods:
    - minmax: Scale to [0, 1] range
    - standard: Standardize to mean=0, std=1
    - robust: Scale using median and IQR (robust to outliers)
    
    Parameters
    ----------
    method : str, default='minmax'
        Scaling method: 'minmax', 'standard', 'robust'
    """
    
    def __init__(self, method: str = 'minmax'):
        """
        Initialize normalizer.
        
        Parameters
        ----------
        method : str
            Scaling method
        """
        self.method = method
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize scaler based on method
        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}. "
                           f"Choose from: minmax, standard, robust")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'DataNormalizer':
        """
        Fit normalizer on data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : DataNormalizer
            Fitted normalizer
        """
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
        
        # Fit scaler
        self.scaler.fit(X_array)
        self.is_fitted = True
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform data using fitted normalizer.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_scaled : np.ndarray, shape (n_samples, n_features)
            Normalized data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before transform")
        
        # Convert to array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Transform
        X_scaled = self.scaler.transform(X_array)
        
        return X_scaled
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit normalizer and transform data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        X_scaled : np.ndarray, shape (n_samples, n_features)
            Normalized data
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse normalization.
        
        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_samples, n_features)
            Normalized data
            
        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Original scale data
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer must be fitted before inverse transform")
        
        return self.scaler.inverse_transform(X_scaled)


def normalize_data(
    X: Union[pd.DataFrame, np.ndarray],
    method: str = 'minmax',
    return_normalizer: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, DataNormalizer]]:
    """
    Convenience function to normalize data.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Data to normalize
    method : str, default='minmax'
        Scaling method
    return_normalizer : bool, default=True
        If True, return (normalized_data, normalizer)
        If False, return only normalized_data
        
    Returns
    -------
    X_scaled : np.ndarray
        Normalized data
    normalizer : DataNormalizer (if return_normalizer=True)
        Fitted normalizer
    """
    normalizer = DataNormalizer(method=method)
    X_scaled = normalizer.fit_transform(X)
    
    if return_normalizer:
        return X_scaled, normalizer
    else:
        return X_scaled
