#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 Ensemble Pipeline
--------------------------
Complete Stage 1 record-level detection pipeline.

Combines 4 detectors with supermajority voting (≥3/4).
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from tqdm import tqdm

from detectors.stage1 import (
    IsolationForestDetector,
    LOFDetector,
    StatisticalEnsembleDetector,
    LSTMAutoencoderDetector,
    GMMDetector,
    HDBSCANDetector
)
from preprocessing.normalizer import DataNormalizer


class Stage1Ensemble:
    """
    Stage 1 Ensemble Detector.
    
    Combines 4 detectors with voting:
    - Isolation Forest
    - Local Outlier Factor
    - Statistical Ensemble
    - LSTM Autoencoder
    
    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration YAML file
    contamination : float, default=0.02
        Expected proportion of outliers
    voting_threshold : int, default=3
        Minimum votes to flag (3/4 = supermajority)
    verbose : bool, default=True
        Print progress messages
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        contamination: float = 0.02,
        voting_threshold: int = 2,
        verbose: bool = True
    ):
        """Initialize Stage 1 ensemble."""
        self.contamination = contamination
        self.verbose = verbose
        
        # Load configuration
        if config_path is not None:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Get voting configuration from config file
        voting_config = self.config.get('voting', {})
        threshold_pct = voting_config.get('threshold_percentage', 0.5)
        # Read weights for each detector (default 1.0)
        self.detector_weights = voting_config.get('weights', {
            'isolation_forest': 1.0,
            'local_outlier_factor': 1.0,
            'statistical_ensemble': 1.0,
            'lstm_autoencoder': 1.0
        })
        # Calculate voting threshold based on percentage of total weight
        self.voting_threshold_percentage = threshold_pct
        self.voting_threshold = voting_threshold  # Fallback
        
        # Initialize components
        self.normalizer = None
        self.detectors = {}
        self.is_fitted = False
        
        # Initialize detectors
        self._init_detectors()
        
        # Update voting threshold based on total weight
        self.detector_names = list(self.detectors.keys())
        self.total_weight = sum(self.detector_weights.get(name, 1.0) for name in self.detector_names)
        self.voting_threshold = self.voting_threshold_percentage * self.total_weight
        if self.verbose:
            print(f"\nWeighted voting threshold: {self.voting_threshold:.2f}/{self.total_weight:.2f} ({self.voting_threshold_percentage*100:.0f}%)")
    
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('stage1', {})
    
    def _default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'preprocessing': {
                'scaler': 'minmax'
            },
            'isolation_forest': {
                'enabled': True,
                'n_estimators': 1000,
                'contamination': self.contamination
            },
            'local_outlier_factor': {
                'enabled': True,
                'n_neighbors': 'sqrt_n',
                'contamination': self.contamination
            },
            'statistical_ensemble': {
                'enabled': True
            },
            'lstm_autoencoder': {
                'enabled': True,
                'epochs': 50,
                'threshold_percentile': 98
            }
        }
    
    def _init_detectors(self):
        """Initialize all detectors based on configuration."""
        # Isolation Forest
        if_config = self.config.get('isolation_forest', {})
        if if_config.get('enabled', True):
            self.detectors['isolation_forest'] = IsolationForestDetector(
                contamination=if_config.get('contamination', self.contamination),
                n_estimators=if_config.get('n_estimators', 1000),
                max_samples=if_config.get('max_samples', 'auto'),
                max_features=if_config.get('max_features', 1.0),
                random_state=if_config.get('random_state', 42),
                n_jobs=if_config.get('n_jobs', -1)
            )
        
        # Local Outlier Factor
        lof_config = self.config.get('local_outlier_factor', {})
        if lof_config.get('enabled', True):
            self.detectors['local_outlier_factor'] = LOFDetector(
                contamination=lof_config.get('contamination', self.contamination),
                n_neighbors=lof_config.get('n_neighbors', 'sqrt_n'),
                metric=lof_config.get('metric', 'manhattan'),
                algorithm=lof_config.get('algorithm', 'auto'),
                leaf_size=lof_config.get('leaf_size', 30),
                n_jobs=lof_config.get('n_jobs', -1),
                decision_threshold_percentile=lof_config.get('decision_threshold_percentile', None)
            )
        
        # Statistical Ensemble
        stat_config = self.config.get('statistical_ensemble', {})
        if stat_config.get('enabled', True):
            self.detectors['statistical_ensemble'] = StatisticalEnsembleDetector(
                mahalanobis_percentile=stat_config.get('mahalanobis', {}).get('threshold_percentile', 99),
                iqr_multiplier=stat_config.get('iqr', {}).get('multiplier', 1.5),
                min_outlier_features=stat_config.get('iqr', {}).get('min_outlier_features', 3),
                zscore_threshold=stat_config.get('zscore', {}).get('threshold', 3.0)
            )
        
        # LSTM Autoencoder
        lstm_config = self.config.get('lstm_autoencoder', {})
        if lstm_config.get('enabled', True):
            arch_config = lstm_config.get('architecture', {})
            train_config = lstm_config.get('training', {})
            detect_config = lstm_config.get('detection', {})
            
            self.detectors['lstm_autoencoder'] = LSTMAutoencoderDetector(
                encoder_layers=arch_config.get('encoder_layers', [64, 32]),
                decoder_layers=arch_config.get('decoder_layers', [32, 64]),
                dropout=arch_config.get('dropout', 0.2),
                activation=arch_config.get('activation', 'tanh'),
                epochs=train_config.get('epochs', 50),
                batch_size=train_config.get('batch_size', 32),
                learning_rate=train_config.get('learning_rate', 0.001),
                validation_split=train_config.get('validation_split', 0.2),
                early_stopping_patience=train_config.get('early_stopping_patience', 10),
                threshold_percentile=detect_config.get('threshold_percentile', 98),
                verbose=train_config.get('verbose', 0)
            )
        
        # Gaussian Mixture Model
        gmm_config = self.config.get('gmm', {})
        if gmm_config.get('enabled', False):
            self.detectors['gmm'] = GMMDetector(
                n_components=gmm_config.get('n_components', 'auto'),
                contamination=gmm_config.get('contamination', self.contamination),
                covariance_type=gmm_config.get('covariance_type', 'full'),
                max_iter=gmm_config.get('max_iter', 100),
                n_init=gmm_config.get('n_init', 10),
                random_state=gmm_config.get('random_state', 42),
                decision_threshold_percentile=gmm_config.get('decision_threshold_percentile', None)
            )
        
        # HDBSCAN
        hdbscan_config = self.config.get('hdbscan', {})
        if hdbscan_config.get('enabled', False):
            self.detectors['hdbscan'] = HDBSCANDetector(
                min_cluster_size=hdbscan_config.get('min_cluster_size', 30),
                min_samples=hdbscan_config.get('min_samples', None),
                contamination=hdbscan_config.get('contamination', self.contamination),
                metric=hdbscan_config.get('metric', 'euclidean'),
                cluster_selection_method=hdbscan_config.get('cluster_selection_method', 'eom'),
                alpha=hdbscan_config.get('alpha', 1.0),
                decision_threshold_percentile=hdbscan_config.get('decision_threshold_percentile', None)
            )
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'Stage1Ensemble':
        """
        Fit Stage 1 ensemble on training data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Training data (will be normalized internally)
            
        Returns
        -------
        self : Stage1Ensemble
            Fitted ensemble
        """
        if self.verbose:
            print("="*80)
            print("STAGE 1: RECORD-LEVEL DETECTION - TRAINING")
            print("="*80)
            print(f"\nData shape: {X.shape[0]} records × {X.shape[1]} features")
            print(f"Enabled detectors: {len(self.detectors)}")
            print(f"Voting threshold: ≥{self.voting_threshold}/{len(self.detectors)}")
        
        # 1. Normalize data
        if self.verbose:
            print("\n[1/5] Normalizing data...")
        
        scaler_method = self.config.get('preprocessing', {}).get('scaler', 'minmax')
        self.normalizer = DataNormalizer(method=scaler_method)
        X_scaled = self.normalizer.fit_transform(X)
        
        if self.verbose:
            print(f"      Scaling method: {scaler_method}")
            print(f"      Data range: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
        
        # 2. Train detectors
        detector_names = list(self.detectors.keys())
        
        for i, (name, detector) in enumerate(self.detectors.items(), 1):
            if self.verbose:
                print(f"\n[{i+1}/5] Training {detector.name}...")
            
            detector.fit(X_scaled)
            
            if self.verbose:
                print(f"      Status: ✓ Fitted")
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n[5/5] Ensemble ready!")
            print("="*80)
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict anomalies using ensemble voting.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        # Normalize
        X_scaled = self.normalizer.transform(X)
        
        # Get predictions from all detectors
        votes = []
        weights = []
        for name, detector in self.detectors.items():
            preds = detector.predict(X_scaled)
            votes.append(preds)
            weights.append(self.detector_weights.get(name, 1.0))
        votes = np.stack(votes, axis=1)  # (n_samples, n_detectors)
        weights = np.array(weights)      # (n_detectors,)
        # Weighted sum per record
        weighted_votes = (votes * weights).sum(axis=1)
        # Apply weighted threshold
        predictions = (weighted_votes >= self.voting_threshold).astype(int)
        return predictions
    
    def predict_with_details(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict anomalies with detailed breakdown.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Data to predict
            
        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Binary predictions (0=normal, 1=anomaly)
        details : pd.DataFrame
            DataFrame with columns:
            - is_suspicious: final prediction
            - confidence: vote_count / n_detectors
            - vote_count: number of detectors that flagged
            - <detector_name>_flag: binary flag from each detector
            - <detector_name>_score: anomaly score from each detector
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")
        
        # Normalize
        X_scaled = self.normalizer.transform(X)
        
        # Get predictions and scores from all detectors
        detector_flags = {}
        detector_scores = {}
        weights = []
        for name, detector in self.detectors.items():
            detector_flags[f'{name}_flag'] = detector.predict(X_scaled)
            detector_scores[f'{name}_score'] = detector.decision_function(X_scaled)
            weights.append(self.detector_weights.get(name, 1.0))
        # Create details DataFrame
        details = pd.DataFrame(detector_flags)
        # Add scores
        for name, scores in detector_scores.items():
            details[name] = scores
        # Compute weighted vote counts and final predictions
        flag_cols = [col for col in details.columns if col.endswith('_flag')]
        flag_matrix = details[flag_cols].values  # (n_samples, n_detectors)
        weights = np.array(weights)              # (n_detectors,)
        weighted_votes = (flag_matrix * weights).sum(axis=1)
        predictions = (weighted_votes >= self.voting_threshold).astype(int)
        # Add summary columns
        details.insert(0, 'is_suspicious', predictions)
        details.insert(1, 'confidence', weighted_votes / self.total_weight)
        details.insert(2, 'vote_count', weighted_votes)
        return predictions, details
    
    def fit_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_details: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        """
        Fit ensemble and predict on same data.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training data
        return_details : bool, default=False
            If True, return (predictions, details)
            
        Returns
        -------
        predictions : np.ndarray
            Binary predictions
        details : pd.DataFrame (if return_details=True)
            Detailed breakdown
        """
        self.fit(X)
        
        if return_details:
            return self.predict_with_details(X)
        else:
            return self.predict(X)
    
    def get_detector_params(self) -> Dict:
        """Get parameters of all detectors."""
        params = {}
        for name, detector in self.detectors.items():
            params[name] = detector.get_params()
        return params
