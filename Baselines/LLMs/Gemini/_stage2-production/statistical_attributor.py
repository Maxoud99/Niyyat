#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: Cell-Level Error Attribution
--------------------------------------
For records flagged in Stage 1, identify which specific cells are erroneous.

Attribution Methods:
1. Counterfactual Analysis: If I fix this cell, does suspicion drop?
2. Local Comparison: How much does this feature deviate from similar clean records?
3. Validator Agreement: Which detectors flagged this record, and which features caused flags?
4. Statistical Deviation: How statistically anomalous is this cell value?

Output: Suspicion score per cell [0-1]
        Matrix: [record_id, feature] -> suspicion_score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class CellLevelAttributor:
    """
    Cell-Level Error Attribution System.
    
    For each flagged record, attributes suspicion scores to individual cells
    using multiple attribution methods.
    
    Parameters
    ----------
    detector_ensemble : Stage1Ensemble
        Trained Stage 1 detector ensemble
    k_neighbors : int, default=10
        Number of neighbors for local comparison
    method_weights : dict, optional
        Weights for each attribution method
    verbose : bool, default=True
        Print progress messages
    """
    
    def __init__(
        self,
        detector_ensemble=None,
        k_neighbors: int = 10,
        method_weights: Optional[Dict[str, float]] = None,
        verbose: bool = True
    ):
        """Initialize cell-level attributor."""
        self.detector_ensemble = detector_ensemble
        self.k_neighbors = k_neighbors
        self.verbose = verbose
        
        # Default method weights
        if method_weights is None:
            self.method_weights = {
                'counterfactual': 0.35,
                'local_comparison': 0.30,
                'validator_agreement': 0.20,
                'statistical_deviation': 0.15
            }
        else:
            self.method_weights = method_weights
        
        # Normalize weights
        total_weight = sum(self.method_weights.values())
        self.method_weights = {k: v/total_weight for k, v in self.method_weights.items()}
        
        # Storage for clean baseline (for comparison)
        self.clean_baseline = None
        self.feature_statistics = {}
        self.nn_model = None
        
    def fit(self, X_clean: pd.DataFrame):
        """
        Fit the attributor on clean data.
        
        Parameters
        ----------
        X_clean : pd.DataFrame
            Clean baseline data for comparison
        """
        if self.verbose:
            print("\n" + "="*60)
            print("FITTING CELL-LEVEL ATTRIBUTOR")
            print("="*60)
        
        self.clean_baseline = X_clean.copy()
        
        # Compute feature statistics
        if self.verbose:
            print("\nComputing feature statistics...")
        
        for col in tqdm(X_clean.columns, disable=not self.verbose, desc="Features"):
            if pd.api.types.is_numeric_dtype(X_clean[col]):
                self.feature_statistics[col] = {
                    'type': 'numeric',
                    'mean': X_clean[col].mean(),
                    'std': X_clean[col].std(),
                    'median': X_clean[col].median(),
                    'q1': X_clean[col].quantile(0.25),
                    'q3': X_clean[col].quantile(0.75),
                    'iqr': X_clean[col].quantile(0.75) - X_clean[col].quantile(0.25),
                    'min': X_clean[col].min(),
                    'max': X_clean[col].max()
                }
            else:
                value_counts = X_clean[col].value_counts()
                self.feature_statistics[col] = {
                    'type': 'categorical',
                    'mode': X_clean[col].mode()[0] if len(X_clean[col].mode()) > 0 else None,
                    'value_counts': value_counts,
                    'unique_values': set(X_clean[col].unique())
                }
        
        # Fit nearest neighbors model for local comparison
        if self.verbose:
            print(f"\nFitting k-NN model (k={self.k_neighbors})...")
        
        # Use normalized features for k-NN
        X_clean_normalized = self._normalize_for_knn(X_clean)
        self.nn_model = NearestNeighbors(n_neighbors=self.k_neighbors, metric='euclidean')
        self.nn_model.fit(X_clean_normalized)
        
        if self.verbose:
            print("✓ Cell-level attributor fitted successfully")
    
    def attribute(
        self,
        X: pd.DataFrame,
        flagged_indices: List[int],
        record_suspicions: Optional[Dict[int, float]] = None
    ) -> pd.DataFrame:
        """
        Attribute suspicion scores to cells in flagged records.
        
        Parameters
        ----------
        X : pd.DataFrame
            Full dataset (including flagged records)
        flagged_indices : list of int
            Indices of records flagged in Stage 1
        record_suspicions : dict, optional
            Record-level suspicion scores from Stage 1
        
        Returns
        -------
        cell_suspicions : pd.DataFrame
            DataFrame with same shape as X[flagged_indices], containing suspicion scores
        """
        if self.verbose:
            print("\n" + "="*60)
            print("STAGE 2: CELL-LEVEL ATTRIBUTION")
            print("="*60)
            print(f"\nAttributing {len(flagged_indices)} flagged records...")
        
        # Initialize cell suspicion matrix
        cell_suspicions = pd.DataFrame(
            0.0,
            index=flagged_indices,
            columns=X.columns
        )
        
        # Process each flagged record
        for idx in tqdm(flagged_indices, disable=not self.verbose, desc="Records"):
            record = X.loc[idx]
            record_suspicion = record_suspicions[idx] if record_suspicions else 0.5
            
            # Compute cell suspicions using multiple methods
            cell_scores = self._attribute_single_record(record, X, record_suspicion)
            cell_suspicions.loc[idx] = cell_scores
        
        if self.verbose:
            print(f"\n✓ Cell attribution complete")
            print(f"  Total cells analyzed: {len(flagged_indices) * len(X.columns)}")
            print(f"  Suspicious cells (>0.5): {(cell_suspicions > 0.5).sum().sum()}")
        
        return cell_suspicions
    
    def _attribute_single_record(
        self,
        record: pd.Series,
        X: pd.DataFrame,
        record_suspicion: float
    ) -> pd.Series:
        """
        Attribute suspicion to cells in a single record.
        
        Returns
        -------
        cell_scores : pd.Series
            Suspicion score for each cell in the record
        """
        cell_scores = pd.Series(0.0, index=record.index)
        
        # Method 1: Counterfactual Analysis
        if 'counterfactual' in self.method_weights and self.detector_ensemble is not None:
            cf_scores = self._counterfactual_score(record, X)
            cell_scores += self.method_weights['counterfactual'] * cf_scores
        
        # Method 2: Local Comparison
        if 'local_comparison' in self.method_weights:
            local_scores = self._local_comparison_score(record, X)
            cell_scores += self.method_weights['local_comparison'] * local_scores
        
        # Method 3: Validator Agreement
        if 'validator_agreement' in self.method_weights and self.detector_ensemble is not None:
            validator_scores = self._validator_agreement_score(record, X)
            cell_scores += self.method_weights['validator_agreement'] * validator_scores
        
        # Method 4: Statistical Deviation
        if 'statistical_deviation' in self.method_weights:
            stat_scores = self._statistical_deviation_score(record)
            cell_scores += self.method_weights['statistical_deviation'] * stat_scores
        
        # Clip to [0, 1]
        cell_scores = cell_scores.clip(0, 1)
        
        return cell_scores
    
    def _counterfactual_score(self, record: pd.Series, X: pd.DataFrame) -> pd.Series:
        """
        Counterfactual analysis: If I fix this cell, does suspicion drop?
        
        Strategy:
        - For each feature, replace the value with a typical clean value
        - Re-score the record with Stage 1 ensemble
        - Contribution = original_suspicion - modified_suspicion
        """
        scores = pd.Series(0.0, index=record.index)
        
        if self.detector_ensemble is None or not self.detector_ensemble.is_fitted:
            return scores
        
        # Get original suspicion
        try:
            original_suspicion = self.detector_ensemble.predict_proba(
                pd.DataFrame([record])
            )[0, 1]  # Probability of being erroneous
        except:
            return scores
        
        # For each feature, try fixing it
        for feature in record.index:
            # Create modified record
            modified_record = record.copy()
            
            # Replace with typical value
            typical_value = self._get_typical_value(feature)
            if typical_value is not None:
                modified_record[feature] = typical_value
                
                # Re-score
                try:
                    modified_suspicion = self.detector_ensemble.predict_proba(
                        pd.DataFrame([modified_record])
                    )[0, 1]
                    
                    # Contribution = drop in suspicion
                    contribution = max(0, original_suspicion - modified_suspicion)
                    scores[feature] = contribution
                except:
                    scores[feature] = 0.0
        
        # Normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        
        return scores
    
    def _local_comparison_score(self, record: pd.Series, X: pd.DataFrame) -> pd.Series:
        """
        Local comparison: How much does this feature deviate from similar clean records?
        
        Strategy:
        - Find k-nearest clean records
        - For each feature, compute deviation from neighbors
        - Higher deviation = higher suspicion
        """
        scores = pd.Series(0.0, index=record.index)
        
        if self.clean_baseline is None or self.nn_model is None:
            return scores
        
        # Find k-nearest neighbors in clean baseline
        record_normalized = self._normalize_for_knn(pd.DataFrame([record]))
        distances, indices = self.nn_model.kneighbors(record_normalized)
        
        neighbors = self.clean_baseline.iloc[indices[0]]
        
        # Compute deviation for each feature
        for feature in record.index:
            if pd.api.types.is_numeric_dtype(self.clean_baseline[feature]):
                # Numeric: compute z-score relative to neighbors
                neighbor_values = neighbors[feature].values
                neighbor_mean = neighbor_values.mean()
                neighbor_std = neighbor_values.std()
                
                if neighbor_std > 0:
                    z_score = abs((record[feature] - neighbor_mean) / neighbor_std)
                    # Convert to [0, 1] using sigmoid
                    scores[feature] = 1 / (1 + np.exp(-z_score + 2))  # Centered at z=2
                else:
                    scores[feature] = 0.0
            else:
                # Categorical: check if value appears in neighbors
                neighbor_values = set(neighbors[feature].values)
                if record[feature] not in neighbor_values:
                    scores[feature] = 1.0  # Complete mismatch
                else:
                    # Partial score based on rarity
                    freq = (neighbors[feature] == record[feature]).mean()
                    scores[feature] = 1 - freq  # Rarer = more suspicious
        
        return scores
    
    def _validator_agreement_score(self, record: pd.Series, X: pd.DataFrame) -> pd.Series:
        """
        Validator agreement: Which features caused detectors to flag this record?
        
        Strategy:
        - For each detector, check which features contribute to the flag
        - Aggregate across detectors
        """
        scores = pd.Series(0.0, index=record.index)
        
        if self.detector_ensemble is None or not hasattr(self.detector_ensemble, 'detectors'):
            return scores
        
        # For each detector, get feature importance (if available)
        detector_contributions = []
        
        for detector_name, detector in self.detector_ensemble.detectors.items():
            # Try to get feature importance from detector
            feature_scores = self._get_detector_feature_scores(detector, record, X)
            if feature_scores is not None:
                detector_contributions.append(feature_scores)
        
        # Aggregate across detectors
        if len(detector_contributions) > 0:
            scores = pd.concat(detector_contributions, axis=1).mean(axis=1)
        
        return scores
    
    def _statistical_deviation_score(self, record: pd.Series) -> pd.Series:
        """
        Statistical deviation: How statistically anomalous is this cell value?
        
        Strategy:
        - For numeric: z-score from clean baseline
        - For categorical: frequency-based rarity score
        """
        scores = pd.Series(0.0, index=record.index)
        
        for feature in record.index:
            if feature not in self.feature_statistics:
                continue
            
            stats_dict = self.feature_statistics[feature]
            
            if stats_dict['type'] == 'numeric':
                # Z-score based suspicion
                mean = stats_dict['mean']
                std = stats_dict['std']
                
                if std > 0:
                    z_score = abs((record[feature] - mean) / std)
                    # Convert to [0, 1] using sigmoid
                    scores[feature] = 1 / (1 + np.exp(-z_score + 3))  # Centered at z=3
                else:
                    scores[feature] = 0.0
            else:
                # Categorical: rarity-based suspicion
                if record[feature] not in stats_dict['unique_values']:
                    scores[feature] = 1.0  # Unknown value
                else:
                    value_counts = stats_dict['value_counts']
                    freq = value_counts.get(record[feature], 0) / value_counts.sum()
                    scores[feature] = 1 - freq  # Rarer = more suspicious
        
        return scores
    
    def _get_typical_value(self, feature: str):
        """Get a typical (clean) value for a feature."""
        if feature not in self.feature_statistics:
            return None
        
        stats_dict = self.feature_statistics[feature]
        
        if stats_dict['type'] == 'numeric':
            return stats_dict['median']  # Use median as typical value
        else:
            return stats_dict['mode']  # Use mode for categorical
    
    def _normalize_for_knn(self, X: pd.DataFrame) -> np.ndarray:
        """Normalize features for k-NN distance calculation."""
        X_normalized = X.copy()
        
        for col in X.columns:
            if col in self.feature_statistics:
                stats = self.feature_statistics[col]
                
                if stats['type'] == 'numeric':
                    # Standard scaling
                    mean = stats['mean']
                    std = stats['std']
                    if std > 0:
                        X_normalized[col] = (X[col] - mean) / std
                else:
                    # One-hot encode categorical (simple label encoding for k-NN)
                    X_normalized[col] = pd.Categorical(X[col]).codes
            else:
                # Feature not in statistics, try to encode
                if pd.api.types.is_numeric_dtype(X[col]):
                    X_normalized[col] = X[col]
                else:
                    X_normalized[col] = pd.Categorical(X[col]).codes
        
        return X_normalized.values
    
    def _get_detector_feature_scores(self, detector, record: pd.Series, X: pd.DataFrame) -> Optional[pd.Series]:
        """
        Get feature-level scores from a detector.
        
        This is detector-specific and may not be available for all detectors.
        """
        # For now, return uniform scores
        # TODO: Implement detector-specific feature importance extraction
        return None
    
    def get_top_suspicious_cells(
        self,
        cell_suspicions: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Get the top-k most suspicious cells across all flagged records.
        
        Parameters
        ----------
        cell_suspicions : pd.DataFrame
            Cell suspicion matrix from attribute()
        top_k : int, default=10
            Number of top cells to return
        
        Returns
        -------
        top_cells : pd.DataFrame
            DataFrame with columns [record_id, feature, suspicion_score]
        """
        # Flatten cell suspicions
        records = []
        for idx in cell_suspicions.index:
            for col in cell_suspicions.columns:
                records.append({
                    'record_id': idx,
                    'feature': col,
                    'suspicion_score': cell_suspicions.loc[idx, col]
                })
        
        top_cells_df = pd.DataFrame(records)
        top_cells_df = top_cells_df.sort_values('suspicion_score', ascending=False)
        
        return top_cells_df.head(top_k)
    
    def summarize_attribution(
        self,
        cell_suspicions: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Summarize cell-level attribution results.
        
        Parameters
        ----------
        cell_suspicions : pd.DataFrame
            Cell suspicion matrix from attribute()
        threshold : float, default=0.5
            Threshold for considering a cell suspicious
        
        Returns
        -------
        summary : dict
            Summary statistics
        """
        suspicious_cells = (cell_suspicions > threshold).sum().sum()
        total_cells = cell_suspicions.size
        
        # Per-feature statistics
        feature_stats = {}
        for col in cell_suspicions.columns:
            feature_stats[col] = {
                'mean_suspicion': cell_suspicions[col].mean(),
                'max_suspicion': cell_suspicions[col].max(),
                'num_suspicious': (cell_suspicions[col] > threshold).sum(),
                'percentage_suspicious': (cell_suspicions[col] > threshold).sum() / len(cell_suspicions) * 100
            }
        
        summary = {
            'total_records': len(cell_suspicions),
            'total_cells': total_cells,
            'suspicious_cells': suspicious_cells,
            'percentage_suspicious': suspicious_cells / total_cells * 100,
            'mean_suspicion': cell_suspicions.mean().mean(),
            'feature_statistics': feature_stats
        }
        
        return summary
