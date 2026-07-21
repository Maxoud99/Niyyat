#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-Based Cell Attribution - Learn from Examples
-------------------------------------------------
Train a classifier to recognize error patterns from labeled examples.
This should achieve much better precision.
"""

import numpy as np
import pandas as pd
from typing import List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MLCellAttributor:
    """
    ML-based cell attribution that LEARNS from examples.
    
    Key idea:
    - Extract features describing each cell in context
    - Train classifier on known errors/non-errors
    - Predict on new records
    
    Should achieve 40-50% precision with good recall.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_classifiers = {}  # One classifier per feature
        self.label_encoders = {}
        self.X_clean = None
        
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_records: np.ndarray,
        cell_masks_train: pd.DataFrame
    ):
        """
        Fit classifiers on training data with known cell-level labels.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training records
        y_train_records : np.ndarray
            Record-level labels (1=erroneous, 0=clean)
        cell_masks_train : pd.DataFrame
            Cell-level ground truth (1=error, 0=clean)
        """
        if self.verbose:
            print("\n" + "="*60)
            print("TRAINING ML-BASED CELL ATTRIBUTOR")
            print("="*60)
        
        # Store clean baseline
        self.X_clean = X_train[y_train_records == 0].copy()
        
        # Get erroneous records with known cell masks
        erroneous_indices = np.where(y_train_records == 1)[0]
        
        if len(erroneous_indices) == 0:
            raise ValueError("No erroneous records in training data!")
        
        X_erroneous = X_train.iloc[erroneous_indices]
        masks_erroneous = cell_masks_train.iloc[erroneous_indices]
        
        print(f"\nTraining data:")
        print(f"  Erroneous records: {len(erroneous_indices)}")
        print(f"  Clean records: {len(self.X_clean)}")
        print(f"  True error cells: {masks_erroneous.sum().sum()}")
        
        # Fit label encoders for categorical features
        for col in X_train.columns:
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                self.label_encoders[col] = le
        
        # Train one classifier per feature
        if self.verbose:
            print("\nTraining per-feature classifiers...")
        
        for col in tqdm(X_train.columns, disable=not self.verbose, desc="Features"):
            if col not in masks_erroneous.columns:
                continue
            
            # Extract features for this column
            X_feat, y_feat = self._extract_features_for_column(
                X_erroneous, masks_erroneous, col
            )
            
            if len(X_feat) == 0 or y_feat.sum() == 0:
                continue  # Skip if no positive examples
            
            # Train classifier
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                n_jobs=-1
            )
            
            clf.fit(X_feat, y_feat)
            self.feature_classifiers[col] = clf
        
        if self.verbose:
            print(f"\n✓ Trained {len(self.feature_classifiers)} classifiers")
    
    def attribute(
        self,
        X: pd.DataFrame,
        flagged_indices: List[int]
    ) -> pd.DataFrame:
        """Predict cell-level errors using trained classifiers."""
        if self.verbose:
            print("\n" + "="*60)
            print("ML-BASED CELL ATTRIBUTION")
            print("="*60)
        
        suspicions = pd.DataFrame(0.0, index=flagged_indices, columns=X.columns)
        
        for col in tqdm(X.columns, disable=not self.verbose, desc="Features"):
            if col not in self.feature_classifiers:
                continue
            
            clf = self.feature_classifiers[col]
            
            # Extract features for all flagged records
            X_feat_all = []
            valid_indices = []
            
            for idx in flagged_indices:
                try:
                    record = X.loc[idx]
                    feat = self._extract_features_single_cell(record, col, X)
                    if feat is not None:
                        X_feat_all.append(feat)
                        valid_indices.append(idx)
                except:
                    continue
            
            if len(X_feat_all) == 0:
                continue
            
            X_feat_all = np.array(X_feat_all)
            
            # Predict probabilities
            probs = clf.predict_proba(X_feat_all)[:, 1]  # Prob of being error
            
            # Assign to suspicions
            for idx, prob in zip(valid_indices, probs):
                suspicions.loc[idx, col] = prob
        
        if self.verbose:
            total_flagged = (suspicions > 0.5).sum().sum()
            print(f"\n✓ Attribution complete")
            print(f"  Cells flagged (>0.5): {total_flagged}")
        
        return suspicions
    
    def _extract_features_for_column(
        self,
        X: pd.DataFrame,
        masks: pd.DataFrame,
        col: str
    ):
        """Extract training features for a specific column."""
        X_feat = []
        y_feat = []
        
        for idx in X.index:
            record = X.loc[idx]
            is_error = masks.loc[idx, col]
            
            feat = self._extract_features_single_cell(record, col, X)
            if feat is not None:
                X_feat.append(feat)
                y_feat.append(is_error)
        
        return np.array(X_feat), np.array(y_feat)
    
    def _extract_features_single_cell(
        self,
        record: pd.Series,
        col: str,
        X_full: pd.DataFrame
    ) -> np.ndarray:
        """
        Extract features describing a single cell in context.
        
        Features:
        1. Value itself (encoded)
        2. Statistical position (percentile, z-score)
        3. Frequency in clean data
        4. Context: values of related features
        5. Neighborhood: does value appear in similar records?
        """
        try:
            features = []
            
            # 1. Value encoding
            val = record[col]
            if col in self.label_encoders:
                # Categorical
                try:
                    val_encoded = self.label_encoders[col].transform([str(val)])[0]
                except:
                    val_encoded = -1  # Unknown value
                features.append(val_encoded)
                
                # Frequency in clean data
                freq = (self.X_clean[col] == val).mean()
                features.append(freq)
                
                # Is it a known value?
                is_known = int(val in self.X_clean[col].unique())
                features.append(is_known)
            else:
                # Numeric
                features.append(val)
                
                # Z-score
                mean = self.X_clean[col].mean()
                std = self.X_clean[col].std()
                if std > 0:
                    z_score = (val - mean) / std
                else:
                    z_score = 0
                features.append(z_score)
                
                # Percentile
                percentile = (self.X_clean[col] < val).mean()
                features.append(percentile)
            
            # 2. Context features (other column values)
            # Add up to 5 other features as context
            other_cols = [c for c in record.index if c != col][:5]
            for other_col in other_cols:
                other_val = record[other_col]
                
                if other_col in self.label_encoders:
                    try:
                        other_encoded = self.label_encoders[other_col].transform([str(other_val)])[0]
                    except:
                        other_encoded = -1
                    features.append(other_encoded)
                else:
                    features.append(other_val)
            
            # Pad if needed
            while len(features) < 10:
                features.append(0)
            
            return np.array(features[:10])  # Fixed size: 10 features
        except:
            return None
