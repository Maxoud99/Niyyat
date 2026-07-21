#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Preprocessing Pipeline for Stage 1 Detection
--------------------------------------------------------
Automatically detects and handles categorical features with one-hot encoding
and optional PCA for dimensionality reduction.

Only applies transformations when categorical columns are detected.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class IntelligentPreprocessor:
    """
    Automatically preprocess datasets for Stage 1 detection.
    
    Features:
    - Detects categorical (string/object) columns automatically
    - Applies one-hot encoding only if categorical columns exist
    - Applies PCA only if dimensionality is high (>50 features)
    - Preserves numeric-only datasets unchanged (except scaling)
    - Maintains 'is_erroneous' target column
    """
    
    def __init__(self, 
                 target_col='is_erroneous',
                 pca_threshold=50,
                 pca_variance=0.95,
                 verbose=True):
        """
        Parameters
        ----------
        target_col : str, default='is_erroneous'
            Name of target column to preserve
        pca_threshold : int, default=50
            Apply PCA if features > this threshold
        pca_variance : float, default=0.95
            Variance to preserve in PCA (0.95 = 95%)
        verbose : bool, default=True
            Print preprocessing steps
        """
        self.target_col = target_col
        self.pca_threshold = pca_threshold
        self.pca_variance = pca_variance
        self.verbose = verbose
        
        # Fitted components
        self.categorical_cols = []
        self.numeric_cols = []
        self.scaler = None
        self.pca = None
        self.feature_names_out = None
        
        # Flags
        self.has_categorical = False
        self.applied_onehot = False
        self.applied_pca = False
        
    def fit_transform(self, df):
        """
        Fit preprocessor and transform dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset with potential categorical columns
            
        Returns
        -------
        X : pd.DataFrame or np.ndarray
            Preprocessed features
        y : np.ndarray
            Target labels
        info : dict
            Preprocessing information
        """
        if self.verbose:
            print("\n" + "="*80)
            print("INTELLIGENT PREPROCESSING PIPELINE")
            print("="*80)
            print(f"\nInput shape: {df.shape}")
        
        # Step 1: Separate target
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset")
        
        y = df[self.target_col].values
        X = df.drop(columns=[self.target_col])
        
        if self.verbose:
            print(f"Features: {X.shape[1]} columns")
            print(f"Target: {len(y)} samples")
        
        # Step 2: Identify column types
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        self.has_categorical = len(self.categorical_cols) > 0
        
        if self.verbose:
            print(f"\n{'─'*80}")
            print("COLUMN TYPE DETECTION")
            print(f"{'─'*80}")
            print(f"Numeric columns: {len(self.numeric_cols)}")
            if len(self.numeric_cols) <= 10:
                print(f"  {self.numeric_cols}")
            print(f"Categorical columns: {len(self.categorical_cols)}")
            if len(self.categorical_cols) > 0:
                print(f"  {self.categorical_cols}")
        
        # Step 3: Handle categorical columns (one-hot encoding)
        if self.has_categorical:
            if self.verbose:
                print(f"\n{'─'*80}")
                print("ONE-HOT ENCODING CATEGORICAL FEATURES")
                print(f"{'─'*80}")
            
            # Strip whitespace from categorical columns
            for col in self.categorical_cols:
                X[col] = X[col].astype(str).str.strip()
            
            # One-hot encode with drop_first to avoid multicollinearity
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=int)
            
            self.applied_onehot = True
            
            if self.verbose:
                print(f"Before encoding: {X.shape[1]} features")
                print(f"After encoding: {X_encoded.shape[1]} features")
                print(f"New features created: {X_encoded.shape[1] - len(self.numeric_cols)}")
            
            X = X_encoded
        else:
            if self.verbose:
                print(f"\n✓ No categorical columns detected - skipping one-hot encoding")
        
        # Step 4: Scale features (always do this)
        if self.verbose:
            print(f"\n{'─'*80}")
            print("FEATURE SCALING")
            print(f"{'─'*80}")
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if self.verbose:
            print(f"Applied StandardScaler (mean=0, std=1)")
        
        # Step 5: PCA for dimensionality reduction (if needed)
        if X_scaled.shape[1] > self.pca_threshold:
            if self.verbose:
                print(f"\n{'─'*80}")
                print("PCA DIMENSIONALITY REDUCTION")
                print(f"{'─'*80}")
                print(f"Features ({X_scaled.shape[1]}) > threshold ({self.pca_threshold})")
                print(f"Applying PCA to preserve {self.pca_variance*100:.0f}% variance...")
            
            self.pca = PCA(n_components=self.pca_variance, random_state=42)
            X_reduced = self.pca.fit_transform(X_scaled)
            
            self.applied_pca = True
            
            if self.verbose:
                print(f"  Before PCA: {X_scaled.shape[1]} features")
                print(f"  After PCA: {X_reduced.shape[1]} components")
                print(f"  Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}")
                print(f"  Dimensionality reduction: {(1 - X_reduced.shape[1]/X_scaled.shape[1])*100:.1f}%")
            
            X_final = X_reduced
            self.feature_names_out = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
        else:
            if self.verbose:
                print(f"\n✓ Features ({X_scaled.shape[1]}) ≤ threshold ({self.pca_threshold}) - skipping PCA")
            
            X_final = X_scaled
            self.feature_names_out = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        # Step 6: Convert to DataFrame for consistency
        if self.feature_names_out is not None:
            X_final = pd.DataFrame(X_final, columns=self.feature_names_out)
        
        # Summary
        if self.verbose:
            print(f"\n{'='*80}")
            print("PREPROCESSING SUMMARY")
            print(f"{'='*80}")
            print(f"Original shape: {df.shape}")
            print(f"Final shape: {X_final.shape[0]} × {X_final.shape[1]}")
            print(f"\nTransformations applied:")
            print(f"  ✓ Target separation")
            print(f"  {'✓' if self.applied_onehot else '✗'} One-hot encoding ({len(self.categorical_cols)} categorical columns)")
            print(f"  ✓ Feature scaling (StandardScaler)")
            print(f"  {'✓' if self.applied_pca else '✗'} PCA dimensionality reduction")
        
        # Build info dict
        info = {
            'original_shape': df.shape,
            'final_shape': X_final.shape,
            'categorical_cols': self.categorical_cols,
            'numeric_cols': self.numeric_cols,
            'has_categorical': self.has_categorical,
            'applied_onehot': self.applied_onehot,
            'applied_pca': self.applied_pca,
            'n_components': X_final.shape[1] if self.applied_pca else None,
            'variance_explained': self.pca.explained_variance_ratio_.sum() if self.applied_pca else None
        }
        
        return X_final, y, info
    
    def save_info(self, output_path):
        """Save preprocessing information to file."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PREPROCESSING INFORMATION\n")
            f.write("="*80 + "\n\n")
            
            f.write("TRANSFORMATIONS APPLIED\n")
            f.write("-"*80 + "\n")
            f.write(f"One-hot encoding: {'Yes' if self.applied_onehot else 'No'}\n")
            f.write(f"PCA reduction: {'Yes' if self.applied_pca else 'No'}\n\n")
            
            if self.has_categorical:
                f.write("CATEGORICAL COLUMNS\n")
                f.write("-"*80 + "\n")
                for col in self.categorical_cols:
                    f.write(f"  - {col}\n")
                f.write("\n")
            
            f.write("NUMERIC COLUMNS\n")
            f.write("-"*80 + "\n")
            for col in self.numeric_cols:
                f.write(f"  - {col}\n")
            f.write("\n")
            
            if self.applied_pca:
                f.write("PCA DETAILS\n")
                f.write("-"*80 + "\n")
                f.write(f"Original features: {len(self.feature_names_out) if not self.applied_pca else self.scaler.n_features_in_}\n")
                f.write(f"Reduced components: {self.pca.n_components_}\n")
                f.write(f"Variance explained: {self.pca.explained_variance_ratio_.sum():.3f}\n")
                f.write(f"Top 5 components variance:\n")
                for i, var in enumerate(self.pca.explained_variance_ratio_[:5], 1):
                    f.write(f"  PC{i}: {var:.3f}\n")


def preprocess_and_save(input_path, output_dir, pca_threshold=50, pca_variance=0.95):
    """
    Preprocess dataset and save results.
    
    Parameters
    ----------
    input_path : str or Path
        Path to input CSV with 'is_erroneous' column
    output_dir : str or Path
        Directory to save preprocessed data and info
    pca_threshold : int, default=50
        Apply PCA if features > threshold
    pca_variance : float, default=0.95
        Variance to preserve in PCA
        
    Returns
    -------
    X : pd.DataFrame
        Preprocessed features
    y : np.ndarray
        Target labels
    info : dict
        Preprocessing information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # Preprocess
    preprocessor = IntelligentPreprocessor(
        pca_threshold=pca_threshold,
        pca_variance=pca_variance,
        verbose=True
    )
    
    X, y, info = preprocessor.fit_transform(df)
    
    # Save preprocessed data
    output_data_path = output_dir / 'preprocessed_data.csv'
    df_output = X.copy()
    df_output['is_erroneous'] = y
    df_output.to_csv(output_data_path, index=False)
    print(f"\n✓ Preprocessed data saved: {output_data_path}")
    
    # Save preprocessing info
    output_info_path = output_dir / 'preprocessing_info.txt'
    preprocessor.save_info(output_info_path)
    print(f"✓ Preprocessing info saved: {output_info_path}")
    
    return X, y, info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess dataset for Stage 1 detection')
    parser.add_argument('--input', '-i', required=True,
                       help='Path to input CSV (must have "is_erroneous" column)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for preprocessed data')
    parser.add_argument('--pca-threshold', type=int, default=50,
                       help='Apply PCA if features > threshold (default: 50)')
    parser.add_argument('--pca-variance', type=float, default=0.95,
                       help='Variance to preserve in PCA (default: 0.95)')
    
    args = parser.parse_args()
    
    X, y, info = preprocess_and_save(
        input_path=args.input,
        output_dir=args.output,
        pca_threshold=args.pca_threshold,
        pca_variance=args.pca_variance
    )
    
    print("\n" + "="*80)
    print("✅ PREPROCESSING COMPLETE")
    print("="*80)
