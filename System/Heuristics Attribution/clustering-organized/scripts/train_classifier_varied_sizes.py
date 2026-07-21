#!/usr/bin/env python3
"""
Random Forest Classifier with Varied Training Set Sizes

This script trains Random Forest classifiers with different training set sizes:
1%, 5%, 10%, 20%, 30%, 50%, and 100% (baseline)

Evaluates how training data size impacts performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import json
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class IntentAttributionClassifier:
    """Random Forest classifier for predicting intent labels."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.feature_encoders = {}
        self.value_encoders = {}
        self.feature_names = []
        self._dummy_columns = []  # Initialize dummy columns list
        
    def create_features(self, df):
        """Create feature vectors from original_value, new_value, and feature_name."""
        df = df.reset_index(drop=True)
        features = pd.DataFrame()
        
        # Encode feature names
        if 'feature_name' not in self.feature_encoders:
            self.feature_encoders['feature_name'] = LabelEncoder()
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].fit_transform(
                df['feature_name'].astype(str)
            )
        else:
            # Handle unseen feature names in test set
            known_features = set(self.feature_encoders['feature_name'].classes_)
            df_copy = df.copy()
            df_copy['feature_name'] = df_copy['feature_name'].astype(str).apply(
                lambda x: x if x in known_features else 'UNKNOWN'
            )
            if 'UNKNOWN' not in known_features:
                # Add UNKNOWN to encoder
                self.feature_encoders['feature_name'].classes_ = np.append(
                    self.feature_encoders['feature_name'].classes_, 'UNKNOWN'
                )
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].transform(
                df_copy['feature_name']
            )
        
        # Encode original values
        if 'original_value' not in self.value_encoders:
            self.value_encoders['original_value'] = LabelEncoder()
            features['original_value_encoded'] = self.value_encoders['original_value'].fit_transform(
                df['original_value'].astype(str)
            )
        else:
            known_values = set(self.value_encoders['original_value'].classes_)
            df_copy = df.copy()
            df_copy['original_value'] = df_copy['original_value'].astype(str).apply(
                lambda x: x if x in known_values else 'UNKNOWN'
            )
            if 'UNKNOWN' not in known_values:
                self.value_encoders['original_value'].classes_ = np.append(
                    self.value_encoders['original_value'].classes_, 'UNKNOWN'
                )
            features['original_value_encoded'] = self.value_encoders['original_value'].transform(
                df_copy['original_value']
            )
        
        # Encode new values
        if 'new_value' not in self.value_encoders:
            self.value_encoders['new_value'] = LabelEncoder()
            features['new_value_encoded'] = self.value_encoders['new_value'].fit_transform(
                df['new_value'].astype(str)
            )
        else:
            known_values = set(self.value_encoders['new_value'].classes_)
            df_copy = df.copy()
            df_copy['new_value'] = df_copy['new_value'].astype(str).apply(
                lambda x: x if x in known_values else 'UNKNOWN'
            )
            if 'UNKNOWN' not in known_values:
                self.value_encoders['new_value'].classes_ = np.append(
                    self.value_encoders['new_value'].classes_, 'UNKNOWN'
                )
            features['new_value_encoded'] = self.value_encoders['new_value'].transform(
                df_copy['new_value']
            )
        
        # Value changed indicator
        features['value_changed'] = (
            df['original_value'].astype(str) != df['new_value'].astype(str)
        ).astype(int)
        
        # One-hot encode feature names (only during training)
        if not self._dummy_columns:  # First time (training)
            # Training: create dummies and store column names
            feature_dummies = pd.get_dummies(df['feature_name'], prefix='feat')
            features = pd.concat([features, feature_dummies], axis=1)
            self.feature_names = features.columns.tolist()
            self._dummy_columns = feature_dummies.columns.tolist()
        else:  # Subsequent calls (testing)
            # Testing: use stored column names and ensure same columns
            feature_dummies = pd.get_dummies(df['feature_name'], prefix='feat')
            # Add missing columns with zeros
            for col in self._dummy_columns:
                if col not in feature_dummies.columns:
                    features[col] = 0
                else:
                    features[col] = feature_dummies[col]
            # Ensure columns are in same order as training
            features = features[self.feature_names]
        
        return features
    
    def train(self, X, y):
        """Train the model."""
        print(f"  Training Random Forest...")
        print(f"    Training samples: {len(X)}")
        print(f"    Features: {X.shape[1]}")
        print(f"    Intentional (1): {(y == 1).sum()}")
        print(f"    Unintentional (-1): {(y == -1).sum()}")
        
        self.model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  ✓ Training complete!")
        return feature_importance
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        try:
            metrics['precision_intentional'] = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            metrics['recall_intentional'] = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            metrics['f1_intentional'] = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        except:
            metrics['precision_intentional'] = 0.0
            metrics['recall_intentional'] = 0.0
            metrics['f1_intentional'] = 0.0
        
        try:
            metrics['precision_unintentional'] = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
            metrics['recall_unintentional'] = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
            metrics['f1_unintentional'] = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        except:
            metrics['precision_unintentional'] = 0.0
            metrics['recall_unintentional'] = 0.0
            metrics['f1_unintentional'] = 0.0
        
        return metrics, y_pred


def load_data(data_dir):
    """Load and prepare data."""
    print(f"\nLoading data from: {data_dir}")
    
    data_path = Path(data_dir)
    masks_file = data_path / 'masks.csv'
    manipulated_file = data_path / 'manipulated_records.csv'
    correct_file = data_path.parent / 'correct_records.csv'
    
    # Load files
    masks = pd.read_csv(masks_file)
    manipulated = pd.read_csv(manipulated_file)
    correct = pd.read_csv(correct_file)
    
    print(f"✓ Loaded masks: {masks.shape}")
    print(f"✓ Loaded manipulated records: {manipulated.shape}")
    print(f"✓ Loaded correct records: {correct.shape}")
    
    # Prepare dataset
    records = []
    feature_cols = masks.columns.tolist()
    
    print(f"\nProcessing {len(masks)} records...")
    
    for idx in tqdm(range(len(masks)), desc="Processing records", unit="record"):
        
        # For each record at index j in correct_records, 
        # there are 3 manipulated variants at indices 3j, 3j+1, 3j+2 in manipulated_records
        original_record_idx = idx // 3
        
        for feature in feature_cols:
            intent_label = masks.iloc[idx][feature]
            
            # Skip if no change (label = 0)
            if intent_label == 0:
                continue
            
            original_value = correct.iloc[original_record_idx][feature]
            new_value = manipulated.iloc[idx][feature]
            
            records.append({
                'record_id': original_record_idx,  # Use original record ID for grouping
                'manipulated_record_id': idx,      # Keep track of which variant
                'feature_name': feature,
                'original_value': original_value,
                'new_value': new_value,
                'intent_label': intent_label
            })
    
    df = pd.DataFrame(records)
    print(f"\n✓ Created dataset with {len(df)} feature changes")
    print(f"  Intentional (1): {(df['intent_label'] == 1).sum()}")
    print(f"  Unintentional (-1): {(df['intent_label'] == -1).sum()}")
    print(f"  Unique records: {df['record_id'].nunique()}")
    
    return df


def split_by_record_id(df, test_size=0.3, random_state=42):
    """Split data by record_id to avoid data leakage."""
    unique_records = df['record_id'].unique()
    
    train_records, test_records = train_test_split(
        unique_records,
        test_size=test_size,
        random_state=random_state
    )
    
    train_df = df[df['record_id'].isin(train_records)].copy()
    test_df = df[df['record_id'].isin(test_records)].copy()
    
    print(f"\nTrain/Test split by record_id:")
    print(f"  Train records: {len(train_records)}")
    print(f"  Test records: {len(test_records)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    
    return train_df, test_df


def subsample_training_data(train_df, fraction, random_state=42):
    """Subsample training data to specified fraction of records."""
    unique_records = train_df['record_id'].unique()
    n_records = int(len(unique_records) * fraction)
    
    np.random.seed(random_state)
    sampled_records = np.random.choice(unique_records, size=n_records, replace=False)
    
    sampled_df = train_df[train_df['record_id'].isin(sampled_records)].copy()
    
    return sampled_df


def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812'
    OUTPUT_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/classifier_varied_sizes'
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Training set fractions to test
    TRAIN_FRACTIONS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
    
    print("\n" + "="*70)
    print(" RANDOM FOREST WITH VARIED TRAINING SET SIZES")
    print("="*70)
    print(f"Dataset: Adult Income Dataset")
    print(f"Training Set Fractions: {[f'{f*100:.0f}%' for f in TRAIN_FRACTIONS]}")
    print(f"Test Set: Fixed at 30% of data")
    print("="*70)
    
    # Load data
    df = load_data(DATA_DIR)
    
    # Split data by record_id (same test set for all experiments)
    train_df_full, test_df = split_by_record_id(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Store results for all fractions
    all_results = []
    
    # Test each training fraction
    for fraction in tqdm(TRAIN_FRACTIONS, desc="Training Progress", unit="fraction"):
        print(f"\n{'='*70}")
        print(f"TRAINING WITH {fraction*100:.0f}% OF TRAINING DATA")
        print(f"{'='*70}")
        
        # Subsample training data
        if fraction < 1.0:
            train_df = subsample_training_data(train_df_full, fraction, RANDOM_STATE)
        else:
            train_df = train_df_full
        
        print(f"  Training records: {train_df['record_id'].nunique()}")
        print(f"  Training samples: {len(train_df)}")
        
        # Initialize classifier
        classifier = IntentAttributionClassifier(random_state=RANDOM_STATE)
        
        # Create features
        print(f"\nCreating features...")
        X_train = classifier.create_features(train_df)
        y_train = train_df['intent_label'].values
        
        X_test = classifier.create_features(test_df)
        y_test = test_df['intent_label'].values
        
        # Train model
        print(f"\nTraining model...")
        feature_importance = classifier.train(X_train, y_train)
        
        # Evaluate model
        print(f"\nEvaluating model...")
        metrics, y_pred = classifier.evaluate(X_test, y_test)
        
        print(f"\n  Results:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 (weighted): {metrics['f1_weighted']:.4f}")
        print(f"    F1 (intentional): {metrics['f1_intentional']:.4f}")
        print(f"    F1 (unintentional): {metrics['f1_unintentional']:.4f}")
        
        # Store results
        result = {
            'train_fraction': fraction,
            'train_records': train_df['record_id'].nunique(),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            **metrics
        }
        all_results.append(result)
        
        # Save model for this fraction
        output_path = Path(OUTPUT_DIR) / f"train_{int(fraction*100)}pct"
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / 'model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(classifier, f)
        
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n  ✓ Saved model and metrics to: {output_path}")
    
    # Create summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: PERFORMANCE vs TRAINING SET SIZE")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(all_results)
    
    print(results_df.to_string(index=False))
    
    # Save summary
    summary_path = Path(OUTPUT_DIR)
    summary_path.mkdir(parents=True, exist_ok=True)
    
    summary_csv = summary_path / 'training_size_comparison.csv'
    results_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved summary to: {summary_csv}")
    
    # Create markdown report
    report_file = summary_path / 'TRAINING_SIZE_REPORT.md'
    with open(report_file, 'w') as f:
        f.write("# Random Forest Performance vs Training Set Size\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report shows how Random Forest classifier performance changes with different training set sizes.\n\n")
        
        f.write("## Results\n\n")
        f.write("| Train % | Train Records | Train Samples | Accuracy | F1 Weighted | F1 Intentional | F1 Unintentional |\n")
        f.write("|---------|---------------|---------------|----------|-------------|----------------|------------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['train_fraction']*100:.0f}% | {row['train_records']} | {row['train_samples']} | ")
            f.write(f"{row['accuracy']:.4f} | {row['f1_weighted']:.4f} | ")
            f.write(f"{row['f1_intentional']:.4f} | {row['f1_unintentional']:.4f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best performance
        best_idx = results_df['f1_weighted'].idxmax()
        best_row = results_df.iloc[best_idx]
        
        f.write(f"- **Best Performance:** {best_row['train_fraction']*100:.0f}% training data\n")
        f.write(f"  - F1 Weighted: {best_row['f1_weighted']:.4f}\n")
        f.write(f"  - Accuracy: {best_row['accuracy']:.4f}\n\n")
        
        # Compare 1% vs 100%
        first_row = results_df.iloc[0]
        last_row = results_df.iloc[-1]
        
        f.write(f"- **Performance Gain (1% → 100%):**\n")
        f.write(f"  - F1 Weighted: {first_row['f1_weighted']:.4f} → {last_row['f1_weighted']:.4f} ")
        f.write(f"({(last_row['f1_weighted'] - first_row['f1_weighted']):.4f} improvement)\n")
        f.write(f"  - Accuracy: {first_row['accuracy']:.4f} → {last_row['accuracy']:.4f} ")
        f.write(f"({(last_row['accuracy'] - first_row['accuracy']):.4f} improvement)\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("For each training fraction, the following files are saved:\n\n")
        f.write("- `train_XX pct/model.pkl` - Trained model\n")
        f.write("- `train_XXpct/metrics.json` - Detailed metrics\n\n")
        f.write("- `training_size_comparison.csv` - Summary comparison table\n")
        f.write("- `TRAINING_SIZE_REPORT.md` - This report\n")
    
    print(f"✓ Saved report to: {report_file}")
    
    print(f"\n{'='*70}")
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
