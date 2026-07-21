#!/usr/bin/env python3
"""
Smart Sampling Strategy for Intent Attribution

Goal: Achieve 85% F1 with <1% training data using:
1. Stratified sampling by feature type and intent label
2. Diversity-based sample selection
3. Enhanced feature engineering (change magnitude, ratios, etc.)
4. Iterative training with uncertainty sampling

This script will test various smart sampling strategies with 0.5%, 0.75%, and 1% of data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.cluster import KMeans
import json
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EnhancedIntentClassifier:
    """Intent classifier with enhanced feature engineering."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=200,  # More trees for better performance with less data
            max_depth=15,      # Deeper trees
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.feature_encoders = {}
        self.value_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self._dummy_columns = []
        
    def create_enhanced_features(self, df, fit=False):
        """Create enhanced feature vectors with change magnitude and patterns."""
        df = df.reset_index(drop=True)
        features = pd.DataFrame()
        
        # 1. Encode feature names
        if fit or 'feature_name' not in self.feature_encoders:
            self.feature_encoders['feature_name'] = LabelEncoder()
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].fit_transform(
                df['feature_name'].astype(str)
            )
        else:
            known_features = set(self.feature_encoders['feature_name'].classes_)
            df_copy = df.copy()
            df_copy['feature_name'] = df_copy['feature_name'].astype(str).apply(
                lambda x: x if x in known_features else 'UNKNOWN'
            )
            if 'UNKNOWN' not in known_features:
                self.feature_encoders['feature_name'].classes_ = np.append(
                    self.feature_encoders['feature_name'].classes_, 'UNKNOWN'
                )
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].transform(
                df_copy['feature_name']
            )
        
        # 2. Encode original values
        if fit or 'original_value' not in self.value_encoders:
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
        
        # 3. Encode new values
        if fit or 'new_value' not in self.value_encoders:
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
        
        # 4. ENHANCED FEATURES: Change magnitude and patterns
        
        # Convert to numeric where possible
        df['original_numeric'] = pd.to_numeric(df['original_value'], errors='coerce').fillna(0)
        df['new_numeric'] = pd.to_numeric(df['new_value'], errors='coerce').fillna(0)
        
        # Change magnitude (absolute difference)
        features['change_magnitude'] = np.abs(df['new_numeric'] - df['original_numeric'])
        
        # Relative change (percentage)
        features['relative_change'] = np.where(
            df['original_numeric'] != 0,
            (df['new_numeric'] - df['original_numeric']) / (df['original_numeric'] + 1e-10),
            0
        )
        
        # Direction of change
        features['change_direction'] = np.sign(df['new_numeric'] - df['original_numeric'])
        
        # Value range features
        features['original_log'] = np.log1p(np.abs(df['original_numeric']))
        features['new_log'] = np.log1p(np.abs(df['new_numeric']))
        
        # Is the change a zero/null?
        features['original_is_zero'] = (df['original_numeric'] == 0).astype(int)
        features['new_is_zero'] = (df['new_numeric'] == 0).astype(int)
        
        # Value magnitude
        features['original_magnitude'] = np.abs(df['original_numeric'])
        features['new_magnitude'] = np.abs(df['new_numeric'])
        
        # Feature type indicators (based on feature name patterns)
        features['is_length_feature'] = df['feature_name'].str.contains('length', case=False).astype(int)
        features['is_count_feature'] = df['feature_name'].str.contains('count', case=False).astype(int)
        features['is_has_feature'] = df['feature_name'].str.contains('has_', case=False).astype(int)
        
        # One-hot encode feature names
        feature_dummies = pd.get_dummies(df['feature_name'], prefix='feat')
        
        if fit:
            self._dummy_columns = feature_dummies.columns.tolist()
        else:
            # Ensure test set has same columns as training set
            for col in self._dummy_columns:
                if col not in feature_dummies.columns:
                    feature_dummies[col] = 0
            feature_dummies = feature_dummies[self._dummy_columns]
        
        features = pd.concat([features, feature_dummies], axis=1)
        
        # Store feature names
        if fit:
            self.feature_names = features.columns.tolist()
        
        return features
    
    def fit(self, df):
        """Train the model."""
        X = self.create_enhanced_features(df, fit=True)
        y = df['intent_label'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, df):
        """Make predictions."""
        X = self.create_enhanced_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df):
        """Get prediction probabilities."""
        X = self.create_enhanced_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)


def load_data(data_dir):
    """Load and prepare the dataset."""
    print("\nLoading data...")
    
    data_path = Path(data_dir)
    masks_file = data_path / 'masks.csv'
    manipulated_file = data_path / 'manipulated_records.csv'
    correct_file = data_path.parent / 'correct_records.csv'
    
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
        original_record_idx = idx // 3
        
        for feature in feature_cols:
            intent_label = masks.iloc[idx][feature]
            
            if intent_label == 0:
                continue
            
            original_value = correct.iloc[original_record_idx][feature]
            new_value = manipulated.iloc[idx][feature]
            
            records.append({
                'record_id': original_record_idx,
                'manipulated_record_id': idx,
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


def stratified_sample_selection(train_df, fraction, random_state=42):
    """
    Smart stratified sampling that ensures:
    1. Balanced representation of intent labels
    2. Coverage of all feature types
    3. Diverse examples using clustering
    """
    print(f"\n{'='*70}")
    print(f"STRATIFIED SAMPLING: {fraction*100:.2f}% of training data")
    print(f"{'='*70}")
    
    unique_records = train_df['record_id'].unique()
    target_records = int(len(unique_records) * fraction)
    
    print(f"Target: {target_records} records from {len(unique_records)} total")
    
    # Strategy: Sample proportionally from each feature type and intent combination
    sampled_record_ids = set()
    
    # Get feature type categories
    train_df['feature_type'] = 'other'
    train_df.loc[train_df['feature_name'].str.contains('length', case=False), 'feature_type'] = 'length'
    train_df.loc[train_df['feature_name'].str.contains('count', case=False), 'feature_type'] = 'count'
    train_df.loc[train_df['feature_name'].str.contains('has_', case=False), 'feature_type'] = 'binary'
    
    # Group by feature type and intent label
    groups = train_df.groupby(['feature_type', 'intent_label'])
    
    print("\nSampling strategy:")
    for (feat_type, intent), group_df in groups:
        group_records = group_df['record_id'].unique()
        n_group_records = len(group_records)
        
        # Proportional sampling
        n_to_sample = max(1, int(n_group_records * fraction))
        
        np.random.seed(random_state)
        sampled = np.random.choice(group_records, size=min(n_to_sample, n_group_records), replace=False)
        sampled_record_ids.update(sampled)
        
        intent_str = "Intentional" if intent == 1 else "Unintentional"
        print(f"  {feat_type:8s} + {intent_str:13s}: {n_to_sample:3d}/{n_group_records:4d} records")
    
    # If we need more records, add random ones
    if len(sampled_record_ids) < target_records:
        remaining_records = set(unique_records) - sampled_record_ids
        n_additional = target_records - len(sampled_record_ids)
        np.random.seed(random_state)
        additional = np.random.choice(list(remaining_records), size=n_additional, replace=False)
        sampled_record_ids.update(additional)
        print(f"\n  Added {n_additional} random records to reach target")
    
    # If we have too many, randomly remove some
    elif len(sampled_record_ids) > target_records:
        np.random.seed(random_state)
        sampled_record_ids = set(np.random.choice(list(sampled_record_ids), size=target_records, replace=False))
    
    sampled_df = train_df[train_df['record_id'].isin(sampled_record_ids)].copy()
    
    print(f"\nFinal sample:")
    print(f"  Records: {len(sampled_record_ids)}")
    print(f"  Samples: {len(sampled_df)}")
    print(f"  Intentional: {(sampled_df['intent_label'] == 1).sum()}")
    print(f"  Unintentional: {(sampled_df['intent_label'] == -1).sum()}")
    
    return sampled_df


def diversity_based_sampling(train_df, fraction, random_state=42):
    """
    Diversity-based sampling using clustering to select diverse examples.
    """
    print(f"\n{'='*70}")
    print(f"DIVERSITY-BASED SAMPLING: {fraction*100:.2f}% of training data")
    print(f"{'='*70}")
    
    unique_records = train_df['record_id'].unique()
    target_records = int(len(unique_records) * fraction)
    
    print(f"Target: {target_records} records from {len(unique_records)} total")
    
    # Create simple feature representation for clustering
    record_features = []
    for rec_id in unique_records:
        rec_df = train_df[train_df['record_id'] == rec_id]
        
        # Aggregate features per record
        feat_vector = [
            len(rec_df),  # Number of changes
            (rec_df['intent_label'] == 1).sum(),  # Intentional changes
            (rec_df['intent_label'] == -1).sum(),  # Unintentional changes
            rec_df['feature_name'].str.contains('length').sum(),  # Length features
            rec_df['feature_name'].str.contains('count').sum(),   # Count features
            rec_df['feature_name'].str.contains('has_').sum(),    # Binary features
        ]
        record_features.append(feat_vector)
    
    # Cluster records
    X_cluster = np.array(record_features)
    n_clusters = min(target_records, len(unique_records))
    
    print(f"Clustering {len(unique_records)} records into {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # Select one record from each cluster (closest to centroid)
    sampled_record_ids = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_records = unique_records[cluster_mask]
        cluster_features = X_cluster[cluster_mask]
        
        if len(cluster_records) == 0:
            continue
        
        # Find record closest to centroid
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        sampled_record_ids.append(cluster_records[closest_idx])
    
    sampled_df = train_df[train_df['record_id'].isin(sampled_record_ids)].copy()
    
    print(f"\nFinal sample:")
    print(f"  Records: {len(sampled_record_ids)}")
    print(f"  Samples: {len(sampled_df)}")
    print(f"  Intentional: {(sampled_df['intent_label'] == 1).sum()}")
    print(f"  Unintentional: {(sampled_df['intent_label'] == -1).sum()}")
    
    return sampled_df


def train_test_split_by_record(df, test_size=0.3, random_state=42):
    """Split dataset by record_id to prevent data leakage."""
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


def evaluate_model(model, test_df, name):
    """Evaluate model and return metrics."""
    print(f"\n{'='*70}")
    print(f"EVALUATING: {name}")
    print(f"{'='*70}")
    
    y_true = test_df['intent_label'].values
    y_pred = model.predict(test_df)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 Score (weighted): {f1_weighted:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    print(f"  Unintentional (-1): {f1_per_class[0]:.4f}")
    print(f"  Intentional (1): {f1_per_class[1]:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Unintentional', 'Intentional']))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_weighted': f1_weighted,
        'f1_intentional': f1_per_class[1],
        'f1_unintentional': f1_per_class[0],
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812'
    OUTPUT_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/smart_sampling'
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    # Test fractions: 0.5%, 0.75%, 1.0%
    TRAIN_FRACTIONS = [0.005, 0.0075, 0.01]
    
    print("\n" + "="*70)
    print("SMART SAMPLING FOR INTENT ATTRIBUTION")
    print("Goal: Achieve 85% F1 with <1% training data")
    print("="*70)
    
    # Load data
    df = load_data(DATA_DIR)
    
    # Split train/test
    train_df, test_df = train_test_split_by_record(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Store all results
    all_results = []
    
    # Test each fraction with both sampling strategies
    for fraction in TRAIN_FRACTIONS:
        print(f"\n\n{'#'*70}")
        print(f"# TESTING WITH {fraction*100:.2f}% OF TRAINING DATA")
        print(f"{'#'*70}\n")
        
        # Strategy 1: Stratified Sampling
        print("\n" + "-"*70)
        print("STRATEGY 1: STRATIFIED SAMPLING")
        print("-"*70)
        
        stratified_sample = stratified_sample_selection(train_df, fraction, RANDOM_STATE)
        
        print("\nTraining classifier with stratified sample...")
        stratified_model = EnhancedIntentClassifier(random_state=RANDOM_STATE)
        stratified_model.fit(stratified_sample)
        
        stratified_metrics = evaluate_model(stratified_model, test_df, f"Stratified {fraction*100:.2f}%")
        
        # Save results
        all_results.append({
            'strategy': 'stratified',
            'train_fraction': fraction,
            'train_records': stratified_sample['record_id'].nunique(),
            'train_samples': len(stratified_sample),
            **stratified_metrics
        })
        
        # Strategy 2: Diversity-Based Sampling
        print("\n" + "-"*70)
        print("STRATEGY 2: DIVERSITY-BASED SAMPLING")
        print("-"*70)
        
        diversity_sample = diversity_based_sampling(train_df, fraction, RANDOM_STATE)
        
        print("\nTraining classifier with diversity sample...")
        diversity_model = EnhancedIntentClassifier(random_state=RANDOM_STATE)
        diversity_model.fit(diversity_sample)
        
        diversity_metrics = evaluate_model(diversity_model, test_df, f"Diversity {fraction*100:.2f}%")
        
        # Save results
        all_results.append({
            'strategy': 'diversity',
            'train_fraction': fraction,
            'train_records': diversity_sample['record_id'].nunique(),
            'train_samples': len(diversity_sample),
            **diversity_metrics
        })
        
        # Save models
        output_path = Path(OUTPUT_DIR) / f"train_{fraction*100:.2f}pct"
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'stratified_model.pkl', 'wb') as f:
            pickle.dump(stratified_model, f)
        
        with open(output_path / 'diversity_model.pkl', 'wb') as f:
            pickle.dump(diversity_model, f)
        
        with open(output_path / 'stratified_metrics.json', 'w') as f:
            json.dump(all_results[-2], f, indent=2)
        
        with open(output_path / 'diversity_metrics.json', 'w') as f:
            json.dump(all_results[-1], f, indent=2)
        
        print(f"\n✓ Saved models and metrics to {output_path}")
    
    # Create summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY: SMART SAMPLING PERFORMANCE")
    print(f"{'='*70}\n")
    
    results_df = pd.DataFrame(all_results)
    print(results_df[['strategy', 'train_fraction', 'train_records', 'train_samples', 
                      'accuracy', 'f1_weighted', 'f1_intentional', 'f1_unintentional']].to_string(index=False))
    
    # Save summary
    summary_path = Path(OUTPUT_DIR)
    summary_path.mkdir(parents=True, exist_ok=True)
    
    summary_csv = summary_path / 'smart_sampling_comparison.csv'
    results_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Saved summary to: {summary_csv}")
    
    # Find best result
    best_idx = results_df['f1_weighted'].idxmax()
    best_result = results_df.iloc[best_idx]
    
    print(f"\n{'='*70}")
    print("BEST PERFORMANCE:")
    print(f"{'='*70}")
    print(f"Strategy: {best_result['strategy']}")
    print(f"Training fraction: {best_result['train_fraction']*100:.2f}%")
    print(f"Training records: {best_result['train_records']}")
    print(f"Training samples: {best_result['train_samples']}")
    print(f"F1 Weighted: {best_result['f1_weighted']:.4f}")
    print(f"F1 Intentional: {best_result['f1_intentional']:.4f}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    
    if best_result['f1_weighted'] >= 0.85:
        print(f"\n🎉 SUCCESS! Achieved {best_result['f1_weighted']:.4f} F1 (>= 85%) with {best_result['train_fraction']*100:.2f}% data!")
    else:
        print(f"\n⚠️  Did not reach 85% F1 target. Best: {best_result['f1_weighted']:.4f}")
        print("   Consider: More feature engineering, active learning, or semi-supervised learning")
    
    print(f"\n{'='*70}")
    print("✓ ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
