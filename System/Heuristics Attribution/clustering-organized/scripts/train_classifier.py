#!/usr/bin/env python3
"""
Random Forest Classifier for Intent Attribution

This script trains a Random Forest classifier to predict whether a feature
manipulation is intentional (1) or unintentional (-1) based on:
- Original feature values
- New (manipulated) feature values  
- Feature names

The classifier is evaluated against baseline strategies and LLM models.
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
            class_weight='balanced'  # Handle class imbalance
        )
        self.feature_encoders = {}
        self.value_encoders = {}
        self.feature_names = []
        
    def create_features(self, df):
        """
        Create feature vectors from original_value, new_value, and feature_name.
        
        Features created:
        - feature_name (encoded)
        - original_value (encoded)
        - new_value (encoded)
        - value_changed (binary: 1 if values differ, 0 if same)
        - feature-specific indicators (one-hot encoding of feature_name)
        """
        # Reset index to ensure proper alignment
        df = df.reset_index(drop=True)
        
        features = pd.DataFrame()
        
        # Encode feature names
        if 'feature_name' not in self.feature_encoders:
            self.feature_encoders['feature_name'] = LabelEncoder()
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].fit_transform(
                df['feature_name'].astype(str)
            )
        else:
            features['feature_name_encoded'] = self.feature_encoders['feature_name'].transform(
                df['feature_name'].astype(str)
            )
        
        # Encode original values
        if 'original_value' not in self.value_encoders:
            self.value_encoders['original_value'] = LabelEncoder()
            features['original_value_encoded'] = self.value_encoders['original_value'].fit_transform(
                df['original_value'].astype(str)
            )
        else:
            # Handle unseen values in test set
            known_values = set(self.value_encoders['original_value'].classes_)
            df_copy = df.copy()
            df_copy['original_value'] = df_copy['original_value'].astype(str).apply(
                lambda x: x if x in known_values else 'UNKNOWN'
            )
            if 'UNKNOWN' not in known_values:
                # Add UNKNOWN to encoder
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
            # Handle unseen values in test set
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
        features['value_changed'] = (df['original_value'].astype(str) != df['new_value'].astype(str)).astype(int)
        
        # One-hot encode feature names for feature-specific patterns
        feature_dummies = pd.get_dummies(df['feature_name'].reset_index(drop=True), prefix='feat')
        
        # Ensure same columns for train and test
        if len(self.feature_names) == 0:
            # First time (training) - save all columns
            features = pd.concat([features.reset_index(drop=True), feature_dummies.reset_index(drop=True)], axis=1)
            self.feature_names = features.columns.tolist()
        else:
            # Test time - align columns with training
            for col in self.feature_names:
                if col.startswith('feat_') and col not in feature_dummies.columns:
                    feature_dummies[col] = 0
            
            # Remove any extra columns not in training
            feature_dummies = feature_dummies[[col for col in self.feature_names if col.startswith('feat_')]]
            
            features = pd.concat([features.reset_index(drop=True), feature_dummies.reset_index(drop=True)], axis=1)
            
            # Ensure column order matches training
            features = features[self.feature_names]
        
        return features
    
    def train(self, X_train, y_train):
        """Train the Random Forest classifier."""
        print(f"\n{'='*60}")
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Class distribution in training set:")
        print(f"  Intentional (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
        print(f"  Unintentional (-1): {(y_train == -1).sum()} ({(y_train == -1).mean()*100:.1f}%)")
        
        self.model.fit(X_train, y_train)
        print("\n✓ Model training complete!")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return feature_importance
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_intentional = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        recall_intentional = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_intentional = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        precision_unintentional = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)
        recall_unintentional = recall_score(y_test, y_pred, pos_label=-1, zero_division=0)
        f1_unintentional = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_intentional': precision_intentional,
            'recall_intentional': recall_intentional,
            'f1_intentional': f1_intentional,
            'precision_unintentional': precision_unintentional,
            'recall_unintentional': recall_unintentional,
            'f1_unintentional': f1_unintentional,
        }
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Test samples: {len(X_test)}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Weighted Precision: {precision:.4f}")
        print(f"  Weighted Recall: {recall:.4f}")
        print(f"  Weighted F1: {f1:.4f}")
        
        print(f"\nIntentional (1) Class Metrics:")
        print(f"  Precision: {precision_intentional:.4f}")
        print(f"  Recall: {recall_intentional:.4f}")
        print(f"  F1: {f1_intentional:.4f}")
        
        print(f"\nUnintentional (-1) Class Metrics:")
        print(f"  Precision: {precision_unintentional:.4f}")
        print(f"  Recall: {recall_unintentional:.4f}")
        print(f"  F1: {f1_unintentional:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"                  Unint  Intent")
        print(f"Actual Unint      {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"       Intent     {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        return metrics, y_pred


def load_data(data_dir):
    """Load and prepare the data."""
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
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
    
    for idx in range(len(masks)):
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(masks)} records", end='\r')
        
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
    print(f"\nClass distribution:")
    print(f"  Intentional (1): {(df['intent_label'] == 1).sum()} ({(df['intent_label'] == 1).mean()*100:.1f}%)")
    print(f"  Unintentional (-1): {(df['intent_label'] == -1).sum()} ({(df['intent_label'] == -1).mean()*100:.1f}%)")
    
    return df


def split_by_record_id(df, test_size=0.3, random_state=42):
    """
    Split data by record_id to avoid data leakage.
    All changes from the same record go into either train or test, not both.
    """
    print(f"\n{'='*60}")
    print("SPLITTING DATA BY RECORD_ID")
    print(f"{'='*60}")
    
    # Get unique record IDs
    unique_records = df['record_id'].unique()
    print(f"Total unique records: {len(unique_records)}")
    
    # Split record IDs
    train_records, test_records = train_test_split(
        unique_records, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Create train/test sets
    train_df = df[df['record_id'].isin(train_records)].copy()
    test_df = df[df['record_id'].isin(test_records)].copy()
    
    print(f"\nTrain set:")
    print(f"  Records: {len(train_records)}")
    print(f"  Feature changes: {len(train_df)}")
    print(f"  Intentional: {(train_df['intent_label'] == 1).sum()}")
    print(f"  Unintentional: {(train_df['intent_label'] == -1).sum()}")
    
    print(f"\nTest set:")
    print(f"  Records: {len(test_records)}")
    print(f"  Feature changes: {len(test_df)}")
    print(f"  Intentional: {(test_df['intent_label'] == 1).sum()}")
    print(f"  Unintentional: {(test_df['intent_label'] == -1).sum()}")
    
    return train_df, test_df


def compare_with_baselines_and_llms(classifier_metrics, results_dir):
    """Compare classifier performance with baselines and LLMs."""
    print(f"\n{'='*60}")
    print("COMPARING WITH BASELINES AND LLMS")
    print(f"{'='*60}")
    
    results_path = Path(results_dir)
    
    # Load baseline results
    baseline_file = results_path / 'baselines' / 'baseline_comparison.csv'
    if baseline_file.exists():
        baselines = pd.read_csv(baseline_file)
        print(f"\n✓ Loaded baseline results: {len(baselines)} strategies")
    else:
        print(f"\n✗ Baseline file not found: {baseline_file}")
        baselines = None
    
    # Load LLM results
    llm_file = results_path / 'analysis' / 'analysis_comparison' / 'summary_csvs' / 'overall_comparison.csv'
    if llm_file.exists():
        llms = pd.read_csv(llm_file)
        print(f"✓ Loaded LLM results: {len(llms)} models")
    else:
        print(f"✗ LLM file not found: {llm_file}")
        llms = None
    
    # Create comparison
    comparison = []
    
    # Add classifier
    comparison.append({
        'Model/Strategy': 'Random Forest Classifier',
        'Type': 'ML Classifier',
        'Accuracy': classifier_metrics['accuracy'],
        'F1_Weighted': classifier_metrics['f1_weighted'],
        'F1_Intentional': classifier_metrics['f1_intentional'],
        'F1_Unintentional': classifier_metrics['f1_unintentional'],
        'Precision_Weighted': classifier_metrics['precision_weighted'],
        'Recall_Weighted': classifier_metrics['recall_weighted']
    })
    
    # Add baselines
    if baselines is not None:
        # Group by strategy (handle multiple runs)
        baseline_summary = baselines.groupby('Strategy').agg({
            'Accuracy': 'mean',
            'Macro_F1': 'mean',
            'Macro_Precision': 'mean',
            'Macro_Recall': 'mean'
        }).reset_index()
        
        for _, row in baseline_summary.iterrows():
            comparison.append({
                'Model/Strategy': row['Strategy'],
                'Type': 'Baseline',
                'Accuracy': row['Accuracy'],
                'F1_Weighted': row['Macro_F1'],
                'F1_Intentional': np.nan,
                'F1_Unintentional': np.nan,
                'Precision_Weighted': row['Macro_Precision'],
                'Recall_Weighted': row['Macro_Recall']
            })
    
    # Add LLMs
    if llms is not None:
        for _, row in llms.iterrows():
            model_name = f"{row['Model']} {row['Trial']}"
            comparison.append({
                'Model/Strategy': model_name,
                'Type': 'LLM',
                'Accuracy': np.nan,  # Not in this file
                'F1_Weighted': row['Macro F1'],
                'F1_Intentional': row['F1 (1)'],
                'F1_Unintentional': row['F1 (-1)'],
                'Precision_Weighted': row['Macro Precision'],
                'Recall_Weighted': row['Macro Recall']
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Sort by F1_Weighted
    comparison_df = comparison_df.sort_values('F1_Weighted', ascending=False)
    
    # Calculate ranks
    print(f"\n{'='*60}")
    print("PERFORMANCE RANKING (by Weighted F1)")
    print(f"{'='*60}")
    
    for idx, row in comparison_df.iterrows():
        rank = comparison_df.index.get_loc(idx) + 1
        print(f"{rank:2d}. {row['Model/Strategy']:40s} ({row['Type']:12s}) - F1: {row['F1_Weighted']:.4f}")
    
    # Find classifier rank
    classifier_rank = comparison_df[comparison_df['Type'] == 'ML Classifier'].index[0] + 1
    total_models = len(comparison_df)
    
    print(f"\n{'='*60}")
    print(f"Random Forest Classifier Rank: {classifier_rank}/{total_models}")
    print(f"{'='*60}")
    
    # Beat statistics
    classifier_f1 = classifier_metrics['f1_weighted']
    
    if baselines is not None:
        baseline_summary = baselines.groupby('Strategy').agg({'Macro_F1': 'mean'}).reset_index()
        baseline_f1s = baseline_summary['Macro_F1'].values
        baselines_beaten = (classifier_f1 > baseline_f1s).sum()
        print(f"\nBaselines beaten: {baselines_beaten}/{len(baseline_f1s)}")
    
    if llms is not None:
        llm_f1s = llms['Macro F1'].values
        llms_beaten = (classifier_f1 > llm_f1s).sum()
        print(f"LLMs beaten: {llms_beaten}/{len(llms)}")
    
    return comparison_df


def save_results(classifier, metrics, feature_importance, comparison_df, output_dir):
    """Save all results and the trained model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save model
    model_file = output_path / 'random_forest_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"✓ Saved model: {model_file}")
    
    # Save metrics
    metrics_file = output_path / 'classifier_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics: {metrics_file}")
    
    # Save feature importance
    importance_file = output_path / 'feature_importance.csv'
    feature_importance.to_csv(importance_file, index=False)
    print(f"✓ Saved feature importance: {importance_file}")
    
    # Save comparison
    comparison_file = output_path / 'classifier_vs_baselines_llms.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"✓ Saved comparison: {comparison_file}")
    
    # Create markdown report
    report_file = output_path / 'CLASSIFIER_REPORT.md'
    with open(report_file, 'w') as f:
        f.write("# Random Forest Classifier - Intent Attribution Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Configuration\n\n")
        f.write("- **Algorithm:** Random Forest\n")
        f.write("- **Number of Trees:** 100\n")
        f.write("- **Max Depth:** 10\n")
        f.write("- **Min Samples Split:** 10\n")
        f.write("- **Min Samples Leaf:** 5\n")
        f.write("- **Class Weight:** Balanced\n")
        f.write("- **Train/Test Split:** 70/30 (by record_id)\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write("### Overall Performance\n\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.4f}\n")
        f.write(f"- **Weighted Precision:** {metrics['precision_weighted']:.4f}\n")
        f.write(f"- **Weighted Recall:** {metrics['recall_weighted']:.4f}\n")
        f.write(f"- **Weighted F1:** {metrics['f1_weighted']:.4f}\n\n")
        
        f.write("### Intentional Class (1)\n\n")
        f.write(f"- **Precision:** {metrics['precision_intentional']:.4f}\n")
        f.write(f"- **Recall:** {metrics['recall_intentional']:.4f}\n")
        f.write(f"- **F1:** {metrics['f1_intentional']:.4f}\n\n")
        
        f.write("### Unintentional Class (-1)\n\n")
        f.write(f"- **Precision:** {metrics['precision_unintentional']:.4f}\n")
        f.write(f"- **Recall:** {metrics['recall_unintentional']:.4f}\n")
        f.write(f"- **F1:** {metrics['f1_unintentional']:.4f}\n\n")
        
        f.write("## Top 10 Most Important Features\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|------------|\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"| {idx+1} | {row['feature']} | {row['importance']:.6f} |\n")
        f.write("\n")
        
        f.write("## Comparison with Baselines and LLMs\n\n")
        f.write("### Top 10 Performers (by Weighted F1)\n\n")
        f.write("| Rank | Model/Strategy | Type | F1 Weighted | Accuracy |\n")
        f.write("|------|----------------|------|-------------|----------|\n")
        for idx, row in comparison_df.head(10).iterrows():
            rank = comparison_df.index.get_loc(idx) + 1
            f.write(f"| {rank} | {row['Model/Strategy']} | {row['Type']} | {row['F1_Weighted']:.4f} | {row['Accuracy']:.4f} |\n")
        f.write("\n")
        
        # Find classifier rank
        classifier_rank = comparison_df[comparison_df['Type'] == 'ML Classifier'].index[0] + 1
        total_models = len(comparison_df)
        f.write(f"**Random Forest Classifier Rank:** {classifier_rank}/{total_models}\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `random_forest_model.pkl` - Trained model (can be loaded with pickle)\n")
        f.write("- `classifier_metrics.json` - Detailed metrics in JSON format\n")
        f.write("- `feature_importance.csv` - Feature importance scores\n")
        f.write("- `classifier_vs_baselines_llms.csv` - Full comparison table\n")
        f.write("- `CLASSIFIER_REPORT.md` - This report\n\n")
        
        f.write("## Usage Example\n\n")
        f.write("```python\n")
        f.write("import pickle\n")
        f.write("import pandas as pd\n\n")
        f.write("# Load the model\n")
        f.write("with open('random_forest_model.pkl', 'rb') as f:\n")
        f.write("    classifier = pickle.load(f)\n\n")
        f.write("# Prepare your data (same format as training)\n")
        f.write("# df should have: record_id, feature_name, original_value, new_value\n")
        f.write("X = classifier.create_features(df)\n\n")
        f.write("# Make predictions\n")
        f.write("predictions = classifier.predict(X)\n")
        f.write("# predictions will be 1 (intentional) or -1 (unintentional)\n")
        f.write("```\n")
    
    print(f"✓ Saved report: {report_file}")
    print(f"\nAll results saved to: {output_path}")


def main():
    """Main execution function."""
    # Configuration
    DATA_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251031_211812'
    RESULTS_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results'
    OUTPUT_DIR = '/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/results/classifier'
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    print("\n" + "="*60)
    print(" RANDOM FOREST INTENT ATTRIBUTION CLASSIFIER")
    print("="*60)
    print(f"Dataset: Adult Income Dataset")
    print(f"Task: Binary Classification (Intentional vs Unintentional)")
    print(f"Train/Test Split: {(1-TEST_SIZE)*100:.0f}/{TEST_SIZE*100:.0f} by record_id")
    print("="*60)
    
    # Load data
    df = load_data(DATA_DIR)
    
    # Split data by record_id
    train_df, test_df = split_by_record_id(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Initialize classifier
    classifier = IntentAttributionClassifier(random_state=RANDOM_STATE)
    
    # Create features
    print(f"\n{'='*60}")
    print("CREATING FEATURES")
    print(f"{'='*60}")
    print("Creating training features...")
    X_train = classifier.create_features(train_df)
    y_train = train_df['intent_label'].values
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    print("Creating test features...")
    X_test = classifier.create_features(test_df)
    y_test = test_df['intent_label'].values
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    print(f"✓ Feature creation complete!")
    print(f"  Training features shape: {X_train.shape}")
    print(f"  Test features shape: {X_test.shape}")
    
    # Train model
    feature_importance = classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred = classifier.evaluate(X_test, y_test)
    
    # Compare with baselines and LLMs
    comparison_df = compare_with_baselines_and_llms(metrics, RESULTS_DIR)
    
    # Save results
    save_results(classifier, metrics, feature_importance, comparison_df, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("✓ CLASSIFIER TRAINING COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
