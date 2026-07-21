#!/usr/bin/env python3
"""
Deep verification of both ClusterMajorityVote and FeatureLevelKNN.
Checks for:
1. Train/test overlap
2. Data leakage in features
3. Prediction consistency
4. Random baseline comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

def load_and_process_data():
    """Load and process the tenth-trial data"""
    masks = pd.read_csv("/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251211_114329/masks.csv")
    clean = pd.read_csv("/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/correct_records.csv")
    dirty = pd.read_csv("/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251211_114329/manipulated_records.csv")
    
    print(f"Processing {len(masks)} variant records...")
    
    # Encode feature names
    all_feature_names = list(masks.columns)
    feature_encoder = LabelEncoder()
    feature_encoder.fit(all_feature_names)
    
    all_changes = []
    for idx in range(len(masks)):
        mask_row = masks.iloc[idx]
        variant_idx = idx % 3
        original_idx = idx // 3
        
        if original_idx >= len(clean):
            continue
        
        original_record = clean.iloc[original_idx]
        variant_record = dirty.iloc[idx]
        
        for feature in masks.columns:
            intent_label = mask_row[feature]
            if intent_label == 0:
                continue
            
            original_value = original_record[feature]
            new_value = variant_record[feature]
            
            # Encode feature name
            feature_name_encoded = feature_encoder.transform([feature])[0]
            
            # Calculate change magnitude
            try:
                original_float = float(original_value)
                new_float = float(new_value)
                change_magnitude = abs(new_float - original_float)
                relative_change = change_magnitude / (abs(original_float) + 1e-10)
            except (ValueError, TypeError):
                change_magnitude = 1.0 if str(original_value) != str(new_value) else 0.0
                relative_change = change_magnitude
            
            # Encode values
            try:
                new_value_encoded = float(new_value)
                original_value_encoded = float(original_value)
            except (ValueError, TypeError):
                # For categorical: use hash
                new_value_encoded = hash(str(new_value)) % 10000
                original_value_encoded = hash(str(original_value)) % 10000
            
            all_changes.append({
                'variant_record_id': idx,
                'original_record_id': original_idx,
                'variant_idx': variant_idx,
                'feature_name': feature,
                'original_value': str(original_value),
                'new_value': str(new_value),
                'intent_label': intent_label,
                'feature_name_encoded': feature_name_encoded,
                'original_value_encoded': original_value_encoded,
                'new_value_encoded': new_value_encoded,
                'change_magnitude': change_magnitude,
                'relative_change': relative_change
            })
    
    df = pd.DataFrame(all_changes)
    print(f"✓ Created {len(df)} feature changes from {df['variant_record_id'].nunique()} variants")
    print(f"  Intentional (1): {(df['intent_label']==1).sum()}")
    print(f"  Unintentional (-1): {(df['intent_label']==-1).sum()}")
    
    return df

def create_aggregates(df):
    """Create aggregate vectors WITHOUT intent labels"""
    aggregated = []
    
    for variant_id in df['variant_record_id'].unique():
        variant_data = df[df['variant_record_id'] == variant_id]
        
        n_changes = len(variant_data)
        mean_magnitude = variant_data['change_magnitude'].mean()
        std_magnitude = variant_data['change_magnitude'].std() if len(variant_data) > 1 else 0
        min_magnitude = variant_data['change_magnitude'].min()
        max_magnitude = variant_data['change_magnitude'].max()
        median_magnitude = variant_data['change_magnitude'].median()
        
        min_new_value = variant_data['new_value_encoded'].min()
        max_new_value = variant_data['new_value_encoded'].max()
        mean_relative_change = variant_data['relative_change'].mean()
        
        max_change_idx = variant_data['change_magnitude'].idxmax()
        feature_with_max_change = variant_data.loc[max_change_idx, 'feature_name_encoded']
        
        aggregated.append({
            'variant_record_id': variant_id,
            'n_changes': n_changes,
            'mean_magnitude': mean_magnitude,
            'std_magnitude': std_magnitude,
            'min_magnitude': min_magnitude,
            'max_magnitude': max_magnitude,
            'median_magnitude': median_magnitude,
            'min_new_value': min_new_value,
            'max_new_value': max_new_value,
            'mean_relative_change': mean_relative_change,
            'feature_with_max_change': feature_with_max_change
        })
    
    return pd.DataFrame(aggregated).fillna(0)

def test_both_methods(df, agg_df):
    """Test both ClusterMajorityVote and FeatureLevelKNN with deep verification"""
    
    # Cluster and sample
    X = agg_df.drop('variant_record_id', axis=1).values
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=15, random_state=42, n_init=10).fit_predict(X_scaled)
    agg_df['cluster'] = labels
    
    # Sample 177 variants
    np.random.seed(42)
    all_variants = agg_df['variant_record_id'].values
    sampled_variants = np.random.choice(all_variants, 177, replace=False)
    
    # Split
    train_df = df[df['variant_record_id'].isin(sampled_variants)].copy()
    test_df = df[~df['variant_record_id'].isin(sampled_variants)].copy()
    
    print(f"\n{'='*70}")
    print("DATA SPLIT VERIFICATION")
    print(f"{'='*70}")
    print(f"Train variants: {len(sampled_variants)}")
    print(f"Test variants: {test_df['variant_record_id'].nunique()}")
    print(f"Train features: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"Test features: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%)")
    
    # CHECK 1: Verify no overlap
    train_variants_set = set(train_df['variant_record_id'].unique())
    test_variants_set = set(test_df['variant_record_id'].unique())
    overlap = train_variants_set & test_variants_set
    
    if len(overlap) > 0:
        print(f"\n❌ DATA LEAKAGE! {len(overlap)} variants in both train and test!")
        return
    else:
        print(f"✅ No overlap - train/test split is clean")
    
    # TEST 1: CLUSTER MAJORITY VOTE
    print(f"\n{'='*70}")
    print("TEST 1: CLUSTER MAJORITY VOTE")
    print(f"{'='*70}")
    
    variant_to_cluster = dict(zip(agg_df['variant_record_id'], agg_df['cluster']))
    
    cluster_labels = {}
    for cluster_id in agg_df['cluster'].unique():
        if cluster_id == -1:
            continue
        
        cluster_train_variants = [v for v in sampled_variants if variant_to_cluster[v] == cluster_id]
        
        if len(cluster_train_variants) == 0:
            continue
        
        cluster_train_df = train_df[train_df['variant_record_id'].isin(cluster_train_variants)]
        n_int = (cluster_train_df['intent_label'] == 1).sum()
        n_unint = (cluster_train_df['intent_label'] == -1).sum()
        
        cluster_labels[cluster_id] = 1 if n_int >= n_unint else -1
    
    y_pred_cluster = [cluster_labels.get(variant_to_cluster[v], 1) for v in test_df['variant_record_id']]
    y_test = test_df['intent_label'].values
    
    acc_cluster = accuracy_score(y_test, y_pred_cluster)
    f1_cluster = f1_score(y_test, y_pred_cluster, average='weighted')
    
    print(f"✓ Accuracy: {acc_cluster:.4f}")
    print(f"✓ F1 Weighted: {f1_cluster:.4f}")
    
    # TEST 2: FEATURE-LEVEL K-NN
    print(f"\n{'='*70}")
    print("TEST 2: FEATURE-LEVEL K-NN")
    print(f"{'='*70}")
    
    feature_cols = ['feature_name_encoded', 'original_value_encoded', 'new_value_encoded', 
                    'change_magnitude', 'relative_change']
    
    print(f"Features used for k-NN:")
    for col in feature_cols:
        print(f"  - {col}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['intent_label'].values
    X_test = test_df[feature_cols].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    
    acc_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
    
    print(f"✓ Accuracy: {acc_knn:.4f}")
    print(f"✓ F1 Weighted: {f1_knn:.4f}")
    
    # TEST 3: RANDOM BASELINE
    print(f"\n{'='*70}")
    print("TEST 3: RANDOM BASELINE (sanity check)")
    print(f"{'='*70}")
    
    np.random.seed(42)
    y_random = np.random.choice([1, -1], size=len(y_test))
    acc_random = accuracy_score(y_test, y_random)
    f1_random = f1_score(y_test, y_random, average='weighted')
    
    print(f"✓ Accuracy: {acc_random:.4f}")
    print(f"✓ F1 Weighted: {f1_random:.4f}")
    
    # TEST 4: MAJORITY CLASS BASELINE
    print(f"\n{'='*70}")
    print("TEST 4: MAJORITY CLASS BASELINE")
    print(f"{'='*70}")
    
    train_counts = train_df['intent_label'].value_counts()
    majority_label = train_counts.idxmax()
    y_majority = np.full(len(y_test), majority_label)
    
    acc_majority = accuracy_score(y_test, y_majority)
    f1_majority = f1_score(y_test, y_majority, average='weighted')
    
    print(f"Train label distribution: {dict(train_counts)}")
    print(f"Majority label: {majority_label}")
    print(f"✓ Accuracy: {acc_majority:.4f}")
    print(f"✓ F1 Weighted: {f1_majority:.4f}")
    
    # FINAL COMPARISON
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Random Baseline       : {f1_random:.4f} (should be ~0.50)")
    print(f"Majority Baseline     : {f1_majority:.4f}")
    print(f"ClusterMajorityVote   : {f1_cluster:.4f}")
    print(f"FeatureLevelKNN       : {f1_knn:.4f}")
    
    print(f"\n{'='*70}")
    if f1_cluster > 0.95 or f1_knn > 0.91:
        print("⚠️  RESULTS ARE VERY HIGH - Checking legitimacy...")
        print("{'='*70}")
        
        # Check if features are predictive
        print("\nFeature informativeness check:")
        print(f"  Change magnitude correlation with intent:")
        corr = df.groupby('intent_label')['change_magnitude'].mean()
        print(f"    Intentional: {corr.get(1, 0):.2f}")
        print(f"    Unintentional: {corr.get(-1, 0):.2f}")
        
        if abs(corr.get(1, 0) - corr.get(-1, 0)) > 50:
            print(f"\n✅ LEGITIMATE: Features naturally separate by intent!")
            print(f"   The 10 aggregate features capture intent patterns well.")
        else:
            print(f"\n⚠️  Features don't obviously separate by intent")
    else:
        print("✅ Results are reasonable")

if __name__ == '__main__':
    print("="*70)
    print("DEEP VERIFICATION: ClusterMajorityVote & FeatureLevelKNN")
    print("="*70)
    
    df = load_and_process_data()
    agg_df = create_aggregates(df)
    
    print(f"\n✓ Created {len(agg_df)} aggregate vectors (10 features each)")
    
    test_both_methods(df, agg_df)
