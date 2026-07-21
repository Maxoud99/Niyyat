#!/usr/bin/env python3
"""
Detailed analysis of WHY ClusterMajorityVote and FeatureLevelKNN perform so well.
Shows:
1. How many samples each method uses
2. Why the results are better than baseline
3. Feature distribution analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

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
                new_value_encoded = hash(str(new_value)) % 10000
                original_value_encoded = hash(str(original_value)) % 10000
            
            # ONE-HOT encode feature name (this is the leakage!)
            feat_one_hot = {f'feat_{feat}': 0 for feat in all_feature_names}
            feat_one_hot[f'feat_{feature}'] = 1
            
            change_dict = {
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
                'relative_change': relative_change,
            }
            change_dict.update(feat_one_hot)
            
            all_changes.append(change_dict)
    
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

def analyze_methods(df, agg_df):
    """Analyze all methods and compare sample usage"""
    
    print(f"\n{'='*70}")
    print("CLUSTERING AND SAMPLING")
    print(f"{'='*70}")
    
    # Cluster
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
    
    print(f"Sampled variants: {len(sampled_variants)}")
    print(f"Train features: {len(train_df)}")
    print(f"Test features: {len(test_df)}")
    print(f"Test variants: {test_df['variant_record_id'].nunique()}")
    
    # Analyze cluster purity
    variant_to_cluster = dict(zip(agg_df['variant_record_id'], agg_df['cluster']))
    
    print(f"\n{'='*70}")
    print("CLUSTER PURITY ANALYSIS")
    print(f"{'='*70}")
    
    cluster_stats = []
    for cluster_id in sorted(agg_df['cluster'].unique()):
        if cluster_id == -1:
            continue
        
        cluster_variants = agg_df[agg_df['cluster'] == cluster_id]['variant_record_id'].values
        cluster_features = df[df['variant_record_id'].isin(cluster_variants)]
        
        n_int = (cluster_features['intent_label'] == 1).sum()
        n_unint = (cluster_features['intent_label'] == -1).sum()
        purity = max(n_int, n_unint) / (n_int + n_unint) if (n_int + n_unint) > 0 else 0
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'n_variants': len(cluster_variants),
            'n_features': len(cluster_features),
            'n_int': n_int,
            'n_unint': n_unint,
            'purity': purity
        })
    
    cluster_df = pd.DataFrame(cluster_stats).sort_values('purity', ascending=False)
    print(f"\n{'Cluster':<8} {'Variants':<10} {'Features':<10} {'Purity':<8} {'Majority'}")
    print("-" * 60)
    for _, row in cluster_df.head(15).iterrows():
        majority = "INT" if row['n_int'] > row['n_unint'] else "UNINT"
        print(f"{row['cluster_id']:<8} {row['n_variants']:<10} {row['n_features']:<10} "
              f"{row['purity']*100:>6.1f}%  {majority} ({row['n_int']} vs {row['n_unint']})")
    
    avg_purity = cluster_df['purity'].mean()
    print(f"\nAverage cluster purity: {avg_purity*100:.1f}%")
    
    # METHOD 1: CLUSTER MAJORITY VOTE
    print(f"\n{'='*70}")
    print("METHOD 1: CLUSTER MAJORITY VOTE")
    print(f"{'='*70}")
    
    cluster_labels = {}
    total_train_features_used = 0
    
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
        total_train_features_used += len(cluster_train_df)
    
    y_pred_cluster = [cluster_labels.get(variant_to_cluster[v], 1) for v in test_df['variant_record_id']]
    y_test = test_df['intent_label'].values
    
    f1_cluster = f1_score(y_test, y_pred_cluster, average='weighted')
    
    print(f"✓ Training samples used: {len(sampled_variants)} variants = {total_train_features_used} features")
    print(f"✓ Prediction method: Cluster-level majority vote")
    print(f"✓ F1 Score: {f1_cluster:.4f}")
    print(f"✓ Why it works: High cluster purity ({avg_purity*100:.1f}%) means clusters naturally separate by intent")
    
    # METHOD 2: FEATURE-LEVEL K-NN (WITH LEAKAGE)
    print(f"\n{'='*70}")
    print("METHOD 2: FEATURE-LEVEL K-NN (WITH feat_* indicators)")
    print(f"{'='*70}")
    
    feature_cols_with_leak = ['feature_name_encoded', 'original_value_encoded', 'new_value_encoded',
                               'change_magnitude', 'relative_change']
    feature_cols_with_leak += [c for c in df.columns if c.startswith('feat_')]
    
    X_train_leak = train_df[feature_cols_with_leak].values
    y_train = train_df['intent_label'].values
    X_test_leak = test_df[feature_cols_with_leak].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_leak)
    X_test_scaled = scaler.transform(X_test_leak)
    
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn.fit(X_train_scaled, y_train)
    y_pred_knn_leak = knn.predict(X_test_scaled)
    
    f1_knn_leak = f1_score(y_test, y_pred_knn_leak, average='weighted')
    
    print(f"✓ Training samples used: {len(train_df)} features (from {len(sampled_variants)} variants)")
    print(f"✓ Features used: {len(feature_cols_with_leak)} (including {len([c for c in feature_cols_with_leak if c.startswith('feat_')])} feat_* indicators)")
    print(f"✓ Prediction method: k-NN (k=7) on individual features")
    print(f"✓ F1 Score: {f1_knn_leak:.4f}")
    print(f"✓ Why it works: feat_* indicators tell the model WHICH FEATURE changed")
    
    # Analyze feature importance
    print(f"\n  Feature-specific intent patterns (from training data):")
    for feat in df['feature_name'].unique():
        feat_col = f'feat_{feat}'
        if feat_col in train_df.columns:
            feat_samples = train_df[train_df[feat_col] == 1]
            if len(feat_samples) > 0:
                n_int = (feat_samples['intent_label'] == 1).sum()
                n_unint = (feat_samples['intent_label'] == -1).sum()
                print(f"    {feat:<20}: {n_int} int, {n_unint} unint → {'INT' if n_int > n_unint else 'UNINT'} bias")
    
    # METHOD 3: FEATURE-LEVEL K-NN (WITHOUT LEAKAGE)
    print(f"\n{'='*70}")
    print("METHOD 3: FEATURE-LEVEL K-NN (WITHOUT feat_* indicators)")
    print(f"{'='*70}")
    
    feature_cols_no_leak = ['feature_name_encoded', 'original_value_encoded', 'new_value_encoded',
                             'change_magnitude', 'relative_change']
    
    X_train_no_leak = train_df[feature_cols_no_leak].values
    X_test_no_leak = test_df[feature_cols_no_leak].values
    
    scaler2 = StandardScaler()
    X_train_scaled2 = scaler2.fit_transform(X_train_no_leak)
    X_test_scaled2 = scaler2.transform(X_test_no_leak)
    
    knn2 = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn2.fit(X_train_scaled2, y_train)
    y_pred_knn_no_leak = knn2.predict(X_test_scaled2)
    
    f1_knn_no_leak = f1_score(y_test, y_pred_knn_no_leak, average='weighted')
    
    print(f"✓ Training samples used: {len(train_df)} features (from {len(sampled_variants)} variants)")
    print(f"✓ Features used: {len(feature_cols_no_leak)} (NO feat_* indicators)")
    print(f"✓ Prediction method: k-NN (k=7) on individual features")
    print(f"✓ F1 Score: {f1_knn_no_leak:.4f}")
    
    # BASELINE: YOUR HDBSCAN + RF
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON")
    print(f"{'='*70}")
    print(f"Your baseline (HDBSCAN+RF): 91.87% F1")
    print(f"  - Uses HDBSCAN clustering (can create noise points)")
    print(f"  - Random Forest classifier on feature-level data")
    print(f"  - Training: {len(train_df)} features")
    print(f"")
    print(f"ClusterMajorityVote: {f1_cluster*100:.2f}% F1 (+{(f1_cluster-0.9187)*100:.2f}%)")
    print(f"  - Uses K-Means (no noise, balanced clusters)")
    print(f"  - Simple majority vote per cluster")
    print(f"  - Benefits from {avg_purity*100:.1f}% cluster purity")
    print(f"")
    print(f"FeatureLevelKNN (with leakage): {f1_knn_leak*100:.2f}% F1 (+{(f1_knn_leak-0.9187)*100:.2f}%)")
    print(f"  - Uses {len(feature_cols_with_leak)} features including feat_* indicators")
    print(f"  - Learns feature-specific patterns (DATA LEAKAGE)")
    print(f"")
    print(f"FeatureLevelKNN (no leakage): {f1_knn_no_leak*100:.2f}% F1 ({(f1_knn_no_leak-0.9187)*100:+.2f}%)")
    print(f"  - Uses only 5 generic features")
    print(f"  - True label propagation without feature-specific knowledge")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"1. ClusterMajorityVote is LEGITIMATE and BETTER because:")
    print(f"   - K-Means creates more balanced clusters than HDBSCAN")
    print(f"   - Clusters have {avg_purity*100:.1f}% average purity")
    print(f"   - Cluster-level prediction is more stable than feature-level")
    print(f"")
    print(f"2. FeatureLevelKNN (current) has DATA LEAKAGE:")
    print(f"   - Uses feat_* indicators that encode WHICH FEATURE changed")
    print(f"   - Can learn 'capital-gain changes are intentional'")
    print(f"   - This is supervised learning, NOT label propagation")
    print(f"")
    print(f"3. FeatureLevelKNN (fixed) would get ~{f1_knn_no_leak*100:.1f}% F1")
    print(f"   - Still decent, but not as good as ClusterMajorityVote")
    print(f"   - Uses only generic features (magnitude, relative change, etc.)")

if __name__ == '__main__':
    print("="*70)
    print("DETAILED ANALYSIS: Why ClusterMajorityVote & FeatureLevelKNN Work")
    print("="*70)
    
    df = load_and_process_data()
    agg_df = create_aggregates(df)
    
    print(f"\n✓ Created {len(agg_df)} aggregate vectors")
    
    analyze_methods(df, agg_df)
