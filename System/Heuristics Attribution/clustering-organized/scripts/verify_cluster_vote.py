#!/usr/bin/env python3
"""
Verification script to check if cluster majority vote is working correctly.
Tests two approaches:
1. Feature-weighted (current implementation) - counts ALL features
2. Variant-weighted (true propagation) - 1 vote per variant
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data():
    """Load tenth-trial data"""
    masks = pd.read_csv("/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/data/raw/run_20251211_114329/masks.csv")
    
    # Process into feature changes
    all_changes = []
    for idx in range(len(masks)):
        for feature in masks.columns:
            intent = masks.iloc[idx][feature]
            if intent != 0:
                all_changes.append({
                    'variant_record_id': idx,
                    'feature': feature,
                    'intent_label': intent
                })
    
    return pd.DataFrame(all_changes)

def create_aggregates(df):
    """Create 10-d aggregate vectors per variant"""
    aggregated = []
    for variant_id in df['variant_record_id'].unique():
        variant_data = df[df['variant_record_id'] == variant_id]
        
        aggregated.append({
            'variant_record_id': variant_id,
            'n_changes': len(variant_data),
            'n_intentional': (variant_data['intent_label'] == 1).sum(),
            'n_unintentional': (variant_data['intent_label'] == -1).sum(),
        })
    
    return pd.DataFrame(aggregated)

def test_both_approaches(df, agg_df):
    """Test feature-weighted vs variant-weighted cluster voting"""
    
    # Cluster
    X = agg_df[['n_changes', 'n_intentional', 'n_unintentional']].values
    X = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=15, random_state=42).fit_predict(X)
    agg_df['cluster'] = labels
    
    # Sample 177 variants (1%)
    np.random.seed(42)
    sampled_variants = np.random.choice(agg_df['variant_record_id'].values, 177, replace=False)
    
    train_df = df[df['variant_record_id'].isin(sampled_variants)]
    test_df = df[~df['variant_record_id'].isin(sampled_variants)]
    
    print(f"Train: {len(train_df)} features from {len(sampled_variants)} variants")
    print(f"Test: {len(test_df)} features from {len(test_df['variant_record_id'].unique())} variants")
    
    # Build variant → cluster mapping
    variant_to_cluster = dict(zip(agg_df['variant_record_id'], agg_df['cluster']))
    
    # APPROACH 1: FEATURE-WEIGHTED (current implementation)
    print(f"\n{'='*70}")
    print("APPROACH 1: FEATURE-WEIGHTED CLUSTER VOTING (current)")
    print(f"{'='*70}")
    
    cluster_labels_feature = {}
    for cluster_id in agg_df['cluster'].unique():
        if cluster_id == -1:
            continue
        
        cluster_train_variants = [v for v in sampled_variants if variant_to_cluster[v] == cluster_id]
        if len(cluster_train_variants) == 0:
            continue
        
        # Count ALL features from sampled variants
        cluster_train_df = train_df[train_df['variant_record_id'].isin(cluster_train_variants)]
        n_int = (cluster_train_df['intent_label'] == 1).sum()
        n_unint = (cluster_train_df['intent_label'] == -1).sum()
        
        cluster_labels_feature[cluster_id] = 1 if n_int >= n_unint else -1
        
        print(f"Cluster {cluster_id}: {len(cluster_train_variants)} variants, "
              f"{n_int} intentional features, {n_unint} unintentional features "
              f"→ Label: {cluster_labels_feature[cluster_id]}")
    
    y_pred_feature = [cluster_labels_feature.get(variant_to_cluster[v], 1) for v in test_df['variant_record_id']]
    y_test = test_df['intent_label'].values
    
    acc = accuracy_score(y_test, y_pred_feature)
    f1 = f1_score(y_test, y_pred_feature, average='weighted')
    f1_int = f1_score(y_test, y_pred_feature, pos_label=1)
    f1_unint = f1_score(y_test, y_pred_feature, pos_label=-1)
    
    print(f"\n✓ Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Weighted: {f1:.4f}")
    print(f"  F1 Intentional: {f1_int:.4f}")
    print(f"  F1 Unintentional: {f1_unint:.4f}")
    
    # APPROACH 2: VARIANT-WEIGHTED (true propagation)
    print(f"\n{'='*70}")
    print("APPROACH 2: VARIANT-WEIGHTED CLUSTER VOTING (true propagation)")
    print(f"{'='*70}")
    
    cluster_labels_variant = {}
    for cluster_id in agg_df['cluster'].unique():
        if cluster_id == -1:
            continue
        
        cluster_train_variants = [v for v in sampled_variants if variant_to_cluster[v] == cluster_id]
        if len(cluster_train_variants) == 0:
            continue
        
        # Count variants with majority intentional vs unintentional
        n_int_variants = 0
        n_unint_variants = 0
        
        for variant_id in cluster_train_variants:
            variant_features = train_df[train_df['variant_record_id'] == variant_id]
            n_int_feat = (variant_features['intent_label'] == 1).sum()
            n_unint_feat = (variant_features['intent_label'] == -1).sum()
            
            if n_int_feat >= n_unint_feat:
                n_int_variants += 1
            else:
                n_unint_variants += 1
        
        cluster_labels_variant[cluster_id] = 1 if n_int_variants >= n_unint_variants else -1
        
        print(f"Cluster {cluster_id}: {len(cluster_train_variants)} variants, "
              f"{n_int_variants} intentional-majority, {n_unint_variants} unintentional-majority "
              f"→ Label: {cluster_labels_variant[cluster_id]}")
    
    y_pred_variant = [cluster_labels_variant.get(variant_to_cluster[v], 1) for v in test_df['variant_record_id']]
    
    acc = accuracy_score(y_test, y_pred_variant)
    f1 = f1_score(y_test, y_pred_variant, average='weighted')
    f1_int = f1_score(y_test, y_pred_variant, pos_label=1)
    f1_unint = f1_score(y_test, y_pred_variant, pos_label=-1)
    
    print(f"\n✓ Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Weighted: {f1:.4f}")
    print(f"  F1 Intentional: {f1_int:.4f}")
    print(f"  F1 Unintentional: {f1_unint:.4f}")
    
    # COMPARISON
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"Feature-weighted (current): {f1_score(y_test, y_pred_feature, average='weighted'):.4f}")
    print(f"Variant-weighted (true):    {f1_score(y_test, y_pred_variant, average='weighted'):.4f}")
    print(f"\nDifference: {f1_score(y_test, y_pred_feature, average='weighted') - f1_score(y_test, y_pred_variant, average='weighted'):.4f}")
    
    # Check cluster label differences
    print(f"\nCluster label differences:")
    for cluster_id in set(cluster_labels_feature.keys()) | set(cluster_labels_variant.keys()):
        feat_label = cluster_labels_feature.get(cluster_id, 'N/A')
        var_label = cluster_labels_variant.get(cluster_id, 'N/A')
        if feat_label != var_label:
            print(f"  Cluster {cluster_id}: feature={feat_label}, variant={var_label} ❌")

if __name__ == '__main__':
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} feature changes from {len(df['variant_record_id'].unique())} variants")
    
    print("\nCreating aggregates...")
    agg_df = create_aggregates(df)
    print(f"Created {len(agg_df)} aggregate vectors")
    
    test_both_approaches(df, agg_df)
