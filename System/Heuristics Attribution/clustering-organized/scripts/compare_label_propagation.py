#!/usr/bin/env python3
"""
LABEL PROPAGATION VS RANDOM FOREST: Intent Classification
Replicates your compare_clustering_algorithms.py structure
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except:
    HDBSCAN_AVAILABLE = False


class LabelPropagationComparison:
    def __init__(self, target_samples=None, random_state=42, mask_path=None, clean_data_path=None, dirty_data_path=None):
        self.target_samples = target_samples
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store custom data paths
        self.mask_path = Path(mask_path) if mask_path else None
        self.clean_data_path = Path(clean_data_path) if clean_data_path else None
        self.dirty_data_path = Path(dirty_data_path) if dirty_data_path else None
        
        script_dir = Path(__file__).parent.resolve()
        parent_dir = script_dir.parent
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = parent_dir / 'outputs' / f'run_label_prop_{self.timestamp}'
        self.results_dir = self.run_dir / 'results'
        self.plots_dir = self.run_dir / 'plots'
        
        for d in [self.results_dir, self.plots_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print("LABEL PROPAGATION VS RANDOM FOREST")
        print(f"{'='*70}")
        print(f"Baseline: HDBSCAN+RF = 91.87% F1")
        print(f"Output: {self.run_dir}")
        
    def load_data(self):
        print(f"\n{'='*70}")
        print("LOADING DATA")
        print(f"{'='*70}")
        
        # Use provided paths (SAME AS YOUR compare_clustering_algorithms.py)
        if not self.mask_path or not self.clean_data_path or not self.dirty_data_path:
            raise ValueError("Must provide --mask-path, --clean-data-path, --dirty-data-path")
        
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {self.mask_path}")
        if not self.clean_data_path.exists():
            raise FileNotFoundError(f"Clean data not found: {self.clean_data_path}")
        if not self.dirty_data_path.exists():
            raise FileNotFoundError(f"Dirty data not found: {self.dirty_data_path}")
        
        print(f"Mask: {self.mask_path}")
        print(f"Clean: {self.clean_data_path}")
        print(f"Dirty: {self.dirty_data_path}")
        
        masks_df = pd.read_csv(self.mask_path)
        correct_df = pd.read_csv(self.clean_data_path)
        manipulated_df = pd.read_csv(self.dirty_data_path)
        
        print(f"✓ Masks: {masks_df.shape}")
        print(f"✓ Manipulated: {manipulated_df.shape}")
        print(f"✓ Correct: {correct_df.shape}")
        
        # Process feature changes (EXACT SAME AS YOUR compare_clustering_algorithms.py)
        feature_cols = masks_df.columns.tolist()
        all_changes = []
        print(f"\nProcessing {len(masks_df)} variant records...")
        
        for idx in tqdm(range(len(masks_df)), desc="Processing", unit="record"):
            original_record_idx = idx // 3  # Each original has 3 variants
            variant_idx = idx % 3  # 0, 1, or 2
            
            for feature in feature_cols:
                intent_label = masks_df.iloc[idx][feature]
                
                if intent_label == 0:
                    continue
                
                original_value = correct_df.iloc[original_record_idx][feature]
                new_value = manipulated_df.iloc[idx][feature]
                
                # Handle numeric vs string values
                try:
                    orig_numeric = float(original_value) if pd.notna(original_value) else 0
                    new_numeric = float(new_value) if pd.notna(new_value) else 0
                    magnitude = abs(new_numeric - orig_numeric)
                except (ValueError, TypeError):
                    # String values - use 1 as placeholder magnitude
                    magnitude = 1
                
                all_changes.append({
                    'variant_record_id': idx,
                    'original_record_id': original_record_idx,
                    'variant_idx': variant_idx,
                    'feature_name': feature,
                    'original_value': original_value,
                    'new_value': new_value,
                    'change_magnitude': magnitude,
                    'intent_label': intent_label
                })
        
        self.df = pd.DataFrame(all_changes)
        
        print(f"\n✓ Dataset: {len(self.df)} feature changes")
        print(f"  Intentional (1): {(self.df['intent_label'] == 1).sum()}")
        print(f"  Unintentional (-1): {(self.df['intent_label'] == -1).sum()}")
        print(f"  Unique variant records: {self.df['variant_record_id'].nunique()}")
        
        # Encode categorical features (SAME AS YOUR CODE)
        for col in ['feature_name', 'original_value', 'new_value']:
            self.df[f'{col}_encoded'] = pd.Categorical(self.df[col].astype(str)).codes
        
        # Convert to numeric where possible for derived features
        def safe_numeric(series):
            return pd.to_numeric(series, errors='coerce').fillna(0)
        
        original_numeric = safe_numeric(self.df['original_value'])
        new_numeric = safe_numeric(self.df['new_value'])
        
        # Add derived features (SAME AS YOUR CODE)
        self.df['relative_change'] = self.df['change_magnitude'] / (original_numeric.abs() + 1)
        self.df['change_direction'] = np.sign(new_numeric - original_numeric)
        self.df['original_log'] = np.log1p(original_numeric.abs())
        self.df['new_log'] = np.log1p(new_numeric.abs())
        self.df['original_magnitude'] = original_numeric.abs()
        self.df['new_magnitude'] = new_numeric.abs()
        
        # NOTE: Removed feat_* one-hot indicators to prevent data leakage
        # Previously: for feat in self.df['feature_name'].unique():
        #     self.df[f'feat_{feat}'] = (self.df['feature_name'] == feat).astype(int)
        # This was allowing the model to learn feature-specific patterns
        # (e.g., "capital-gain changes are intentional, sex changes are unintentional")
        # which is supervised learning, not label propagation.
        
        # Store total dataset info
        self.total_variants = self.df['variant_record_id'].nunique()
        if self.target_samples is None:
            self.target_samples = max(1, int(self.total_variants * 0.01))
        
        print(f"\n⚙️  Target: {self.target_samples} variants ({self.target_samples/self.total_variants*100:.2f}%)")
        
    def create_aggregate_features(self):
        """Create aggregate features (NO INTENT LABELS - SAME AS YOUR CODE)"""
        print(f"\n{'='*70}")
        print("CREATING AGGREGATE FEATURES")
        print(f"{'='*70}")
        
        aggregated = []
        for variant_id in tqdm(self.df['variant_record_id'].unique(), desc="Aggregating"):
            variant_data = self.df[self.df['variant_record_id'] == variant_id]
            
            # Basic counts (NO INTENT LABELS - avoid data leakage!)
            n_changes = len(variant_data)
            
            # Change magnitude statistics
            mean_magnitude = variant_data['change_magnitude'].mean()
            std_magnitude = variant_data['change_magnitude'].std() if len(variant_data) > 1 else 0
            min_magnitude = variant_data['change_magnitude'].min()
            max_magnitude = variant_data['change_magnitude'].max()
            median_magnitude = variant_data['change_magnitude'].median()
            
            # Use encoded values
            min_new_value_encoded = variant_data['new_value_encoded'].min()
            max_new_value_encoded = variant_data['new_value_encoded'].max()
            
            # Additional derived features
            mean_relative_change = variant_data['relative_change'].mean()
            
            # Feature with largest magnitude change
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
                'min_new_value_encoded': min_new_value_encoded,
                'max_new_value_encoded': max_new_value_encoded,
                'mean_relative_change': mean_relative_change,
                'feature_with_max_change_encoded': feature_with_max_change,
            })
        
        self.agg_df = pd.DataFrame(aggregated).fillna(0)
        print(f"✓ Created {len(self.agg_df)} aggregate vectors ({len(self.agg_df.columns)-1} features)")
        
    def cluster_and_sample(self, algorithm='kmeans'):
        print(f"\n{'='*70}")
        print(f"CLUSTERING: {algorithm.upper()}")
        print(f"{'='*70}")
        
        X = self.agg_df.drop('variant_record_id', axis=1).values
        X = StandardScaler().fit_transform(X)
        
        if algorithm == 'kmeans':
            labels = KMeans(n_clusters=15, random_state=self.random_state, n_init=10).fit_predict(X)
        elif algorithm == 'dbscan':
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=5).fit(X)
            dists, _ = nn.kneighbors(X)
            eps = np.percentile(dists[:, -1], 90) * 0.5
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
        elif algorithm == 'hdbscan' and HDBSCAN_AVAILABLE:
            labels = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1).fit_predict(X)
        else:
            raise ValueError(f"Unknown: {algorithm}")
        
        self.agg_df['cluster'] = labels
        
        n_clusters = len([x for x in set(labels) if x != -1])
        n_noise = (labels == -1).sum()
        print(f"✓ Clusters: {n_clusters}, Noise: {n_noise}")
        
        # Proportional sampling (SAME AS YOUR CODE)
        sampled_variants = []
        unique_clusters = [c for c in set(labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_variants = self.agg_df[self.agg_df['cluster'] == cluster_id]['variant_record_id'].values
            cluster_size = len(cluster_variants)
            n_to_sample = max(1, int(self.target_samples * (cluster_size / len(self.agg_df))))
            
            if n_to_sample > cluster_size:
                n_to_sample = cluster_size
            
            sampled = np.random.choice(cluster_variants, n_to_sample, replace=False)
            sampled_variants.extend(sampled)
        
        print(f"✓ Sampled {len(sampled_variants)} variants")
        
        # Split data
        train_df = self.df[self.df['variant_record_id'].isin(sampled_variants)].copy()
        test_df = self.df[~self.df['variant_record_id'].isin(sampled_variants)].copy()
        
        print(f"Train: {len(train_df)} features ({len(train_df)/len(self.df)*100:.2f}%)")
        print(f"Test: {len(test_df)} features")
        
        return train_df, test_df
        
    def test_random_forest(self, train_df, test_df):
        print(f"\n{'='*70}")
        print("RANDOM FOREST")
        print(f"{'='*70}")
        
        feature_cols = [c for c in train_df.columns if c not in ['variant_record_id', 'original_record_id', 'variant_idx', 'feature_name', 'original_value', 'new_value', 'intent_label']]
        
        X_train = train_df[feature_cols].values
        y_train = train_df['intent_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['intent_label'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        start = time.time()
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
        runtime = time.time() - start
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_int = f1_score(y_test, y_pred, pos_label=1)
        f1_unint = f1_score(y_test, y_pred, pos_label=-1)
        
        print(f"✓ Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  F1 Intentional: {f1_int:.4f}")
        print(f"  F1 Unintentional: {f1_unint:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'method': 'RandomForest',
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_intentional': f1_int,
            'f1_unintentional': f1_unint,
            'runtime': runtime
        }
    
    def test_cluster_majority_vote(self, train_df, test_df):
        """Approach 1: Assign cluster's majority label to all variants in that cluster"""
        print(f"\n{'='*70}")
        print("APPROACH 1: CLUSTER MAJORITY VOTE")
        print(f"{'='*70}")
        
        train_variants = train_df['variant_record_id'].unique()
        
        # Get cluster assignment for each variant
        variant_to_cluster = dict(zip(self.agg_df['variant_record_id'].values, 
                                      self.agg_df['cluster'].values))
        
        # For each cluster, compute majority label from sampled variants
        cluster_labels = {}
        cluster_stats = []  # For analysis
        
        for cluster_id in self.agg_df['cluster'].unique():
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get sampled variants in this cluster
            cluster_train_variants = [v for v in train_variants if variant_to_cluster.get(v) == cluster_id]
            
            # Get all variants in cluster (for purity analysis)
            cluster_all_variants = self.agg_df[self.agg_df['cluster'] == cluster_id]['variant_record_id'].values
            cluster_all_df = self.df[self.df['variant_record_id'].isin(cluster_all_variants)]
            
            if len(cluster_train_variants) == 0:
                cluster_labels[cluster_id] = None  # No sampled variants
                continue
            
            # Get all features from sampled variants in this cluster
            cluster_train_df = train_df[train_df['variant_record_id'].isin(cluster_train_variants)]
            
            # Majority vote from ALL features
            n_intentional = (cluster_train_df['intent_label'] == 1).sum()
            n_unintentional = (cluster_train_df['intent_label'] == -1).sum()
            
            cluster_labels[cluster_id] = 1 if n_intentional >= n_unintentional else -1
            
            # Compute cluster purity (on ALL variants, including test)
            all_int = (cluster_all_df['intent_label'] == 1).sum()
            all_unint = (cluster_all_df['intent_label'] == -1).sum()
            purity = max(all_int, all_unint) / (all_int + all_unint) if (all_int + all_unint) > 0 else 0
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_all_variants),
                'train_variants': len(cluster_train_variants),
                'train_int_features': n_intentional,
                'train_unint_features': n_unintentional,
                'assigned_label': cluster_labels[cluster_id],
                'total_int_features': all_int,
                'total_unint_features': all_unint,
                'purity': purity
            })
        
        # Print cluster analysis
        print(f"\nCluster Analysis (sorted by size):")
        print(f"{'Cluster':<8} {'Size':<6} {'Train':<6} {'Label':<6} {'Purity':<8} {'Details'}")
        print(f"{'-'*70}")
        for stats in sorted(cluster_stats, key=lambda x: x['size'], reverse=True):
            details = f"Train: {stats['train_int_features']} int, {stats['train_unint_features']} unint | "
            details += f"Total: {stats['total_int_features']} int, {stats['total_unint_features']} unint"
            print(f"{stats['cluster_id']:<8} {stats['size']:<6} {stats['train_variants']:<6} "
                  f"{'+1' if stats['assigned_label']==1 else '-1':<6} {stats['purity']*100:>6.1f}%  {details}")
        
        # Assign cluster labels to all test variants
        start = time.time()
        y_pred = []
        for vid in test_df['variant_record_id'].values:
            cluster = variant_to_cluster.get(vid, -1)
            label = cluster_labels.get(cluster, np.random.choice([1, -1]))  # Random if no label
            y_pred.append(label)
        
        y_pred = np.array(y_pred)
        y_test = test_df['intent_label'].values
        runtime = time.time() - start
        
        # Count unlabeled clusters
        unlabeled_clusters = sum(1 for c, l in cluster_labels.items() if l is None)
        total_clusters = len([c for c in self.agg_df['cluster'].unique() if c != -1])
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_int = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_unint = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        
        print(f"✓ Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  F1 Intentional: {f1_int:.4f}")
        print(f"  F1 Unintentional: {f1_unint:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Clusters with labels: {total_clusters - unlabeled_clusters}/{total_clusters}")
        
        return {
            'method': 'ClusterMajorityVote',
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_intentional': f1_int,
            'f1_unintentional': f1_unint,
            'runtime': runtime
        }
    
    def test_feature_level_voting(self, train_df, test_df):
        """Approach 2: Use feature-level k-NN voting (original feature-level propagation)"""
        print(f"\n{'='*70}")
        print("APPROACH 2: FEATURE-LEVEL K-NN VOTING")
        print(f"{'='*70}")
        
        feature_cols = [c for c in train_df.columns if c not in ['variant_record_id', 'original_record_id', 'variant_idx', 'feature_name', 'original_value', 'new_value', 'intent_label']]
        
        print(f"Using {len(feature_cols)} features:")
        for col in feature_cols:
            print(f"  - {col}")
        
        X_train = train_df[feature_cols].values
        y_train = train_df['intent_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['intent_label'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Simple k-NN voting
        from sklearn.neighbors import KNeighborsClassifier
        
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        runtime = time.time() - start
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_int = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_unint = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        
        print(f"✓ Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  F1 Intentional: {f1_int:.4f}")
        print(f"  F1 Unintentional: {f1_unint:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return {
            'method': 'FeatureLevelKNN',
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_intentional': f1_int,
            'f1_unintentional': f1_unint,
            'runtime': runtime
        }
    
    def test_label_propagation(self, train_df, test_df):
        """Approach 3: sklearn LabelPropagation (graph-based, variant-level)"""
        print(f"\n{'='*70}")
        print("APPROACH 3: LABEL PROPAGATION (GRAPH-BASED, VARIANT-LEVEL)")
        print(f"{'='*70}")
        
        # Step 1: Get variant-level labels from training data
        train_variants = train_df['variant_record_id'].unique()
        test_variants = test_df['variant_record_id'].unique()
        
        # Create mapping: variant_id -> majority intent label
        train_variant_labels = {}
        for vid in train_variants:
            labels = train_df[train_df['variant_record_id'] == vid]['intent_label'].values
            # Use majority vote (though should be same for all features in a variant)
            train_variant_labels[vid] = np.bincount(labels.astype(int) + 1).argmax() - 1  # Convert back to 1/-1
        
        # Step 2: Get variant-level aggregate features (the 10-d features we clustered on)
        train_agg = self.agg_df[self.agg_df['variant_record_id'].isin(train_variants)]
        test_agg = self.agg_df[self.agg_df['variant_record_id'].isin(test_variants)]
        
        # Align labels
        y_train_variants = np.array([train_variant_labels[vid] for vid in train_agg['variant_record_id'].values])
        
        # Step 3: Extract features (drop variant_record_id and cluster)
        feat_cols = [c for c in train_agg.columns if c not in ['variant_record_id', 'cluster']]
        X_train_agg = train_agg[feat_cols].values
        X_test_agg = test_agg[feat_cols].values
        
        # Step 4: Apply Label Propagation at VARIANT level
        X_all = np.vstack([X_train_agg, X_test_agg])
        y_all = np.concatenate([y_train_variants, np.full(len(X_test_agg), -999)])
        
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        
        start = time.time()
        lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
        lp.fit(X_all_scaled, y_all)
        y_pred_variants = lp.predict(X_all_scaled[len(X_train_agg):])
        runtime = time.time() - start
        
        # Step 5: Map variant predictions back to ALL features in those variants
        test_variant_ids = test_agg['variant_record_id'].values
        variant_predictions = dict(zip(test_variant_ids, y_pred_variants))
        
        y_pred = np.array([variant_predictions[vid] for vid in test_df['variant_record_id'].values])
        y_test = test_df['intent_label'].values
        
        # Count unlabeled VARIANTS
        unlabeled_variants = (y_pred_variants == -999).sum()
        total_test_variants = len(test_agg)
        
        # For unlabeled variants, randomly assign to features
        if unlabeled_variants > 0:
            unlabeled_mask = y_pred == -999
            y_pred[unlabeled_mask] = np.random.choice([1, -1], size=unlabeled_mask.sum())
        
        # Calculate metrics on ALL test FEATURES
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_int = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_unint = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        
        print(f"✓ Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  F1 Intentional: {f1_int:.4f}")
        print(f"  F1 Unintentional: {f1_unint:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Train variants: {len(train_variants)}, Test variants: {len(test_variants)}")
        if unlabeled_variants > 0:
            print(f"  ⚠️  Unlabeled variants: {unlabeled_variants}/{total_test_variants} ({unlabeled_variants/total_test_variants*100:.1f}%)")
        
        return {
            'method': 'LabelPropagation',
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_intentional': f1_int,
            'f1_unintentional': f1_unint,
            'runtime': runtime
        }
    
    def test_label_spreading(self, train_df, test_df):
        print(f"\n{'='*70}")
        print("LABEL SPREADING (VARIANT-LEVEL)")
        print(f"{'='*70}")
        
        # Step 1: Get variant-level labels from training data
        train_variants = train_df['variant_record_id'].unique()
        test_variants = test_df['variant_record_id'].unique()
        
        # Create mapping: variant_id -> majority intent label
        train_variant_labels = {}
        for vid in train_variants:
            labels = train_df[train_df['variant_record_id'] == vid]['intent_label'].values
            train_variant_labels[vid] = np.bincount(labels.astype(int) + 1).argmax() - 1
        
        # Step 2: Get variant-level aggregate features
        train_agg = self.agg_df[self.agg_df['variant_record_id'].isin(train_variants)]
        test_agg = self.agg_df[self.agg_df['variant_record_id'].isin(test_variants)]
        
        # Align labels
        y_train_variants = np.array([train_variant_labels[vid] for vid in train_agg['variant_record_id'].values])
        
        # Step 3: Extract features
        feat_cols = [c for c in train_agg.columns if c not in ['variant_record_id', 'cluster']]
        X_train_agg = train_agg[feat_cols].values
        X_test_agg = test_agg[feat_cols].values
        
        # Step 4: Apply Label Spreading at VARIANT level
        X_all = np.vstack([X_train_agg, X_test_agg])
        y_all = np.concatenate([y_train_variants, np.full(len(X_test_agg), -999)])
        
        scaler = StandardScaler()
        X_all_scaled = scaler.fit_transform(X_all)
        
        start = time.time()
        ls = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2, max_iter=1000)
        ls.fit(X_all_scaled, y_all)
        y_pred_variants = ls.predict(X_all_scaled[len(X_train_agg):])
        runtime = time.time() - start
        
        # Step 5: Map variant predictions back to ALL features
        test_variant_ids = test_agg['variant_record_id'].values
        variant_predictions = dict(zip(test_variant_ids, y_pred_variants))
        
        y_pred = np.array([variant_predictions[vid] for vid in test_df['variant_record_id'].values])
        y_test = test_df['intent_label'].values
        
        # Count unlabeled VARIANTS
        unlabeled_variants = (y_pred_variants == -999).sum()
        total_test_variants = len(test_agg)
        
        # For unlabeled variants, randomly assign
        if unlabeled_variants > 0:
            unlabeled_mask = y_pred == -999
            y_pred[unlabeled_mask] = np.random.choice([1, -1], size=unlabeled_mask.sum())
        
        # Calculate metrics on ALL test FEATURES
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_int = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_unint = f1_score(y_test, y_pred, pos_label=-1, zero_division=0)
        
        print(f"✓ Results:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
        print(f"  F1 Intentional: {f1_int:.4f}")
        print(f"  F1 Unintentional: {f1_unint:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Train variants: {len(train_variants)}, Test variants: {len(test_variants)}")
        if unlabeled_variants > 0:
            print(f"  ⚠️  Unlabeled variants: {unlabeled_variants}/{total_test_variants} ({unlabeled_variants/total_test_variants*100:.1f}%)")
        
        return {
            'method': 'LabelSpreading',
            'accuracy': acc,
            'f1_weighted': f1_weighted,
            'f1_intentional': f1_int,
            'f1_unintentional': f1_unint,
            'runtime': runtime
        }
    
    def save_results(self, algorithm, results, train_df, test_df):
        output_file = self.results_dir / f'{algorithm}_results.txt'
        
        # Calculate sample counts
        train_variants = train_df['variant_record_id'].nunique()
        train_features = len(train_df)
        test_variants = test_df['variant_record_id'].nunique()
        test_features = len(test_df)
        total_variants = self.df['variant_record_id'].nunique()
        total_features = len(self.df)
        
        train_int = (train_df['intent_label'] == 1).sum()
        train_unint = (train_df['intent_label'] == -1).sum()
        
        with open(output_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"LABEL PROPAGATION VS RANDOM FOREST - {algorithm.upper()}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write(f"DATASET:\n")
            f.write(f"{'='*70}\n")
            f.write(f"Total: {total_features} features from {total_variants} variants\n")
            f.write(f"  Intentional: {(self.df['intent_label']==1).sum()}\n")
            f.write(f"  Unintentional: {(self.df['intent_label']==-1).sum()}\n\n")
            
            f.write(f"TRAINING DATA (1% sampling):\n")
            f.write(f"{'='*70}\n")
            f.write(f"Sampled: {train_variants} variants ({train_variants/total_variants*100:.2f}%)\n")
            f.write(f"  Training features: {train_features} ({train_features/total_features*100:.2f}%)\n")
            f.write(f"    Intentional labels: {train_int}\n")
            f.write(f"    Unintentional labels: {train_unint}\n")
            f.write(f"  Test features: {test_features} ({test_features/total_features*100:.2f}%)\n")
            f.write(f"    Test variants: {test_variants}\n\n")
            
            f.write(f"BASELINE:\n")
            f.write(f"{'='*70}\n")
            f.write(f"Method: HDBSCAN+RF\n")
            f.write(f"Training: {train_variants} variants, {train_features} features\n")
            f.write(f"  ({train_int} intentional + {train_unint} unintentional labels)\n")
            f.write(f"F1 Score: 91.87% (91.53% int, 92.17% unint)\n\n")
            
            f.write(f"RESULTS (all use SAME {train_features} training samples):\n")
            f.write(f"{'='*70}\n")
            for res in results:
                f.write(f"\n{res['method']}:\n")
                f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
                f.write(f"  F1 Weighted: {res['f1_weighted']:.4f}\n")
                f.write(f"  F1 Intentional: {res['f1_intentional']:.4f}\n")
                f.write(f"  F1 Unintentional: {res['f1_unintentional']:.4f}\n")
                f.write(f"  Runtime: {res['runtime']:.2f}s\n")
            
            best = max(results, key=lambda x: x['f1_weighted'])
            f.write(f"\n{'='*70}\n")
            f.write(f"WINNER: {best['method']} = {best['f1_weighted']:.4f}\n")
            
            if best['f1_weighted'] < 0.9187:
                f.write(f"❌ Below baseline - Random Forest wins\n")
            else:
                improvement = (best['f1_weighted'] - 0.9187) * 100
                f.write(f"✅ Beats baseline by {improvement:.2f} percentage points!\n")
        
        print(f"\n✓ Saved: {output_file}")
        
        # Also save JSON
        json_file = self.results_dir / f'{algorithm}_results.json'
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': self.timestamp,
                'algorithm': algorithm,
                'dataset': {
                    'total_features': total_features,
                    'total_variants': total_variants,
                    'train_variants': train_variants,
                    'train_features': train_features,
                    'train_intentional': int(train_int),
                    'train_unintentional': int(train_unint),
                    'test_features': test_features,
                    'test_variants': test_variants
                },
                'baseline': {'f1': 0.9187, 'f1_int': 0.9153, 'f1_unint': 0.9217},
                'results': results,
                'winner': best
            }, f, indent=2)
        
    def run(self, algorithm='kmeans'):
        self.load_data()
        self.create_aggregate_features()
        train_df, test_df = self.cluster_and_sample(algorithm)
        
        # Test Random Forest (baseline)
        rf_results = self.test_random_forest(train_df, test_df)
        
        # Test all 3 label propagation approaches
        approach1_results = self.test_cluster_majority_vote(train_df, test_df)
        approach2_results = self.test_feature_level_voting(train_df, test_df)
        approach3_results = self.test_label_propagation(train_df, test_df)
        
        all_results = [rf_results, approach1_results, approach2_results, approach3_results]
        self.save_results(algorithm, all_results, train_df, test_df)
        
        # Calculate sample counts
        train_variants = train_df['variant_record_id'].nunique()
        train_features = len(train_df)
        test_variants = test_df['variant_record_id'].nunique()
        test_features = len(test_df)
        total_variants = self.df['variant_record_id'].nunique()
        total_features = len(self.df)
        
        # Count labels in training
        train_int = (train_df['intent_label'] == 1).sum()
        train_unint = (train_df['intent_label'] == -1).sum()
        
        print(f"\n{'='*70}")
        print("TRAINING DATA SUMMARY")
        print(f"{'='*70}")
        print(f"Dataset: {total_features} features from {total_variants} variants")
        print(f"  Intentional: {(self.df['intent_label']==1).sum()}")
        print(f"  Unintentional: {(self.df['intent_label']==-1).sum()}")
        print(f"\nSampling: {train_variants} variants ({train_variants/total_variants*100:.2f}%)")
        print(f"  Training features: {train_features} ({train_features/total_features*100:.2f}%)")
        print(f"    ├─ Intentional labels: {train_int}")
        print(f"    └─ Unintentional labels: {train_unint}")
        print(f"  Test features: {test_features} ({test_features/total_features*100:.2f}%)")
        print(f"    └─ Test variants: {test_variants}")
        
        print(f"\n{'='*70}")
        print("METHOD COMPARISON")
        print(f"{'='*70}")
        print(f"Baseline (HDBSCAN+RF):")
        print(f"  Training: {train_variants} variants, {train_features} features")
        print(f"  F1 Score: 91.87% (91.53% int, 92.17% unint)")
        print(f"  Method: HDBSCAN clustering + Random Forest classifier")
        print(f"\nAll methods use SAME training data:")
        print(f"  {train_variants} variants = {train_features} labeled features")
        print(f"  {train_int} intentional + {train_unint} unintentional labels")
        
        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        for res in all_results:
            print(f"  {res['method']:20s}: {res['f1_weighted']:.4f} ({res['f1_intentional']:.4f} int, {res['f1_unintentional']:.4f} unint) [{res['runtime']:.2f}s]")
        
        best = max(all_results, key=lambda x: x['f1_weighted'])
        print(f"\n🏆 Winner: {best['method']} = {best['f1_weighted']:.4f}")
        
        if best['f1_weighted'] < 0.9187:
            print(f"❌ Below baseline - Random Forest wins")
        else:
            print(f"✅ Beats or matches baseline!")
            improvement = (best['f1_weighted'] - 0.9187) * 100
            print(f"   Improvement: +{improvement:.2f} percentage points")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['kmeans', 'dbscan', 'hdbscan'], default='kmeans')
    parser.add_argument('--target-samples', type=int, default=None)
    parser.add_argument('--mask-path', type=str, required=True, help='Path to masks.csv')
    parser.add_argument('--clean-data-path', type=str, required=True, help='Path to correct_records.csv')
    parser.add_argument('--dirty-data-path', type=str, required=True, help='Path to manipulated_records.csv')
    args = parser.parse_args()
    
    comp = LabelPropagationComparison(
        target_samples=args.target_samples,
        mask_path=args.mask_path,
        clean_data_path=args.clean_data_path,
        dirty_data_path=args.dirty_data_path
    )
    comp.run(algorithm=args.algorithm)
