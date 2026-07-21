"""
CLUSTERING ALGORITHM COMPARISON: NORMAL vs SMART SAMPLING
==========================================================

Tests 6 clustering algorithms with TWO sampling strategies each:
  
NORMAL SAMPLING:
  1. Cluster the data
  2. Proportional allocation to clusters
  3. Random selection within each cluster
  
SMART SAMPLING (USER'S IDEA):
  1. Cluster the data
  2. Proportional allocation to clusters (1-10 reps per cluster based on size)
  3. Stratified sampling WITHIN each cluster (by intent label)

Clustering algorithms tested:
1. K-Means: Fixed K, spherical clusters
2. DBSCAN: Density-based, finds arbitrary shapes, handles outliers
3. Hierarchical Clustering (Ward): Creates dendrogram, ward linkage
4. Hierarchical Clustering (Average): Creates dendrogram, average linkage
5. Gaussian Mixture Models: Probabilistic, soft cluster assignments
6. HDBSCAN: Hierarchical DBSCAN, automatically finds optimal clusters

Goal: Compare Normal vs Smart sampling across different clustering algorithms
"""

# Fix matplotlib backend issue - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, silhouette_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import time

warnings.filterwarnings('ignore')

# Try to import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("⚠️  HDBSCAN not available. Install with: pip install hdbscan")


class ClusteringComparison:
    """Compare different clustering algorithms for smart sampling"""
    
    def __init__(self, target_samples=None, random_state=42, 
                 mask_path=None, clean_data_path=None, dirty_data_path=None,
                 enable_detailed_logging=True):
        from datetime import datetime
        
        self.target_samples = target_samples  # Will be set to 1% of dataset if None
        self.random_state = random_state
        self.enable_detailed_logging = enable_detailed_logging
        
        # Create timestamped output folders - use script-relative paths
        # This ensures outputs always go to clustering-organized/outputs/
        # regardless of where the script is executed from
        script_dir = Path(__file__).parent.resolve()  # scripts/
        parent_dir = script_dir.parent  # clustering-organized/
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = parent_dir / 'outputs' / f'run_{self.timestamp}'
        self.results_dir = self.run_dir / 'results'
        self.plots_dir = self.run_dir / 'plots'
        self.logs_dir = self.run_dir / 'logs'
        
        # Create all output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Store custom data paths
        self.mask_path = Path(mask_path) if mask_path else None
        self.clean_data_path = Path(clean_data_path) if clean_data_path else None
        self.dirty_data_path = Path(dirty_data_path) if dirty_data_path else None
        
        # Store results
        self.all_results = []
        self.cluster_info = {}
        
        # Initialize detailed logging system (if enabled)
        self.loggers = {}
        if self.enable_detailed_logging:
            self._setup_detailed_logging()
    
    def _setup_detailed_logging(self):
        """Setup detailed logging with separate files for each component"""
        if not self.enable_detailed_logging:
            return
            
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log files for each component
        log_configs = {
            'main': f'00_main_pipeline_{timestamp}.log',
            'data_loading': f'01_data_loading_{timestamp}.log',
            'feature_creation': f'02_feature_creation_{timestamp}.log',
            'kmeans': f'03_kmeans_{timestamp}.log',
            'dbscan': f'04_dbscan_{timestamp}.log',
            'hierarchical_ward': f'05_hierarchical_ward_{timestamp}.log',
            'hierarchical_average': f'06_hierarchical_average_{timestamp}.log',
            'hierarchical_complete': f'07_hierarchical_complete_{timestamp}.log',
            'spectral': f'08_spectral_{timestamp}.log',
            'gmm': f'09_gmm_{timestamp}.log',
            'hdbscan': f'10_hdbscan_{timestamp}.log',
            'normal_sampling': f'11_normal_sampling_{timestamp}.log',
            'smart_sampling': f'12_smart_sampling_{timestamp}.log',
            'evaluation': f'13_evaluation_{timestamp}.log',
            'visualization': f'14_visualization_{timestamp}.log',
        }
        
        # Open all log files
        for key, filename in log_configs.items():
            filepath = self.logs_dir / filename
            self.loggers[key] = open(filepath, 'w', buffering=1)  # Line buffered
            
        # Write headers
        for key, log_file in self.loggers.items():
            log_file.write(f"{'='*80}\n")
            log_file.write(f"{key.upper().replace('_', ' ')} LOG\n")
            log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n\n")
            
    def _log(self, component, message, also_print=True):
        """Write to specific component log and optionally print to console"""
        if self.enable_detailed_logging and component in self.loggers:
            self.loggers[component].write(f"{message}\n")
            self.loggers[component].flush()
        
        if also_print:
            print(message)
            
    def _close_logs(self):
        """Close all log files"""
        if not self.enable_detailed_logging:
            return
            
        from datetime import datetime
        
        for key, log_file in self.loggers.items():
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"{'='*80}\n")
            log_file.close()
        
    def load_data(self):
        """Load and process the dataset"""
        self._log('main', "\n" + "="*70)
        self._log('main', "LOADING DATA")
        self._log('main', "="*70)
        
        self._log('data_loading', "Starting data loading process...")
        
        # Use custom paths if provided, otherwise auto-detect
        if self.mask_path and self.clean_data_path and self.dirty_data_path:
            # Custom paths provided
            self._log('data_loading', "Using custom data paths:")
            self._log('data_loading', f"  Mask: {self.mask_path}")
            self._log('data_loading', f"  Clean: {self.clean_data_path}")
            self._log('data_loading', f"  Dirty: {self.dirty_data_path}")
            
            if not self.mask_path.exists():
                raise FileNotFoundError(f"Mask file not found: {self.mask_path}")
            if not self.clean_data_path.exists():
                raise FileNotFoundError(f"Clean data file not found: {self.clean_data_path}")
            if not self.dirty_data_path.exists():
                raise FileNotFoundError(f"Dirty data file not found: {self.dirty_data_path}")
            
            print(f"Using custom data paths:")
            print(f"  Mask: {self.mask_path}")
            print(f"  Clean: {self.clean_data_path}")
            print(f"  Dirty: {self.dirty_data_path}")
            
            masks_df = pd.read_csv(self.mask_path)
            correct_df = pd.read_csv(self.clean_data_path)
            manipulated_df = pd.read_csv(self.dirty_data_path)
        else:
            # Auto-detect data directory
            self._log('data_loading', "Auto-detecting data directory...")
            
            # Get script directory to build absolute paths
            script_dir = Path(__file__).parent.resolve()  # scripts/
            parent_dir = script_dir.parent  # clustering-organized/
            llms_baseline_dir = parent_dir.parent  # llms_baseline/
            
            data_paths = [
                # From clustering-organized/scripts/ - symlinked data
                parent_dir / 'data' / 'raw' / 'run_20251031_211812',
                # From llms_baseline/adult_income_dataset/tenth-trial/
                llms_baseline_dir / 'adult_income_dataset' / 'tenth-trial' / 'data' / 'raw' / 'run_20251031_211812',
                # Legacy paths
                Path('data/raw/run_20251031_211812'),
                Path('../adult_income_dataset/tenth-trial/data/raw/run_20251031_211812'),
                Path('.'),
            ]
            
            self._log('data_loading', f"Searching in {len(data_paths)} potential locations...", also_print=False)
            
            data_dir = None
            for path in data_paths:
                self._log('data_loading', f"  Checking: {path.resolve()}", also_print=False)
                if (path / 'masks.csv').exists():
                    data_dir = path
                    self._log('data_loading', f"  ✓ Found data in: {path}")
                    print(f"Found data in: {path}")
                    break
            
            if data_dir is None:
                print(f"\n❌ Searched for data in:")
                for path in data_paths:
                    print(f"   - {path.resolve()}")
                raise FileNotFoundError("Could not find masks.csv in expected locations. "
                                       "Use --mask-path, --clean-data-path, --dirty-data-path to specify files.")
            
            print(f"Auto-detected data directory: {data_dir}")
            
            masks_df = pd.read_csv(data_dir / 'masks.csv')
            manipulated_df = pd.read_csv(data_dir / 'manipulated_records.csv')
            
            # correct_records.csv is in parent directory
            correct_path = data_dir.parent / 'correct_records.csv'
            if not correct_path.exists():
                correct_path = Path('data/raw/correct_records.csv')
            correct_df = pd.read_csv(correct_path)
        
        self._log('data_loading', f"Loaded masks: {masks_df.shape}")
        self._log('data_loading', f"Loaded manipulated records: {manipulated_df.shape}")
        self._log('data_loading', f"Loaded correct records: {correct_df.shape}")
        
        print(f"✓ Loaded masks: {masks_df.shape}")
        print(f"✓ Loaded manipulated records: {manipulated_df.shape}")
        print(f"✓ Loaded correct records: {correct_df.shape}")
        
        # Process into feature changes
        feature_cols = masks_df.columns.tolist()
        
        self._log('data_loading', f"\nProcessing {len(masks_df)} variant records into feature changes...")
        self._log('data_loading', f"Features to process: {len(feature_cols)}")
        
        all_changes = []
        print(f"\nProcessing {len(masks_df)} variant records...")
        
        for idx in tqdm(range(len(masks_df)), desc="Processing records", unit="record"):
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
                    'variant_record_id': idx,  # Each variant is treated separately
                    'original_record_id': original_record_idx,
                    'variant_idx': variant_idx,
                    'feature_name': feature,
                    'original_value': original_value,
                    'new_value': new_value,
                    'change_magnitude': magnitude,
                    'intent_label': intent_label
                })
        
        self.df = pd.DataFrame(all_changes)
        
        self._log('data_loading', f"\n✓ Created dataset with {len(self.df)} feature changes")
        self._log('data_loading', f"  Intentional (1): {(self.df['intent_label'] == 1).sum()}")
        self._log('data_loading', f"  Unintentional (-1): {(self.df['intent_label'] == -1).sum()}")
        self._log('data_loading', f"  Unique variant records: {self.df['variant_record_id'].nunique()}")
        
        print(f"\n✓ Created dataset with {len(self.df)} feature changes")
        print(f"  Intentional (1): {(self.df['intent_label'] == 1).sum()}")
        print(f"  Unintentional (-1): {(self.df['intent_label'] == -1).sum()}")
        print(f"  Unique variant records: {self.df['variant_record_id'].nunique()}")
        
        # Encode categorical features
        self._log('data_loading', "Encoding categorical features...", also_print=False)
        for col in ['feature_name', 'original_value', 'new_value']:
            self.df[f'{col}_encoded'] = pd.Categorical(self.df[col].astype(str)).codes
        
        # Convert to numeric where possible for derived features
        def safe_numeric(series):
            return pd.to_numeric(series, errors='coerce').fillna(0)
        
        original_numeric = safe_numeric(self.df['original_value'])
        new_numeric = safe_numeric(self.df['new_value'])
        
        # Add derived features
        self._log('data_loading', "Adding derived features (relative_change, change_direction)...", also_print=False)
        self.df['relative_change'] = self.df['change_magnitude'] / (original_numeric.abs() + 1)
        self.df['change_direction'] = np.sign(new_numeric - original_numeric)
        
        self._log('data_loading', "Data loading complete!\n")
        self.df['original_log'] = np.log1p(original_numeric.abs())
        self.df['new_log'] = np.log1p(new_numeric.abs())
        self.df['original_magnitude'] = original_numeric.abs()
        self.df['new_magnitude'] = new_numeric.abs()
        
        # Add feature type indicators
        for feat in self.df['feature_name'].unique():
            self.df[f'feat_{feat}'] = (self.df['feature_name'] == feat).astype(int)
        
        # Store total dataset info for evaluation on ALL unseen data
        self.total_variants = self.df['variant_record_id'].nunique()
        self.total_samples = len(self.df)
        
        # Set target_samples to 1% of total variants if not specified
        if self.target_samples is None:
            self.target_samples = max(1, int(self.total_variants * 0.01))
            print(f"\n⚙️  Auto-calculated target_samples: {self.target_samples} (1% of {self.total_variants} variants)")
            self._log('data_loading', f"Auto-calculated target_samples: {self.target_samples} (1% of {self.total_variants} variants)")
        
        print(f"\nTotal dataset:")
        print(f"  Total variants: {self.total_variants}")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Target samples for training: {self.target_samples} ({self.target_samples/self.total_variants*100:.2f}% of variants)")
        print(f"  Intentional: {(self.df['intent_label'] == 1).sum()}")
        print(f"  Unintentional: {(self.df['intent_label'] == -1).sum()}")
        
    def create_aggregate_features(self, df):
        """
        Create aggregate features per variant for clustering.
        
        CRITICAL: Does NOT use intent labels to avoid data leakage!
        Intent labels are what we're trying to predict, so they shouldn't
        influence the clustering.
        
        Creates 10 features per variant:
        - 1 count feature (n_changes)
        - 5 magnitude statistics
        - 2 encoded value statistics  
        - 2 derived features
        """
        aggregated = []
        
        for variant_id in tqdm(df['variant_record_id'].unique(), 
                              desc="Aggregating features"):
            variant_data = df[df['variant_record_id'] == variant_id]
            
            # Basic counts (NO INTENT LABELS - avoid data leakage!)
            n_changes = len(variant_data)
            
            # Change magnitude statistics (use pre-computed if available)
            if 'change_magnitude' in variant_data.columns:
                mean_magnitude = variant_data['change_magnitude'].mean()
                std_magnitude = variant_data['change_magnitude'].std() if len(variant_data) > 1 else 0
                min_magnitude = variant_data['change_magnitude'].min()
                max_magnitude = variant_data['change_magnitude'].max()
                median_magnitude = variant_data['change_magnitude'].median()
            else:
                # Fallback to computing
                mean_magnitude = 0
                std_magnitude = 0
                min_magnitude = 0
                max_magnitude = 0
                median_magnitude = 0
            
            # Use encoded values if available (better than string comparisons)
            if 'new_value_encoded' in variant_data.columns:
                min_new_value_encoded = variant_data['new_value_encoded'].min()
                max_new_value_encoded = variant_data['new_value_encoded'].max()
            else:
                min_new_value_encoded = 0
                max_new_value_encoded = 0
            
            # Additional derived features (if available)
            mean_relative_change = variant_data.get('relative_change', pd.Series([0])).mean()
            
            # Feature with largest magnitude change
            if 'change_magnitude' in variant_data.columns and len(variant_data) > 0:
                max_change_idx = variant_data['change_magnitude'].idxmax()
                feature_with_max_change = variant_data.loc[max_change_idx, 'feature_name_encoded'] if 'feature_name_encoded' in variant_data.columns else 0
            else:
                feature_with_max_change = 0
            
            # Build feature dictionary - 10 features (removed intent labels to avoid data leakage)
            features = {
                'variant_record_id': variant_id,
                
                # Count features
                'n_changes': n_changes,
                
                # Magnitude statistics
                'mean_magnitude': mean_magnitude,
                'std_magnitude': std_magnitude,
                'min_magnitude': min_magnitude,
                'max_magnitude': max_magnitude,
                'median_magnitude': median_magnitude,
                
                # Encoded value statistics
                'min_new_value_encoded': min_new_value_encoded,
                'max_new_value_encoded': max_new_value_encoded,
                
                # Derived features
                'mean_relative_change': mean_relative_change,
                'feature_with_max_change_encoded': feature_with_max_change,
            }
            
            aggregated.append(features)
        
        agg_df = pd.DataFrame(aggregated).fillna(0)
        
        print(f"\n✓ Created {len(agg_df)} aggregate feature vectors")
        print(f"  Feature dimension: {len(agg_df.columns) - 1}")  # Exclude variant_record_id
        
        return agg_df
    
    def cluster_kmeans(self, features, n_clusters=15):
        """K-Means clustering (baseline)"""
        print(f"\n{'='*70}")
        print("K-MEANS CLUSTERING (BASELINE)")
        print(f"{'='*70}")
        
        self._log('kmeans', f"\n{'='*70}")
        self._log('kmeans', "K-MEANS CLUSTERING STARTED", also_print=False)
        self._log('kmeans', f"{'='*70}", also_print=False)
        self._log('kmeans', f"Parameters:", also_print=False)
        self._log('kmeans', f"  n_clusters: {n_clusters}", also_print=False)
        self._log('kmeans', f"  random_state: {self.random_state}", also_print=False)
        self._log('kmeans', f"  n_init: 10", also_print=False)
        self._log('kmeans', f"Feature matrix shape: {features.shape}", also_print=False)
        
        start_time = time.time()
        
        self._log('kmeans', "Fitting K-Means model...", also_print=False)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(features)
        
        self._log('kmeans', "Computing metrics...", also_print=False)
        silhouette = silhouette_score(features, labels)
        runtime = time.time() - start_time
        
        # Log cluster distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        self._log('kmeans', f"\nCluster distribution:", also_print=False)
        for label, count in zip(unique_labels, counts):
            self._log('kmeans', f"  Cluster {label}: {count} points ({count/len(labels)*100:.1f}%)", also_print=False)
        
        self._log('kmeans', f"\nResults:", also_print=False)
        self._log('kmeans', f"  Silhouette Score: {silhouette:.4f}", also_print=False)
        self._log('kmeans', f"  Runtime: {runtime:.2f}s", also_print=False)
        self._log('kmeans', f"  Clusters found: {len(unique_labels)}", also_print=False)
        self._log('kmeans', f"  Inertia: {kmeans.inertia_:.2f}", also_print=False)
        
        print(f"✓ K-Means with K={n_clusters}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Clusters found: {len(np.unique(labels))}")
        
        return labels, {
            'algorithm': 'K-Means',
            'n_clusters': len(np.unique(labels)),
            'silhouette': silhouette,
            'runtime': runtime,
            'params': {'n_clusters': n_clusters}
        }
    
    def cluster_dbscan(self, features, eps=None, min_samples=5):
        """DBSCAN clustering"""
        print(f"\n{'='*70}")
        print("DBSCAN CLUSTERING")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Auto-tune eps if not provided
        if eps is None:
            # Use k-distance graph to estimate eps
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=min_samples)
            neighbors_fit = neighbors.fit(features)
            distances, indices = neighbors_fit.kneighbors(features)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 90)  # Use 90th percentile
            
            # Safety check: eps must be > 0
            if eps <= 0 or np.isnan(eps):
                # Fallback: use mean distance or a small default
                mean_dist = np.mean(distances[distances > 0]) if np.any(distances > 0) else 0.5
                eps = max(mean_dist, 0.5)  # Ensure at least 0.5
                print(f"  ⚠️  Auto-tuned eps was {np.percentile(distances, 90):.6f}, using fallback: {eps:.4f}")
            else:
                print(f"  Auto-tuned eps: {eps:.4f}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features)
        
        # Calculate metrics (excluding noise points for silhouette)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(labels) - 1:
            # Only calculate silhouette if we have valid clusters
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                silhouette = silhouette_score(features[valid_mask], labels[valid_mask])
            else:
                silhouette = -1.0
        else:
            silhouette = -1.0
        
        runtime = time.time() - start_time
        
        print(f"✓ DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        return labels, {
            'algorithm': 'DBSCAN',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'runtime': runtime,
            'params': {'eps': eps, 'min_samples': min_samples}
        }
    
    def cluster_hierarchical(self, features, n_clusters=15, linkage_method='ward'):
        """Hierarchical Clustering with dendrogram"""
        print(f"\n{'='*70}")
        print("HIERARCHICAL CLUSTERING")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        labels = hierarchical.fit_predict(features)
        
        silhouette = silhouette_score(features, labels)
        runtime = time.time() - start_time
        
        print(f"✓ Hierarchical Clustering with {linkage_method} linkage")
        print(f"  Clusters: {n_clusters}")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        # Create dendrogram (sample for visualization if too large)
        if len(features) > 1000:
            sample_indices = np.random.choice(len(features), 1000, replace=False)
            sample_features = features[sample_indices]
            print(f"  Creating dendrogram with sample of 1000 points...")
        else:
            sample_features = features
        
        linkage_matrix = linkage(sample_features, method=linkage_method)
        
        # Plot dendrogram
        plt.figure(figsize=(15, 8))
        dendrogram(linkage_matrix, 
                   truncate_mode='lastp',
                   p=30,
                   show_leaf_counts=True)
        plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
        plt.xlabel('Cluster Size')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'dendrogram_{linkage_method}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved dendrogram to dendrogram_{linkage_method}.png")
        
        return labels, {
            'algorithm': f'Hierarchical-{linkage_method}',
            'n_clusters': len(np.unique(labels)),
            'silhouette': silhouette,
            'runtime': runtime,
            'params': {'n_clusters': n_clusters, 'linkage': linkage_method}
        }
    
    def cluster_gmm(self, features, n_components=15):
        """Gaussian Mixture Models"""
        print(f"\n{'='*70}")
        print("GAUSSIAN MIXTURE MODELS")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            random_state=self.random_state,
            n_init=10
        )
        labels = gmm.fit_predict(features)
        
        # Get soft cluster probabilities
        probabilities = gmm.predict_proba(features)
        
        silhouette = silhouette_score(features, labels)
        runtime = time.time() - start_time
        
        # Additional GMM-specific metrics
        bic = gmm.bic(features)
        aic = gmm.aic(features)
        
        print(f"✓ Gaussian Mixture Model with {n_components} components")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  BIC: {bic:.2f}")
        print(f"  AIC: {aic:.2f}")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Converged: {gmm.converged_}")
        
        return labels, {
            'algorithm': 'GMM',
            'n_clusters': len(np.unique(labels)),
            'silhouette': silhouette,
            'bic': bic,
            'aic': aic,
            'runtime': runtime,
            'converged': gmm.converged_,
            'probabilities': probabilities,
            'params': {'n_components': n_components}
        }
    
    def cluster_hdbscan(self, features, min_cluster_size=5, min_samples=5):
        """HDBSCAN clustering"""
        if not HDBSCAN_AVAILABLE:
            print("\n⚠️  HDBSCAN not available, skipping...")
            return None, None
        
        print(f"\n{'='*70}")
        print("HDBSCAN CLUSTERING")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(features)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(labels) - 1:
            valid_mask = labels != -1
            if valid_mask.sum() > 1:
                silhouette = silhouette_score(features[valid_mask], labels[valid_mask])
            else:
                silhouette = -1.0
        else:
            silhouette = -1.0
        
        runtime = time.time() - start_time
        
        print(f"✓ HDBSCAN")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Runtime: {runtime:.2f}s")
        
        # Plot cluster stability
        if hasattr(clusterer, 'cluster_persistence_'):
            persistence = clusterer.cluster_persistence_
            print(f"  Cluster persistence scores available: {len(persistence)}")
        
        return labels, {
            'algorithm': 'HDBSCAN',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette,
            'runtime': runtime,
            'params': {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}
        }
    
    def normal_sampling(self, agg_df, labels, cluster_info):
        """
        NORMAL CLUSTERING SAMPLING:
        1. Cluster the data
        2. Sample proportionally from each cluster (simple proportional allocation)
        3. No stratification - just random selection within each cluster
        """
        self._log('normal_sampling', "Starting normal sampling...", also_print=False)
        
        # Handle noise points (-1 labels) for DBSCAN/HDBSCAN
        valid_mask = labels != -1
        valid_indices = np.where(valid_mask)[0]
        valid_labels = labels[valid_mask]
        
        self._log('normal_sampling', f"Total points: {len(labels)}, Valid (non-noise): {len(valid_indices)}", also_print=False)
        
        if len(valid_indices) == 0:
            print("⚠️  No valid clusters found!")
            self._log('normal_sampling', "⚠️  No valid clusters found!")
            return []
        
        # Get variant IDs for valid points
        valid_variants = agg_df.iloc[valid_indices]['variant_record_id'].values
        
        # Count variants per cluster
        unique_labels = np.unique(valid_labels)
        cluster_sizes = {label: (valid_labels == label).sum() for label in unique_labels}
        
        self._log('normal_sampling', f"Number of clusters: {len(unique_labels)}", also_print=False)
        self._log('normal_sampling', f"Cluster sizes: {cluster_sizes}", also_print=False)
        
        # Simple proportional allocation
        total_variants = len(valid_indices)
        allocation = {}
        
        for label in unique_labels:
            base_allocation = (cluster_sizes[label] / total_variants) * self.target_samples
            allocation[label] = max(1, int(round(base_allocation)))
        
        self._log('normal_sampling', f"Initial allocation: {allocation}", also_print=False)
        
        for label in unique_labels:
            base_allocation = (cluster_sizes[label] / total_variants) * self.target_samples
            allocation[label] = max(1, int(round(base_allocation)))
        
        # Adjust to match target
        total_allocated = sum(allocation.values())
        if total_allocated > self.target_samples:
            # Remove from largest clusters
            while total_allocated > self.target_samples:
                largest_cluster = max(allocation.items(), key=lambda x: x[1])[0]
                if allocation[largest_cluster] > 1:
                    allocation[largest_cluster] -= 1
                    total_allocated -= 1
                else:
                    break
        elif total_allocated < self.target_samples:
            # Add to largest clusters
            while total_allocated < self.target_samples:
                largest_cluster = max(cluster_sizes.items(), key=lambda x: x[1])[0]
                allocation[largest_cluster] += 1
                total_allocated += 1
        
        # Sample randomly from each cluster
        selected_variants = []
        cluster_details = {}  # Store detailed cluster information
        
        for label in unique_labels:
            n_samples = allocation[label]
            cluster_mask = valid_labels == label
            cluster_variant_ids = valid_variants[cluster_mask]
            
            self._log('normal_sampling', f"\nCluster {label}:", also_print=False)
            self._log('normal_sampling', f"  Total variants in cluster: {len(cluster_variant_ids)}", also_print=False)
            self._log('normal_sampling', f"  Allocated samples: {n_samples}", also_print=False)
            self._log('normal_sampling', f"  All variant IDs in cluster: {sorted(cluster_variant_ids.tolist())}", also_print=False)
            
            # Random selection
            if len(cluster_variant_ids) >= n_samples:
                sampled = np.random.choice(cluster_variant_ids, size=n_samples, replace=False)
            else:
                sampled = cluster_variant_ids
                self._log('normal_sampling', f"  Warning: Cluster has fewer variants than allocated", also_print=False)
            
            # Log picked representatives
            self._log('normal_sampling', f"  PICKED REPRESENTATIVES: {sorted(sampled.tolist())}", also_print=False)
            
            # Store cluster details
            cluster_details[int(label)] = {
                'all_variants': sorted(cluster_variant_ids.tolist()),
                'picked_representatives': sorted(sampled.tolist()),
                'cluster_size': len(cluster_variant_ids),
                'num_picked': len(sampled)
            }
            
            selected_variants.extend(sampled)
        
        print(f"\n  Selected {len(selected_variants)} variants from {len(unique_labels)} clusters")
        
        # Get training samples
        train_sample = self.df[self.df['variant_record_id'].isin(selected_variants)]
        
        self._log('normal_sampling', f"\nFinal selection:", also_print=False)
        self._log('normal_sampling', f"  Total variants: {len(selected_variants)}", also_print=False)
        self._log('normal_sampling', f"  Total samples: {len(train_sample)}", also_print=False)
        self._log('normal_sampling', f"  Intentional: {(train_sample['intent_label'] == 1).sum()}", also_print=False)
        self._log('normal_sampling', f"  Unintentional: {(train_sample['intent_label'] == -1).sum()}", also_print=False)
        
        # Log detailed cluster summary
        self._log('normal_sampling', f"\n{'='*80}", also_print=False)
        self._log('normal_sampling', f"CLUSTER MEMBERSHIP & REPRESENTATIVES SUMMARY", also_print=False)
        self._log('normal_sampling', f"{'='*80}", also_print=False)
        for cluster_id, details in sorted(cluster_details.items()):
            self._log('normal_sampling', f"\nCluster {cluster_id}:", also_print=False)
            self._log('normal_sampling', f"  Size: {details['cluster_size']} variants", also_print=False)
            self._log('normal_sampling', f"  Picked: {details['num_picked']} representatives", also_print=False)
            self._log('normal_sampling', f"  All members: {details['all_variants']}", also_print=False)
            self._log('normal_sampling', f"  Representatives: {details['picked_representatives']}", also_print=False)
        self._log('normal_sampling', f"{'='*80}\n", also_print=False)
        
        print(f"  Total samples: {len(train_sample)}")
        print(f"  Intentional: {(train_sample['intent_label'] == 1).sum()}, "
              f"Unintentional: {(train_sample['intent_label'] == -1).sum()}")
        
        return selected_variants
    
    def smart_sampling(self, agg_df, labels, cluster_info):
        """
        USER'S SMART SAMPLING IDEA:
        1. Cluster the data
        2. Proportional allocation to clusters (1-10 reps per cluster)
        3. Stratified sampling WITHIN each cluster (by intent label)
        """
        self._log('smart_sampling', "Starting smart sampling...", also_print=False)
        
        # Handle noise points (-1 labels) for DBSCAN/HDBSCAN
        valid_mask = labels != -1
        valid_indices = np.where(valid_mask)[0]
        valid_labels = labels[valid_mask]
        
        self._log('smart_sampling', f"Total points: {len(labels)}, Valid (non-noise): {len(valid_indices)}", also_print=False)
        
        if len(valid_indices) == 0:
            print("⚠️  No valid clusters found!")
            self._log('smart_sampling', "⚠️  No valid clusters found!")
            return []
        
        # Count variants per cluster
        unique_labels = np.unique(valid_labels)
        cluster_sizes = {label: (valid_labels == label).sum() for label in unique_labels}
        
        self._log('smart_sampling', f"Number of clusters: {len(unique_labels)}", also_print=False)
        self._log('smart_sampling', f"Cluster sizes: {cluster_sizes}", also_print=False)
        
        # STEP 1: Proportional allocation (minimum 1, maximum 10 per cluster)
        MIN_REPS = 1
        MAX_REPS = 10
        total_variants = len(valid_indices)
        allocation = {}
        
        self._log('smart_sampling', f"Allocation constraints: MIN={MIN_REPS}, MAX={MAX_REPS}", also_print=False)
        
        for label in unique_labels:
            base_allocation = (cluster_sizes[label] / total_variants) * self.target_samples
            allocation[label] = max(MIN_REPS, min(MAX_REPS, round(base_allocation)))
        
        self._log('smart_sampling', f"Initial allocation: {allocation}", also_print=False)
        
        # Adjust to reach exact target
        current_total = sum(allocation.values())
        while current_total < self.target_samples and len(allocation) > 0:
            # Add to largest cluster (ignore MAX_REPS when filling to target)
            largest = max(allocation.items(), key=lambda x: cluster_sizes[x[0]])[0]
            allocation[largest] += 1
            current_total += 1
        
        while current_total > self.target_samples:
            # Remove from largest allocation (if > MIN_REPS)
            largest = max(allocation.items(), key=lambda x: x[1])[0]
            if allocation[largest] > MIN_REPS:
                allocation[largest] -= 1
                current_total -= 1
            else:
                break
        
        # STEP 2: Stratified sampling WITHIN each cluster
        selected_variants = []
        intentional_samples = 0
        unintentional_samples = 0
        cluster_details = {}  # Store detailed cluster information
        
        for label, n_reps in allocation.items():
            # Get all variants in this cluster
            cluster_mask = valid_labels == label
            cluster_indices = valid_indices[cluster_mask]
            cluster_variant_ids = agg_df.iloc[cluster_indices]['variant_record_id'].values
            
            self._log('smart_sampling', f"\n{'─'*80}", also_print=False)
            self._log('smart_sampling', f"Cluster {label}:", also_print=False)
            self._log('smart_sampling', f"  Total variants in cluster: {len(cluster_variant_ids)}", also_print=False)
            self._log('smart_sampling', f"  Allocated samples: {n_reps}", also_print=False)
            self._log('smart_sampling', f"  All variant IDs in cluster: {sorted(cluster_variant_ids.tolist())}", also_print=False)
            
            # Get intent labels for variants in this cluster from ALL data
            cluster_data = self.df[self.df['variant_record_id'].isin(cluster_variant_ids)]
            
            # Calculate intent ratio per variant
            variant_intents = {}
            for vid in cluster_variant_ids:
                variant_changes = cluster_data[cluster_data['variant_record_id'] == vid]
                intentional_count = (variant_changes['intent_label'] == 1).sum()
                total_count = len(variant_changes)
                variant_intents[vid] = intentional_count / total_count if total_count > 0 else 0
            
            # Split into intentional-dominant vs unintentional-dominant
            intentional_variants = [vid for vid, ratio in variant_intents.items() if ratio >= 0.5]
            unintentional_variants = [vid for vid, ratio in variant_intents.items() if ratio < 0.5]
            
            self._log('smart_sampling', f"  Intent breakdown:", also_print=False)
            self._log('smart_sampling', f"    Intentional-dominant variants: {len(intentional_variants)}", also_print=False)
            self._log('smart_sampling', f"      IDs: {sorted(intentional_variants)}", also_print=False)
            self._log('smart_sampling', f"    Unintentional-dominant variants: {len(unintentional_variants)}", also_print=False)
            self._log('smart_sampling', f"      IDs: {sorted(unintentional_variants)}", also_print=False)
            
            # Stratified sampling within cluster
            if len(intentional_variants) > 0 and len(unintentional_variants) > 0:
                # Both classes present - sample proportionally
                n_intentional = max(1, round(n_reps * len(intentional_variants) / len(cluster_variant_ids)))
                n_unintentional = n_reps - n_intentional
                
                self._log('smart_sampling', f"  Stratified sampling (both classes present):", also_print=False)
                self._log('smart_sampling', f"    Sampling {n_intentional} intentional, {n_unintentional} unintentional", also_print=False)
                
                # Sample from each group
                sampled_int = np.random.choice(intentional_variants, 
                                              size=min(n_intentional, len(intentional_variants)), 
                                              replace=False)
                sampled_unint = np.random.choice(unintentional_variants,
                                                size=min(n_unintentional, len(unintentional_variants)),
                                                replace=False)
                
                selected = list(sampled_int) + list(sampled_unint)
            elif len(intentional_variants) > 0:
                # Only intentional variants
                self._log('smart_sampling', f"  Only intentional variants available", also_print=False)
                selected = list(np.random.choice(intentional_variants, 
                                                size=min(n_reps, len(intentional_variants)), 
                                                replace=False))
            elif len(unintentional_variants) > 0:
                # Only unintentional variants
                self._log('smart_sampling', f"  Only unintentional variants available", also_print=False)
                selected = list(np.random.choice(unintentional_variants,
                                                size=min(n_reps, len(unintentional_variants)),
                                                replace=False))
            else:
                selected = []
            
            # Log picked representatives
            self._log('smart_sampling', f"  PICKED REPRESENTATIVES: {sorted(selected)}", also_print=False)
            
            # Store cluster details
            cluster_details[int(label)] = {
                'all_variants': sorted(cluster_variant_ids.tolist()),
                'intentional_variants': sorted(intentional_variants),
                'unintentional_variants': sorted(unintentional_variants),
                'picked_representatives': sorted(selected),
                'cluster_size': len(cluster_variant_ids),
                'num_picked': len(selected)
            }
            
            self._log('smart_sampling', f"  Selected {len(selected)} variants total", also_print=False)
            
            selected_variants.extend(selected)
            
            # Count actual samples
            for vid in selected:
                variant_samples = cluster_data[cluster_data['variant_record_id'] == vid]
                intentional_samples += (variant_samples['intent_label'] == 1).sum()
                unintentional_samples += (variant_samples['intent_label'] == -1).sum()
        
        self._log('smart_sampling', f"\nFinal selection:", also_print=False)
        self._log('smart_sampling', f"  Total variants: {len(selected_variants)}", also_print=False)
        self._log('smart_sampling', f"  Total samples: {intentional_samples + unintentional_samples}", also_print=False)
        self._log('smart_sampling', f"  Intentional: {intentional_samples}", also_print=False)
        self._log('smart_sampling', f"  Unintentional: {unintentional_samples}", also_print=False)
        
        # Log detailed cluster summary
        self._log('smart_sampling', f"\n{'='*80}", also_print=False)
        self._log('smart_sampling', f"CLUSTER MEMBERSHIP & REPRESENTATIVES SUMMARY", also_print=False)
        self._log('smart_sampling', f"{'='*80}", also_print=False)
        for cluster_id, details in sorted(cluster_details.items()):
            self._log('smart_sampling', f"\nCluster {cluster_id}:", also_print=False)
            self._log('smart_sampling', f"  Size: {details['cluster_size']} variants", also_print=False)
            self._log('smart_sampling', f"  Picked: {details['num_picked']} representatives", also_print=False)
            self._log('smart_sampling', f"  All members: {details['all_variants']}", also_print=False)
            self._log('smart_sampling', f"  Intentional-dominant: {details['intentional_variants']}", also_print=False)
            self._log('smart_sampling', f"  Unintentional-dominant: {details['unintentional_variants']}", also_print=False)
            self._log('smart_sampling', f"  Representatives: {details['picked_representatives']}", also_print=False)
        self._log('smart_sampling', f"{'='*80}\n", also_print=False)
        
        print(f"\n  Selected {len(selected_variants)} variants from {len(unique_labels)} clusters")
        print(f"  Total samples: {intentional_samples + unintentional_samples}")
        print(f"  Intentional: {intentional_samples}, Unintentional: {unintentional_samples}")
        
        return selected_variants
    
    def train_and_evaluate(self, algorithm_name, selected_variants):
        """Train classifier and evaluate on ALL UNSEEN data"""
        # Prepare training data from selected variants
        train_sample = self.df[self.df['variant_record_id'].isin(selected_variants)]
        
        # Prepare test data: ALL data EXCEPT selected training variants
        test_sample = self.df[~self.df['variant_record_id'].isin(selected_variants)]
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['variant_record_id', 'original_record_id', 
                                     'feature_name', 'original_value', 'new_value', 
                                     'intent_label']]
        
        X_train = train_sample[feature_cols]
        y_train = train_sample['intent_label']
        X_test = test_sample[feature_cols]
        y_test = test_sample['intent_label']
        
        # Calculate percentages
        pct_train_variants = (len(selected_variants) / self.total_variants) * 100
        pct_train_samples = (len(train_sample) / self.total_samples) * 100
        pct_test_samples = (len(test_sample) / self.total_samples) * 100
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_intent = f1_score(y_test, y_pred, pos_label=1)
        f1_unintent = f1_score(y_test, y_pred, pos_label=-1)
        
        print(f"\n{'='*70}")
        print(f"EVALUATION: {algorithm_name}")
        print(f"{'='*70}")
        print(f"  Training variants: {len(selected_variants)} ({pct_train_variants:.2f}% of {self.total_variants})")
        print(f"  Training samples: {len(train_sample)} ({pct_train_samples:.2f}% of {self.total_samples})")
        print(f"  Test samples: {len(test_sample)} ({pct_test_samples:.2f}% of {self.total_samples})")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Weighted: {f1:.4f}")
        print(f"  F1 Intentional: {f1_intent:.4f}")
        print(f"  F1 Unintentional: {f1_unintent:.4f}")
        
        return {
            'algorithm': algorithm_name,
            'n_variants': len(selected_variants),
            'n_train_samples': len(train_sample),
            'n_test_samples': len(test_sample),
            'total_variants': self.total_variants,
            'total_samples': self.total_samples,
            'pct_train_variants': pct_train_variants,
            'pct_train_samples': pct_train_samples,
            'pct_test_samples': pct_test_samples,
            'accuracy': accuracy,
            'f1_weighted': f1,
            'f1_intentional': f1_intent,
            'f1_unintentional': f1_unintent
        }
    
    def run_comparison(self):
        """Run complete comparison of all clustering algorithms"""
        print("\n" + "="*70)
        print("CLUSTERING ALGORITHM COMPARISON")
        print("NORMAL vs SMART SAMPLING")
        print("="*70)
        if self.target_samples is None:
            print(f"Target samples: Auto (1% of dataset)")
        else:
            print(f"Target samples: {self.target_samples}")
        print(f"Random state: {self.random_state}")
        print("\nTesting TWO strategies for EACH clustering algorithm:")
        print("\n  NORMAL SAMPLING:")
        print("    1. Cluster the data")
        print("    2. Proportional allocation to clusters")
        print("    3. Random selection within clusters")
        print("\n  SMART SAMPLING (User's Idea):")
        print("    1. Cluster the data")
        print("    2. Proportional allocation (1-10 reps per cluster)")
        print("    3. Stratified sampling within clusters (by intent)")
        
        # Load data (this will auto-calculate target_samples if None)
        self.load_data()
        
        # Create aggregate features from ALL data
        print(f"\n{'='*70}")
        print("CREATING AGGREGATE FEATURES FROM ALL DATA")
        print(f"{'='*70}")
        
        agg_df = self.create_aggregate_features(self.df)
        
        # Prepare features for clustering
        feature_cols = [col for col in agg_df.columns if col != 'variant_record_id']
        features = agg_df[feature_cols].values
        
        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"✓ Created {len(agg_df)} aggregate feature vectors")
        print(f"  Feature dimension: {features_scaled.shape[1]}")
        
        # Test all algorithms WITH BOTH SAMPLING STRATEGIES
        algorithms = [
            ('K-Means', lambda: self.cluster_kmeans(features_scaled, n_clusters=15)),
            ('DBSCAN', lambda: self.cluster_dbscan(features_scaled)),
            ('Hierarchical-Ward', lambda: self.cluster_hierarchical(features_scaled, n_clusters=15, linkage_method='ward')),
            ('Hierarchical-Average', lambda: self.cluster_hierarchical(features_scaled, n_clusters=15, linkage_method='average')),
            ('GMM', lambda: self.cluster_gmm(features_scaled, n_components=15)),
            ('HDBSCAN', lambda: self.cluster_hdbscan(features_scaled)),
        ]
        
        results = []
        
        for algo_idx, (name, cluster_func) in enumerate(algorithms, 1):
            try:
                self._log('main', f"\n{'='*70}")
                self._log('main', f"ALGORITHM {algo_idx}/{len(algorithms)}: {name}")
                self._log('main', f"{'='*70}")
                
                labels, cluster_info = cluster_func()
                
                if labels is None:
                    self._log('main', f"⚠️  {name} returned no labels, skipping...")
                    continue
                
                # ===== TEST 1: NORMAL SAMPLING =====
                print(f"\n{'─'*70}")
                print(f"Testing {name} with NORMAL SAMPLING")
                print(f"{'─'*70}")
                
                self._log('main', f"\n--- NORMAL SAMPLING for {name} ---")
                self._log('normal_sampling', f"\n{'='*70}")
                self._log('normal_sampling', f"NORMAL SAMPLING: {name}")
                self._log('normal_sampling', f"{'='*70}")
                self._log('normal_sampling', f"Cluster info: {cluster_info}")
                
                selected_variants_normal = self.normal_sampling(agg_df, labels, cluster_info)
                
                self._log('normal_sampling', f"Selected {len(selected_variants_normal)} variants")
                self._log('normal_sampling', f"Variants: {selected_variants_normal[:10]}{'...' if len(selected_variants_normal) > 10 else ''}", also_print=False)
                
                if len(selected_variants_normal) > 0:
                    self._log('evaluation', f"\n{'='*70}")
                    self._log('evaluation', f"EVALUATING: {name} (Normal)")
                    self._log('evaluation', f"{'='*70}")
                    
                    eval_results_normal = self.train_and_evaluate(
                        f"{name} (Normal)", 
                        selected_variants_normal
                    )
                    
                    self._log('evaluation', f"Results: {eval_results_normal}", also_print=False)
                    
                    combined_normal = {
                        **cluster_info,
                        **eval_results_normal,
                        'sampling_strategy': 'Normal'
                    }
                    # Update algorithm name to include strategy
                    combined_normal['algorithm'] = f"{name} (Normal)"
                    results.append(combined_normal)
                    
                    self._log('main', f"✓ {name} (Normal) - F1: {eval_results_normal.get('f1_weighted', 0):.4f}")
                else:
                    print(f"⚠️  No variants selected for {name} (Normal), skipping...")
                    self._log('main', f"⚠️  No variants selected for {name} (Normal)")
                
                # ===== TEST 2: SMART SAMPLING =====
                print(f"\n{'─'*70}")
                print(f"Testing {name} with SMART SAMPLING")
                print(f"{'─'*70}")
                
                self._log('main', f"\n--- SMART SAMPLING for {name} ---")
                self._log('smart_sampling', f"\n{'='*70}")
                self._log('smart_sampling', f"SMART SAMPLING: {name}")
                self._log('smart_sampling', f"{'='*70}")
                self._log('smart_sampling', f"Cluster info: {cluster_info}")
                
                selected_variants_smart = self.smart_sampling(agg_df, labels, cluster_info)
                
                self._log('smart_sampling', f"Selected {len(selected_variants_smart)} variants")
                self._log('smart_sampling', f"Variants: {selected_variants_smart[:10]}{'...' if len(selected_variants_smart) > 10 else ''}", also_print=False)
                
                if len(selected_variants_smart) > 0:
                    self._log('evaluation', f"\n{'='*70}")
                    self._log('evaluation', f"EVALUATING: {name} (Smart)")
                    self._log('evaluation', f"{'='*70}")
                    
                    eval_results_smart = self.train_and_evaluate(
                        f"{name} (Smart)", 
                        selected_variants_smart
                    )
                    
                    self._log('evaluation', f"Results: {eval_results_smart}", also_print=False)
                    
                    combined_smart = {
                        **cluster_info,
                        **eval_results_smart,
                        'sampling_strategy': 'Smart'
                    }
                    # Update algorithm name to include strategy
                    combined_smart['algorithm'] = f"{name} (Smart)"
                    results.append(combined_smart)
                    
                    self._log('main', f"✓ {name} (Smart) - F1: {eval_results_smart.get('f1_weighted', 0):.4f}")
                else:
                    print(f"⚠️  No variants selected for {name} (Smart), skipping...")
                    self._log('main', f"⚠️  No variants selected for {name} (Smart)")
                
            except Exception as e:
                error_msg = f"⚠️  Error with {name}: {e}"
                print(error_msg)
                self._log('main', error_msg)
                self._log('main', "Traceback:", also_print=False)
                import traceback
                tb_str = traceback.format_exc()
                self._log('main', tb_str, also_print=False)
                traceback.print_exc()
                continue
        
        self.all_results = results
        
        # Save results
        self._log('main', f"\n{'='*70}")
        self._log('main', "SAVING RESULTS")
        self._log('main', f"{'='*70}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_dir / 'algorithm_comparison.csv', index=False)
        
        self._log('main', f"✓ Saved {len(results)} results to algorithm_comparison.csv")
        print(f"\n✓ Saved results to {self.results_dir / 'algorithm_comparison.csv'}")
        
        # Create visualizations
        self._log('visualization', "Creating visualizations...")
        self.create_visualizations(results_df)
        
        # Create comprehensive comparison plots
        self.plot_comparison_results(results_df)
        
        # Print summary
        self.print_summary(results_df)
        
        # Close all log files
        self._log('main', f"\n{'='*70}")
        self._log('main', "PIPELINE COMPLETE")
        self._log('main', f"Total algorithms tested: {len(results)}")
        self._log('main', f"{'='*70}")
        self._close_logs()
        
        return results_df
    
    def create_visualizations(self, results_df):
        """Create comparison visualizations"""
        print(f"\n{'='*70}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*70}")
        
        self._log('visualization', f"\n{'='*70}")
        self._log('visualization', "CREATING VISUALIZATIONS")
        self._log('visualization', f"{'='*70}")
        self._log('visualization', f"Results to visualize: {len(results_df)}")
        
        # Filter out algorithms with invalid results
        valid_df = results_df[results_df['f1_weighted'] > 0].copy()
        
        self._log('visualization', f"Valid results (F1 > 0): {len(valid_df)}")
        
        if len(valid_df) == 0:
            print("⚠️  No valid results to visualize")
            self._log('visualization', "⚠️  No valid results to visualize")
            return
        
        # Sort by F1 score
        valid_df = valid_df.sort_values('f1_weighted', ascending=False)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. F1 Score comparison
        ax = axes[0, 0]
        x_pos = np.arange(len(valid_df))
        ax.barh(x_pos, valid_df['f1_weighted'], color='steelblue', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(valid_df['algorithm'])
        ax.set_xlabel('F1 Score (Weighted)')
        ax.set_title('F1 Score Comparison by Algorithm')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(valid_df['f1_weighted']):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center')
        
        # 2. Silhouette Score comparison
        ax = axes[0, 1]
        ax.barh(x_pos, valid_df['silhouette'], color='coral', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(valid_df['algorithm'])
        ax.set_xlabel('Silhouette Score')
        ax.set_title('Clustering Quality (Silhouette Score)')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(valid_df['silhouette']):
            ax.text(v + 0.005, i, f'{v:.4f}', va='center')
        
        # 3. Runtime comparison
        ax = axes[1, 0]
        ax.barh(x_pos, valid_df['runtime'], color='green', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(valid_df['algorithm'])
        ax.set_xlabel('Runtime (seconds)')
        ax.set_title('Computational Efficiency')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(valid_df['runtime']):
            ax.text(v + 0.5, i, f'{v:.2f}s', va='center')
        
        # 4. Per-class F1 scores
        ax = axes[1, 1]
        width = 0.35
        ax.barh(x_pos - width/2, valid_df['f1_intentional'], width, 
                label='Intentional', color='#2ecc71', alpha=0.8)
        ax.barh(x_pos + width/2, valid_df['f1_unintentional'], width,
                label='Unintentional', color='#e74c3c', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(valid_df['algorithm'])
        ax.set_xlabel('F1 Score')
        ax.set_title('Per-Class F1 Scores')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison plot to algorithm_comparison.png")
    
    def print_summary(self, results_df):
        """Print summary of results and save to file"""
        summary_file = self.results_dir / 'detailed_summary.txt'
        
        # Open file for writing
        with open(summary_file, 'w') as f:
            def print_and_log(text):
                """Print to console and write to file"""
                print(text)
                f.write(text + '\n')
            
            print_and_log(f"\n{'='*110}")
            print_and_log("FINAL SUMMARY - NORMAL vs SMART SAMPLING")
            print_and_log(f"{'='*110}")
            
            valid_df = results_df[results_df['f1_weighted'] > 0].copy()
            valid_df = valid_df.sort_values('f1_weighted', ascending=False)
            
            print_and_log("\nRanking by F1 Score:")
            print_and_log("-" * 110)
            print_and_log(f"{'Algorithm':35s} | {'Strategy':8s} | {'F1':6s} | {'Variants':8s} | {'%Vars':7s} | "
                          f"{'Samples':7s} | {'%Smpls':8s} | {'Test':7s}")
            print_and_log("-" * 110)
            for idx, row in valid_df.iterrows():
                strategy = row.get('sampling_strategy', 'N/A')
                print_and_log(f"{row['algorithm']:35s} | {strategy:8s} | {row['f1_weighted']:6.4f} | "
                              f"{row['n_variants']:8.0f} | {row['pct_train_variants']:6.2f}% | "
                              f"{row['n_train_samples']:7.0f} | {row['pct_train_samples']:7.2f}% | "
                              f"{row['n_test_samples']:7.0f}")
            
            if len(valid_df) > 0:
                best = valid_df.iloc[0]
                print_and_log(f"\n{'='*110}")
                print_and_log("🏆 BEST OVERALL RESULT")
                print_and_log(f"{'='*110}")
                print_and_log(f"Algorithm: {best['algorithm']}")
                print_and_log(f"Sampling Strategy: {best.get('sampling_strategy', 'N/A')}")
                print_and_log(f"F1 Score: {best['f1_weighted']:.4f}")
                print_and_log(f"F1 Intentional: {best['f1_intentional']:.4f}")
                print_and_log(f"F1 Unintentional: {best['f1_unintentional']:.4f}")
                print_and_log(f"Silhouette Score: {best['silhouette']:.4f}")
                print_and_log(f"Number of Clusters: {best['n_clusters']:.0f}")
                print_and_log(f"Variants Selected: {best['n_variants']:.0f} / {best['total_variants']:.0f} ({best['pct_train_variants']:.2f}%)")
                print_and_log(f"Training Samples: {best['n_train_samples']:.0f} / {best['total_samples']:.0f} ({best['pct_train_samples']:.2f}%)")
                print_and_log(f"Test Samples: {best['n_test_samples']:.0f} ({best['pct_test_samples']:.2f}%)")
                print_and_log(f"Runtime: {best['runtime']:.2f}s")
                
                # Compare Normal vs Smart for each base algorithm
                print_and_log(f"\n{'='*110}")
                print_and_log("📊 NORMAL vs SMART SAMPLING COMPARISON")
                print_and_log(f"{'='*110}")
                
                # Group by base algorithm
                base_algorithms = set()
                for algo in valid_df['algorithm']:
                    base_algo = algo.replace(' (Normal)', '').replace(' (Smart)', '')
                    base_algorithms.add(base_algo)
                
                for base_algo in sorted(base_algorithms):
                    normal_row = valid_df[valid_df['algorithm'] == f"{base_algo} (Normal)"]
                    smart_row = valid_df[valid_df['algorithm'] == f"{base_algo} (Smart)"]
                    
                    if len(normal_row) > 0 and len(smart_row) > 0:
                        normal_f1 = normal_row.iloc[0]['f1_weighted']
                        smart_f1 = smart_row.iloc[0]['f1_weighted']
                        diff = smart_f1 - normal_f1
                        pct_change = (diff / normal_f1) * 100 if normal_f1 > 0 else 0
                        
                        winner = "🔥 SMART" if diff > 0 else "⚡ NORMAL"
                        
                        print_and_log(f"\n{base_algo}:")
                        print_and_log(f"  Normal:  {normal_f1:.4f} F1 ({normal_row.iloc[0]['n_variants']:.0f} variants [{normal_row.iloc[0]['pct_train_variants']:.2f}%], "
                                      f"{normal_row.iloc[0]['n_train_samples']:.0f} samples [{normal_row.iloc[0]['pct_train_samples']:.2f}%])")
                        print_and_log(f"  Smart:   {smart_f1:.4f} F1 ({smart_row.iloc[0]['n_variants']:.0f} variants [{smart_row.iloc[0]['pct_train_variants']:.2f}%], "
                                      f"{smart_row.iloc[0]['n_train_samples']:.0f} samples [{smart_row.iloc[0]['pct_train_samples']:.2f}%])")
                        print_and_log(f"  Diff:    {diff:+.4f} ({pct_change:+.2f}%) - {winner}")
        
        print(f"\n✓ Detailed summary saved to: {summary_file}")
    
    def plot_comparison_results(self, results_df, save_path=None):
        """
        Create comprehensive comparison plots for Normal vs Smart sampling
        
        Args:
            results_df: DataFrame with comparison results
            save_path: Optional custom save path (default: plots_dir)
        """
        if save_path is None:
            save_path = self.plots_dir
        
        print(f"\n{'='*70}")
        print("CREATING COMPREHENSIVE COMPARISON PLOTS")
        print(f"{'='*70}")
        
        # Filter valid results
        valid_df = results_df[results_df['f1_weighted'] > 0].copy()
        
        if len(valid_df) == 0:
            print("⚠️  No valid results to plot")
            return
        
        # Separate Normal and Smart results
        normal_df = valid_df[valid_df['sampling_strategy'] == 'Normal'].copy()
        smart_df = valid_df[valid_df['sampling_strategy'] == 'Smart'].copy()
        
        # Extract base algorithm names
        normal_df['base_algo'] = normal_df['algorithm'].str.replace(' (Normal)', '')
        smart_df['base_algo'] = smart_df['algorithm'].str.replace(' (Smart)', '')
        
        # === PLOT 1: Side-by-side F1 Score Comparison ===
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort by normal F1 (descending)
        sorted_algos = normal_df.sort_values('f1_weighted', ascending=True)['base_algo'].tolist()
        
        y_pos = np.arange(len(sorted_algos))
        width = 0.35
        
        normal_f1 = [normal_df[normal_df['base_algo'] == algo]['f1_weighted'].values[0] for algo in sorted_algos]
        smart_f1 = [smart_df[smart_df['base_algo'] == algo]['f1_weighted'].values[0] if algo in smart_df['base_algo'].values else 0 for algo in sorted_algos]
        
        bars1 = ax.barh(y_pos - width/2, normal_f1, width, label='Normal (Proportional Random)', 
                        color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.barh(y_pos + width/2, smart_f1, width, label='Smart (Stratified by Intent)', 
                        color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_algos, fontsize=11)
        ax.set_xlabel('Weighted F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Normal vs Smart Sampling Comparison\n(Evaluated on ALL Unseen Data)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1.0)
        
        # Add value labels
        for i, (n_val, s_val) in enumerate(zip(normal_f1, smart_f1)):
            ax.text(n_val + 0.01, i - width/2, f'{n_val:.3f}', va='center', fontsize=9, fontweight='bold')
            ax.text(s_val + 0.01, i + width/2, f'{s_val:.3f}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'f1_comparison_normal_vs_smart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved F1 comparison plot")
        
        # === PLOT 2: Win/Loss Heatmap ===
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create difference matrix
        diff_data = []
        for algo in sorted_algos:
            n_f1 = normal_df[normal_df['base_algo'] == algo]['f1_weighted'].values[0]
            s_f1 = smart_df[smart_df['base_algo'] == algo]['f1_weighted'].values[0] if algo in smart_df['base_algo'].values else 0
            diff = n_f1 - s_f1  # Positive = Normal wins
            diff_data.append([diff * 100])  # Convert to percentage points
        
        diff_array = np.array(diff_data)
        
        # Create heatmap
        im = ax.imshow(diff_array, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
        
        ax.set_yticks(np.arange(len(sorted_algos)))
        ax.set_yticklabels(sorted_algos, fontsize=11)
        ax.set_xticks([0])
        ax.set_xticklabels(['Normal - Smart\n(% points)'], fontsize=11)
        ax.set_title('Performance Difference Heatmap\n(Green = Normal Wins, Red = Smart Wins)', 
                     fontsize=13, fontweight='bold', pad=15)
        
        # Add text annotations
        for i in range(len(sorted_algos)):
            text = ax.text(0, i, f'{diff_data[i][0]:+.2f}%', ha='center', va='center', 
                          fontsize=11, fontweight='bold',
                          color='white' if abs(diff_data[i][0]) > 3 else 'black')
        
        plt.colorbar(im, ax=ax, label='Difference (percentage points)')
        plt.tight_layout()
        plt.savefig(save_path / 'difference_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved difference heatmap")
        
        # === PLOT 3: Sample Efficiency ===
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for idx, algo in enumerate(sorted_algos):
            n_row = normal_df[normal_df['base_algo'] == algo]
            s_row = smart_df[smart_df['base_algo'] == algo]
            
            if len(n_row) > 0:
                ax.scatter(n_row['pct_train_samples'].values[0], n_row['f1_weighted'].values[0], 
                          s=200, marker='o', color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5,
                          label='Normal' if idx == 0 else '')
                ax.text(n_row['pct_train_samples'].values[0], n_row['f1_weighted'].values[0] + 0.01, 
                       algo, fontsize=9, ha='center')
            
            if len(s_row) > 0:
                ax.scatter(s_row['pct_train_samples'].values[0], s_row['f1_weighted'].values[0], 
                          s=200, marker='^', color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5,
                          label='Smart' if idx == 0 else '')
        
        ax.set_xlabel('Training Data Used (% of total samples)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weighted F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Sample Efficiency: F1 Score vs Training Data Size', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0.75, 0.95)
        
        plt.tight_layout()
        plt.savefig(save_path / 'sample_efficiency.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved sample efficiency plot")
        
        # === PLOT 4: Multi-metric Comparison ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # F1 Weighted
        ax = axes[0, 0]
        x_pos = np.arange(len(sorted_algos))
        ax.barh(x_pos - width/2, normal_f1, width, label='Normal', color='#3498db', alpha=0.8)
        ax.barh(x_pos + width/2, smart_f1, width, label='Smart', color='#e74c3c', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_algos, fontsize=9)
        ax.set_xlabel('F1 Weighted', fontsize=10, fontweight='bold')
        ax.set_title('F1 Weighted Comparison', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Intentional F1
        ax = axes[0, 1]
        normal_f1_int = [normal_df[normal_df['base_algo'] == algo]['f1_intentional'].values[0] for algo in sorted_algos]
        smart_f1_int = [smart_df[smart_df['base_algo'] == algo]['f1_intentional'].values[0] if algo in smart_df['base_algo'].values else 0 for algo in sorted_algos]
        ax.barh(x_pos - width/2, normal_f1_int, width, label='Normal', color='#2ecc71', alpha=0.8)
        ax.barh(x_pos + width/2, smart_f1_int, width, label='Smart', color='#e67e22', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_algos, fontsize=9)
        ax.set_xlabel('F1 Intentional', fontsize=10, fontweight='bold')
        ax.set_title('F1 Intentional Comparison', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Unintentional F1
        ax = axes[1, 0]
        normal_f1_unint = [normal_df[normal_df['base_algo'] == algo]['f1_unintentional'].values[0] for algo in sorted_algos]
        smart_f1_unint = [smart_df[smart_df['base_algo'] == algo]['f1_unintentional'].values[0] if algo in smart_df['base_algo'].values else 0 for algo in sorted_algos]
        ax.barh(x_pos - width/2, normal_f1_unint, width, label='Normal', color='#9b59b6', alpha=0.8)
        ax.barh(x_pos + width/2, smart_f1_unint, width, label='Smart', color='#f39c12', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_algos, fontsize=9)
        ax.set_xlabel('F1 Unintentional', fontsize=10, fontweight='bold')
        ax.set_title('F1 Unintentional Comparison', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # Training Data %
        ax = axes[1, 1]
        normal_pct = [normal_df[normal_df['base_algo'] == algo]['pct_train_samples'].values[0] for algo in sorted_algos]
        smart_pct = [smart_df[smart_df['base_algo'] == algo]['pct_train_samples'].values[0] if algo in smart_df['base_algo'].values else 0 for algo in sorted_algos]
        ax.barh(x_pos - width/2, normal_pct, width, label='Normal', color='#1abc9c', alpha=0.8)
        ax.barh(x_pos + width/2, smart_pct, width, label='Smart', color='#e74c3c', alpha=0.8)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(sorted_algos, fontsize=9)
        ax.set_xlabel('Training Data Used (%)', fontsize=10, fontweight='bold')
        ax.set_title('Training Data Usage', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        plt.suptitle('Multi-Metric Comparison: Normal vs Smart Sampling', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path / 'multi_metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved multi-metric comparison")
        
        print(f"\n✅ All comparison plots saved to: {save_path}")
        print(f"   - f1_comparison_normal_vs_smart.png")
        print(f"   - difference_heatmap.png")
        print(f"   - sample_efficiency.png")
        print(f"   - multi_metric_comparison.png")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare clustering algorithms with Normal vs Smart sampling - Dataset Independent!',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default dataset path (auto-detected)
  python compare_clustering_algorithms.py
  
  # Specify individual data files
  python compare_clustering_algorithms.py \\
    --mask-path /path/to/masks.csv \\
    --clean-data-path /path/to/correct_records.csv \\
    --dirty-data-path /path/to/manipulated_records.csv
  
  # Legacy: Specify directory (deprecated, use individual files instead)
  python compare_clustering_algorithms.py --dataset_path /path/to/data/run_XXXXX
  
  # Change number of target samples
  python compare_clustering_algorithms.py --target_samples 200
  
  # Full example with all options
  python compare_clustering_algorithms.py \\
    --mask-path data/raw/run_20251031_211812/masks.csv \\
    --clean-data-path data/raw/correct_records.csv \\
    --dirty-data-path data/raw/run_20251031_211812/manipulated_records.csv \\
    --target_samples 150 \\
    --random_state 42
        """
    )
    
    # Data file arguments (recommended)
    parser.add_argument(
        '--mask-path',
        type=str,
        default=None,
        help='Path to masks.csv file containing intent labels (1=intentional, -1=unintentional, 0=no change)'
    )
    
    parser.add_argument(
        '--clean-data-path',
        type=str,
        default=None,
        help='Path to correct_records.csv file containing original/clean data'
    )
    
    parser.add_argument(
        '--dirty-data-path',
        type=str,
        default=None,
        help='Path to manipulated_records.csv file containing modified/dirty data'
    )
    
    # Legacy directory argument (for backward compatibility)
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='[DEPRECATED] Path to directory containing masks.csv and manipulated_records.csv. '
             'Use --mask-path, --clean-data-path, --dirty-data-path instead.'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--target_samples',
        type=int,
        default=None,
        help='Target number of samples to select (default: 1%% of dataset variants, auto-calculated)'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Logging control
    parser.add_argument(
        '--logging',
        type=lambda x: str(x).lower() in ['true', '1', 'yes'],
        default=False,
        help='Enable detailed logging to separate files (default: False). Use --logging True to enable.'
    )
    
    args = parser.parse_args()
    
    # Determine which paths to use
    mask_path = None
    clean_path = None
    dirty_path = None
    
    # Priority 1: Individual file paths
    if args.mask_path and args.clean_data_path and args.dirty_data_path:
        mask_path = args.mask_path
        clean_path = args.clean_data_path
        dirty_path = args.dirty_data_path
        print(f"Using individual file paths:")
        print(f"  Mask: {mask_path}")
        print(f"  Clean: {clean_path}")
        print(f"  Dirty: {dirty_path}")
    
    # Priority 2: Legacy dataset_path (for backward compatibility)
    elif args.dataset_path:
        dataset_dir = Path(args.dataset_path)
        if not dataset_dir.exists():
            print(f"❌ Error: Dataset directory does not exist: {dataset_dir}")
            return
        
        mask_path = dataset_dir / 'masks.csv'
        dirty_path = dataset_dir / 'manipulated_records.csv'
        clean_path = dataset_dir.parent / 'correct_records.csv'
        
        if not clean_path.exists():
            clean_path = Path('data/raw/correct_records.csv')
        
        if not mask_path.exists():
            print(f"❌ Error: masks.csv not found in: {dataset_dir}")
            return
        
        print(f"⚠️  Using legacy --dataset_path (consider using individual file paths)")
        print(f"  Mask: {mask_path}")
        print(f"  Clean: {clean_path}")
        print(f"  Dirty: {dirty_path}")
    
    # Priority 3: Auto-detect (None provided)
    else:
        print("No paths specified, will auto-detect dataset...")
    
    # Create comparison instance with paths
    comparison = ClusteringComparison(
        target_samples=args.target_samples,
        random_state=args.random_state,
        mask_path=mask_path,
        clean_data_path=clean_path,
        dirty_data_path=dirty_path,
        enable_detailed_logging=args.logging
    )
    
    # Setup timestamped logging
    from datetime import datetime
    import sys
    import io
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = comparison.logs_dir / f"comparison_all_data_evaluation_{timestamp}.log"
    
    # Create a tee-like logger that writes to both file and console
    class TeeLogger:
        def __init__(self, *files):
            self.files = files
        
        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()
        
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Open log file
    log_file = open(log_filename, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeLogger(sys.stdout, log_file)
    
    try:
        # Print execution info with timestamp
        print(f"\n{'='*70}")
        print(f"EXECUTION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_filename}")
        print(f"Output directory: {comparison.run_dir}")
        print(f"{'='*70}\n")
        
        # Run comparison
        results = comparison.run_comparison()
        
        print(f"\n{'='*70}")
        print("✓ COMPARISON COMPLETE!")
        print(f"{'='*70}")
        print(f"\n📁 All outputs saved to: {comparison.run_dir}")
        print(f"   ├── results/  - CSV and TXT result files")
        print(f"   ├── plots/    - PNG visualization files")
        print(f"   └── logs/     - Execution log file")
        print(f"\nCommand used:")
        if mask_path:
            print(f"  Mask: {mask_path}")
            print(f"  Clean: {clean_path}")
            print(f"  Dirty: {dirty_path}")
        else:
            print(f"  Dataset: auto-detected")
        print(f"  Target samples: {args.target_samples}")
        print(f"  Random state: {args.random_state}")
        
        print(f"\n{'='*70}")
        print(f"EXECUTION COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log saved to: {log_filename}")
        print(f"{'='*70}\n")
        
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()


if __name__ == '__main__':
    main()
