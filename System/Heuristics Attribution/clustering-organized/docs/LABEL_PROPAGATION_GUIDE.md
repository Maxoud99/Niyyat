# Label Propagation vs Random Forest Comparison Guide

## 📋 Overview

This guide documents the **Label Propagation vs Random Forest** comparison framework for **cell-level error detection** in the Adult Income dataset (v2).

**Purpose:** Compare semi-supervised learning approaches (label propagation) against traditional supervised learning (Random Forest) using minimal labeled data (~1% of records).

---

## 🎯 Problem Statement

### Context
- **Dataset:** Adult Income v2 with 4000 records × 15 features = 60,000 cells
- **Error Rate:** ~2.35% of cells contain errors
- **Challenge:** Detect cell-level errors using only ~1% labeled records (40 out of 4000)

### Goal
Evaluate whether **label propagation** (semi-supervised) can match or outperform **Random Forest** (supervised) when labeled data is scarce.

---

## 🔬 Methodology

### Three Approaches Compared

#### **Approach 1: Graph-Based Label Propagation**
Uses sklearn's semi-supervised learning algorithms:

**Method 1a: LabelPropagation (Hard Labels)**
- Builds k-NN similarity graph (k=7)
- Iteratively propagates labels through graph edges
- Hard label assignment (no uncertainty)
- Max iterations: 1000

**Method 1b: LabelSpreading (Soft Labels)**  
- Similar to LabelPropagation but with clamping
- Alpha parameter (α=0.2): balance between label consistency
- More robust to noise
- Soft label probabilities

**Characteristics:**
- ✅ Leverages unlabeled data structure
- ✅ Graph-based similarity
- ✅ No training required
- ⚠️ Computationally expensive for large datasets
- ⚠️ Sensitive to k-NN parameter

---

#### **Approach 2: Cluster-Constrained Propagation**
Custom algorithm that propagates within cluster boundaries:

**Algorithm:**
```
1. Cluster records using aggregate features
2. Sample representatives (~1%) from clusters  
3. Calculate majority label per cluster from labeled cells
4. Assign cluster's majority label to all unlabeled cells in that cluster
```

**Characteristics:**
- ✅ Respects cluster structure
- ✅ Very fast (negligible runtime)
- ✅ Simple and interpretable
- ⚠️ Coarse-grained (cluster-level decisions)
- ⚠️ May miss within-cluster variability

---

#### **Approach 3: Random Forest Classifier (Baseline)**
Traditional supervised learning:

**Configuration:**
- 200 trees
- Max depth: 15
- Class weight: balanced (handles 2.35% error imbalance)
- Features: feature_name, cell_value, value_log, value_magnitude

**Characteristics:**
- ✅ Proven performance
- ✅ Handles class imbalance well
- ✅ Fast training and prediction
- ⚠️ Requires labeled training data
- ⚠️ No use of unlabeled data structure

---

## 📊 Workflow

### Step 1: Data Loading
```
Input:
- combined_dataset_no_id_v2.csv (4000 records × 15 features)
- ground_truth_masks_v2.csv (0=correct, 1=error)

Output:
- 60,000 cell-level samples
- Error labels for evaluation
```

### Step 2: Feature Engineering
```
Per-cell features:
- feature_name_encoded (which feature)
- cell_value_encoded (categorical encoding)
- value_log (log transform of numeric values)
- value_magnitude (absolute numeric values)

Aggregate features per record (for clustering):
- n_cells, mean/std/min/max of encoded values
- mean/std/min/max of numeric values
- Feature diversity metrics
```

### Step 3: Clustering
```
Algorithms tested:
1. K-Means (n_clusters=15)
2. DBSCAN (eps auto-tuned, min_samples=5)

Purpose:
- Group similar records
- Enable stratified sampling
- Structure for label propagation
```

### Step 4: Proportional Sampling
```
NO DATA LEAKAGE: Error labels NOT used for sampling

1. Count records per cluster
2. Allocate samples proportionally to cluster size
3. Random selection within each cluster
4. Result: ~40 records (600 cells) labeled, 59,400 cells unlabeled
```

### Step 5: Apply Methods
```
For each clustering algorithm:
  1. LabelPropagation (graph-based)
  2. LabelSpreading (graph-based with clamping)  
  3. ClusterConstrained (majority vote per cluster)
  4. RandomForest (supervised baseline)
```

### Step 6: Evaluation
```
Metrics (on 59,400 unlabeled test cells):
- Accuracy
- F1 Weighted (overall)
- F1 Error (class 1)
- F1 Correct (class 0)
- Runtime
```

---

## 📈 Results Summary

### Test Run (40 records sampled, 600 cells labeled):

| Method              | Algorithm | F1 Weighted | Accuracy | F1 Error | F1 Correct | Runtime |
|---------------------|-----------|-------------|----------|----------|------------|---------|
| RandomForest        | K-Means   | **0.9745**  | 0.9760   | **0.4219** | 0.9878   | 0.37s   |
| LabelSpreading      | DBSCAN    | 0.9678      | **0.9777** | 0.0982 | **0.9887** | 1.67s   |
| LabelPropagation    | DBSCAN    | 0.9678      | **0.9777** | 0.0969 | **0.9887** | 2.34s   |
| LabelPropagation    | K-Means   | 0.9677      | 0.9775   | 0.0961 | 0.9886     | 2.42s   |
| LabelSpreading      | K-Means   | 0.9677      | 0.9775   | 0.0961 | 0.9886     | 2.16s   |
| ClusterConstrained  | DBSCAN    | 0.9649      | 0.9765   | 0.0000 | 0.9881     | ~0s     |
| ClusterConstrained  | K-Means   | 0.9649      | 0.9765   | 0.0000 | 0.9881     | ~0s     |
| RandomForest        | DBSCAN    | 0.9532      | 0.9443   | 0.2071 | 0.9711     | 0.38s   |

### Average Performance by Method:

| Method              | Avg F1  | Avg Accuracy | Notes                                    |
|---------------------|---------|--------------|------------------------------------------|
| LabelSpreading      | 0.9677  | 0.9776       | Best overall accuracy & correct class    |
| LabelPropagation    | 0.9677  | 0.9776       | Tied with LabelSpreading                 |
| ClusterConstrained  | 0.9649  | 0.9765       | Fast but fails on error class (F1=0)     |
| RandomForest        | 0.9638  | 0.9602       | **Best error detection (F1_error=0.42)** |

---

## 🔍 Key Insights

### 1. **Random Forest Excels at Error Detection**
- **F1 Error: 0.42** (vs ~0.10 for label propagation)
- Better precision-recall balance for the minority class (errors)
- Supervised learning leverages labeled error patterns effectively

### 2. **Label Propagation Better for Overall Accuracy**
- **Accuracy: 97.77%** (vs 94.43-97.60% for RF)
- Excellent on correct class (F1_correct ~0.99)
- Graph-based similarity captures data structure

### 3. **Cluster-Constrained Fails on Errors**
- **F1 Error: 0.00** (predicts no errors)
- Majority vote too coarse-grained
- Error cells (2.35%) drown out in majority correct cells (97.65%)
- Fast but not suitable for imbalanced classification

### 4. **Trade-offs**
| Method              | Error Detection | Overall Accuracy | Speed | Complexity |
|---------------------|-----------------|------------------|-------|------------|
| RandomForest        | ⭐⭐⭐⭐⭐     | ⭐⭐⭐⭐        | ⭐⭐⭐⭐⭐ | Low       |
| LabelPropagation    | ⭐⭐            | ⭐⭐⭐⭐⭐      | ⭐⭐      | Medium    |
| LabelSpreading      | ⭐⭐            | ⭐⭐⭐⭐⭐      | ⭐⭐      | Medium    |
| ClusterConstrained  | ⭐              | ⭐⭐⭐⭐⭐      | ⭐⭐⭐⭐⭐ | Low       |

---

## 🚀 Usage

### Basic Usage
```bash
cd /path/to/clustering-organized/scripts

python3 compare_label_propagation.py \
  --data-path /path/to/combined_dataset_no_id_v2.csv \
  --mask-path /path/to/ground_truth_masks_v2.csv \
  --target-samples 40 \
  --random-state 42
```

### Auto-Detection (if paths are standard)
```bash
python3 compare_label_propagation.py --target-samples 40
```

### Parameters
- `--data-path`: Path to combined dataset CSV (default: auto-detect)
- `--mask-path`: Path to ground truth masks CSV (default: auto-detect)
- `--target-samples`: Number of records to sample (default: 1% of dataset)
- `--random-state`: Random seed for reproducibility (default: 42)

---

## 📁 Output Structure

```
clustering-organized/outputs/run_label_prop_YYYYMMDD_HHMMSS/
├── results/
│   ├── comparison_results.csv      # All method results
│   ├── comparison_results.json     # JSON format
│   └── summary.txt                 # Human-readable summary
├── plots/
│   └── method_comparison.png       # 4-panel comparison plot
└── logs/
    └── (empty, for future logging)
```

---

## 🎨 Visualizations

The script generates a 4-panel comparison plot:

1. **F1 Score Comparison** (horizontal bar chart)
   - Shows weighted F1 for all method-algorithm combinations
   - Color-coded by method type

2. **Accuracy Comparison** (horizontal bar chart)
   - Overall accuracy on test set

3. **Per-Class F1 Scores** (grouped horizontal bar chart)
   - F1 Error (red) vs F1 Correct (green)
   - Shows class imbalance handling

4. **Runtime Comparison** (horizontal bar chart)
   - Computational efficiency

---

## 💡 Recommendations

### For Production Error Detection
**Use Random Forest** when:
- ✅ Error detection is critical (minimize false negatives)
- ✅ Speed matters (training + inference < 1s)
- ✅ Interpretability needed (feature importance available)
- ✅ Some labeled data available (~1% sufficient)

### For High-Accuracy Classification
**Use LabelPropagation/LabelSpreading** when:
- ✅ Overall accuracy is primary goal
- ✅ Correct class detection is critical
- ✅ Computational resources available (2-3s runtime)
- ✅ Unlabeled data structure should be exploited

### Avoid Cluster-Constrained When:
- ❌ Dealing with imbalanced classes (errors << correct)
- ❌ Fine-grained predictions needed (not cluster-level)
- ❌ Minority class detection is important

---

## 🔧 Technical Details

### Dependencies
```python
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

### Data Format
**combined_dataset_no_id_v2.csv:**
```csv
age,workclass,fnlwgt,education,education-num,...
55,Private,199713,9th,5,...
20,State-gov,243986,Some-college,1,...
```

**ground_truth_masks_v2.csv:**
```csv
age,workclass,fnlwgt,education,education-num,...
0,0,0,0,0,...
0,0,0,0,1,...  # 1 = error in education-num
```

### Cell-Level Representation
Each (record, feature) pair becomes a sample:
```
Record 0, age        -> cell_value=55, error_label=0
Record 0, workclass  -> cell_value=Private, error_label=0
...
Record 1, education-num -> cell_value=1, error_label=1
```

---

## 🐛 Known Limitations

1. **Class Imbalance:** Only 2.35% errors makes evaluation challenging
2. **Cluster-Constrained:** Fails completely on minority class
3. **Graph Construction:** k-NN (k=7) may not be optimal for all datasets
4. **Memory:** Storing 60,000 cells in memory (not an issue for this dataset)
5. **Scalability:** Graph-based methods slow for >100K cells

---

## 🔮 Future Improvements

1. **Hybrid Approach:** Combine RF for error detection + LP for correct class
2. **Adaptive k-NN:** Auto-tune k parameter for LabelPropagation
3. **Weighted Cluster-Constrained:** Use error rate instead of majority vote
4. **Active Learning:** Iteratively select most uncertain samples for labeling
5. **Ensemble:** Combine multiple methods with weighted voting
6. **Feature Engineering:** Add more domain-specific features
7. **Cross-Validation:** Test across multiple random seeds and sample sizes

---

## 📚 References

1. Zhu, X., & Ghahramani, Z. (2002). Learning from labeled and unlabeled data with label propagation.
2. Zhou, D., et al. (2004). Learning with local and global consistency.
3. Scikit-learn Label Propagation: https://scikit-learn.org/stable/modules/semi_supervised.html

---

## 👥 Contact

For questions or suggestions, contact your research team or open an issue in the repository.

---

**Last Updated:** January 28, 2026  
**Script Version:** 1.0  
**Author:** Research Team
