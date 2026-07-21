# Detailed Logging System Guide

## 📝 Overview

The clustering comparison pipeline now includes a **comprehensive logging system** that creates separate log files for each component of the pipeline. This makes debugging, analysis, and understanding the results much easier!

## 🗂️ Log File Structure

Each run creates **15 separate log files** in the `outputs/run_YYYYMMDD_HHMMSS/logs/` directory:

```
outputs/run_20251210_133629/logs/
├── 00_main_pipeline_20251210_133629.log              # Overall pipeline flow
├── 01_data_loading_20251210_133629.log               # Data loading & preprocessing
├── 02_feature_creation_20251210_133629.log           # Feature engineering
├── 03_kmeans_20251210_133629.log                     # K-Means algorithm details
├── 04_dbscan_20251210_133629.log                     # DBSCAN algorithm details
├── 05_hierarchical_ward_20251210_133629.log          # Hierarchical (Ward) details
├── 06_hierarchical_average_20251210_133629.log       # Hierarchical (Average) details
├── 07_hierarchical_complete_20251210_133629.log      # Hierarchical (Complete) details
├── 08_spectral_20251210_133629.log                   # Spectral clustering details
├── 09_gmm_20251210_133629.log                        # GMM algorithm details
├── 10_hdbscan_20251210_133629.log                    # HDBSCAN algorithm details
├── 11_normal_sampling_20251210_133629.log            # Normal sampling strategy
├── 12_smart_sampling_20251210_133629.log             # Smart sampling strategy
├── 13_evaluation_20251210_133629.log                 # Model evaluation & metrics
├── 14_visualization_20251210_133629.log              # Visualization creation
└── comparison_all_data_evaluation_20251210_133629.log # Console output (everything)
```

## 📊 What Each Log Contains

### **00_main_pipeline.log** - Overall Flow
```
Purpose: High-level overview of the entire pipeline execution
Contains:
  - Which algorithms are being tested
  - Success/failure status for each algorithm + strategy
  - F1 scores for quick comparison
  - Error messages if any algorithm fails
  - Pipeline start/end timestamps

Example:
  ======================================================================
  ALGORITHM 1/6: K-Means
  ======================================================================
  
  --- NORMAL SAMPLING for K-Means ---
  ✓ K-Means (Normal) - F1: 0.6301
  
  --- SMART SAMPLING for K-Means ---
  ✓ K-Means (Smart) - F1: 0.6467
```

### **01_data_loading.log** - Data Loading
```
Purpose: Track data loading and preprocessing
Contains:
  - Data paths searched
  - Data found location
  - Dataset shapes (masks, clean, dirty)
  - Number of feature changes created
  - Intentional vs. Unintentional distribution
  - Encoding details
  
Example:
  Auto-detecting data directory...
  Searching in 5 potential locations...
  ✓ Found data in: /path/to/data
  Loaded masks: (19539, 15)
  ✓ Created dataset with 28256 feature changes
    Intentional (1): 13291
    Unintentional (-1): 14965
```

### **03_kmeans.log** (and other algorithm logs)
```
Purpose: Detailed algorithm execution information
Contains:
  - Algorithm parameters
  - Feature matrix shape
  - Cluster distribution (size & percentage)
  - Silhouette score
  - Runtime
  - Algorithm-specific metrics (e.g., inertia for K-Means)

Example:
  ======================================================================
  K-MEANS CLUSTERING STARTED
  ======================================================================
  Parameters:
    n_clusters: 15
    random_state: 42
    n_init: 10
  Feature matrix shape: (18207, 15)
  
  Cluster distribution:
    Cluster 0: 2981 points (16.4%)
    Cluster 1: 2629 points (14.4%)
    ...
  
  Results:
    Silhouette Score: 0.5293
    Runtime: 2.82s
    Inertia: 17982.39
```

### **11_normal_sampling.log** - Normal Sampling Strategy
```
Purpose: Track proportional random sampling from clusters
Contains:
  - Total points and valid (non-noise) points
  - Number of clusters
  - Cluster sizes
  - Allocation strategy
  - Per-cluster sampling details
  - Final selection statistics

Example:
  Starting normal sampling...
  Total points: 18207, Valid (non-noise): 18207
  Number of clusters: 15
  Cluster sizes: {0: 2981, 1: 2629, ...}
  Initial allocation: {0: 2, 1: 2, ...}
  
  Cluster 0: sampling 1 from 2981 variants
  Cluster 1: sampling 1 from 2629 variants
  ...
  
  Final selection:
    Total variants: 15
    Total samples: 29
    Intentional: 13
    Unintentional: 16
```

### **12_smart_sampling.log** - Smart Sampling Strategy
```
Purpose: Track intent-stratified sampling within clusters
Contains:
  - Same basic info as normal sampling
  - Allocation constraints (MIN=1, MAX=10)
  - Per-cluster intentional/unintentional variant counts
  - Stratified selection details
  - Final intent distribution

Example:
  Starting smart sampling...
  Allocation constraints: MIN=1, MAX=10
  Initial allocation: {0: 2, 1: 2, ...}
  
  Cluster 0: selected 1 variants (allocation=1)
    Intentional variants: 0, Unintentional: 2981
  Cluster 2: selected 1 variants (allocation=1)
    Intentional variants: 644, Unintentional: 0
  ...
  
  Final selection:
    Total variants: 15
    Total samples: 27
    Intentional: 13
    Unintentional: 14
```

### **13_evaluation.log** - Model Evaluation
```
Purpose: Track Random Forest training and evaluation
Contains:
  - Training set size and intent distribution
  - Test set size and intent distribution
  - Feature dimensions
  - Model parameters (Random Forest settings)
  - Classification metrics per algorithm+strategy
  - Confusion matrix values
  
Example:
  ======================================================================
  EVALUATING: K-Means (Normal)
  ======================================================================
  Results: {'accuracy': 0.7234, 'precision_weighted': 0.7189, 
           'recall_weighted': 0.7234, 'f1_weighted': 0.6301, ...}
```

### **14_visualization.log** - Visualization Creation
```
Purpose: Track plot generation
Contains:
  - Number of results to visualize
  - Valid results count
  - Plot types created
  - Save locations
  
Example:
  ======================================================================
  CREATING VISUALIZATIONS
  ======================================================================
  Results to visualize: 12
  Valid results (F1 > 0): 12
```

## 🔍 How to Use These Logs for Debugging

### **1. Pipeline Failed? Check Main Log First**
```bash
cat outputs/run_*/logs/00_main_pipeline*.log
```
Look for error messages, which algorithm failed, and the traceback.

### **2. Algorithm Producing Poor Results?**
```bash
# Check the algorithm-specific log
cat outputs/run_*/logs/03_kmeans*.log

# Look for:
#   - Cluster distribution (are clusters balanced?)
#   - Silhouette score (good clustering quality?)
#   - Runtime (timeout issues?)
```

### **3. Sampling Strategy Not Working?**
```bash
# Check sampling logs
cat outputs/run_*/logs/11_normal_sampling*.log
cat outputs/run_*/logs/12_smart_sampling*.log

# Look for:
#   - How many variants selected from each cluster?
#   - Is the intent distribution balanced?
#   - Are some clusters too small?
```

### **4. Data Loading Issues?**
```bash
cat outputs/run_*/logs/01_data_loading*.log

# Look for:
#   - Which data path was used?
#   - Correct dataset shapes?
#   - Intent label distribution
```

### **5. Compare Algorithm Performance**
```bash
# Quick comparison using main log
grep "F1:" outputs/run_*/logs/00_main_pipeline*.log

# Example output:
#   ✓ K-Means (Normal) - F1: 0.6301
#   ✓ K-Means (Smart) - F1: 0.6467
#   ✓ DBSCAN (Normal) - F1: 0.5234
#   ✓ DBSCAN (Smart) - F1: 0.5678
```

## 📈 Common Debugging Scenarios

### **Scenario 1: Low F1 Score for Specific Algorithm**

**Steps:**
1. Check algorithm log: `cat 03_kmeans*.log`
   - Is silhouette score low? → Poor clustering quality
   - Are clusters very imbalanced? → Sampling issue

2. Check sampling log: `cat 11_normal_sampling*.log`
   - Are we getting enough samples from each cluster?
   - Is intent distribution balanced?

3. Check evaluation log: `cat 13_evaluation*.log`
   - How's the training set distribution?
   - Is the test set large enough?

### **Scenario 2: DBSCAN/HDBSCAN Fails**

**Steps:**
1. Check algorithm log: `cat 04_dbscan*.log`
   ```
   Look for:
     - Too many noise points? (>50%)
     - Too few clusters? (<3)
     - Auto-tuned eps value reasonable?
   ```

2. If too much noise:
   - Check feature matrix shape in log
   - Consider adjusting eps or min_samples parameters

### **Scenario 3: Different Results Between Runs**

**Steps:**
1. Check if `random_state` is consistent:
   ```bash
   grep "random_state" outputs/run_*/logs/00_main_pipeline*.log
   ```

2. Compare cluster distributions:
   ```bash
   grep "Cluster distribution" outputs/run_*/logs/03_kmeans*.log
   ```

### **Scenario 4: Slow Execution**

**Steps:**
1. Check runtime for each algorithm:
   ```bash
   grep "Runtime:" outputs/run_*/logs/0*.log
   ```

2. Identify bottleneck:
   - Data loading slow? → Check 01_data_loading.log
   - Specific algorithm slow? → Check that algorithm's log
   - Evaluation slow? → Check 13_evaluation.log

## 🎯 Best Practices

### **1. Always Check Logs After Each Run**
```bash
# Quick overview
ls -lh outputs/run_LATEST/logs/

# Check for errors
grep -i "error\|warning\|failed" outputs/run_*/logs/00_main_pipeline*.log
```

### **2. Compare Multiple Runs**
```bash
# Compare F1 scores across runs
for dir in outputs/run_*/logs/; do
    echo "=== $dir ==="
    grep "F1:" "$dir/00_main_pipeline"*.log | head -5
done
```

### **3. Archive Important Runs**
```bash
# Keep logs for successful runs
mkdir -p analysis/good_runs/
cp -r outputs/run_20251210_133629/logs analysis/good_runs/
```

### **4. Use Logs for Reports**
```bash
# Extract key metrics for reporting
cat outputs/run_*/logs/00_main_pipeline*.log | \
    grep "F1:" > report_f1_scores.txt
```

## 🛠️ Log File Format

All logs follow a consistent format:

```
================================================================================
[COMPONENT NAME] LOG
Started: YYYY-MM-DD HH:MM:SS
================================================================================

[Detailed component-specific content here]

================================================================================
Completed: YYYY-MM-DD HH:MM:SS
================================================================================
```

## 💡 Tips & Tricks

### **Quick Log Analysis Commands**

```bash
# Count how many times each algorithm was run
grep "ALGORITHM" outputs/run_*/logs/00_main_pipeline*.log | wc -l

# Find best performing algorithm
grep "F1:" outputs/run_*/logs/00_main_pipeline*.log | sort -t':' -k2 -nr | head -1

# Check cluster sizes for K-Means
grep "Cluster [0-9]:" outputs/run_*/logs/03_kmeans*.log

# See intent distribution in sampling
grep "Intentional\|Unintentional" outputs/run_*/logs/11_normal_sampling*.log

# Monitor progress (during run)
tail -f outputs/run_LATEST/logs/00_main_pipeline*.log
```

### **Automated Analysis Script**

```bash
#!/bin/bash
# analyze_logs.sh - Quick log analysis

RUN_DIR=$1

echo "=== Log Analysis for $RUN_DIR ==="
echo

echo "1. F1 Scores:"
grep "F1:" "$RUN_DIR/logs/00_main_pipeline"*.log
echo

echo "2. Algorithm Runtimes:"
grep "Runtime:" "$RUN_DIR/logs/0"*.log
echo

echo "3. Cluster Counts:"
grep "Clusters found:" "$RUN_DIR/logs/0"*.log
echo

echo "4. Sampling Sizes:"
grep "Total variants:" "$RUN_DIR/logs/1"*.log
```

## 📚 Related Documentation

- **[COMPLETE_PIPELINE_EXPLANATION.md](COMPLETE_PIPELINE_EXPLANATION.md)** - Understand what each step does
- **[TIMESTAMPED_OUTPUTS_GUIDE.md](TIMESTAMPED_OUTPUTS_GUIDE.md)** - Managing output folders
- **[RUNNING_THE_SCRIPT.md](RUNNING_THE_SCRIPT.md)** - How to run the pipeline

## 🎉 Benefits of This Logging System

✅ **Easy Debugging** - Quickly find where things went wrong  
✅ **Algorithm Analysis** - Deep dive into each clustering algorithm  
✅ **Sampling Transparency** - See exactly what was sampled and why  
✅ **Performance Tracking** - Monitor runtimes and identify bottlenecks  
✅ **Reproducibility** - Complete record of every run  
✅ **Comparative Analysis** - Easy to compare different runs  

---

**Happy Debugging! 🐛🔍**
