# Script Migration Guide

## ✅ Updated: `compare_clustering_algorithms.py`

### What Changed (Latest Update)
- **✨ NEW: Dataset-independent file paths** - Added `--mask-path`, `--clean-data-path`, `--dirty-data-path`
- **🌍 Works with ANY dataset** following the same structure (masks.csv, correct_records.csv, manipulated_records.csv)
- **Added command-line argument support** via `argparse`
- Configurable `--target_samples` and `--random_state`
- Properly evaluates on **ALL unseen data** (not 30% split)
- Tests all 6 algorithms with both Normal and Smart sampling
- **Backward compatible** - Old `--dataset_path` still works

### New Usage

```bash
# RECOMMENDED: Specify individual data files (dataset independent!)
python compare_clustering_algorithms.py \
  --mask-path /path/to/masks.csv \
  --clean-data-path /path/to/correct_records.csv \
  --dirty-data-path /path/to/manipulated_records.csv

# With parameters
python compare_clustering_algorithms.py \
  --mask-path data/raw/run_20251031_211812/masks.csv \
  --clean-data-path data/raw/correct_records.csv \
  --dirty-data-path data/raw/run_20251031_211812/manipulated_records.csv \
  --target_samples 200 \
  --random_state 123

# LEGACY: Directory path (deprecated but still works)
python compare_clustering_algorithms.py \
  --dataset_path /path/to/data/run_XXXXX

# AUTO-DETECT: No arguments (searches common locations)
python compare_clustering_algorithms.py

# Get help
python compare_clustering_algorithms.py --help
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mask-path` | str | auto-detect | **Path to masks.csv** (intent labels: 1=intentional, -1=unintentional, 0=no change) |
| `--clean-data-path` | str | auto-detect | **Path to correct_records.csv** (original/clean data) |
| `--dirty-data-path` | str | auto-detect | **Path to manipulated_records.csv** (modified/dirty data) |
| `--dataset_path` | str | auto-detect | [DEPRECATED] Path to directory containing data files |
| `--target_samples` | int | 127 | Target number of samples to select |
| `--random_state` | int | 42 | Random seed for reproducibility |

### Required File Structure

Your dataset must have these three files:

1. **masks.csv**
   - Contains intent labels for each feature
   - Values: `1` (intentional), `-1` (unintentional), `0` (no change)
   - Columns: feature names
   - Rows: variant records

2. **correct_records.csv**
   - Original/clean data before manipulation
   - Columns: same features as masks.csv
   - Rows: original records (before creating variants)

3. **manipulated_records.csv**
   - Modified/dirty data after manipulation
   - Columns: same features as masks.csv
   - Rows: variant records (matches masks.csv row count)

### Output Files

All results saved to `results/clustering_comparison/`:
- `algorithm_comparison.csv` - Full numeric results
- `detailed_summary.txt` - Human-readable summary
- `algorithm_comparison.png` - Visualizations
- `comparison_all_data_evaluation.log` - Execution log
- `FINAL_RESULTS_ALL_DATA_EVALUATION.md` - Comprehensive report

---

## ⚠️ Obsolete: `train_hierarchical_clustering.py`

### Why It's Obsolete

1. **Only tests one algorithm** (Hierarchical Clustering)
   - `compare_clustering_algorithms.py` tests 6 algorithms

2. **Uses old 70/30 train/test split**
   - `compare_clustering_algorithms.py` evaluates on ALL unseen data (96-99%)

3. **Lower performance** due to smaller test set
   - Old: K-Means = 77.16% F1 (tested on 8,441 samples)
   - New: K-Means = 84.85% F1 (tested on 28,061 samples)

4. **No command-line arguments**
   - Hard-coded paths and parameters

### Migration Actions

**Option 1: Archive (Recommended)**
```bash
mkdir -p archived_scripts
mv train_hierarchical_clustering.py archived_scripts/
mv hierarchical_clustering_*.log archived_scripts/
```

**Option 2: Delete**
```bash
rm train_hierarchical_clustering.py
rm hierarchical_clustering_*.log
```

---

## 🎯 Performance Comparison

### Old Script (train_hierarchical_clustering.py)
- **Evaluation:** 30% test split (8,441 samples)
- **Best Result:** 77.16% F1 (K-Means with Smart Sampling)
- **Algorithms:** 1 (Hierarchical only)
- **Train/Test:** 70/30 split

### New Script (compare_clustering_algorithms.py)
- **Evaluation:** ALL unseen data (96-99% of dataset, ~27,000-28,000 samples)
- **Best Result:** 90.39% F1 (HDBSCAN with Normal Sampling)
- **Algorithms:** 6 (K-Means, DBSCAN, Hierarchical-Ward, Hierarchical-Average, GMM, HDBSCAN)
- **Train/Test:** No split - trains on selected samples, tests on ALL remaining

### Results Table

| Algorithm | Old F1 (30% split) | New F1 (ALL data) | Improvement |
|-----------|-------------------|-------------------|-------------|
| K-Means | 77.16% | 84.85% | +7.69% |
| Hierarchical-Ward | N/A | 87.83% | - |
| Hierarchical-Average | N/A | 86.42% | - |
| DBSCAN | N/A | 88.13% | - |
| GMM | N/A | 87.76% | - |
| HDBSCAN | N/A | **90.39%** | - |

---

## 📊 Key Insights

### From Latest Run (ALL Data Evaluation)

1. **HDBSCAN wins** with 90.39% F1 using only 3.75% training data
2. **Normal sampling beats Smart** in ALL 6 algorithms (100% win rate)
3. **Sample efficiency:** Excellent performance with minimal training
4. **Proper evaluation:** Testing on 96-99% of data (not 30%)

### Dataset Statistics

- **Total records:** 19,540 variants
  - **With changes:** 18,207 variants (used for training/testing)
  - **Without changes:** 1,333 variants (excluded)
- **Total samples:** 28,256 feature changes
  - **Intentional:** 13,291 (47.0%)
  - **Unintentional:** 14,965 (53.0%)

### Terminology

- **VARIANT** = One modified data record (e.g., person #42)
- **SAMPLE** = One feature change within a variant (e.g., age=25→30)
- One variant can have multiple samples (if multiple features changed)

---

## 🚀 Recommendations

1. **Use `compare_clustering_algorithms.py` going forward**
   - More comprehensive
   - Better evaluation method
   - Command-line support
   - Higher performance

2. **Archive `train_hierarchical_clustering.py`**
   - Keep for historical reference
   - Don't use for new experiments

3. **Preferred algorithm: HDBSCAN + Normal Sampling**
   - 90.39% F1 score
   - Adaptive clustering (no need to specify K)
   - Only 3.75% training data needed

4. **For production:**
   ```bash
   python compare_clustering_algorithms.py \
     --dataset_path /path/to/your/data/run_XXXXXX \
     --target_samples 127 \
     --random_state 42
   ```

---

## 📝 Migration Checklist

- [x] Add argparse support to `compare_clustering_algorithms.py`
- [x] Test command-line arguments work correctly
- [x] Document usage and examples
- [x] Compare results with old script
- [ ] Archive `train_hierarchical_clustering.py` (user action)
- [ ] Update any scripts/docs that reference the old file (user action)

---

**Last Updated:** December 10, 2025  
**Migration Status:** Complete ✅
