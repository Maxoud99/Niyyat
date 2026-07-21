# Clustering-Organized

**Location:** `/home/mohamed/error_injector/llms_baseline/clustering-organized/`

This is a **dataset-independent** clustering comparison framework, originally developed in the tenth-trial experiments and now relocated to `llms_baseline` for broader applicability.

## 📁 Folder Structure

```
clustering-organized/
├── scripts/                    # All Python scripts (9 files)
├── docs/                       # All documentation (7 files)
├── outputs/                    # ⭐ TIMESTAMPED OUTPUT FOLDERS
│   ├── run_20251210_123550/    # Each run gets its own folder
│   │   ├── results/            # CSV and TXT results
│   │   ├── plots/              # PNG visualizations
│   │   └── logs/               # Execution log
│   ├── run_20251210_140000/    # Another run
│   └── run_20251210_150000/    # Yet another run
└── data/                       # Symlink to ../data
```

## ⚡ Key Feature: Timestamped Outputs

**Each run automatically creates a timestamped folder** (format: `run_YYYYMMDD_HHMMSS/`):
- ✅ **No results overwritten** - Every run is preserved
- ✅ **Easy comparison** - Compare multiple runs side-by-side
- ✅ **Complete history** - Track all experiments over time
- ✅ **Self-contained** - Each run has results + plots + logs

📖 **See [TIMESTAMPED_OUTPUTS_GUIDE.md](TIMESTAMPED_OUTPUTS_GUIDE.md) for details**

## 🚀 Quick Start

### **Method 1: Using wrapper script (EASIEST - Run from anywhere!)**
```bash
cd /home/mohamed/error_injector/llms_baseline/clustering-organized
./run_comparison.sh
```

### **Method 2: From scripts directory**
```bash
cd /home/mohamed/error_injector/llms_baseline/clustering-organized/scripts
python3 compare_clustering_algorithms.py
```

### **Method 3: From anywhere using absolute path**
```bash
python3 /home/mohamed/error_injector/llms_baseline/clustering-organized/scripts/compare_clustering_algorithms.py
```

**Output:** Creates `outputs/run_YYYYMMDD_HHMMSS/` with all results, plots, and logs!

**✨ NEW: Outputs now always go to `clustering-organized/outputs/` regardless of where you run the script from!**

**📖 See [RUNNING_THE_SCRIPT.md](RUNNING_THE_SCRIPT.md) for detailed instructions**

### **With custom parameters:**
```bash
./run_comparison.sh \
  --target_samples 127 \
  --logging True  # ⭐ NEW: Enable detailed logging (15 log files)
```

### **With custom data paths:**
```bash
./run_comparison.sh \
  --mask-path /path/to/masks.csv \
  --clean-data-path /path/to/correct_records.csv \
  --dirty-data-path /path/to/manipulated_records.csv \
  --logging False  # ⭐ NEW: Disable detailed logging (only 1 log file)
```

### **⭐ NEW Logging Features:**
- `--logging True` (default): Creates 15 detailed log files with cluster membership & representatives
- `--logging False`: Creates only 1 console log file (fast, minimal)
- **📖 See [LOGGING_FEATURES_SUMMARY.md](docs/LOGGING_FEATURES_SUMMARY.md) for complete guide**

### **⭐ NEW: Label Propagation for Intent Classification**
```bash
# Compare label propagation vs Random Forest for intent classification
python3 scripts/compare_label_propagation.py --target-samples 40

# With custom data paths
python3 scripts/compare_label_propagation.py \
  --data-path /path/to/combined_dataset_no_id_v2.csv \
  --mask-path /path/to/ground_truth_masks_v2.csv \
  --target-samples 80
```

**Compares 3 approaches for cell-level intent classification (intentional vs unintentional):**
1. Graph-Based Label Propagation (LabelPropagation & LabelSpreading)
2. Cluster-Constrained Propagation (majority voting)
3. Random Forest Classifier (supervised baseline)

**🎯 Result:** Random Forest wins (4x better F1 for intentional class, 5x faster)

**📖 See [docs/propagation/](docs/propagation/) for complete documentation:**
- **README.md** - Documentation index and navigation guide
- **RESULTS_SUMMARY.md** ⭐ - Complete experimental results and analysis
- **LABEL_PROPAGATION_GUIDE.md** - Technical deep-dive (380+ lines)
- **LABEL_PROPAGATION_README.md** - Quick start guide
- **LABEL_PROPAGATION_IMPLEMENTATION_SUMMARY.md** - Implementation details

## 📊 Main Scripts

### Core Analysis
- **compare_clustering_algorithms.py** ⭐ - Main script comparing Normal vs Smart sampling across 8 clustering algorithms
- **compare_label_propagation.py** ⭐ NEW! - Compare label propagation vs Random Forest for cell-level error detection
- **baseline_guessing_strategies.py** - Baseline random guessing strategies
- **compare_baselines_vs_llms.py** - Compare baselines against LLM performance

### Training & Evaluation
- **train_classifier.py** - Train Random Forest classifier on clustered samples
- **train_smart_sampling.py** - Train using smart (stratified) sampling
- **train_classifier_varied_sizes.py** - Evaluate performance across different training sizes

### Visualization
- **plot_complete_comparison.py** - Generate comprehensive comparison plots
- **plot_training_size_comparison.py** - Visualize training size impact
- **generate_error_count_analysis.py** - Error distribution analysis

## 📄 Key Documentation

### Core Documentation
- **README.md** - This file
- **COMPLETE_PIPELINE_EXPLANATION.md** ⭐ - 1200+ line detailed pipeline guide
- **TIMESTAMPED_OUTPUTS_GUIDE.md** ⭐ - Complete guide to timestamped output folders

### ⭐ Label Propagation Documentation (`docs/propagation/`)
**Research:** Can label propagation replace Random Forest for intent classification?  
**Answer:** NO - Random Forest is 4x better for intentional class detection

- **README.md** ⭐ START HERE - Documentation index and navigation
- **RESULTS_SUMMARY.md** ⭐ - Complete experimental results (13KB)
  - Research question answered
  - Detailed performance comparison
  - Why label propagation fails
  - Production recommendations
- **LABEL_PROPAGATION_GUIDE.md** - Technical deep-dive (380+ lines)
- **LABEL_PROPAGATION_README.md** - Quick start guide
- **LABEL_PROPAGATION_IMPLEMENTATION_SUMMARY.md** - Implementation checklist
- **DELIVERABLES_CHECKLIST.md** - Project validation

**Key Finding:** Random Forest (F1=0.42-0.44) outperforms Label Propagation (F1=0.10-0.27) for intent classification by 4x, and is 5x faster.

### Logging Documentation ⭐ NEW
- **LOGGING_FEATURES_SUMMARY.md** ⭐ - Quick summary of new logging features
- **NEW_LOGGING_FEATURES_GUIDE.md** ⭐ - Complete guide to `--logging` toggle & cluster details
- **DETAILED_LOGGING_GUIDE.md** - Complete guide to 15 separate log files per run
- **LOGGING_QUICK_REFERENCE.md** - Quick command reference

### Results & Analysis
- **NORMAL_VS_SMART_SAMPLING_RESULTS.md** - Detailed results comparison
- **FINAL_RESULTS_ALL_DATA_EVALUATION.md** - Final evaluation results
- **BASELINES_COMPLETE_GUIDE.md** - Complete guide to baseline strategies

### Technical Documentation
- **RELOCATION_SUMMARY.md** - Move to llms_baseline details
- **CLEANUP_AND_ENHANCEMENT_SUMMARY.md** - Recent improvements log
- **SCRIPT_MIGRATION.md** - Migration guide for dataset independence
- **CLEANUP_PROPOSAL.md** - Original cleanup plan

## 📈 Outputs (Per Run)

Each run creates a timestamped folder: `outputs/run_YYYYMMDD_HHMMSS/`

### Logs (`outputs/run_*/logs/`) - **15 DETAILED LOG FILES! 🎯**
- `00_main_pipeline_*.log` - Overall pipeline execution & F1 scores
- `01_data_loading_*.log` - Data loading details & statistics
- `02_feature_creation_*.log` - Feature engineering logs
- `03_kmeans_*.log` - K-Means algorithm details
- `04_dbscan_*.log` - DBSCAN algorithm details
- `05_hierarchical_ward_*.log` - Hierarchical (Ward) details
- `06_hierarchical_average_*.log` - Hierarchical (Average) details
- `07_hierarchical_complete_*.log` - Hierarchical (Complete) details
- `08_spectral_*.log` - Spectral clustering details
- `09_gmm_*.log` - Gaussian Mixture Model details
- `10_hdbscan_*.log` - HDBSCAN algorithm details
- `11_normal_sampling_*.log` - Normal sampling strategy logs
- `12_smart_sampling_*.log` - Smart sampling strategy logs
- `13_evaluation_*.log` - Model evaluation & metrics
- `14_visualization_*.log` - Visualization creation logs
- `comparison_all_data_evaluation_*.log` - Complete console output

**📖 See [DETAILED_LOGGING_GUIDE.md](docs/DETAILED_LOGGING_GUIDE.md) for usage examples!**

### Plots (`outputs/run_*/plots/`)
- `f1_comparison_normal_vs_smart.png` - Side-by-side F1 comparison
- `difference_heatmap.png` - Win/loss heatmap (green=Normal, red=Smart)
- `sample_efficiency.png` - F1 vs training data percentage
- `multi_metric_comparison.png` - 4-panel comprehensive view
- `algorithm_comparison.png` - Original per-class comparison
- `dendrogram_ward.png` - Ward linkage dendrogram
- `dendrogram_average.png` - Average linkage dendrogram

### Results (`outputs/run_*/results/`)
- `algorithm_comparison.csv` - Detailed metrics for all 16 algorithm variations
- `detailed_summary.txt` - Human-readable summary with tables

## 🔧 Requirements

Install required packages:
```bash
pip3 install --user pandas numpy scikit-learn matplotlib seaborn hdbscan
```

## 💡 Key Features

1. **Dataset Independent** - Works with any tabular dataset
2. **Timestamped Logging** - Auto-generates timestamped log files
3. **Comprehensive Plotting** - 4 publication-quality visualizations
4. **8 Clustering Algorithms** - K-Means, Agglomerative (Ward/Average/Complete), DBSCAN, OPTICS, Spectral, HDBSCAN
5. **2 Sampling Strategies** - Normal (Proportional Random) vs Smart (Stratified by Intent)

## 📝 Usage Examples

### Basic run (auto-detects data):
```bash
cd scripts
python3 compare_clustering_algorithms.py
```

### Custom target samples:
```bash
python3 compare_clustering_algorithms.py --target_samples 200
```

### Custom random seed:
```bash
python3 compare_clustering_algorithms.py --random_state 123
```

### Full custom configuration:
```bash
python3 compare_clustering_algorithms.py \
  --target_samples 150 \
  --random_state 42 \
  --mask-path ../data/raw/run_20251031_211812/masks.csv \
  --clean-data-path ../data/raw/run_20251031_211812/correct_records.csv \
  --dirty-data-path ../data/raw/run_20251031_211812/manipulated_records.csv
```

## 📊 Expected Results

The script will generate:
- Console output with progress and results
- Timestamped log file in `outputs/logs/`
- 7 visualization plots in `outputs/plots/`
- 2 result files in `outputs/results/`

## 🎯 Typical Workflow

1. **Run comparison:**
   ```bash
   cd scripts
   python3 compare_clustering_algorithms.py
   ```

2. **Check results:**
   ```bash
   cat ../outputs/results/detailed_summary.txt
   ```

3. **View visualizations:**
   ```bash
   ls -lh ../outputs/plots/
   ```

4. **Review logs:**
   ```bash
   tail -100 ../outputs/logs/comparison_all_data_evaluation_*.log
   ```

## 📌 Notes

- Original files remain in `tenth-trial/` root directory
- This folder is a clean, organized copy
- Data folder is symlinked (not copied) to save space
- Scripts automatically create `results/clustering_comparison/` folder when run
- All paths are relative to script location for portability

## 🔗 Related Folders

- **../data/** - Raw data files (masks, correct_records, manipulated_records)
- **../archived_obsolete/** - Archived old files from cleanup
- **../results/** - Original results folder (still exists in root)

---

Created: December 10, 2025
Location: `/home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/clustering-organized/`
