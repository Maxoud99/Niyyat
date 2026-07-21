# Timestamped Outputs Guide

## Overview

The clustering comparison script now automatically creates **timestamped output folders** for each run. This ensures that:
- ✅ No results are ever overwritten
- ✅ You can compare multiple runs side-by-side
- ✅ Each run is completely self-contained
- ✅ Easy to track experiments over time

## Folder Structure

Each run creates a timestamped folder under `outputs/`:

```
clustering-organized/
└── outputs/
    ├── run_20251210_123550/        # Timestamp: YYYYMMDD_HHMMSS
    │   ├── results/                # CSV and TXT result files
    │   │   ├── algorithm_comparison.csv
    │   │   └── detailed_summary.txt
    │   ├── plots/                  # All PNG visualizations
    │   │   ├── f1_comparison_normal_vs_smart.png
    │   │   ├── difference_heatmap.png
    │   │   ├── sample_efficiency.png
    │   │   ├── multi_metric_comparison.png
    │   │   ├── algorithm_comparison.png
    │   │   ├── dendrogram_ward.png
    │   │   └── dendrogram_average.png
    │   └── logs/                   # Execution log
    │       └── comparison_all_data_evaluation_20251210_123550.log
    ├── run_20251210_140000/        # Another run
    │   ├── results/
    │   ├── plots/
    │   └── logs/
    └── run_20251210_150000/        # Yet another run
        ├── results/
        ├── plots/
        └── logs/
```

## Timestamp Format

- **Format:** `YYYYMMDD_HHMMSS`
- **Example:** `run_20251210_123550`
  - Year: 2025
  - Month: 12 (December)
  - Day: 10
  - Hour: 12 (noon)
  - Minute: 35
  - Second: 50

## What Gets Saved

### Results Folder (`results/`)
- **algorithm_comparison.csv** - Detailed metrics for all 16 algorithm variations (8 algorithms × 2 sampling strategies)
- **detailed_summary.txt** - Human-readable summary with tables and statistics

### Plots Folder (`plots/`)
- **f1_comparison_normal_vs_smart.png** - Side-by-side F1 score comparison
- **difference_heatmap.png** - Color-coded win/loss heatmap (green=Normal wins, red=Smart wins)
- **sample_efficiency.png** - F1 score vs training data percentage
- **multi_metric_comparison.png** - 4-panel comprehensive view
- **algorithm_comparison.png** - Original per-class F1 comparison
- **dendrogram_ward.png** - Hierarchical clustering dendrogram (Ward linkage)
- **dendrogram_average.png** - Hierarchical clustering dendrogram (Average linkage)

### Logs Folder (`logs/`)
- **comparison_all_data_evaluation_YYYYMMDD_HHMMSS.log** - Complete execution log with all console output

## Usage Examples

### Basic Run
```bash
cd clustering-organized/scripts
python3 compare_clustering_algorithms.py
```

**Output:**
```
Run directory: ../outputs/run_20251210_123550/
```

### Multiple Runs
```bash
# Run 1
python3 compare_clustering_algorithms.py --target_samples 100
# Creates: outputs/run_20251210_120000/

# Run 2 (different parameters)
python3 compare_clustering_algorithms.py --target_samples 150
# Creates: outputs/run_20251210_120130/

# Run 3 (different random seed)
python3 compare_clustering_algorithms.py --random_state 999
# Creates: outputs/run_20251210_120245/
```

Each run is completely independent!

### Finding Latest Run
```bash
cd clustering-organized
ls -lt outputs/run_* | head -1
```

### Comparing Multiple Runs
```bash
# Compare results from two runs
diff outputs/run_20251210_120000/results/algorithm_comparison.csv \
     outputs/run_20251210_130000/results/algorithm_comparison.csv

# Or view them side-by-side
cat outputs/run_20251210_120000/results/detailed_summary.txt
cat outputs/run_20251210_130000/results/detailed_summary.txt
```

## Benefits

### 1. **No Data Loss**
Every run is preserved. You'll never accidentally overwrite important results.

### 2. **Easy Comparison**
Compare results from different:
- Target sample sizes
- Random seeds
- Data sources
- Dates/times

### 3. **Experiment Tracking**
Keep a history of all experiments:
```bash
ls -lh outputs/
# run_20251210_120000/  - Initial baseline
# run_20251210_130000/  - Increased samples to 200
# run_20251210_140000/  - Different random seed
# run_20251210_150000/  - New dataset
```

### 4. **Clean Organization**
Each run is self-contained with:
- All results
- All plots
- Complete execution log

## Finding Your Results

### Latest Run
The script prints the output directory at startup and completion:
```
EXECUTION STARTED: 2025-12-10 12:35:50
Output directory: ../outputs/run_20251210_123550
```

### Specific File
```bash
# Find latest comparison CSV
find outputs -name "algorithm_comparison.csv" -type f | sort | tail -1

# Find all F1 comparison plots
find outputs -name "f1_comparison_normal_vs_smart.png" -type f

# Find logs from today
find outputs -name "*.log" -type f -mtime 0
```

### Browse All Runs
```bash
cd clustering-organized/outputs
ls -ltr  # List by time (oldest first)
```

## Cleanup Old Runs

If you accumulate many runs, you can clean up old ones:

### Keep Only Recent Runs
```bash
cd clustering-organized/outputs
# Keep last 5 runs, remove older ones
ls -t run_* | tail -n +6 | xargs rm -rf
```

### Remove Failed Runs
```bash
# Find runs with no results
for dir in outputs/run_*/; do
    if [ ! -f "$dir/results/algorithm_comparison.csv" ]; then
        echo "Incomplete run: $dir"
        # rm -rf "$dir"  # Uncomment to delete
    fi
done
```

### Archive Old Runs
```bash
# Archive runs older than 30 days
mkdir -p archived_runs
find outputs -name "run_*" -type d -mtime +30 -exec mv {} archived_runs/ \;
```

## Notes

- **Timestamp Precision:** Second-level precision means you can run multiple times per minute
- **Timezone:** Timestamps use local system time
- **Disk Space:** Each run takes ~5-10 MB (mostly plots)
- **Parallel Runs:** Safe to run multiple instances simultaneously (different timestamps)

## Example Session

```bash
mohamed@vega:~/tenth-trial/clustering-organized/scripts$ python3 compare_clustering_algorithms.py

======================================================================
EXECUTION STARTED: 2025-12-10 12:35:50
Log file: ../outputs/run_20251210_123550/logs/comparison_all_data_evaluation_20251210_123550.log
Output directory: ../outputs/run_20251210_123550
======================================================================

[... execution ...]

======================================================================
✓ COMPARISON COMPLETE!
======================================================================

📁 All outputs saved to: ../outputs/run_20251210_123550
   ├── results/  - CSV and TXT result files
   ├── plots/    - PNG visualization files
   └── logs/     - Execution log file

Command used:
  Dataset: auto-detected
  Target samples: 127
  Random state: 42

======================================================================
EXECUTION COMPLETED: 2025-12-10 12:42:15
======================================================================
Log saved to: ../outputs/run_20251210_123550/logs/comparison_all_data_evaluation_20251210_123550.log
```

## Quick Reference

| What | Where |
|------|-------|
| All run outputs | `outputs/run_YYYYMMDD_HHMMSS/` |
| CSV results | `outputs/run_*/results/algorithm_comparison.csv` |
| Summary text | `outputs/run_*/results/detailed_summary.txt` |
| Visualizations | `outputs/run_*/plots/*.png` |
| Execution log | `outputs/run_*/logs/*.log` |
| Latest run | `ls -t outputs/run_* \| head -1` |
| All plots | `find outputs -name "*.png"` |
| Today's runs | `find outputs -name "run_*" -mtime 0` |

---

**Last Updated:** December 10, 2025
