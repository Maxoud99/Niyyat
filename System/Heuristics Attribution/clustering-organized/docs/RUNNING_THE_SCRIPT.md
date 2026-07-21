# Running the Clustering Comparison Script

## Quick Start

### Method 1: From clustering-organized directory (RECOMMENDED)
```bash
cd /path/to/tenth-trial/clustering-organized
./run_comparison.sh
```

### Method 2: From scripts directory
```bash
cd /path/to/tenth-trial/clustering-organized/scripts
python3 compare_clustering_algorithms.py
```

### Method 3: From anywhere using absolute path
```bash
python3 /full/path/to/clustering-organized/scripts/compare_clustering_algorithms.py
```

## Data Auto-Detection

The script automatically finds data in this order:

1. **clustering-organized/data/** (symlink to ../data)
2. **tenth-trial/data/raw/run_20251031_211812/**
3. **Relative paths** (if running from tenth-trial/)

### Auto-Detection Works From:
- ✅ `clustering-organized/` directory
- ✅ `clustering-organized/scripts/` directory  
- ✅ Any directory using absolute path
- ✅ Any directory using wrapper script

## Custom Data Paths

If auto-detection fails or you want to use different data:

```bash
python3 compare_clustering_algorithms.py \
  --mask-path /path/to/masks.csv \
  --clean-data-path /path/to/correct_records.csv \
  --dirty-data-path /path/to/manipulated_records.csv
```

## Output Location

Outputs are always created relative to the **script location**, NOT the current directory:

```
clustering-organized/
└── outputs/
    └── run_YYYYMMDD_HHMMSS/
        ├── results/
        ├── plots/
        └── logs/
```

This ensures outputs are always in the same place, regardless of where you run the script from.

## Examples

### Run with default settings from home directory:
```bash
cd ~
/path/to/clustering-organized/run_comparison.sh
```

### Run with custom target samples:
```bash
cd /path/to/clustering-organized
./run_comparison.sh --target_samples 200
```

### Run with custom random seed:
```bash
cd /path/to/clustering-organized/scripts
python3 compare_clustering_algorithms.py --random_state 999
```

### Run with custom data from anywhere:
```bash
python3 /path/to/scripts/compare_clustering_algorithms.py \
  --mask-path ~/my_data/masks.csv \
  --clean-data-path ~/my_data/correct.csv \
  --dirty-data-path ~/my_data/dirty.csv
```

## Troubleshooting

### "Could not find masks.csv"

**Solution 1:** Use the wrapper script
```bash
cd /path/to/clustering-organized
./run_comparison.sh
```

**Solution 2:** Specify paths explicitly
```bash
python3 compare_clustering_algorithms.py \
  --mask-path /full/path/to/masks.csv \
  --clean-data-path /full/path/to/correct_records.csv \
  --dirty-data-path /full/path/to/manipulated_records.csv
```

**Solution 3:** Run from the scripts directory
```bash
cd /path/to/clustering-organized/scripts
python3 compare_clustering_algorithms.py
```

### "RuntimeError: main thread is not in main loop"

This is fixed! The script now uses `matplotlib.use('Agg')` for non-interactive plotting.

### Outputs not where expected

Outputs are ALWAYS in:
```
clustering-organized/outputs/run_YYYYMMDD_HHMMSS/
```

regardless of where you run the script from.

## Full Example

```bash
# Navigate to clustering-organized
cd /home/mohamed/error_injector/llms_baseline/adult_income_dataset/tenth-trial/clustering-organized

# Run with default settings (auto-detects data)
./run_comparison.sh

# Or with custom parameters
./run_comparison.sh --target_samples 150 --random_state 42

# Check outputs
ls -lh outputs/
ls -lh outputs/run_20251210_*/
```

## Command-Line Arguments

```
--target_samples N      Number of samples to select (default: 127)
--random_state N        Random seed for reproducibility (default: 42)
--mask-path PATH        Path to masks.csv (auto-detected if not provided)
--clean-data-path PATH  Path to correct_records.csv (auto-detected if not provided)
--dirty-data-path PATH  Path to manipulated_records.csv (auto-detected if not provided)
```

## Notes

- **Symlink:** `clustering-organized/data` points to `tenth-trial/data` (saves disk space)
- **Auto-detection:** Uses script location to find data, works from any directory
- **Outputs:** Always saved to `clustering-organized/outputs/run_YYYYMMDD_HHMMSS/`
- **Logs:** Execution log includes full command and data paths used
