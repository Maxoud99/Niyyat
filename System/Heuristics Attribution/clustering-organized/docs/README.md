# Tenth Trial - Adult Income Dataset

Comprehensive error injection experiment with LLM-based intent attribution analysis.

## 📁 Directory Structure

```
tenth-trial/
│
├── 📄 README.md                    # This file - main guide
│
├── 📊 data/                        # All datasets
│   ├── raw/                        # Original/generated data
│   │   ├── correct_records.csv               # Clean dataset (19,539 records)
│   │   └── run_20251031_211812/              # Error injection run
│   │       ├── manipulated_records.csv       # Records with injected errors
│   │       ├── masks.csv                     # Error masks
│   │       ├── masks-blind.csv               # Blind evaluation masks
│   │       ├── metadata.jsonl                # Error metadata
│   │       ├── analysis_output/              # Analysis results
│   │       └── failures/                     # Failed injections
│   │
│   ├── sliced/                     # Data sliced by different criteria
│   │   ├── by_error_count/                   # Sliced by number of errors
│   │   ├── by_manipulation_type/             # Sliced by error type
│   │   ├── sliced_data_structure.txt         # Structure documentation
│   │   └── README.md                         # Slicing methodology
│   │
│   └── error_analysis/             # Error analysis CSVs
│       ├── error_count_per_record.csv        # Errors per record
│       ├── accuracy_by_error_count_detailed.csv
│       ├── correlation_error_count_accuracy.csv
│       └── model_performance_by_error_count.csv  # 14MB detailed results
│
├── 🤖 results/                     # Model outputs and analysis
│   ├── model_outputs/              # Raw LLM outputs
│   │   └── local-llms/                       # Local model results
│   │       ├── first-trial-llama/
│   │       ├── first-trial-mixtral/
│   │       ├── first-trial-R1-deepseek/
│   │       ├── first-trial-qwen/
│   │       ├── first-trial-gemini-pro/
│   │       └── [6 more model trials...]
│   │
│   └── analysis/                   # Comparative analysis
│       └── analysis_comparison/              # 5 models × 3 trials comparison
│           ├── visualizations/               # 8 PNG charts
│           ├── summary_csvs/                 # Aggregated results
│           ├── documentation/                # 8 analysis docs
│           ├── scripts/                      # 7 analysis scripts
│           ├── raw_data/                     # Per-model CSVs
│           ├── README.md                     # Detailed guide
│           ├── NAVIGATION.txt                # Quick reference
│           └── COMMANDS.txt                  # Command cheatsheet
│
├── 🔧 scripts/                     # Analysis scripts
│   └── generate_error_count_analysis.py      # Error count analysis
│
└── 📚 documentation/               # Documentation & prompts
    ├── ANALYSIS_INDEX.md                     # Analysis overview
    └── prompt_for_generating_attacks.txt     # Attack generation prompt
```

## 🎯 Quick Start

### View Main Analysis Results
```bash
cd results/analysis/analysis_comparison
cat README.md  # Complete guide
cat NAVIGATION.txt  # Quick reference
```

### View Visualizations
```bash
cd results/analysis/analysis_comparison/visualizations
ls -lh *.png
```

### Access Data
```bash
# Original data
cd data/raw
head correct_records.csv

# Error analysis
cd data/error_analysis
column -t -s, error_count_per_record.csv | less -S

# Sliced data
cd data/sliced/by_error_count
ls -lh
```

## 📊 Dataset Overview

- **Original Records**: 19,539 clean records
- **Features**: 15 (age, workclass, education, marital-status, occupation, etc.)
- **Error Types**: 4 manipulation types
  - Unintentional errors
  - Gain-targeted manipulations
  - Fairness masking
  - Obfuscation DMV (Disguised Missing Values)

## 🤖 Models Tested

**5 LLM Models × 3 Trial Types = 15 Configurations**

### Models:
1. LLAMA
2. MIXTRAL
3. DEEPSEEK-R1
4. QWEN
5. GEMINI

### Trial Types:
1. **Bare Minimum** - Basic prompt
2. **With Info** - Enhanced with context
3. **Info + Few-Shot** - Full examples provided

## 📈 Key Findings

**Best Overall Performance:**
- Model: GEMINI
- Trial: Bare Minimum
- Macro F1: 0.8181

**Best by Error Type:**
- Unintentional: MIXTRAL (Bare) - F1 = 0.989
- Gain Targeted: QWEN (Few-Shot) - F1 = 0.997
- Fairness Masking: QWEN (Few-Shot) - F1 = 0.997
- Obfuscation DMV: DEEPSEEK-R1 (Few-Shot) - F1 = 0.973

See `results/analysis/analysis_comparison/` for complete analysis.

## 🔍 Data Files Description

### Raw Data
- `correct_records.csv` - Clean baseline dataset
- `manipulated_records.csv` - Records with injected errors
- `masks.csv` - Binary masks indicating error locations
- `metadata.jsonl` - Error injection metadata

### Error Analysis
- `error_count_per_record.csv` - Number of errors per record
- `accuracy_by_error_count_detailed.csv` - Performance vs error count
- `model_performance_by_error_count.csv` - Detailed model performance (14MB)

### Sliced Data
Data organized by:
- Error count (0, 1, 2, 3+ errors)
- Manipulation type (4 types)

## 🔧 Running Analysis

```bash
# Generate error count analysis
cd scripts
python generate_error_count_analysis.py

# Regenerate comparison visualizations
cd results/analysis/analysis_comparison/scripts
python create_improved_comprehensive.py
```

## 📝 Documentation

- `ANALYSIS_INDEX.md` - Comprehensive analysis overview
- `prompt_for_generating_attacks.txt` - Attack generation methodology
- See `results/analysis/analysis_comparison/documentation/` for detailed analysis docs

## 🎨 Visualizations

8 publication-quality visualizations available in:
`results/analysis/analysis_comparison/visualizations/`

1. Comprehensive comparison (4 styles)
2. Manipulation type analysis
3. Feature-level analysis
4. Performance by error count
5. Learning curves

## 📞 Navigation

| Need | Location |
|------|----------|
| Analysis results | `results/analysis/analysis_comparison/` |
| Model outputs | `results/model_outputs/local-llms/` |
| Raw data | `data/raw/` |
| Error analysis | `data/error_analysis/` |
| Sliced data | `data/sliced/` |
| Scripts | `scripts/` |
| Documentation | `documentation/` |

## 🔗 Quick Links

**Main Analysis**: `results/analysis/analysis_comparison/README.md`  
**Visualizations**: `results/analysis/analysis_comparison/visualizations/`  
**Summary CSVs**: `results/analysis/analysis_comparison/summary_csvs/`  
**Documentation**: `results/analysis/analysis_comparison/documentation/`  

---

**Experiment Date**: October 31, 2024  
**Dataset**: Adult Income (UCI ML Repository)  
**Total Records**: 19,539  
**Models Tested**: 5 (15 configurations)  
**Last Updated**: November 24, 2025
