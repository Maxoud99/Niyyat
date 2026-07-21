# Quick Logging Reference Card

## 📁 Where Are The Logs?

```
outputs/run_YYYYMMDD_HHMMSS/logs/
```

## 🗂️ 15 Log Files Created Per Run

| # | Log File | Purpose |
|---|----------|---------|
| 00 | `main_pipeline_*.log` | Overall flow, F1 scores, errors |
| 01 | `data_loading_*.log` | Data loading & preprocessing |
| 02 | `feature_creation_*.log` | Feature engineering |
| 03 | `kmeans_*.log` | K-Means clustering details |
| 04 | `dbscan_*.log` | DBSCAN clustering details |
| 05 | `hierarchical_ward_*.log` | Hierarchical (Ward) details |
| 06 | `hierarchical_average_*.log` | Hierarchical (Average) details |
| 07 | `hierarchical_complete_*.log` | Hierarchical (Complete) details |
| 08 | `spectral_*.log` | Spectral clustering details |
| 09 | `gmm_*.log` | GMM clustering details |
| 10 | `hdbscan_*.log` | HDBSCAN clustering details |
| 11 | `normal_sampling_*.log` | Normal sampling strategy |
| 12 | `smart_sampling_*.log` | Smart sampling strategy |
| 13 | `evaluation_*.log` | Model evaluation & metrics |
| 14 | `visualization_*.log` | Plot generation |
| -- | `comparison_all_data_*.log` | Complete console output |

## 🔍 Quick Commands

### See All Logs
```bash
ls -lh outputs/run_*/logs/
```

### Main Pipeline Overview
```bash
cat outputs/run_*/logs/00_main_pipeline*.log
```

### All F1 Scores
```bash
grep "F1:" outputs/run_*/logs/00_main_pipeline*.log
```

### Check Specific Algorithm
```bash
cat outputs/run_*/logs/03_kmeans*.log
```

### Compare Sampling Strategies
```bash
cat outputs/run_*/logs/11_normal_sampling*.log
cat outputs/run_*/logs/12_smart_sampling*.log
```

### Find Errors
```bash
grep -i "error\|warning" outputs/run_*/logs/*.log
```

### Monitor Running Pipeline
```bash
tail -f outputs/run_*/logs/00_main_pipeline*.log
```

## 📖 What's Logged?

### For Each Algorithm (03-10):
- ✓ All parameters
- ✓ Feature matrix shape
- ✓ Cluster distribution
- ✓ Quality metrics (Silhouette, etc.)
- ✓ Runtime

### For Sampling (11-12):
- ✓ Cluster sizes
- ✓ Allocation strategy
- ✓ Per-cluster sampling
- ✓ Intent distribution
- ✓ Final selection

### For Evaluation (13):
- ✓ Training/test split
- ✓ Model parameters
- ✓ All metrics (F1, Precision, Recall, Accuracy)

## 🎯 Common Use Cases

| Need | Command |
|------|---------|
| Quick overview | `cat 00_main_pipeline*.log` |
| Best algorithm | `grep "F1:" 00_main_pipeline*.log \| sort -t':' -k2 -nr` |
| Runtimes | `grep "Runtime:" 0*.log` |
| Cluster sizes | `grep "Cluster [0-9]:" 03_kmeans*.log` |
| Sampling stats | `grep "Final selection:" 1*.log` |
| Errors | `grep -i error *.log` |

## 📚 Full Guide

See: [DETAILED_LOGGING_GUIDE.md](DETAILED_LOGGING_GUIDE.md) for complete documentation!

---

**Location:** `/home/mohamed/error_injector/llms_baseline/clustering-organized/outputs/run_*/logs/`
