# Label Propagation Comparison - Quick Reference

## 🚀 Quick Start

```bash
cd /home/mohamed/error_injector/llms_baseline/error_detection_system/src/attribution/clustering-organized/scripts

# Run with default settings (1% sampling)
python3 compare_label_propagation.py

# Run with custom sample size
python3 compare_label_propagation.py --target-samples 40
```

## 📊 What It Does

Compares **3 approaches** for cell-level error detection:

1. **Graph-Based Label Propagation** (2 methods)
   - LabelPropagation (hard labels)
   - LabelSpreading (soft labels with clamping)

2. **Cluster-Constrained Propagation**
   - Majority voting within clusters

3. **Random Forest Classifier**
   - Supervised learning baseline

## 🎯 Key Results (40 samples, 600 cells labeled)

| Method | Best F1 | Best For | Speed |
|--------|---------|----------|-------|
| **RandomForest** | 0.9745 | **Error Detection** (F1_error=0.42) | ⚡ Fast (0.37s) |
| **LabelSpreading** | 0.9678 | **Overall Accuracy** (97.77%) | 🐢 Slow (1.67s) |
| **LabelPropagation** | 0.9677 | **Correct Class** (F1=0.99) | 🐢 Slow (2.34s) |
| **ClusterConstrained** | 0.9649 | Fast prediction | ⚡ Very Fast (~0s) |

## 💡 Recommendation

**For error detection:** Use **Random Forest** (4x better at detecting errors)  
**For accuracy:** Use **LabelSpreading** (highest overall accuracy)

## 📁 Outputs

Results saved to: `clustering-organized/outputs/run_label_prop_YYYYMMDD_HHMMSS/`
- `results/comparison_results.csv` - All metrics
- `plots/method_comparison.png` - Visual comparison
- `results/summary.txt` - Detailed summary

## 📖 Full Documentation

See [LABEL_PROPAGATION_GUIDE.md](../docs/LABEL_PROPAGATION_GUIDE.md) for complete documentation.

## 🔗 Related Scripts

- `compare_clustering_algorithms.py` - Original clustering comparison
- `train_classifier.py` - Train Random Forest only
- `baseline_guessing_strategies.py` - Random guessing baselines
