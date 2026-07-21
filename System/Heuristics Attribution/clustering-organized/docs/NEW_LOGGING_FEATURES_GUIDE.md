# New Logging Features Guide

## 🎛️ Feature 1: Toggle Detailed Logging

### Overview
You can now enable or disable the detailed logging system using the `--logging` argument.

### Usage

**Enable detailed logging (default):**
```bash
python3 scripts/compare_clustering_algorithms.py --logging True
# OR
python3 scripts/compare_clustering_algorithms.py --logging true
# OR
python3 scripts/compare_clustering_algorithms.py --logging 1
# OR
python3 scripts/compare_clustering_algorithms.py --logging yes
```

**Disable detailed logging:**
```bash
python3 scripts/compare_clustering_algorithms.py --logging False
# OR
python3 scripts/compare_clustering_algorithms.py --logging false
# OR
python3 scripts/compare_clustering_algorithms.py --logging 0
# OR
python3 scripts/compare_clustering_algorithms.py --logging no
```

### What Changes?

#### With `--logging True` (Default):
✅ **15 detailed log files created:**
- 00_main_pipeline_*.log
- 01_data_loading_*.log
- 02_feature_creation_*.log
- 03-10: Algorithm-specific logs
- 11_normal_sampling_*.log
- 12_smart_sampling_*.log
- 13_evaluation_*.log
- 14_visualization_*.log
- comparison_all_data_*.log (console output)

```
outputs/run_YYYYMMDD_HHMMSS/logs/
├── 00_main_pipeline_20251210_134531.log
├── 01_data_loading_20251210_134531.log
├── 02_feature_creation_20251210_134531.log
├── 03_kmeans_20251210_134531.log
├── ... (12 more log files)
└── comparison_all_data_evaluation_20251210_134531.log
```

#### With `--logging False`:
✅ **Only 1 log file created:**
- comparison_all_data_*.log (console output only)

```
outputs/run_YYYYMMDD_HHMMSS/logs/
└── comparison_all_data_evaluation_20251210_134547.log
```

### When to Use Each Mode?

**Use `--logging True` when:**
- 🐛 Debugging issues
- 📊 Analyzing algorithm behavior
- 🔍 Understanding sampling decisions
- 📈 Tracking performance metrics
- 📝 Creating detailed reports
- 🧪 Experimenting with parameters

**Use `--logging False` when:**
- ⚡ Quick test runs
- 💾 Saving disk space
- 🚀 Production runs where only results matter
- 📦 Minimal overhead needed

### Example Commands

```bash
# Quick test without detailed logs
./run_comparison.sh --target_samples 20 --logging False

# Full analysis with detailed logs
./run_comparison.sh --target_samples 127 --logging True

# With custom data and detailed logging
python3 scripts/compare_clustering_algorithms.py \
  --mask-path /path/to/masks.csv \
  --clean-data-path /path/to/clean.csv \
  --dirty-data-path /path/to/dirty.csv \
  --target_samples 100 \
  --logging True
```

---

## 📋 Feature 2: Cluster Elements & Representatives Logging

### Overview
When detailed logging is enabled, the system now logs:
1. **All variant IDs** in each cluster
2. **Picked representative samples** from each cluster
3. **Intent breakdown** (for smart sampling)

### What's Logged

#### For Normal Sampling (11_normal_sampling_*.log):

```
Cluster 0:
  Total variants in cluster: 2981
  Allocated samples: 2
  All variant IDs in cluster: [14, 23, 45, 67, ..., 18305]
  PICKED REPRESENTATIVES: [145, 2341]

Cluster 1:
  Total variants in cluster: 2629
  Allocated samples: 2
  All variant IDs in cluster: [12, 34, 56, ..., 18456]
  PICKED REPRESENTATIVES: [567, 1234]

...

================================================================================
CLUSTER MEMBERSHIP & REPRESENTATIVES SUMMARY
================================================================================

Cluster 0:
  Size: 2981 variants
  Picked: 2 representatives
  All members: [14, 23, 45, 67, 89, ..., 18305]
  Representatives: [145, 2341]

Cluster 1:
  Size: 2629 variants
  Picked: 2 representatives
  All members: [12, 34, 56, 78, ..., 18456]
  Representatives: [567, 1234]

...
================================================================================
```

#### For Smart Sampling (12_smart_sampling_*.log):

```
────────────────────────────────────────────────────────────────────────────────
Cluster 0:
  Total variants in cluster: 2981
  Allocated samples: 2
  All variant IDs in cluster: [14, 23, 45, 67, ..., 18305]
  Intent breakdown:
    Intentional-dominant variants: 0
      IDs: []
    Unintentional-dominant variants: 2981
      IDs: [14, 23, 45, ..., 18305]
  Only unintentional variants available
  PICKED REPRESENTATIVES: [145, 2341]
  Selected 2 variants total

────────────────────────────────────────────────────────────────────────────────
Cluster 2:
  Total variants in cluster: 644
  Allocated samples: 1
  All variant IDs in cluster: [123, 234, 345, ..., 11335]
  Intent breakdown:
    Intentional-dominant variants: 644
      IDs: [123, 234, 345, ..., 11335]
    Unintentional-dominant variants: 0
      IDs: []
  Only intentional variants available
  PICKED REPRESENTATIVES: [456]
  Selected 1 variants total

...

================================================================================
CLUSTER MEMBERSHIP & REPRESENTATIVES SUMMARY
================================================================================

Cluster 0:
  Size: 2981 variants
  Picked: 2 representatives
  All members: [14, 23, 45, ..., 18305]
  Intentional-dominant: []
  Unintentional-dominant: [14, 23, 45, ..., 18305]
  Representatives: [145, 2341]

Cluster 2:
  Size: 644 variants
  Picked: 1 representatives
  All members: [123, 234, 345, ..., 11335]
  Intentional-dominant: [123, 234, 345, ..., 11335]
  Unintentional-dominant: []
  Representatives: [456]

...
================================================================================
```

### Information Provided

For **each cluster**, you get:

1. **Cluster Size**: Total number of variant IDs in the cluster
2. **All Members**: Complete list of all variant IDs in the cluster
3. **Allocated Samples**: How many representatives to pick
4. **Picked Representatives**: Which variant IDs were selected

For **Smart Sampling**, additionally:

5. **Intentional-dominant variants**: Variant IDs with ≥50% intentional changes
6. **Unintentional-dominant variants**: Variant IDs with <50% intentional changes
7. **Stratification strategy**: How sampling balanced the two groups

### Use Cases

#### 1. Verify Sampling Coverage
```bash
# Check if all clusters are represented
grep "Cluster" outputs/run_*/logs/11_normal_sampling*.log

# See which clusters were empty
grep "Size: 0" outputs/run_*/logs/11_normal_sampling*.log
```

#### 2. Analyze Representative Selection
```bash
# See which variants were picked
grep "PICKED REPRESENTATIVES" outputs/run_*/logs/11_normal_sampling*.log

# Compare normal vs smart representatives
diff \
  <(grep "PICKED REPRESENTATIVES" outputs/run_*/logs/11_normal_sampling*.log) \
  <(grep "PICKED REPRESENTATIVES" outputs/run_*/logs/12_smart_sampling*.log)
```

#### 3. Understand Intent Distribution
```bash
# See intent breakdown per cluster
grep -A2 "Intent breakdown:" outputs/run_*/logs/12_smart_sampling*.log

# Find clusters with only intentional variants
grep "Only intentional" outputs/run_*/logs/12_smart_sampling*.log
```

#### 4. Track Specific Variant
```bash
# Find which cluster a variant belongs to
grep "variant_id: 1234" outputs/run_*/logs/11_normal_sampling*.log

# Was this variant picked as representative?
grep "1234" outputs/run_*/logs/11_normal_sampling*.log | grep "PICKED"
```

#### 5. Analyze Cluster Composition
```bash
# Extract cluster sizes
grep "Size:" outputs/run_*/logs/11_normal_sampling*.log

# Find large clusters
grep "Size:" outputs/run_*/logs/11_normal_sampling*.log | \
  awk '{if ($2 > 1000) print}'

# Count clusters
grep "^Cluster" outputs/run_*/logs/11_normal_sampling*.log | wc -l
```

### Example Analysis Session

```bash
cd /home/mohamed/error_injector/llms_baseline/clustering-organized/outputs/run_LATEST/logs/

# 1. Quick overview of sampling
echo "=== Normal Sampling ==="
grep "CLUSTER MEMBERSHIP" -A50 11_normal_sampling*.log | grep "Cluster\|Size:\|Representatives:"

echo -e "\n=== Smart Sampling ==="
grep "CLUSTER MEMBERSHIP" -A50 12_smart_sampling*.log | grep "Cluster\|Size:\|Representatives:"

# 2. Compare cluster sizes
echo -e "\n=== Cluster Size Comparison ==="
paste \
  <(grep "Size:" 11_normal_sampling*.log | awk '{print $2}') \
  <(grep "Size:" 12_smart_sampling*.log | awk '{print $2}')

# 3. Check intent stratification
echo -e "\n=== Intent Stratification ==="
grep "Intentional-dominant:\|Unintentional-dominant:" 12_smart_sampling*.log

# 4. Verify all representatives are unique
echo -e "\n=== Representative Uniqueness ==="
grep "PICKED REPRESENTATIVES" 11_normal_sampling*.log | \
  sed 's/.*: \[//; s/\]//' | tr ',' '\n' | sort | uniq -d
# If output is empty, all representatives are unique!
```

### Benefits

✅ **Full Traceability**: Know exactly which variants are in each cluster  
✅ **Representative Verification**: See which variants were chosen and why  
✅ **Intent Analysis**: Understand intent distribution per cluster (smart sampling)  
✅ **Debugging**: Quickly identify sampling issues  
✅ **Reproducibility**: Complete record of all sampling decisions  
✅ **Comparison**: Easy to compare normal vs smart sampling choices  

---

## 🎯 Combined Example

```bash
# Run with both features
cd /home/mohamed/error_injector/llms_baseline/clustering-organized

# Full detailed logging with cluster details
python3 scripts/compare_clustering_algorithms.py \
  --target_samples 127 \
  --logging True

# Check the detailed cluster information
cat outputs/run_LATEST/logs/11_normal_sampling*.log | \
  grep -A10 "CLUSTER MEMBERSHIP"

cat outputs/run_LATEST/logs/12_smart_sampling*.log | \
  grep -A15 "CLUSTER MEMBERSHIP"

# Quick run without detailed logs
python3 scripts/compare_clustering_algorithms.py \
  --target_samples 50 \
  --logging False

# Only console output saved, no cluster details
ls outputs/run_LATEST/logs/
# Output: comparison_all_data_evaluation_20251210_134547.log
```

---

## 📚 Related Documentation

- **[DETAILED_LOGGING_GUIDE.md](DETAILED_LOGGING_GUIDE.md)** - Complete logging system guide
- **[LOGGING_QUICK_REFERENCE.md](LOGGING_QUICK_REFERENCE.md)** - Quick command reference
- **[README.md](../README.md)** - Main project documentation

---

## 🎉 Summary

### Feature 1: `--logging True/False`
- **Purpose**: Toggle detailed logging on/off
- **Default**: True (enabled)
- **Impact**: 15 logs vs 1 log
- **Use**: `--logging False` for quick runs, `--logging True` for analysis

### Feature 2: Cluster Elements & Representatives
- **Purpose**: Log all cluster members and picked samples
- **Availability**: Only when `--logging True`
- **Location**: `11_normal_sampling_*.log` and `12_smart_sampling_*.log`
- **Content**: All variant IDs, picked representatives, intent breakdown
- **Benefit**: Complete sampling traceability

**Both features work together to give you full control over logging detail level!** 🚀
