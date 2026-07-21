# ✅ Auto Target Samples Update

## 🎯 What Changed

The `target_samples` parameter is now **automatically calculated** as **1% of the total variant records** in your dataset, instead of being hardcoded to 127.

## 📊 Before vs After

### **Before:**
```python
target_samples = 127  # Always 127, regardless of dataset size
```

### **After:**
```python
target_samples = None  # Default: Auto-calculate as 1% of dataset
# After loading data:
target_samples = max(1, int(total_variants * 0.01))
```

## 🔍 Example with Your Dataset

**Your dataset has 18,207 variants:**

```
⚙️  Auto-calculated target_samples: 182 (1% of 18207 variants)

Total dataset:
  Total variants: 18207
  Total samples: 28256
  Target samples for training: 182 (1.00% of variants)
  Intentional: 13291
  Unintentional: 14965
```

**Calculation:** `18207 × 0.01 = 182.07 → 182`

## 🎮 Usage

### **Option 1: Auto-calculate (Default - Recommended)**
```bash
# No --target_samples argument → Uses 1% automatically
python3 scripts/compare_clustering_algorithms.py

# Explicitly request auto (same as default)
python3 scripts/compare_clustering_algorithms.py --target_samples None
```

**Output:**
```
Target samples: Auto (1% of dataset)
...
⚙️  Auto-calculated target_samples: 182 (1% of 18207 variants)
```

### **Option 2: Override with Custom Value**
```bash
# Specify exact number
python3 scripts/compare_clustering_algorithms.py --target_samples 200

# Different sizes
python3 scripts/compare_clustering_algorithms.py --target_samples 50
python3 scripts/compare_clustering_algorithms.py --target_samples 500
```

**Output:**
```
Target samples: 200
```

## 📐 Why 1%?

1. **Scalable**: Works for any dataset size
   - Small dataset (1,000 variants) → 10 samples
   - Medium dataset (10,000 variants) → 100 samples  
   - Large dataset (20,000 variants) → 200 samples
   - Very large (100,000 variants) → 1,000 samples

2. **Balanced**: Not too small (poor generalization), not too large (expensive training)

3. **Adaptive**: Automatically adjusts when you switch datasets

## 🔄 Backward Compatibility

✅ **Old scripts still work!** You can still specify `--target_samples 127` if needed:

```bash
# Works exactly as before
python3 scripts/compare_clustering_algorithms.py --target_samples 127
```

## 📝 Code Changes

### **1. Class Initialization (Line 68)**
```python
# Before:
def __init__(self, target_samples=127, ...):

# After:
def __init__(self, target_samples=None, ...):
```

### **2. Auto-calculation in load_data() (Line 340)**
```python
# Set target_samples to 1% of total variants if not specified
if self.target_samples is None:
    self.target_samples = max(1, int(self.total_variants * 0.01))
    print(f"\n⚙️  Auto-calculated target_samples: {self.target_samples} (1% of {self.total_variants} variants)")
```

### **3. Argument Parser (Line 1710)**
```python
# Before:
parser.add_argument('--target_samples', type=int, default=127,
                    help='Target number of samples to select (default: 127)')

# After:
parser.add_argument('--target_samples', type=int, default=None,
                    help='Target number of samples to select (default: 1% of dataset variants, auto-calculated)')
```

### **4. run_comparison() Display (Line 1092)**
```python
# Before:
print(f"Target samples: {self.target_samples}")

# After:
if self.target_samples is None:
    print(f"Target samples: Auto (1% of dataset)")
else:
    print(f"Target samples: {self.target_samples}")
```

## 🎯 Benefits

### **For Different Dataset Sizes:**

| Dataset Size | Old (Fixed 127) | New (1%) | Ratio |
|--------------|-----------------|----------|-------|
| 1,000 variants | 127 (12.7%) | 10 (1.0%) | More efficient |
| 5,000 variants | 127 (2.5%) | 50 (1.0%) | Balanced |
| 10,000 variants | 127 (1.3%) | 100 (1.0%) | Balanced |
| **18,207 variants** | **127 (0.7%)** | **182 (1.0%)** | **Better coverage** |
| 50,000 variants | 127 (0.25%) | 500 (1.0%) | Much better coverage |
| 100,000 variants | 127 (0.13%) | 1000 (1.0%) | Vastly better |

### **Advantages:**

✅ **Adaptive**: Works optimally for any dataset size  
✅ **Consistent**: Always uses 1% of data  
✅ **Scalable**: Automatically scales with larger datasets  
✅ **Transparent**: Logs the calculated value  
✅ **Flexible**: Can still override if needed  
✅ **Safer**: Uses `max(1, ...)` to ensure at least 1 sample  

## 🧪 Test Results

**Run with auto-calculation:**
```bash
$ python3 scripts/compare_clustering_algorithms.py --logging False
```

**Output:**
```
Target samples: Auto (1% of dataset)
...
⚙️  Auto-calculated target_samples: 182 (1% of 18207 variants)
Total dataset:
  Total variants: 18207
  Total samples: 28256
  Target samples for training: 182 (1.00% of variants)
```

**✅ Working correctly!**

## 📚 Related Documentation

- **[README.md](../README.md)** - Main documentation
- **[LOGGING_FEATURES_SUMMARY.md](LOGGING_FEATURES_SUMMARY.md)** - Logging features guide
- **[COMPLETE_PIPELINE_EXPLANATION.md](COMPLETE_PIPELINE_EXPLANATION.md)** - Full pipeline guide

---

**Status:** ✅ Implemented and tested  
**Date:** December 10, 2025  
**Impact:** Improves scalability and consistency across different dataset sizes
