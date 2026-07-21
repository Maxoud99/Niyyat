# DBSCAN eps=0.0 Fix

## Problem

DBSCAN was failing with this error:
```
The 'eps' parameter of DBSCAN must be a float in the range (0.0, inf). Got np.float64(0.0) instead.
```

## Root Cause

The auto-tuning logic calculated `eps` as the 90th percentile of k-nearest neighbor distances. When the feature data has very low variance or many identical/near-identical points after standardization, all distances become 0.0, making eps = 0.0, which is invalid for DBSCAN.

## Solution

Added a safety check with fallback logic:

```python
# Auto-tune eps if not provided
if eps is None:
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(features)
    distances, indices = neighbors_fit.kneighbors(features)
    distances = np.sort(distances[:, -1])
    eps = np.percentile(distances, 90)  # Use 90th percentile
    
    # Safety check: eps must be > 0
    if eps <= 0 or np.isnan(eps):
        # Fallback: use mean distance or a small default
        mean_dist = np.mean(distances[distances > 0]) if np.any(distances > 0) else 0.5
        eps = max(mean_dist, 0.5)  # Ensure at least 0.5
        print(f"  ⚠️  Auto-tuned eps was {np.percentile(distances, 90):.6f}, using fallback: {eps:.4f}")
    else:
        print(f"  Auto-tuned eps: {eps:.4f}")
```

## Fallback Strategy

1. **First attempt:** Use mean of non-zero distances
2. **Fallback:** Use minimum value of 0.5 if all distances are zero
3. **Safety:** Ensures `eps >= 0.5` in all cases

## When This Occurs

This typically happens when:
- Feature data has very low variance after standardization
- Many data points are identical or near-identical
- Dataset has high redundancy in the feature space
- Aggressive feature scaling produces uniform values

## Impact

- ✅ DBSCAN no longer crashes with eps=0.0
- ✅ Uses reasonable fallback (0.5) that allows clustering
- ✅ Warns user when fallback is used
- ⚠️  May produce different clustering results than optimal eps
- ℹ️  Users can override with manual `--eps` parameter if needed

## Testing

```python
# Test with zero distances (identical points)
features = np.zeros((100, 10))
# Result: eps = 0.0 → fallback to 0.5 ✓

# Test with small non-zero distances
features = np.random.randn(100, 10) * 0.001
# Result: eps calculated from 90th percentile ✓
```

## Date Fixed

December 10, 2025
