# Rank Selection Implementations: v1 vs v2 (MATLAB-Matched)

This guide explains two implementations of automatic rank selection for CSMF:
1. **v1**: Simplified Python implementation
2. **v2**: Exact translation of MATLAB authors' code

---

## Quick Start with v2 (Recommended - Matches MATLAB)

```python
import numpy as np
from csmf import rank_selection_pipeline_v2, csmf

# Prepare datasets (keep separate, not concatenated)
X1 = np.random.rand(100, 50)   # Dataset 1
X2 = np.random.rand(100, 60)   # Dataset 2
X3 = np.random.rand(100, 40)   # Dataset 3

# Automatic rank selection (MATLAB-matched)
vec_para, analysis = rank_selection_pipeline_v2(
    [X1, X2, X3],              # Pass as list, not concatenated
    min_rank=2,
    max_rank=12,
    n_repeats=30,              # Higher = more stable
    correlations_cutoff=0.7,   # Threshold for common factors
    verbose=1
)
# Output: [r_common, r_specific_1, r_specific_2, r_specific_3]

# Run CSMF with detected ranks
X_concat = np.hstack([X1, X2, X3])
W, H, _, _, _ = csmf(
    X_concat,
    vec_n=[50, 60, 40],
    vec_para=vec_para,
    verbose=1
)
```

---

## Implementation Comparison

### v1 (Simple Python)
```
Data (concatenated) 
    ↓
NeNMF on concatenated matrix with varying ranks
    ↓
Correlation-based stability for each rank
    ↓
Find elbow point
    ↓
Learn common/specific via correlation matching
```

**Pros:**
- Simple, fast
- Works with concatenated data

**Cons:**
- Not exactly matching MATLAB
- May miss true structure from individual datasets

### v2 (MATLAB-Matched) ✅ **RECOMMENDED**
```
Datasets (kept separate)
    ↓
NeNMF independently on each dataset with varying ranks
    ↓
Amari distance-based stability for each rank per dataset
    ↓
Find local minima/maxima in stability curve
    ↓
Select optimal rank per dataset
    ↓
Learn common/specific via correlation matching
```

**Pros:**
- Exact MATLAB implementation
- Better identifies true structure from each dataset
- More robust peak detection

**Cons:**
- Slightly slower
- More parameters to tune

---

## Detailed v2 Pipeline

### Step 1: Independent NeNMF on Each Dataset

```python
from csmf import nenmf_rank_sweep_per_dataset

# Run NeNMF independently on each dataset
nenmf_results = nenmf_rank_sweep_per_dataset(
    [X1, X2, X3],              # Datasets kept separate
    min_rank=2,
    max_rank=12,
    n_repeats=30,              # Per rank repetitions
    max_iter=200
)
# nenmf_results[dataset_idx][rank] = List of W matrices from 30 runs
```

**Why separate?** MATLAB implementation runs NMF independently on each dataset's columns to capture dataset-specific structure.

### Step 2: Stability via Amari Distance

```python
from csmf import analyze_stability_per_dataset

stability_scores = analyze_stability_per_dataset(
    nenmf_results,
    min_rank=2,
    max_rank=12,
    verbose=1
)
# stability_scores[dataset_idx][rank] = Amari-distance-based stability
```

**Amari Distance:**
- Measures factor correspondence via correlation matrix
- Unlike correlation, symmetric in matching quality
- Lower Amari = better correspondence between runs

### Step 3: Peak Detection for Optimal Ranks

```python
from csmf import find_optimal_ranks

optimal_ranks = find_optimal_ranks(
    stability_scores,
    min_rank=2,
    max_rank=12,
    verbose=1
)
# optimal_ranks[dataset_idx] = Detected optimal rank via peak detection
```

**Key difference from v1:** Uses `find_peaks()` to locate local minima/maxima rather than just finding an elbow.

### Step 4: Select Best Factorizations

```python
W_matrices = []
for dataset_idx, optimal_rank in optimal_ranks.items():
    W_list = nenmf_results[dataset_idx][optimal_rank]
    W_best, _ = select_best_factorization(W_list)
    
    # Normalize
    W_best_norm = W_best / np.linalg.norm(W_best, axis=0, keepdims=True)
    W_matrices.append(W_best_norm)
```

### Step 5: Learn Common/Specific Ranks

```python
from csmf import learn_common_specific_ranks_from_correlations

vec_para = learn_common_specific_ranks_from_correlations(
    W_matrices,
    correlations_cutoff=0.7,
    verbose=1
)
# Returns [r_common, r_specific_1, r_specific_2, r_specific_3]
```

**Algorithm:**
1. Find highly correlated factors (>cutoff) across datasets
2. Use Hungarian algorithm for optimal matching
3. Iteratively verify common rank with all datasets
4. Return final [r_common, r_specific_1, ..., r_specific_K]

---

## Parameter Guide

### `rank_selection_pipeline_v2()` Parameters

| Parameter | TypeDefault | Range | Notes |
|-----------|-------------|-------|-------|
| `X_datasets` | List[ndarray] | - | Separate datasets, NOT concatenated |
| `min_rank` | int | 3 | 2-5 typical for small data |
| `max_rank` | int | 20 | 2-3× expected common rank |
| `n_repeats` | int | 30 | 20-50 recommended (stability) |
| `correlations_cutoff` | float | 0.7 | 0.5-0.9 typical |
| `verbose` | int | 1 | 0=silent, 1=summary, 2=detailed |

### Choosing Parameters

**n_repeats** - Higher = more stable but slower
- 10-20: Fast, acceptable
- 30-50: Good balance (recommended)
- 50+: Very stable but slow

**correlations_cutoff** - Controls common/specific separation
- 0.5: Lenient (many factors = common)
- 0.7: Moderate (recommended)
- 0.9: Strict (few factors = common)

**min_rank/max_rank** - Range to search
- Too narrow: May miss true rank
- Too wide: Slower, more noise
- Good start: min=2, max=2×(expected common rank)

---

## v1 vs v2 Usage

### When to use v1 (Simple):
- Quick exploration
- Small datasets
- Already concatenated data
- Don't need MATLAB matching

### When to use v2 (MATLAB-Matched):
```python
# v2 - MATLAB matched (RECOMMENDED)
vec_para, analysis = rank_selection_pipeline_v2([X1, X2, X3], ...)

# v1 - Simple (for comparison)
X_concat = np.hstack([X1, X2, X3])
vec_para, _ = rank_selection_pipeline(X_concat, [n1, n2, n3], ...)
```

---

## Complete Example: v2 Pipeline

```python
import numpy as np
import matplotlib.pyplot as plt
from csmf import rank_selection_pipeline_v2, csmf

# 1. Generate data
np.random.seed(42)
X1 = np.abs(np.random.randn(100, 50)) + 0.5
X2 = np.abs(np.random.randn(100, 60)) + 0.5
X3 = np.abs(np.random.randn(100, 40)) + 0.5

X_datasets = [X1, X2, X3]
X_concat = np.hstack(X_datasets)
vec_n = [50, 60, 40]

# 2. Rank selection (v2)
print("Running rank selection (MATLAB v2)...")
vec_para, analysis = rank_selection_pipeline_v2(
    X_datasets,
    min_rank=2,
    max_rank=12,
    n_repeats=30,
    correlations_cutoff=0.7,
    verbose=1
)

# 3. Plot stability for each dataset
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ds_idx, ax in enumerate(axes):
    scores = analysis['stability_scores'][ds_idx]
    ranks = sorted(scores.keys())
    stabilities = [scores[r] for r in ranks]
    
    ax.plot(ranks, stabilities, 'b-o')
    optimal = analysis['optimal_ranks'][ds_idx]
    ax.axvline(optimal, color='r', linestyle='--', alpha=0.7)
    ax.set_title(f'Dataset {ds_idx + 1}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Stability')

plt.tight_layout()
plt.savefig('stability_analysis.png')
plt.show()

# 4. Run CSMF
print(f"\nRunning CSMF with detected ranks: {vec_para}")
W, H, _, _, _ = csmf(
    X_concat,
    vec_n=vec_n,
    vec_para=vec_para,
    iter_outer=200,
    verbose=1
)

# 5. Analyze results
r_c = vec_para[0]
W_common = W[:, :r_c]
print(f"\nResults:")
print(f"  Common rank: {r_c}")
print(f"  Specific ranks: {vec_para[1:]}")
print(f"  W_common shape: {W_common.shape}")
```

---

## Troubleshooting

**Q: Detected ranks don't look right?**
- A: Try different `min_rank`/`max_rank` range
- A: Increase `n_repeats` (20→50)
- A: Adjust `correlations_cutoff` (try 0.6 or 0.8)

**Q: Too slow?**
- A: Decrease `n_repeats` (30→10)
- A: Decrease `max_rank`
- A: Use v1 instead of v2

**Q: Always detecting 0 common factors?**
- A: Try stricter cutoff (0.7→0.9)
- A: Data truly has no common patterns

**Q: All factors marked as common?**
- A: Try lenient cutoff (0.7→0.5)
- A: Increase `n_repeats` for better stability

---

## Mathematical Details: Amari Distance

The Amari distance measures factor correspondence:

$$d(A) = \frac{1}{2K}\left(\sum_j \left(1 - \max_i |A_{ij}|\right) + \sum_i \left(1 - \max_j |A_{ij}|\right)\right)$$

where A is the absolute correlation matrix between two factor sets.

- **d = 0**: Perfect correspondence (matrices represent same factors)
- **d = 1**: No correspondence (factors scrambled)

See MATLAB `staNMF_rank.m` for reference implementation.

---

## References

- Original MATLAB code: `staNMF_rank.m`, `learning_common_specific_ranks.m`
- Zhang et al. (2019) - Section "Determining the ranks"
- Amari distance: From blind source separation literature
