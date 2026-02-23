#!/usr/bin/env python
"""
CPU CSMF Test - Validates CPU implementation correctness and performance.
This test generates synthetic multi-dataset data and runs CSMF factorization.
"""

import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from csmf.csmf import csmf

def generate_test_data(n_datasets=3, n_features=100, n_samples=[50, 60, 40],
                       rank_common=3, rank_specific=[2, 2, 2], noise=0.1, seed=42):
    """Generate synthetic CSMF test data"""
    np.random.seed(seed)
    K = len(n_datasets) if isinstance(n_datasets, list) else len(n_samples)
    
    # True factors
    W_c_true = np.abs(np.random.randn(n_features, rank_common))
    W_s_true = [np.abs(np.random.randn(n_features, rank_specific[k])) for k in range(K)]
    H_c_true = [np.abs(np.random.randn(rank_common, n_samples[k])) * 0.8 + 0.5 for k in range(K)]
    H_s_true = [np.abs(np.random.randn(rank_specific[k], n_samples[k])) * 0.5 + 0.1 for k in range(K)]
    
    # Generate data
    X_list = []
    for k in range(K):
        X_clean = W_c_true @ H_c_true[k] + W_s_true[k] @ H_s_true[k]
        noise_mat = noise * np.random.randn(n_features, n_samples[k]) * np.std(X_clean)
        X = np.maximum(X_clean + noise_mat, 0)
        X_list.append(X)
    
    return X_list, (W_c_true, W_s_true, H_c_true, H_s_true)

def align_and_scale_factors(W_true, W_inferred):
    """Align inferred factors to true factors and scale to match magnitude"""
    r_true, r_inf = W_true.shape[1], W_inferred.shape[1]
    r = min(r_true, r_inf)
    
    # Normalize for correlation
    W_true_n = W_true[:, :r] / (np.linalg.norm(W_true[:, :r], axis=0, keepdims=True) + 1e-10)
    W_inf_n = W_inferred[:, :r] / (np.linalg.norm(W_inferred[:, :r], axis=0, keepdims=True) + 1e-10)
    
    # Optimal assignment
    corr_mat = np.abs(W_true_n.T @ W_inf_n)
    _, col_idx = linear_sum_assignment(-corr_mat)
    
    # Align factors
    W_aligned = W_inferred[:, col_idx].copy()
    
    # Check for sign flips and scale each factor individually
    for i in range(r):
        # NMF has arbitrary sign ambiguity - check if factor needs flipping
        signed_corr = (W_true_n[:, i] * W_inf_n[:, col_idx[i]]).sum()
        if signed_corr < 0:  # Negative correlation means factor is flipped
            W_aligned[:, i] *= -1
        
        # Scale based on norm
        scale = np.linalg.norm(W_true[:, i]) / (np.linalg.norm(W_aligned[:, i]) + 1e-10)
        W_aligned[:, i] *= scale
    
    return W_true[:, :r], W_aligned, col_idx

def compute_correlation(W_true, W_inferred):
    """Compute mean absolute correlation after optimal alignment"""
    r_true, r_inf = W_true.shape[1], W_inferred.shape[1]
    r = min(r_true, r_inf)
    
    # Normalize
    W_true_n = W_true[:, :r] / (np.linalg.norm(W_true[:, :r], axis=0, keepdims=True) + 1e-10)
    W_inf_n = W_inferred[:, :r] / (np.linalg.norm(W_inferred[:, :r], axis=0, keepdims=True) + 1e-10)
    
    # Assignment
    corr_mat = np.abs(W_true_n.T @ W_inf_n)
    _, col_idx = linear_sum_assignment(-corr_mat)
    
    return np.mean([corr_mat[i, col_idx[i]] for i in range(r)])

def create_scatter_plots(W_true, W_inferred, title, filename):
    """Create scatter plots comparing true vs inferred factors with optimal alignment"""
    r = W_true.shape[1]
    cols = (r + 1) // 2
    
    fig, axes = plt.subplots(2, cols, figsize=(12, 8))
    if r == 1:
        axes = axes.reshape(2, 1)
    axes = axes.flatten()
    
    fig.suptitle(f'{title} Factor Validation', fontsize=14, fontweight='bold')
    
    # Find optimal alignment using Hungarian algorithm
    W_true_n = W_true / (np.linalg.norm(W_true, axis=0, keepdims=True) + 1e-10)
    W_inf_n = W_inferred / (np.linalg.norm(W_inferred, axis=0, keepdims=True) + 1e-10)
    
    corr_mat = np.abs(W_true_n.T @ W_inf_n)
    _, col_idx = linear_sum_assignment(-corr_mat)
    
    # Plot aligned factors
    for i in range(r):
        ax = axes[i]
        j = col_idx[i]  # Optimal index for factor i
        
        ax.scatter(W_true[:, i], W_inferred[:, j], alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
        
        # Diagonal line for perfect match
        min_val = min(W_true[:, i].min(), W_inferred[:, j].min())
        max_val = max(W_true[:, i].max(), W_inferred[:, j].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect')
        
        # Metrics
        corr = np.corrcoef(W_true[:, i], W_inferred[:, j])[0, 1]
        rmse = np.sqrt(np.mean((W_true[:, i] - W_inferred[:, j])**2))
        
        ax.set_xlabel('True Factor', fontsize=9)
        ax.set_ylabel('Inferred Factor', fontsize=9)
        ax.set_title(f'Factor {i+1}\nr={corr:.3f}, RMSE={rmse:.3f}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused
    for i in range(r, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.close()

print("\n" + "="*70)
print("TEST: CPU CSMF Implementation")
print("="*70)

# Generate data
print("\n[1] Generating test data...")
X_list, (W_c_true, W_s_true, H_c_true, H_s_true) = generate_test_data()
print(f"  ✓ {len(X_list)} datasets, {X_list[0].shape[0]} features")
print(f"  ✓ Samples: {[X.shape[1] for X in X_list]}")

# Run CSMF
print("\n[2] Running CPU CSMF (100 iterations)...")
X_concat = np.hstack(X_list)
vec_n = [X.shape[1] for X in X_list]
vec_para = [3, 2, 2, 2]

start = time.time()
W, H, n_iter, elapsed, history = csmf(
    X_concat, vec_n, vec_para, 
    iter_outer=100, max_iter_nenm=300, verbose=0
)
elapsed = time.time() - start

# Extract factors
W_c = W[:, :3]
W_s = [W[:, 3:5], W[:, 5:7], W[:, 7:9]]

sum_n = [sum(vec_n[:i+1]) for i in range(len(vec_n))]
H_c = [H[:3, sum_n[k-1] if k > 0 else 0:sum_n[k]] for k in range(len(X_list))]
H_s = []
row_start = 3
for k in range(len(X_list)):
    H_s.append(H[row_start:row_start+2, sum_n[k-1] if k > 0 else 0:sum_n[k]])
    row_start += 2

print(f"  ✓ Time: {elapsed:.2f}s")
print(f"  ✓ NeNMF iterations: {history['niter']}")

# Evaluate
print("\n[3] Evaluating factor quality...")

# Align factors for comparison
W_c_true_aligned, W_c_aligned, _ = align_and_scale_factors(W_c_true, W_c)
corr_wc = compute_correlation(W_c_true, W_c)
print(f"  W_c correlation: {corr_wc:.4f}")

print("  W_s correlations:")
W_s_aligned = []
for k in range(len(X_list)):
    W_s_true_aligned, W_s_k_aligned, _ = align_and_scale_factors(W_s_true[k], W_s[k])
    W_s_aligned.append(W_s_k_aligned)
    corr = compute_correlation(W_s_true[k], W_s[k])
    print(f"    Dataset {k}: {corr:.4f}")

print("\n[4] Reconstruction errors...")
recon_errors = []
for k in range(len(X_list)):
    recon = W_c @ H_c[k] + W_s[k] @ H_s[k]
    err = np.linalg.norm(X_list[k] - recon) / np.linalg.norm(X_list[k])
    recon_errors.append(err)
    accuracy = 100 * (1 - err)
    print(f"  Dataset {k}: {err:.4f} ({accuracy:.1f}%)")

print("\n[5] Creating scatter plot visualizations...")
create_scatter_plots(W_c_true_aligned, W_c_aligned, 'W_c Common', 'outputs/test_cpu_wc_scatter.png')
for k in range(len(X_list)):
    W_s_true_k_aligned, W_s_k_aligned, _ = align_and_scale_factors(W_s_true[k], W_s[k])
    create_scatter_plots(W_s_true_k_aligned, W_s_k_aligned, f'W_s[{k}] Specific', f'outputs/test_cpu_ws{k}_scatter.png')

print("\n" + "="*70)
if corr_wc > 0.97 and all(e < 0.05 for e in recon_errors):
    print("✓ TEST PASSED - CPU CSMF working correctly")
else:
    print("⚠ TEST ISSUES:")
    if corr_wc <= 0.97:
        print(f"  - W_c correlation {corr_wc:.4f} below threshold")
    if any(e >= 0.05 for e in recon_errors):
        print(f"  - High reconstruction errors")
print("="*70 + "\n")
