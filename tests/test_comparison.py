#!/usr/bin/env python
"""
Comparison Test - Runs GPU and CPU CSMF and creates side-by-side factor comparison plots.
Shows GPU vs CPU for each true factor on the same basis for fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from csmf.gpu.gpu_csmf import gpu_csmf
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

def compute_correlation(W_true, W_inferred):
    """Compute correlation using optimal Hungarian alignment"""
    r = min(W_true.shape[1], W_inferred.shape[1])
    
    # Normalize for correlation
    W_true_n = W_true[:, :r] / (np.linalg.norm(W_true[:, :r], axis=0, keepdims=True) + 1e-10)
    W_inf_n = W_inferred[:, :r] / (np.linalg.norm(W_inferred[:, :r], axis=0, keepdims=True) + 1e-10)
    
    # Optimal assignment
    corr_mat = np.abs(W_true_n.T @ W_inf_n)
    _, col_idx = linear_sum_assignment(-corr_mat)
    
    return np.mean([corr_mat[i, col_idx[i]] for i in range(r)])

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
        
        # Scale based on norm to match true factor magnitude
        scale = np.linalg.norm(W_true[:, i]) / (np.linalg.norm(W_aligned[:, i]) + 1e-10)
        W_aligned[:, i] *= scale
    
    return W_true[:, :r], W_aligned, col_idx

def create_gpu_vs_cpu_scatter(W_true, W_gpu, W_cpu, title, filename):
    """Create side-by-side scatter plots comparing GPU vs CPU for each true factor"""
    r = W_true.shape[1]
    
    # Step 1: Align GPU to ground truth
    W_true_aligned_gpu, W_gpu_scaled, col_idx_gpu = align_and_scale_factors(W_true, W_gpu)
    
    # Step 2: Reorder CPU using GPU's permutation (consistency!)
    # This ensures both GPU and CPU show same true factor in each row
    W_gpu_perm = W_gpu[:, col_idx_gpu].copy()
    m, r_w = W_cpu.shape
    W_cpu_work = W_cpu[:, :r]
    
    # Align CPU to GPU (not to ground truth) to ensure same factor order
    W_gpu_n = W_gpu_perm / (np.linalg.norm(W_gpu_perm, axis=0, keepdims=True) + 1e-10)
    W_cpu_n = W_cpu_work / (np.linalg.norm(W_cpu_work, axis=0, keepdims=True) + 1e-10)
    corr_mat_gc = np.abs(W_gpu_n.T @ W_cpu_n)
    _, col_idx_cpu_to_gpu = linear_sum_assignment(-corr_mat_gc)
    
    # Reorder CPU to match GPU's factor order
    W_cpu_reordered = W_cpu_work[:, col_idx_cpu_to_gpu].copy()
    
    # Now scale both to ground truth for fair visual comparison
    for i in range(r):
        # Scale GPU
        scale_gpu = np.linalg.norm(W_true_aligned_gpu[:, i]) / (np.linalg.norm(W_gpu_scaled[:, i]) + 1e-10)
        W_gpu_scaled[:, i] *= scale_gpu
        
        # Scale CPU
        scale_cpu = np.linalg.norm(W_true_aligned_gpu[:, i]) / (np.linalg.norm(W_cpu_reordered[:, i]) + 1e-10)
        W_cpu_reordered[:, i] *= scale_cpu
    
    # Create side-by-side plots (GPU on left, CPU on right)
    fig, axes = plt.subplots(r, 2, figsize=(14, 5*r))
    if r == 1:
        axes = axes.reshape(1, 2)
    
    fig.suptitle(f'{title} - GPU vs CPU Factor Comparison\n(Same True Factor, Different Implementations)', 
                 fontsize=14, fontweight='bold')
    
    for i in range(r):
        # GPU subplot (left)
        ax_gpu = axes[i, 0]
        ax_gpu.scatter(W_true_aligned_gpu[:, i], W_gpu_scaled[:, i], alpha=0.6, s=50, 
                       edgecolors='blue', linewidth=0.5, label='GPU')
        min_val = min(W_true_aligned_gpu[:, i].min(), W_gpu_scaled[:, i].min())
        max_val = max(W_true_aligned_gpu[:, i].max(), W_gpu_scaled[:, i].max())
        ax_gpu.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.4, label='Perfect')
        
        corr_gpu = np.corrcoef(W_true_aligned_gpu[:, i], W_gpu_scaled[:, i])[0, 1]
        rmse_gpu = np.sqrt(np.mean((W_true_aligned_gpu[:, i] - W_gpu_scaled[:, i])**2))
        
        ax_gpu.set_xlabel('True Factor', fontsize=11, fontweight='bold')
        ax_gpu.set_ylabel('GPU Factor', fontsize=11, fontweight='bold')
        ax_gpu.set_title(f'Factor {i} (Ground Truth: {col_idx_gpu[i]}) - GPU\nr={corr_gpu:.4f}, RMSE={rmse_gpu:.4f}', 
                        fontsize=11, color='blue', fontweight='bold')
        ax_gpu.grid(True, alpha=0.3)
        ax_gpu.legend(loc='upper left', fontsize=9)
        
        # CPU subplot (right)
        ax_cpu = axes[i, 1]
        ax_cpu.scatter(W_true_aligned_gpu[:, i], W_cpu_reordered[:, i], alpha=0.6, s=50, 
                       edgecolors='red', linewidth=0.5, label='CPU')
        min_val = min(W_true_aligned_gpu[:, i].min(), W_cpu_reordered[:, i].min())
        max_val = max(W_true_aligned_gpu[:, i].max(), W_cpu_reordered[:, i].max())
        ax_cpu.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.4, label='Perfect')
        
        corr_cpu = np.corrcoef(W_true_aligned_gpu[:, i], W_cpu_reordered[:, i])[0, 1]
        rmse_cpu = np.sqrt(np.mean((W_true_aligned_gpu[:, i] - W_cpu_reordered[:, i])**2))
        
        ax_cpu.set_xlabel('True Factor', fontsize=11, fontweight='bold')
        ax_cpu.set_ylabel('CPU Factor', fontsize=11, fontweight='bold')
        ax_cpu.set_title(f'Factor {i} (Ground Truth: {col_idx_gpu[i]}) - CPU\nr={corr_cpu:.4f}, RMSE={rmse_cpu:.4f}', 
                        fontsize=11, color='red', fontweight='bold')
        ax_cpu.grid(True, alpha=0.3)
        ax_cpu.legend(loc='upper left', fontsize=9)
        
        # Print comparison stats
        diff = abs(corr_gpu - corr_cpu)
        better = "GPU" if corr_gpu > corr_cpu else "CPU" if corr_cpu > corr_gpu else "TIED"
        print(f"  Factor {i}: GPU r={corr_gpu:.4f} vs CPU r={corr_cpu:.4f} (diff={diff:.4f}, {better} better)")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}\n")
    plt.close()

print("\n" + "="*80)
print("GPU vs CPU CSMF COMPARISON TEST")
print("="*80)

# Generate data
print("\n[1] Generating test data...")
X_list, (W_c_true, W_s_true, H_c_true, H_s_true) = generate_test_data()
print(f"  ✓ {len(X_list)} datasets, {X_list[0].shape[0]} features")
print(f"  ✓ Samples: {[X.shape[1] for X in X_list]}")

# Run GPU CSMF
print("\n[2] Running GPU CSMF...")
start = time.time()
result_gpu = gpu_csmf(
    X_list,
    rank_common=3,
    rank_specific=2,
    n_iter_outer=100,
    n_iter_inner=300,  # Match CPU's max_iter_nenm=300
    verbose=0
)
gpu_time = time.time() - start
print(f"  ✓ Completed in {gpu_time:.2f}s")

W_c_gpu = result_gpu['W_c']
W_s_gpu = result_gpu['W_s']
H_c_gpu = result_gpu['H_c']
H_s_gpu = result_gpu['H_s']

# Run CPU CSMF
print("\n[3] Running CPU CSMF...")
X_concat = np.hstack(X_list)
vec_n = [X.shape[1] for X in X_list]
vec_para = [3, 2, 2, 2]  # 3 common, 2 specific for each of 3 datasets

start = time.time()
W_concat, H_concat, n_iter, elapsed, history = csmf(
    X_concat, vec_n, vec_para, 
    iter_outer=100, max_iter_nenm=300, verbose=0
)
cpu_time = time.time() - start
print(f"  ✓ Completed in {cpu_time:.2f}s")

# Extract factors from concatenated result
W_c_cpu = W_concat[:, :3]
W_s_cpu = [W_concat[:, 3:5], W_concat[:, 5:7], W_concat[:, 7:9]]

sum_n = [sum(vec_n[:i+1]) for i in range(len(vec_n))]
H_c_cpu = [H_concat[:3, sum_n[k-1] if k > 0 else 0:sum_n[k]] for k in range(len(X_list))]
H_s_cpu = []
row_start = 3
for k in range(len(X_list)):
    H_s_cpu.append(H_concat[row_start:row_start+2, sum_n[k-1] if k > 0 else 0:sum_n[k]])
    row_start += 2

# Evaluate and compare
print("\n[4] Evaluating and comparing factors...")
print("\n  Common Factors (W_c):")
corr_wc_gpu = compute_correlation(W_c_true, W_c_gpu)
corr_wc_cpu = compute_correlation(W_c_true, W_c_cpu)
print(f"    GPU: {corr_wc_gpu:.4f}")
print(f"    CPU: {corr_wc_cpu:.4f}")
print(f"    Diff: {abs(corr_wc_gpu - corr_wc_cpu):+.4f}")

print("\n  Specific Factors (W_s):")
for k in range(len(X_list)):
    corr_ws_gpu = compute_correlation(W_s_true[k], W_s_gpu[k])
    corr_ws_cpu = compute_correlation(W_s_true[k], W_s_cpu[k])
    print(f"    Dataset {k}: GPU={corr_ws_gpu:.4f}, CPU={corr_ws_cpu:.4f}, Diff={abs(corr_ws_gpu - corr_ws_cpu):+.4f}")

# Create scatter plot visualizations
print("\n[5] Creating GPU vs CPU comparison scatter plots...")
create_gpu_vs_cpu_scatter(W_c_true, W_c_gpu, W_c_cpu, 'W_c Common Factors', 
                         'outputs/test_comparison_wc_scatter.png')

for k in range(len(X_list)):
    create_gpu_vs_cpu_scatter(W_s_true[k], W_s_gpu[k], W_s_cpu[k], 
                             f'W_s[{k}] Specific Factors',
                             f'outputs/test_comparison_ws{k}_scatter.png')

print("\n[6] Reconstruction errors:")
print("  GPU:")
for k in range(len(X_list)):
    recon = W_c_gpu @ H_c_gpu[k] + W_s_gpu[k] @ H_s_gpu[k]
    err = np.linalg.norm(X_list[k] - recon) / np.linalg.norm(X_list[k])
    accuracy = 100 * (1 - err)
    print(f"    Dataset {k}: {err:.4f} ({accuracy:.1f}%)")

print("  CPU:")
for k in range(len(X_list)):
    recon = W_c_cpu @ H_c_cpu[k] + W_s_cpu[k] @ H_s_cpu[k]
    err = np.linalg.norm(X_list[k] - recon) / np.linalg.norm(X_list[k])
    accuracy = 100 * (1 - err)
    print(f"    Dataset {k}: {err:.4f} ({accuracy:.1f}%)")

print("\n[7] Performance comparison:")
print(f"  GPU speed: {gpu_time:.2f}s")
print(f"  CPU speed: {cpu_time:.2f}s")
print(f"  GPU speedup: {cpu_time/gpu_time:.2f}x faster" if gpu_time < cpu_time 
      else f"  CPU speedup: {gpu_time/cpu_time:.2f}x faster")

print("\n" + "="*80)
print("✓ COMPARISON COMPLETE - Check PNG files for visual factor alignment")
print("="*80 + "\n")
