#!/usr/bin/env python
"""
Compare CPU vs GPU algorithms on factor recovery

Shows which version performs better on synthetic data
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from csmf import csmf as csmf_cpu
from csmf.gpu import gpu_csmf

print("Comparing CPU vs GPU implementations on factor recovery\n")

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
m, n1, n2 = 100, 50, 80

W_common_true = np.random.rand(m, 4)
H_common_1_true = np.random.rand(4, n1)
H_common_2_true = np.random.rand(4, n2)

W_specific_1_true = np.random.rand(m, 2)
H_specific_1_true = np.random.rand(2, n1)

W_specific_2_true = np.random.rand(m, 3)
H_specific_2_true = np.random.rand(3, n2)

# Generate data
X1_clean = W_common_true @ H_common_1_true + W_specific_1_true @ H_specific_1_true
X2_clean = W_common_true @ H_common_2_true + W_specific_2_true @ H_specific_2_true

# Add noise
noise_level = 0.1
X1 = X1_clean + noise_level * np.random.randn(m, n1) * np.std(X1_clean)
X2 = X2_clean + noise_level * np.random.randn(m, n2) * np.std(X2_clean)

X1 = np.maximum(X1, 0)
X2 = np.maximum(X2, 0)

X = np.hstack([X1, X2])
vec_n = [n1, n2]
vec_para = [4, 2, 3]

print(f"Data shape: {X.shape}")
print(f"True ranks: common=4, specific_1=2, specific_2=3\n")

# ============================================================================
# CPU Version
# ============================================================================
print("=" * 70)
print("CPU CSMF")
print("=" * 70)
try:
    W_cpu, H_cpu, n_iter_cpu, elapsed_cpu, _ = csmf_cpu(
        X, vec_n=vec_n, vec_para=vec_para,
        iter_outer=100, max_iter_nenm=100, verbose=0
    )
    err_cpu = np.linalg.norm(X - W_cpu @ H_cpu, 'fro')
    print(f"Reconstruction error: {err_cpu:.6f}")
    print(f"Time: {elapsed_cpu:.3f}s")
    print(f"W shape (common|specific): {W_cpu.shape}")
    
    # Get common factors
    W_cpu_common = W_cpu[:, :4]
    
    # Compute correlation with true common factors
    correlations_cpu = []
    for i in range(4):
        corr = np.corrcoef(W_cpu_common[:, i], W_common_true[:, i])[0, 1]
        correlations_cpu.append(corr)
    
    print(f"Common factor correlations: {[f'{c:.4f}' for c in correlations_cpu]}")
    print(f"Average correlation: {np.mean(np.abs(correlations_cpu)):.4f}\n")
except Exception as e:
    print(f"Error: {e}\n")

# ============================================================================
# GPU Version
# ============================================================================
print("=" * 70)
print("GPU CSMF")
print("=" * 70)
try:
    W_gpu, H_gpu, n_iter_gpu, elapsed_gpu, _ = gpu_csmf(
        X, vec_n=vec_n, vec_para=vec_para,
        iter_outer=100, max_iter_nenm=100, verbose=0, device='cuda'
    )
    err_gpu = np.linalg.norm(X - W_gpu @ H_gpu, 'fro')
    print(f"Reconstruction error: {err_gpu:.6f}")
    print(f"Time: {elapsed_gpu:.3f}s")
    print(f"W shape (common|specific): {W_gpu.shape}")
    
    # Get common factors
    W_gpu_common = W_gpu[:, :4]
    
    # Compute correlation with true common factors
    correlations_gpu = []
    for i in range(4):
        corr = np.corrcoef(W_gpu_common[:, i], W_common_true[:, i])[0, 1]
        correlations_gpu.append(corr)
    
    print(f"Common factor correlations: {[f'{c:.4f}' for c in correlations_gpu]}")
    print(f"Average correlation: {np.mean(np.abs(correlations_gpu)):.4f}\n")
    
    # Compare CPU vs GPU
    print("=" * 70)
    print("COMPARISON: CPU vs GPU")
    print("=" * 70)
    print(f"Reconstruction error difference: {abs(err_cpu - err_gpu):.6f}")
    print(f"Speed ratio (GPU/CPU): {elapsed_gpu / elapsed_cpu:.2f}x")
    
except Exception as e:
    print(f"Error (likely CUDA not available): {e}")
    print("GPU version requires CUDA and PyTorch with GPU support\n")
