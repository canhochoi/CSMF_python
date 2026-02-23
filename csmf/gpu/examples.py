"""
GPU CSMF Package Examples and Tests

Demonstrates usage of all GPU-accelerated algorithms:
- GPU CSMF
- GPU iNMF
- GPU jNMF

Run with: python -m csmf.gpu.examples
"""

import numpy as np
import scipy.sparse as sp
from csmf.gpu import gpu_csmf, gpu_inmf, gpu_jnmf, GPUConfig


def example_1_gpu_csmf():
    """Example 1: Basic GPU CSMF"""
    print("\n" + "="*70)
    print("EXAMPLE 1: GPU CSMF - Common & Specific Matrix Factorization")
    print("="*70)
    
    np.random.seed(42)
    
    # Create synthetic datasets
    m, n1, n2, n3 = 100, 50, 60, 70
    X1 = np.random.rand(m, n1)
    X2 = np.random.rand(m, n2)
    X3 = np.random.rand(m, n3)
    
    print(f"\nDatasets: {len([X1, X2, X3])}")
    print(f"  X1: {X1.shape}")
    print(f"  X2: {X2.shape}")
    print(f"  X3: {X3.shape}")
    
    # GPU CSMF
    print(f"\nRunning GPU CSMF...")
    result = gpu_csmf(
        [X1, X2, X3],
        rank_common=5,
        rank_specific=3,
        n_iter_outer=20,
        n_iter_inner=20,
        verbose=1
    )
    
    print(f"\nResults:")
    print(f"  W_c (common): {result['W_c'].shape}")
    print(f"  W_s shapes: {[w.shape for w in result['W_s']]}")
    print(f"  H_c shapes: {[h.shape for h in result['H_c']]}")
    print(f"  Final objective: {result['history']['obj'][-1]:.6e}")
    print(f"  Total time: {result['total_time']:.3f}s")


def example_2_gpu_inmf():
    """Example 2: GPU iNMF"""
    print("\n" + "="*70)
    print("EXAMPLE 2: GPU iNMF - Integrative NMF")
    print("="*70)
    
    np.random.seed(42)
    
    # Create synthetic datasets with some shared patterns
    m, rank_c, rank_s = 100, 5, 3
    
    # Create shared common factors
    W_c_true = np.random.rand(m, rank_c)
    
    X_list = []
    for n_samples in [50, 60, 70]:
        # Common part
        H_c = np.random.rand(rank_c, n_samples)
        # Specific part
        W_s = np.random.rand(m, rank_s)
        H_s = np.random.rand(rank_s, n_samples)
        X = W_c_true @ H_c + W_s @ H_s + np.random.randn(m, n_samples) * 0.01
        X = np.abs(X)
        X_list.append(X)
    
    print(f"\nDatasets: {len(X_list)}")
    for i, X in enumerate(X_list):
        print(f"  X{i+1}: {X.shape}")
    
    # GPU iNMF
    print(f"\nRunning GPU iNMF...")
    result = gpu_inmf(
        X_list,
        rank_common=rank_c,
        rank_specific=rank_s,
        correlation_threshold=0.5,
        n_iter_nmf=20,
        verbose=1
    )
    
    print(f"\nResults:")
    print(f"  W_c (common): {result['W_c'].shape}")
    print(f"  W_s shapes: {[w.shape for w in result['W_s']]}")
    print(f"  Total time: {result['total_time']:.3f}s")


def example_3_gpu_jnmf():
    """Example 3: GPU jNMF"""
    print("\n" + "="*70)
    print("EXAMPLE 3: GPU jNMF - Joint NMF")
    print("="*70)
    
    np.random.seed(42)
    
    # Create synthetic datasets
    m, n1, n2, n3 = 100, 50, 60, 70
    X1 = np.random.rand(m, n1)
    X2 = np.random.rand(m, n2)
    X3 = np.random.rand(m, n3)
    
    print(f"\nDatasets: {len([X1, X2, X3])}")
    print(f"  X1: {X1.shape}")
    print(f"  X2: {X2.shape}")
    print(f"  X3: {X3.shape}")
    
    # GPU jNMF
    print(f"\nRunning GPU jNMF...")
    result = gpu_jnmf(
        [X1, X2, X3],
        rank_common=5,
        rank_specific=3,
        cut=0.5,
        n_iter_nmf=20,
        verbose=1
    )
    
    print(f"\nResults:")
    print(f"  W_c (common): {result['W_c'].shape}")
    print(f"  W_s shapes: {[w.shape for w in result['W_s']]}")
    print(f"  Total time: {result['total_time']:.3f}s")


def example_4_sparse_matrices():
    """Example 4: Using sparse matrices"""
    print("\n" + "="*70)
    print("EXAMPLE 4: GPU Algorithms with Sparse Matrices")
    print("="*70)
    
    np.random.seed(42)
    
    # Create sparse datasets (typical for single-cell: 85% sparse)
    m, n1, n2 = 100, 50, 60
    sparsity = 0.85
    
    # Create sparse data
    X1_dense = np.random.exponential(0.1, (m, n1))
    X1_dense[np.random.rand(m, n1) > (1-sparsity)] = 0
    X1 = sp.csr_matrix(X1_dense)
    
    X2_dense = np.random.exponential(0.1, (m, n2))
    X2_dense[np.random.rand(m, n2) > (1-sparsity)] = 0
    X2 = sp.csr_matrix(X2_dense)
    
    print(f"\nSparse datasets (sparsity: {sparsity:.1%}):")
    print(f"  X1: {X1.shape}, nnz={X1.nnz}")
    print(f"  X2: {X2.shape}, nnz={X2.nnz}")
    print(f"  Memory: {(X1.data.nbytes + X2.data.nbytes) / 1e6:.1f} MB (sparse)")
    print(f"          {X1_dense.nbytes / 1e6:.1f} MB (dense)")
    
    # GPU CSMF with sparse
    print(f"\nRunning GPU CSMF with sparse matrices...")
    result = gpu_csmf(
        [X1, X2],
        rank_common=5,
        rank_specific=3,
        n_iter_outer=10,
        n_iter_inner=10,
        verbose=0
    )
    
    print(f"✓ Sparse matrices handled correctly")
    print(f"  W_c: {result['W_c'].shape}")
    print(f"  Time: {result['total_time']:.3f}s")


def example_5_configuration():
    """Example 5: GPU Configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 5: GPU Configuration and Device Management")
    print("="*70)
    
    # Create config
    config = GPUConfig()
    
    print(f"\nGPU Configuration:")
    print(f"  Device type: {config.device.type}")
    print(f"  Use fp16: {config.use_fp16}")
    print(f"  Sparse threshold: {config.sparse_threshold}")
    
    config.info()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GPU CSMF PACKAGE EXAMPLES")
    print("="*70)
    
    try:
        example_1_gpu_csmf()
        example_2_gpu_inmf()
        example_3_gpu_jnmf()
        example_4_sparse_matrices()
        example_5_configuration()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED ✓")
        print("="*70)
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
