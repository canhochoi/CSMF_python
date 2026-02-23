"""
Example script demonstrating all CSMF algorithms

This script provides complete examples of:
1. Single dataset NMF (NeNMF)
2. Multi-dataset with common+specific patterns (CSMF)
3. Automatic pattern detection (iNMF)
4. Simple joint factorization (jNMF)

NEW FEATURE: Factor Comparison Scatter Plots
------------------------------------------
Compares inferred factors against ground truth values using:
- Hungarian algorithm for optimal factor alignment
- Automatic sign correction (handles NMF sign ambiguity)
- Magnitude scaling for fair comparison
- Correlation coefficients quantifying recovery quality

Generated scatter plots:
- csmf_common_factors.png: CSMF common factor recovery
- csmf_specific1_factors.png: CSMF dataset-specific factors
- inmf_common_factors.png: iNMF pattern discovery
- jnmf_common_factors.png: jNMF common factors
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from csmf import nenmf, csmf, inmf, jnmf
from csmf import compute_reconstruction_error


def generate_synthetic_data():
    """
    Generate synthetic multi-dataset example (matching test_cpu.py methodology).
    
    Creates three related datasets with:
    - Shared biological processes (common patterns)
    - Dataset-specific processes
    - Realistic noise (10%)
    - Uses random factors via np.abs(np.random.randn(...))
    """
    np.random.seed(42)
    
    m = 100  # features
    n1 = 50  # samples dataset 1
    n2 = 60  # samples dataset 2
    n3 = 40  # samples dataset 3
    
    # True factors - truly random (matching test_cpu.py)
    W_common_true = np.abs(np.random.randn(m, 3))
    H_common_1_true = np.abs(np.random.randn(3, n1)) * 0.8 + 0.5
    H_common_2_true = np.abs(np.random.randn(3, n2)) * 0.8 + 0.5
    H_common_3_true = np.abs(np.random.randn(3, n3)) * 0.8 + 0.5
    
    W_specific_1_true = np.abs(np.random.randn(m, 2))
    H_specific_1_true = np.abs(np.random.randn(2, n1)) * 0.5 + 0.1
    
    W_specific_2_true = np.abs(np.random.randn(m, 2))
    H_specific_2_true = np.abs(np.random.randn(2, n2)) * 0.5 + 0.1
    
    W_specific_3_true = np.abs(np.random.randn(m, 2))
    H_specific_3_true = np.abs(np.random.randn(2, n3)) * 0.5 + 0.1
    
    # Generate clean data
    X1_clean = W_common_true @ H_common_1_true + W_specific_1_true @ H_specific_1_true
    X2_clean = W_common_true @ H_common_2_true + W_specific_2_true @ H_specific_2_true
    X3_clean = W_common_true @ H_common_3_true + W_specific_3_true @ H_specific_3_true
    
    # Add realistic noise (10%)
    noise_level = 0.1
    X1 = X1_clean + noise_level * np.random.randn(m, n1) * np.std(X1_clean)
    X2 = X2_clean + noise_level * np.random.randn(m, n2) * np.std(X2_clean)
    X3 = X3_clean + noise_level * np.random.randn(m, n3) * np.std(X3_clean)
    
    # Ensure non-negativity
    X1 = np.maximum(X1, 0)
    X2 = np.maximum(X2, 0)
    X3 = np.maximum(X3, 0)
    
    # Concatenate for multi-dataset analysis
    X = np.hstack([X1, X2, X3])
    
    print("=" * 70)
    print("SYNTHETIC DATA GENERATED")
    print("=" * 70)
    print(f"Dataset 1: {m} features × {n1} samples")
    print(f"Dataset 2: {m} features × {n2} samples")
    print(f"Dataset 3: {m} features × {n3} samples")
    print(f"True ranks: r_common=3, r_specific_1=2, r_specific_2=2, r_specific_3=2")
    print(f"Noise level: {noise_level*100:.1f}%")
    print()
    
    # Return true factors for comparison
    true_factors = {
        'W_common': W_common_true,
        'W_specific_1': W_specific_1_true,
        'W_specific_2': W_specific_2_true,
        'W_specific_3': W_specific_3_true,
    }
    
    return X, [n1, n2, n3], (X1, X2, X3), true_factors


def example_1_nenmf(X):
    """
    Example 1: Single dataset NMF using NeNMF algorithm
    
    Best for: Single dataset factorization
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: NeNMF - Single Dataset NMF")
    print("=" * 70)
    print("Algorithm: Nesterov's Optimal Gradient Method")
    print("Use case: Single dataset factorization")
    print()
    
    # Concatenate all data for single factorization
    X_concat = X
    
    # Factorize with rank 9 (4 common + 2 + 3 specific, if we aggregated)
    W, H, n_iter, elapsed, history = nenmf(
        X_concat, r=9,
        max_iter=200,
        min_iter=10,
        tol=1e-6,
        verbose=2
    )
    
    print(f"\nResults:")
    print(f"  W shape: {W.shape}")
    print(f"  H shape: {H.shape}")
    print(f"  Iterations: {n_iter}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Final error: {np.linalg.norm(X_concat - W @ H, 'fro'):.6f}")
    
    return W, H, history


def example_2_csmf(X, vec_n):
    """
    Example 2: CSMF - Common and Specific Matrix Factorization
    
    Best for: Multiple related datasets with shared + unique patterns
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: CSMF - Common and Specific Factorization")
    print("=" * 70)
    print("Algorithm: Alternating factorization of common + specific components")
    print("Use case: Multiple datasets with shared biological processes")
    print()
    
    vec_para = [3, 2, 2, 2]  # [r_common, r_specific_1, r_specific_2, r_specific_3]
    
    W, H, n_iter, elapsed, history = csmf(
        X,
        vec_n=vec_n,
        vec_para=vec_para,
        iter_outer=100,
        max_iter_nenm=300,
        tol=1e-6,
        verbose=2
    )
    
    print(f"\nResults:")
    print(f"  W shape: {W.shape} [W_c({vec_para[0]})|W_s1({vec_para[1]})|W_s2({vec_para[2]})|W_s3({vec_para[3]})]")
    print(f"  H shape: {H.shape}")
    print(f"  Outer iterations: {n_iter}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Final error: {np.linalg.norm(X - W @ H, 'fro'):.6f}")
    
    # Extract and display components
    W_common = W[:, :vec_para[0]]
    W_specific_1 = W[:, vec_para[0]:vec_para[0]+vec_para[1]]
    W_specific_2 = W[:, vec_para[0]+vec_para[1]:vec_para[0]+vec_para[1]+vec_para[2]]
    W_specific_3 = W[:, vec_para[0]+vec_para[1]+vec_para[2]:]
    
    print(f"\nComponent Analysis:")
    print(f"  Common basis variance: {np.var(W_common):.6f}")
    print(f"  Specific basis 1 variance: {np.var(W_specific_1):.6f}")
    print(f"  Specific basis 2 variance: {np.var(W_specific_2):.6f}")
    print(f"  Specific basis 3 variance: {np.var(W_specific_3):.6f}")
    
    return W, H, history


def example_3_inmf(X, vec_n):
    """
    Example 3: iNMF - Integrative NMF with Correlation Analysis
    
    Best for: Automatic discovery of common patterns via correlation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: iNMF - Integrative NMF")
    print("=" * 70)
    print("Algorithm: Correlation-based pattern matching + Hungarian algorithm")
    print("Use case: Automatic discovery of commonly expressed factors")
    print()
    
    vec_para = [3, 2, 2, 2]
    
    W, H, err, elapsed = inmf(
        X,
        vec_n=vec_n,
        vec_para=vec_para,
        max_iter_nenm=100,
        tol=1e-6,
        verbose=1
    )
    
    print(f"\nResults:")
    print(f"  W shape: {W.shape}")
    print(f"  H shape: {H.shape}")
    print(f"  Reconstruction error: {err:.6f}")
    print(f"  Time: {elapsed:.3f}s")
    
    return W, H


def example_4_jnmf(X, vec_n):
    """
    Example 4: jNMF - Joint NMF with Thresholding
    
    Best for: Simple scenarios where most patterns are truly common
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: jNMF - Joint NMF")
    print("=" * 70)
    print("Algorithm: Joint factorization + thresholding for component filtering")
    print("Use case: Simpler multi-dataset factorization")
    print()
    
    vec_para = [3, 1, 1, 1]  # Simpler ranks for jNMF
    
    W, H, err, elapsed = jnmf(
        X,
        vec_para=vec_para,
        vec_n=vec_n,
        cut=0.5,  # Z-score threshold
        max_iter_nenm=100,
        tol=1e-6,
        verbose=1
    )
    
    print(f"\nResults:")
    print(f"  W shape: {W.shape}")
    print(f"  H shape: {H.shape}")
    print(f"  Reconstruction error: {err:.6f}")
    print(f"  Time: {elapsed:.3f}s")
    
    return W, H


def compare_algorithms(X, vec_n, individual_datasets):
    """
    Compare all algorithms and provide summary statistics
    """
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 70)
    print()
    
    X1, X2, X3 = individual_datasets
    sum_n = np.cumsum(vec_n)
    
    results = {}
    
    # NeNMF (single dataset)
    W_nenm, H_nenm, _, _, _ = nenmf(X, r=9, max_iter=200, verbose=0)
    err_nenm = np.linalg.norm(X - W_nenm @ H_nenm, 'fro')
    results['NeNMF'] = {
        'error': err_nenm,
        'w_shape': W_nenm.shape,
        'h_shape': H_nenm.shape
    }
    
    # CSMF
    W_csmf, H_csmf, _, _, _ = csmf(X, vec_n=vec_n, vec_para=[3, 2, 2, 2], 
                                    iter_outer=100, max_iter_nenm=300, verbose=0)
    err_csmf = np.linalg.norm(X - W_csmf @ H_csmf, 'fro')
    results['CSMF'] = {
        'error': err_csmf,
        'w_shape': W_csmf.shape,
        'h_shape': H_csmf.shape
    }
    
    # iNMF
    W_inmf, H_inmf, err_inmf, _ = inmf(X, vec_n=vec_n, vec_para=[3, 2, 2, 2], 
                                       max_iter_nenm=100, verbose=0)
    results['iNMF'] = {
        'error': err_inmf,
        'w_shape': W_inmf.shape,
        'h_shape': H_inmf.shape
    }
    
    # jNMF
    W_jnmf, H_jnmf, err_jnmf, _ = jnmf(X, vec_para=[3, 1, 1, 1], vec_n=vec_n,
                                       max_iter_nenm=100, verbose=0)
    results['jNMF'] = {
        'error': err_jnmf,
        'w_shape': W_jnmf.shape,
        'h_shape': H_jnmf.shape
    }
    
    # Print comparison table
    print(f"{'Algorithm':<12} {'Reconstruction Error':<22} {'W Shape':<15} {'H Shape':<15}")
    print("-" * 64)
    for algo_name, stats in results.items():
        print(f"{algo_name:<12} {stats['error']:<22.6f} {str(stats['w_shape']):<15} {str(stats['h_shape']):<15}")
    
    print()
    print("Interpretation:")
    print("- Lower reconstruction error = better factorization")
    print("- W-factor shapes differ based on algorithm's assumptions about common/specific patterns")
    print("- CSMF typically best: explicitly models common + specific components")
    
    return results


def align_and_scale_factors(W_true, W_inferred):
    """
    Align inferred factors to true factors using normalized correlation (Hungarian algorithm).
    Matches test_cpu.py methodology for robust factor alignment.
    
    Handles:
    - Factor permutation (optimal assignment via Hungarian)
    - Sign ambiguity in NMF (normalized dot product test)
    - Magnitude scaling (norm-based)
    """
    r_true, r_inf = W_true.shape[1], W_inferred.shape[1]
    r = min(r_true, r_inf)
    
    # Normalize for correlation-based alignment
    W_true_n = W_true[:, :r] / (np.linalg.norm(W_true[:, :r], axis=0, keepdims=True) + 1e-10)
    W_inf_n = W_inferred[:, :r] / (np.linalg.norm(W_inferred[:, :r], axis=0, keepdims=True) + 1e-10)
    
    # Compute correlation matrix via normalized dot product
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


def plot_factor_comparison(inferred_W, true_W, factor_name, output_filename):
    """
    Create scatter plots comparing inferred vs true factors.
    Uses test_cpu.py methodology for robust factor alignment.
    """
    # Align and scale inferred factors to true factors
    true_W_aligned, inferred_W_aligned, permutation = align_and_scale_factors(true_W, inferred_W)
    
    n_factors = true_W_aligned.shape[1]
    n_cols = min(4, n_factors)  # At most 4 plots per row
    n_rows = (n_factors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    correlations = []
    for idx in range(n_factors):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        x = true_W_aligned[:, idx]
        y = inferred_W_aligned[:, idx]
        
        # Compute correlation
        corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=30)
        
        # Add diagonal reference line
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, 'r--', alpha=0.5, linewidth=1, label='Perfect agreement')
        
        ax.set_xlabel('True Factor', fontsize=10)
        ax.set_ylabel('Inferred Factor', fontsize=10)
        ax.set_title(f'Factor {idx} (r={corr:.4f})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_factors, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{factor_name}: Inferred vs True Factors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, dpi=100, bbox_inches='tight')
    print(f"  Factor comparison plot saved: outputs/{output_filename}")
    plt.close()
    
    # Print correlation statistics
    avg_corr = np.mean(correlations)
    print(f"  {factor_name} - Average correlation: {avg_corr:.4f} (range: {min(correlations):.4f} to {max(correlations):.4f})")


def plot_convergence(histories):
    """
    Plot convergence history for algorithms
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    if 'nenmf_hist' in histories:
        hist = histories['nenmf_hist']
        
        # Objective function
        axes[0].semilogy(hist['f'])
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Reconstruction Error')
        axes[0].set_title('NeNMF: Objective Function')
        axes[0].grid(True, alpha=0.3)
        
        # Stopping criterion
        axes[1].semilogy(hist['p'])
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Projected Gradient Norm')
        axes[1].set_title('NeNMF: Convergence Criterion')
        axes[1].grid(True, alpha=0.3)
        
        # Time
        axes[2].plot(hist['t'])
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Elapsed Time (s)')
        axes[2].set_title('NeNMF: Computation Time')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/convergence_history.png', dpi=100, bbox_inches='tight')
    print("\nConvergence history plot saved as 'outputs/convergence_history.png'")
    plt.close()


def main():
    """
    Run all examples
    """
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "CSMF: Common and Specific Matrix Factorization".center(68) + "║")
    print("║" + "Complete Example and Algorithm Comparison".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Generate synthetic data
    X, vec_n, individual_datasets, true_factors = generate_synthetic_data()
    
    # Run examples
    histories = {}
    
    print("\n[Running Example 1/4: NeNMF]")
    W_nenm, H_nenm, hist_nenm = example_1_nenmf(X)
    histories['nenmf_hist'] = hist_nenm
    
    print("\n[Running Example 2/4: CSMF]")
    W_csmf, H_csmf, hist_csmf = example_2_csmf(X, vec_n)
    
    print("\n[Running Example 3/4: iNMF]")
    W_inmf, H_inmf = example_3_inmf(X, vec_n)
    
    print("\n[Running Example 4/4: jNMF]")
    W_jnmf, H_jnmf = example_4_jnmf(X, vec_n)
    
    # Algorithm comparison
    results = compare_algorithms(X, vec_n, individual_datasets)
    
    # Define ranks for factor extraction
    r_common = true_factors['W_common'].shape[1]  # 3
    r_specific_1 = true_factors['W_specific_1'].shape[1]  # 2
    r_specific_2 = true_factors['W_specific_2'].shape[1]  # 2
    r_specific_3 = true_factors['W_specific_3'].shape[1]  # 2
    
    # Plot factor comparisons with ground truth
    print("\n[Comparing inferred factors with ground truth]")
    print("\nCSMF Factor Comparison:")
    # Extract ONLY common factors (first r_common columns)
    W_csmf_common = W_csmf[:, :r_common]
    plot_factor_comparison(
        W_csmf_common,
        true_factors['W_common'],
        "CSMF: Common Factors",
        "outputs/csmf_common_factors.png"
    )
    
    # Extract specific factors from CSMF
    W_csmf_specific_1 = W_csmf[:, r_common:r_common+r_specific_1]
    
    if W_csmf_specific_1.shape[1] > 0:
        plot_factor_comparison(
            W_csmf_specific_1,
            true_factors['W_specific_1'],
            "CSMF: Dataset 1 Specific Factors",
            "outputs/csmf_specific1_factors.png"
        )
    
    print("\niNMF Factor Comparison:")
    # iNMF returns all factors - extract only first r_common for common factor comparison
    W_inmf_common = W_inmf[:, :r_common]
    plot_factor_comparison(
        W_inmf_common,
        true_factors['W_common'],
        "iNMF: Common Factors",
        "outputs/inmf_common_factors.png"
    )
    
    print("\njNMF Factor Comparison:")
    # jNMF returns all factors - extract only first r_common for common factor comparison
    W_jnmf_common = W_jnmf[:, :min(r_common, W_jnmf.shape[1])]
    plot_factor_comparison(
        W_jnmf_common,
        true_factors['W_common'],
        "jNMF: Common Factors",
        "outputs/jnmf_common_factors.png"
    )
    
    # Plot results
    print("\n[Generating convergence visualizations]")
    plot_convergence(histories)
    
    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. NeNMF: Fast gradient method, good for single datasets")
    print("2. CSMF: Explicit common/specific split, best for multi-dataset analysis")
    print("3. iNMF: Correlation-based matching, automatic pattern discovery")
    print("4. jNMF: Simple joint factorization, best when most patterns are common")
    print("\nFor more details, see README.md and individual function docstrings.")
    print()


if __name__ == "__main__":
    main()

