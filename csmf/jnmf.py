"""
Joint Non-negative Matrix Factorization (jNMF)

This module implements jNMF - a simple approach where all basis vectors
are assumed to be common (shared) across datasets, but coefficients vary 
per dataset.

Mathematical Background:
-----------------------
Problem: Given K data matrices X¹, X², ..., Xᴷ, decompose as:

.. math::
    X^k ≈ W H^k

where:
- W: Shared basis matrix (same for all datasets)
- H^k: Dataset-specific coefficient matrix

Objective:

.. math::
    \\min_W, \\{H^k\\} \\sum_k \\|X^k - W H^k\\|_F^2

subject to W, H^k >= 0

Algorithm:
----------
jNMF differs from CSMF by assuming ALL patterns are common:

1. **Initialize**: Factorize concatenated data [X¹|X²|...|Xᴷ] ≈ W @ H_init
   
2. **Threshold common components**: 
   - Remove insignificant elements from W and H using z-score threshold
   - This helps focus on interpretable patterns
   
3. **Extract specific residuals**: R^k = max(X^k - W @ H^{c,k}, 0)
   
4. **Specific factorization**: For each dataset, factor residuals
   R^k ≈ W^{s,k} H^{s,k}

5. **Assemble**: W = [W_common | W_specific_1 | ... | W_specific_K]

Key Differences from CSMF:
- iNMF uses correlation-based matching
- jNMF uses direct thresholding of joint factorization
- Simpler, faster, but assumes less flexibility
- Better when most patterns truly common

References:
-----------
Complementary to CSMF. Useful when:
- Data is highly similar across datasets
- Common biological processes dominate
- Interpretability is priority over accuracy

Author: Python implementation of jNMF algorithm based on Zhang et al. (2019)
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from scipy import stats
from csmf.nenmf import nenmf


def jnmf(
    X: np.ndarray,
    vec_para: List[int],
    vec_n: List[int],
    cut: float = 0.5,
    max_iter_nenm: int = 100,
    min_iter_nenm: int = 2,
    max_time_nenm: float = 100000,
    tol: float = 1e-6,
    verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    r"""
    Joint Non-negative Matrix Factorization.

    Factorizes multiple datasets with shared common basis and
    dataset-specific coefficients plus dataset-specific basis.

    Parameters
    ----------
    X : np.ndarray
        Concatenated data matrix of shape (m, sum(vec_n))
        [X¹ | X² | ... | Xᴷ]

    vec_para : list of int
        Target ranks [r_c, r_{s,1}, r_{s,2}, ..., r_{s,K}]
        - r_c: common rank (shared by all datasets)
        - r_{s,k}: specific rank for dataset k
        - Length must be K + 1

    vec_n : list of int
        Number of samples in each dataset.
        Length K.
        Example: [50, 80] for two datasets

    cut : float, optional
        Z-score threshold for component selection. Default: 0.5.
        - Elements with |z-score| <= cut are zeroed out
        - Higher cut → sparser but less informative patterns
        - Typical range: 0.3 - 1.5

    max_iter_nenm : int, optional
        Maximum NeNMF iterations. Default: 100

    min_iter_nenm : int, optional
        Minimum NeNMF iterations. Default: 2

    max_time_nenm : float, optional
        Maximum time (seconds) for NeNMF. Default: 100000

    tol : float, optional
        Convergence tolerance. Default: 1e-6

    verbose : int, optional
        Verbosity level. Default: 0

    Returns
    -------
    W : np.ndarray
        Basis matrix [W_common | W_specific_1 | ... | W_specific_K]

    H : np.ndarray
        Coefficient matrix arranged in blocks:
        - Row 0 to r_c: Common coefficients
        - Row r_c to r_c+r_{s,1}: Specific coefficients for dataset 1
        - Etc.

    err : float
        Reconstruction error (Frobenius norm squared)

    elapse : float
        Elapsed time in seconds

    Notes
    -----
    **Algorithm Details:**

    1. **Joint Factorization**:
       - Treat concatenation [X¹|X²|...|Xᴷ] as single matrix
       - Factorize with common basis: [X¹|X²] ≈ W @ [H¹|H²]
    
    2. **Thresholding for Sparsity**:
       - Compute z-scores: z = (x - mean(x)) / std(x)
       - Zero out elements where |z| <= cut
       - Aims to keep only significant patterns
       - Mathematical form:
       
       .. math::
           W_thresh[i,j] = \\begin{cases}
               W[i,j] & \\text{if } |z[i,j]| > cut \\\\
               0 & \\text{otherwise}
           \\end{cases}

    3. **Decomposition into Common + Specific**:
       - Common part: Full W and H after thresholding
       - Residuals: R^k = X^k - W_c @ H^{c,k}
       - Factorize residuals for specific components
    
    4. **Final Assembly**:
       - W = [W_c | W_{s,1} | W_{s,2} | ... | W_{s,K}]
       - H = [H_c; H_{s,1}; H_{s,2}; ...; H_{s,K}]

    **Advantages:**
    
    - Simplest multi-dataset NMF approach
    - Computationally efficient
    - Automatic identification of common components
    - Good for highly similar datasets
    
    **Limitations:**
    
    - Assumes all patterns are common
    - Cannot model dataset-specific basis patterns
    - Thresholding parameter tuning can be tricky
    - Less flexible than CSMF or iNMF

    **Parameter Selection:**
    
    - **cut (threshold)**:
      - 0.3: Very permissive, keeps many patterns
      - 0.5: Default, moderate sparsity
      - 1.0: Strict, removes noisy patterns
      - Adjust based on data noise level
    
    - **Ranks**:
      - r_c: Number of truly common processes
      - r_{s,k}: Remaining factors unique to dataset k
      
    Examples
    --------
    >>> import numpy as np
    >>> from csmf import jnmf
    
    # Two similar datasets
    >>> X1 = np.random.rand(100, 50)
    >>> X2 = np.random.rand(100, 80)
    >>> X = np.hstack([X1, X2])
    
    # Factorize
    >>> W, H, err, t = jnmf(
    ...     X,
    ...     vec_para=[4, 1, 1],  # 4 common, 1 specific each
    ...     vec_n=[50, 80],
    ...     cut=0.5
    ... )
    
    >>> W.shape  # (100, 6): 4 common + 1 + 1 specific
    (100, 6)

    Raises
    ------
    ValueError
        If dimensions or parameters invalid
    """
    
    if len(vec_n) <= 1:
        raise ValueError("jNMF requires at least 2 datasets")
    
    m = X.shape[0]
    num_datasets = len(vec_n)
    
    if len(vec_para) != num_datasets + 1:
        raise ValueError(f"vec_para must have length {num_datasets + 1}")
    
    # Cumsum helpers
    sum_n = np.cumsum(vec_n)
    sum_para = np.cumsum(vec_para)
    
    # Extract individual matrices
    X_record = []
    start_idx = 0
    for k in range(num_datasets):
        end_idx = sum_n[k]
        X_record.append(X[:, start_idx:end_idx])
        start_idx = end_idx
    
    start_time = time.time()
    
    # ========================================================================
    # Step 1: Joint Factorization of All Data
    # ========================================================================
    # Factorize concatenated data to rank r_c (common rank)
    W_c, H_concat, _, _, _ = nenmf(
        X, vec_para[0],
        max_iter=max_iter_nenm,
        min_iter=min_iter_nenm,
        max_time=max_time_nenm,
        tol=tol,
        verbose=0
    )
    
    # ========================================================================
    # Step 2: Threshold to identify significant components
    # ========================================================================
    # z-score normalization: emphasize important features
    # Z-score: (x - mean) / std
    
    # For W: normalize each row
    W_c_zscore = np.zeros_like(W_c)
    for i in range(m):
        w_row = W_c[i, :]
        if np.std(w_row) > 1e-10:
            W_c_zscore[i, :] = (w_row - np.mean(w_row)) / np.std(w_row)
    
    # For H: normalize each column (transpose for row-wise operations)
    H_concat_T = H_concat.T  # Now shape (sum_n, r_c)
    H_concat_zscore_T = np.zeros_like(H_concat_T)
    for j in range(H_concat_T.shape[1]):  # For each basis vector
        h_col = H_concat_T[:, j]
        if np.std(h_col) > 1e-10:
            H_concat_zscore_T[:, j] = (h_col - np.mean(h_col)) / np.std(h_col)
    H_concat_zscore = H_concat_zscore_T.T  # Transpose back
    
    # Threshold: zero out components below threshold
    W_c[np.abs(W_c_zscore) <= cut] = 0
    H_concat[np.abs(H_concat_zscore) <= cut] = 0
    
    # Normalize after thresholding
    col_sums = np.sum(W_c, axis=0)
    col_sums[col_sums == 0] = 1
    W_c = W_c / col_sums
    H_concat = H_concat * col_sums[:, np.newaxis]
    
    # ========================================================================
    # Step 3: Extract common coefficients per dataset
    # ========================================================================
    H_c_record = []
    start_idx = 0
    for k in range(num_datasets):
        end_idx = sum_n[k]
        H_c_record.append(H_concat[:, start_idx:end_idx])
        start_idx = end_idx
    
    # ========================================================================
    # Step 4: Compute dataset-specific residuals and factorize
    # ========================================================================
    W_s_record = []
    H_s_record = []
    
    for k in range(num_datasets):
        # Residual = Data - Common Part
        residual = np.maximum(X_record[k] - W_c @ H_c_record[k], 1e-10)
        
        if vec_para[k + 1] > 0:
            # Factorize residual
            W_s_k, H_s_k, _, _, _ = nenmf(
                residual, vec_para[k + 1],
                max_iter=max_iter_nenm,
                min_iter=min_iter_nenm,
                max_time=max_time_nenm,
                tol=tol,
                verbose=0
            )
            W_s_record.append(W_s_k)
            
            # Store H_s for later assembly
            # Use a cell array-like approach
            H_s_k_padded = np.zeros((vec_para[k + 1], sum(vec_n)))
            col_start = sum_n[k - 1] if k > 0 else 0
            col_end = sum_n[k]
            H_s_k_padded[:, col_start:col_end] = H_s_k
            H_s_record.append(H_s_k_padded)
        else:
            W_s_record.append(np.zeros((m, 0)))
    
    # ========================================================================
    # Step 5: Assemble final factors
    # ========================================================================
    # W = [W_common | W_specific_1 | W_specific_2 | ... | W_specific_K]
    W = np.hstack([W_c] + W_s_record)
    
    # H structure:
    # Row 0 : vec_para[0] - common coefficients (concatenated)
    # Row vec_para[0] : vec_para[0]+vec_para[1] - specific for dataset 1
    # Etc.
    
    H = np.zeros((sum_para[-1], sum(vec_n)))
    
    # Fill common coefficients
    col_idx = 0
    for k in range(num_datasets):
        col_end = col_idx + vec_n[k]
        H[:vec_para[0], col_idx:col_end] = H_c_record[k]
        col_idx = col_end
    
    # Fill specific coefficients (diagonal structure for simplicity)
    for k in range(num_datasets):
        row_start = vec_para[0] + sum([vec_para[i + 1] for i in range(k)])
        row_end = row_start + vec_para[k + 1]
        col_start = sum_n[k - 1] if k > 0 else 0
        col_end = sum_n[k]
        
        if row_end > row_start:  # Only if rank > 0
            # Get H_s for this dataset
            h_s_idx = k
            if h_s_idx < len(H_s_record):
                H[row_start:row_end, col_start:col_end] = H_s_record[h_s_idx][:vec_para[k + 1], col_start:col_end]
    
    # Final normalization
    col_sums = np.sum(W, axis=0)
    col_sums[col_sums == 0] = 1
    W = W / col_sums
    H = H * col_sums[:, np.newaxis]
    
    # ========================================================================
    # Compute reconstruction error
    # ========================================================================
    X0 = X
    err = np.linalg.norm(X0 - W @ H, 'fro') ** 2
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"jNMF completed in {elapsed:.3f}s with reconstruction error {err:.6f}")
    
    return W, H, err, elapsed
