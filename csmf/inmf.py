"""
Integrative Non-negative Matrix Factorization (iNMF)

This module implements iNMF - a method that systematically extracts
common patterns from multiple datasets by identifying correlated basis vectors.

Mathematical Background:
-----------------------
Problem: Given K data matrices X¹, X², ..., Xᴷ, decompose as:

.. math::
    X^k ≈ W^c H^{c,k} + \\sum_{i=1}^k W^{s,i} H^{s,i,k}

where W^c is common to all datasets and W^{s,i} is specific to dataset i.

Key Innovation: iNMF uses correlation analysis to identify which basis vectors
are common across datasets:

1. Independently factorize each dataset to rank r_i
2. Compute pairwise correlations between basis matrices
3. Use Hungarian algorithm to find best matching
4. Average highly correlated vectors as common basis
5. Remaining unmatched vectors form specific basis

References:
-----------
Related to integrative analysis approaches in multi-omics studies.
Soft et al. (2012) discuss integrative NMF for multi-tissue data.

Author: Python implementation of iNMF algorithm based on Zhang et al. (2019)
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from csmf.nenmf import nenmf
from csmf.utils.hungarian import hungarian


def inmf(
    X: np.ndarray,
    vec_n: List[int],
    vec_para: List[int],
    max_iter_nenm: int = 100,
    min_iter_nenm: int = 2,
    max_time_nenm: float = 100000,
    tol: float = 1e-3,
    verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    r"""
    Integrative Non-negative Matrix Factorization.

    Decomposes multiple datasets by finding correlated basis vectors
    and treating them as common patterns.

    Parameters
    ----------
    X : np.ndarray
        Concatenated data matrix of shape (m, sum(vec_n))
        [X¹ | X² | ... | Xᴷ]

    vec_n : list of int
        Number of samples in each dataset

    vec_para : list of int
        Target ranks. Length K+1: [r_c, r_{s,1}, r_{s,2}, ..., r_{s,K}]
        where r_c is common rank and r_{s,k} is specific rank for dataset k

    max_iter_nenm : int, optional
        Maximum NeNMF iterations. Default: 100

    min_iter_nenm : int, optional
        Minimum NeNMF iterations. Default: 2

    max_time_nenm : float, optional
        Max time for NeNMF in seconds. Default: 100000

    tol : float, optional
        Convergence tolerance. Default: 1e-3

    verbose : int, optional
        Verbosity level. Default: 0

    Returns
    -------
    W : np.ndarray
        Factorized basis matrix

    H : np.ndarray
        Factorized coefficient matrix

    err : float
        Reconstruction error (Frobenius norm squared)

    elapse : float
        Elapsed time in seconds

    Notes
    -----
    Algorithm Steps:
    
    1. **Independent Factorization**: Each dataset independently:
       X^k ≈ W^k @ H^k where combined rank = r_c + r_{s,k}
    
    2. **Correlation Analysis**: 
       - Compute pairwise correlations between basis vectors
       - Use correlation distance: d = 1 - correlation
    
    3. **Optimal Matching**:
       - Hungarian algorithm finds best pairing of basis vectors
       - High correlation pairs are candidates for common basis
    
    4. **Common Basis Selection**:
       - Select top-matching pairs with correlation > threshold
       - Average matched vectors as common basis
       - Unmatched vectors become specific basis
    
    5. **Assembly**: Organize into W=[W^c|W^{s,1}|...|W^{s,K}]
    """
    
    if len(vec_n) <= 1:
        raise ValueError("iNMF requires at least 2 datasets")
    
    m = X.shape[0]
    num_datasets = len(vec_n)
    
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
    # Step 1: Independently factorize each dataset
    # ========================================================================
    W_record = []
    H_record = []
    
    # For iNMF: combined rank includes both common and specific
    combined_ranks = [vec_para[0] + vec_para[k + 1] for k in range(num_datasets)]
    
    for k in range(num_datasets):
        W_k, H_k, _, _, _ = nenmf(
            X_record[k], combined_ranks[k],
            max_iter=max_iter_nenm,
            min_iter=min_iter_nenm,
            max_time=max_time_nenm,
            tol=tol,
            verbose=0
        )
        
        # Normalize
        col_sums = np.sum(W_k, axis=0)
        col_sums[col_sums == 0] = 1
        W_k = W_k / col_sums
        H_k = H_k * col_sums[:, np.newaxis]
        
        W_record.append(W_k)
        H_record.append(H_k)
    
    # ========================================================================
    # Step 2-4: Find common basis via correlation analysis
    # ========================================================================
    # Compute correlation between first two datasets
    C = np.corrcoef(W_record[0].T, W_record[1].T)[:combined_ranks[0], combined_ranks[0]:]
    
    # Create distance matrix (1 - correlation)
    D = 1 - C
    
    # Hungarian algorithm for optimal matching
    matching, _ = hungarian(D)
    matched_rows, matched_cols = np.where(matching == 1)
    
    # Compute matching costs
    costs = np.array([D[matched_rows[i], matched_cols[i]] for i in range(len(matched_rows))])
    sorted_idx = np.argsort(costs)
    
    # Select top matches as common basis (corresponding to vec_para[0])
    n_common = vec_para[0]
    common_idx_w1 = matched_rows[sorted_idx[:n_common]]
    common_idx_w2 = matched_cols[sorted_idx[:n_common]]
    
    # Common basis is average of matched vectors
    W_c = (W_record[0][:, common_idx_w1] + W_record[1][:, common_idx_w2]) / 2
    
    # Common H for first two datasets
    H_c_1 = H_record[0][common_idx_w1, :]
    H_c_2 = H_record[1][common_idx_w2, :]
    
    # Mark which rows are used in matching
    matching[common_idx_w1, common_idx_w2] = 0
    unmatched_rows, unmatched_cols = np.where(matching == 1)
    
    # Specific basis for datasets 1 and 2
    spec_idx_1 = np.setdiff1d(np.arange(combined_ranks[0]), common_idx_w1)
    spec_idx_2 = np.setdiff1d(np.arange(combined_ranks[1]), common_idx_w2)
    
    W_s_1 = W_record[0][:, spec_idx_1]
    W_s_2 = W_record[1][:, spec_idx_2]
    
    H_s_1 = H_record[0][spec_idx_1, :]
    H_s_2 = H_record[1][spec_idx_2, :]
    
    # ========================================================================
    # For K > 2 datasets, iteratively match with common basis
    # ========================================================================
    H_c_record = [H_c_1, H_c_2]
    W_s_record = [W_s_1, W_s_2]
    H_s_record = [H_s_1, H_s_2]
    
    for k in range(2, num_datasets):
        # Correlate current common basis with new dataset
        C_k = np.corrcoef(W_c.T, W_record[k].T)[:n_common, combined_ranks[k]:]
        D_k = 1 - C_k
        
        matching_k, _ = hungarian(D_k)
        matched_rows_k, matched_cols_k = np.where(matching_k == 1)
        costs_k = np.array([D_k[matched_rows_k[i], matched_cols_k[i]] 
                           for i in range(len(matched_rows_k))])
        sorted_idx_k = np.argsort(costs_k)
        
        # Update common basis (average with new dataset)
        matched_idx_c = matched_rows_k[sorted_idx_k[:n_common]]
        matched_idx_k = matched_cols_k[sorted_idx_k[:n_common]]
        
        W_c = (W_c + W_record[k][:, matched_idx_k]) / 2
        
        # Store common H for this dataset
        H_c_k = H_record[k][matched_idx_k, :]
        H_c_record.append(H_c_k)
        
        # Specific components
        spec_idx_k = np.setdiff1d(np.arange(combined_ranks[k]), matched_idx_k)
        W_s_record.append(W_record[k][:, spec_idx_k])
        H_s_record.append(H_record[k][spec_idx_k, :])
    
    # ========================================================================
    # Assemble final factors
    # ========================================================================
    W = np.hstack([W_c] + W_s_record)
    
    H = np.zeros((sum_para[-1], sum(vec_n)))
    
    # Common H
    col_idx = 0
    for k in range(num_datasets):
        col_end = col_idx + vec_n[k]
        H[:n_common, col_idx:col_end] = H_c_record[k]
        col_idx = col_end
    
    # Specific H
    for k in range(num_datasets):
        row_start = n_common + sum([len(H_s_record[i]) for i in range(k)])
        row_end = row_start + len(H_s_record[k])
        col_start = sum_n[k - 1] if k > 0 else 0
        col_end = sum_n[k]
        H[row_start:row_end, col_start:col_end] = H_s_record[k]
    
    # Normalize
    col_sums = np.sum(W, axis=0)
    col_sums[col_sums == 0] = 1
    W = W / col_sums
    H = H * col_sums[:, np.newaxis]
    
    # Compute error
    X0 = X
    err = np.linalg.norm(X0 - W @ H, 'fro') ** 2
    
    elapsed = time.time() - start_time
    
    return W, H, err, elapsed
