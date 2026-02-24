"""
Rank Selection via SVD Scree Plot Analysis

This module provides an alternative rank selection pipeline based on Singular
Value Decomposition (SVD). The core idea is to identify the "elbow" in the
scree plot of singular values, which indicates the transition from signal to
noise.

Workflow:
1.  For each data block, compute its SVD.
2.  Find the "elbow" in the scree plot of singular values to determine the
    initial optimal rank for that block.
3.  Run NMF at these optimal ranks to get the factor matrices (W).
4.  Use the same correlation-based method as the MATLAB implementation to
    decompose the initial ranks into common and specific parts.
"""

import numpy as np
from typing import List, Dict, Tuple
from csmf.nenmf import nenmf as nmf
from csmf.utils.rank_selection import learn_common_specific_ranks_from_correlations


def find_elbow(y: np.ndarray) -> int:
    """
    Finds the rank cutoff by detecting the largest gap (ratio) between
    consecutive singular values.

    The biggest drop in the scree plot indicates the transition from signal
    to noise. Using ratios (y[i] / y[i+1]) makes the detection scale-invariant.

    Parameters
    ----------
    y : np.ndarray
        A 1D array of singular values in descending order.

    Returns
    -------
    int
        Estimated rank (1-based).
    """
    if len(y) <= 1:
        return 1
    ratios = y[:-1] / (y[1:] + 1e-10)
    return int(np.argmax(ratios)) + 1


def rank_selection_svd_pipeline(
    X_datasets: List[np.ndarray],
    correlations_cutoff: float = 0.75,
    verbose: int = 1
) -> Tuple[List[int], Dict]:
    """
    Complete rank selection pipeline using SVD elbow analysis.

    Parameters
    ----------
    X_datasets : List[np.ndarray]
        List of data matrices for each dataset.
    correlations_cutoff : float
        Threshold for common factor identification.
    verbose : int
        Verbosity level.

    Returns
    -------
    vec_para : List[int]
        [r_common, r_specific_1, ..., r_specific_K]
    analysis : Dict
        Detailed analysis including singular values and optimal ranks.
    """
    if verbose >= 1:
        print("=" * 70)
        print("RANK SELECTION PIPELINE (SVD-based)")
        print("=" * 70)

    num_datasets = len(X_datasets)
    singular_values_all = {}
    optimal_ranks = {}

    # Step 1: SVD and Elbow Detection for each dataset
    if verbose >= 1:
        print("\nStep 1: SVD and Elbow Detection per dataset...")

    for i in range(num_datasets):
        U, s, Vt = np.linalg.svd(X_datasets[i], full_matrices=False)
        singular_values_all[i] = s
        
        # Find elbow to determine initial rank
        # We consider a max of 20 ranks for elbow detection
        max_rank_to_consider = min(len(s), 20)
        initial_rank = find_elbow(s[:max_rank_to_consider])
        optimal_ranks[i] = initial_rank
        
        if verbose >= 1:
            print(f"  Dataset {i+1}: Found elbow at rank={initial_rank}")

    # Step 2: Run NMF at optimal ranks to get W matrices
    if verbose >= 1:
        print("\nStep 2: Running NMF at optimal ranks...")
    
    W_matrices = []
    for i in range(num_datasets):
        opt_rank = optimal_ranks[i]
        W, H, _, _, _ = nmf(X_datasets[i], r=opt_rank)
        
        # Normalize W for correlation analysis
        W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
        W_matrices.append(W_norm)
        
        if verbose >= 1:
            print(f"  Dataset {i+1}: NMF at rank={opt_rank}, W shape={W.shape}")

    # Step 3: Learn common/specific ranks from correlations
    if verbose >= 1:
        print("\nStep 3: Learning common/specific ranks...")

    vec_para = learn_common_specific_ranks_from_correlations(
        W_matrices, correlations_cutoff, verbose=verbose
    )

    if verbose >= 1:
        print("\n" + "=" * 70)
        print(f"RECOMMENDED RANKS: {vec_para}")
        print("=" * 70)

    analysis = {
        'singular_values': singular_values_all,
        'optimal_ranks': optimal_ranks,
        'W_matrices': W_matrices,
    }

    return vec_para, analysis
