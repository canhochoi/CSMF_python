"""
Rank Selection via SVD Scree Plot Analysis

Workflow:
1.  SVD each dataset → per-dataset total rank r_k (elbow of scree plot)
    and score subspace U_k (n × r_k orthonormal columns).
2.  Stack score subspaces: U_stack = [U_1 | U_2 | ... | U_K] (n × Σr_k).
    SVD of U_stack: joint directions appear in all K blocks → singular value ≈ √K.
    Individual directions appear in only one block → singular value ≈ 1.
    Common rank r_J = number of singular values ≥ (1 + √K) / 2.
3.  Specific rank for dataset k = max(r_k - r_J, 1).
4.  Return vec_para = [r_J, r_specific_1, ..., r_specific_K].
"""

import numpy as np
from typing import List, Dict, Tuple


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
    verbose: int = 1
) -> Tuple[List[int], Dict]:
    """
    Complete rank selection pipeline using SVD elbow analysis.

    Common rank is detected by SVD of the horizontally concatenated matrix.
    Specific ranks are detected by SVD of each individual dataset.

    Parameters
    ----------
    X_datasets : List[np.ndarray]
        List of data matrices for each dataset (shape: n_features × n_samples).
    verbose : int
        Verbosity level.

    Returns
    -------
    vec_para : List[int]
        [r_common, r_specific_1, ..., r_specific_K]
    analysis : Dict
        Detailed analysis including singular values and detected ranks.
    """
    if verbose >= 1:
        print("=" * 70)
        print("RANK SELECTION PIPELINE (SVD-based)")
        print("=" * 70)

    num_datasets = len(X_datasets)
    K = num_datasets
    singular_values_all = {}
    optimal_ranks = {}
    score_subspaces = {}

    # Step 1: SVD per dataset → per-dataset total rank and score subspace
    if verbose >= 1:
        print("\nStep 1: SVD per dataset → per-dataset total rank...")

    for i in range(num_datasets):
        U, s, _ = np.linalg.svd(X_datasets[i], full_matrices=False)
        singular_values_all[i] = s
        max_rank_k = min(len(s), 20)
        r_k = find_elbow(s[:max_rank_k])
        optimal_ranks[i] = r_k
        score_subspaces[i] = U[:, :r_k]   # n × r_k orthonormal scores
        if verbose >= 1:
            print(f"  Dataset {i+1}: total rank (elbow) = {r_k}")

    # Step 2: Stack score subspaces → common rank
    # Joint directions appear in all K blocks  → singular value ≈ √K
    # Individual directions appear in one block → singular value ≈ 1
    # Threshold = (1 + √K) / 2 cleanly separates the two groups.
    if verbose >= 1:
        print("\nStep 2: Stacking score subspaces → common rank...")

    U_stack = np.hstack([score_subspaces[i] for i in range(num_datasets)])
    _, s_stack, _ = np.linalg.svd(U_stack, full_matrices=False)
    threshold = (1.0 + np.sqrt(K)) / 2
    r_common = int(np.sum(s_stack >= threshold))
    r_common = max(r_common, 1)

    if verbose >= 1:
        print(f"  Stacked scores shape: {U_stack.shape}")
        print(f"  Threshold (1+√{K})/2 = {threshold:.3f}")
        print(f"  Singular values of stacked scores (top {min(len(s_stack), 15)}): "
              f"{np.round(s_stack[:min(len(s_stack), 15)], 3).tolist()}")
        print(f"  Common rank (values ≥ threshold): {r_common}")

    # Step 3: Specific rank = max(per-dataset rank - common rank, 1)
    if verbose >= 1:
        print("\nStep 3: Computing specific ranks...")

    r_specific = []
    for i in range(num_datasets):
        r_s = max(optimal_ranks[i] - r_common, 1)
        r_specific.append(r_s)
        if verbose >= 1:
            print(f"  Dataset {i+1}: {optimal_ranks[i]} (total) - {r_common} (common) = {r_s} (specific)")

    vec_para = [r_common] + r_specific

    if verbose >= 1:
        print("\n" + "=" * 70)
        print(f"RECOMMENDED RANKS: {vec_para}")
        print("=" * 70)

    analysis = {
        'singular_values': singular_values_all,
        'singular_values_stack': s_stack,
        'optimal_ranks': optimal_ranks,
        'r_common': r_common,
        'threshold': threshold,
    }

    return vec_para, analysis
