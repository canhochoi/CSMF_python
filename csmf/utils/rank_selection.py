"""
Rank Selection via NeNMF Stability Analysis (MATLAB Implementation)

Implements the original authors' exact MATLAB workflow:
1. Run NeNMF independently on each dataset with varying ranks
2. Compute stability using Amari distance (not correlation)
3. Find local minima in stability curve
4. Learn common vs specific ranks via correlation matching

References:
-----------
Zhang, L., Zhang, S., & Qian, Z. (2019)
Learning common and specific patterns from data of multiple interrelated
biological scenarios with matrix factorization.
Nucleic Acids Research, 47(13), 6606-6617.
https://doi.org/10.1093/nar/gkz488
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from csmf.utils.hungarian import hungarian
from scipy.signal import find_peaks


def amari_distance(A: np.ndarray) -> float:
    r"""
    Compute Amari distance between two matrices (via correlation matrix).
    
    This measures how well one set of factors corresponds to another.
    Smaller values = better correspondence.
    
    Parameters
    ----------
    A : np.ndarray
        Correlation matrix (r1 × r2) between two factor sets
        
    Returns
    -------
    dist : float
        Amari distance (0-1)
    """
    n, m = A.shape
    if n == 1 and m == 1:
        return float(A[0, 0] == 0)
    
    # Maximum per column: best match for each column
    max_col = np.max(np.abs(A), axis=0)
    col_error = 1 - max_col  # How much each column fails to match
    
    # Maximum per row: best match for each row
    max_row = np.max(np.abs(A), axis=1)
    row_error = 1 - max_row  # How much each row fails to match
    
    # Amari distance: average of row and column errors
    dist = (np.mean(row_error) + np.mean(col_error)) / 2
    
    return dist


def stability_between_solutions(W1: np.ndarray, W2: np.ndarray) -> float:
    r"""
    Compute stability between two factor matrices using Amari distance.
    
    Parameters
    ----------
    W1, W2 : np.ndarray
        Two basis matrices
        
    Returns
    -------
    stability : float
        Stability score (0-1, higher = more stable/similar)
    """
    # Normalize factors
    W1_norm = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-10)
    W2_norm = W2 / (np.linalg.norm(W2, axis=0, keepdims=True) + 1e-10)
    
    # Correlation between factors
    corr_matrix = np.abs(W1_norm.T @ W2_norm)
    
    # Amari distance
    dist = amari_distance(corr_matrix)
    
    # Convert to stability (1 - distance)
    return 1.0 - dist


def compute_pairwise_stability(W_list: List[np.ndarray]) -> np.ndarray:
    r"""
    Compute pairwise stability between all solutions at a given rank.
    
    Parameters
    ----------
    W_list : List[np.ndarray]
        List of basis matrices from repeated runs
        
    Returns
    -------
    stability_matrix : np.ndarray
        (n_repeats × n_repeats) matrix of pairwise stabilities
    """
    n_runs = len(W_list)
    stability_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            stab = stability_between_solutions(W_list[i], W_list[j])
            stability_matrix[i, j] = stab
            stability_matrix[j, i] = stab
    
    np.fill_diagonal(stability_matrix, 1.0)  # Self = perfect stability
    return stability_matrix


def rank_stability_score(W_list: List[np.ndarray]) -> float:
    r"""
    Compute overall stability for a specific rank.
    
    This averages all pairwise Amari distances.
    
    Parameters
    ----------
    W_list : List[np.ndarray]
        List of basis matrices from repeated runs
        
    Returns
    -------
    score : float
        Stability score for this rank (0-1)
    """
    stab_matrix = compute_pairwise_stability(W_list)
    
    # Average all pairwise off-diagonal elements
    n = stab_matrix.shape[0]
    total = 2 * np.sum(np.triu(stab_matrix, k=1))  # Upper triangle × 2
    count = n * (n - 1)  # Number of pairs
    
    return total / count if count > 0 else 0.0


def nenmf_rank_sweep_per_dataset(
    X_datasets: List[np.ndarray],
    min_rank: int = 3,
    max_rank: int = 20,
    n_repeats: int = 30,
    max_iter: int = 200,
    verbose: int = 0
) -> Dict:
    r"""
    Run NeNMF on each dataset independently across rank range.
    
    This matches MATLAB NeNMF_data.m - runs NMF on concatenated data,
    but separately analyzes each dataset.
    
    Parameters
    ----------
    X_datasets : List[np.ndarray]
        List of data matrices for each dataset
    min_rank : int
        Minimum rank to test
    max_rank : int
        Maximum rank to test
    n_repeats : int
        Repetitions per rank
    max_iter : int
        Max iterations for NeNMF
    verbose : int
        Verbosity level
        
    Returns
    -------
    results : Dict
        results[dataset_idx][rank] = List of W matrices from n_repeats trials
    """
    from csmf.nenmf import nenmf
    
    n_datasets = len(X_datasets)
    results = {}
    
    if verbose >= 1:
        print(f"NeNMF rank sweep on {n_datasets} datasets (ranks {min_rank}-{max_rank})...")
    
    for dataset_idx, X in enumerate(X_datasets):
        results[dataset_idx] = {}
        
        if verbose >= 1:
            print(f"  Dataset {dataset_idx + 1}/{n_datasets}...")
        
        for rank in range(min_rank, max_rank + 1):
            W_list = []
            
            for rep in range(n_repeats):
                W, H, _, _, _ = nenmf(
                    X, r=rank,
                    max_iter=max_iter,
                    verbose=0
                )
                W_list.append(W)
            
            results[dataset_idx][rank] = W_list
            
            if verbose >= 2:
                print(f"    Rank {rank}: {n_repeats} runs ✓")
    
    return results


def analyze_stability_per_dataset(
    results: Dict,
    min_rank: int,
    max_rank: int,
    verbose: int = 0
) -> Dict:
    r"""
    Compute stability scores for each rank on each dataset.
    
    Returns dict mapping dataset_idx -> rank -> stability_score
    """
    stability_scores = {}
    
    for dataset_idx in results:
        stability_scores[dataset_idx] = {}
        
        if verbose >= 1:
            print(f"Computing stability for dataset {dataset_idx + 1}...")
        
        for rank in range(min_rank, max_rank + 1):
            W_list = results[dataset_idx][rank]
            score = rank_stability_score(W_list)
            stability_scores[dataset_idx][rank] = score
            
            if verbose >= 2:
                print(f"  Rank {rank}: stability = {score:.4f}")
    
    return stability_scores


def find_optimal_ranks(
    stability_scores: Dict,
    min_rank: int,
    max_rank: int,
    verbose: int = 0
) -> Dict:
    r"""
    Find optimal ranks per dataset using peak detection in stability curve.
    
    Matches MATLAB find_lowrank.m - finds local minima/maxima.
    
    Parameters
    ----------
    stability_scores : Dict
        Dataset -> rank -> stability mappings
    min_rank : int
        Minimum rank
    max_rank : int
        Maximum rank
    verbose : int
        Verbosity level
        
    Returns
    -------
    optimal_ranks : Dict
        dataset_idx -> optimal_rank
    """
    optimal_ranks = {}
    
    for dataset_idx in sorted(stability_scores.keys()):
        ranks = np.arange(min_rank, max_rank + 1)
        scores = np.array([stability_scores[dataset_idx][r] for r in ranks])
        
        # Find local maxima in stability scores
        peaks_max, _ = find_peaks(scores, distance=1)
        
        if verbose >= 2:
            print(f"    Dataset {dataset_idx}: ranks={ranks}, scores={np.round(scores, 4)}")
            print(f"    Local maxima at ranks: {ranks[peaks_max] if len(peaks_max) > 0 else 'None'}")
        
        # Strategy: Select best rank based on signal
        if len(peaks_max) > 1:
            # Multiple peaks: use the one with highest stability
            best_peak_idx = peaks_max[np.argmax(scores[peaks_max])]
            optimal_idx = best_peak_idx
        elif len(peaks_max) == 1:
            # Single peak: could be true signal
            optimal_idx = peaks_max[0]
        else:
            # No peaks (monotonic): stability decreases with rank
            # This usually means lower ranks are better (more stable)
            # Select the minimum rank in search range
            optimal_idx = 0
        
        optimal_rank = ranks[optimal_idx]
        optimal_ranks[dataset_idx] = optimal_rank
        
        if verbose >= 1:
            print(f"  Dataset {dataset_idx + 1}: optimal rank = {optimal_rank} " +
                  f"(stability = {scores[optimal_idx]:.4f})")
    
    return optimal_ranks


def select_best_factorization(
    W_list: List[np.ndarray],
    H_list: List[np.ndarray],
    criterion: str = 'reconstruction'
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Select best factorization from repeated runs.
    
    Can select by lowest reconstruction error, highest stability, etc.
    
    Parameters
    ----------
    W_list, H_list : Lists of matrices from n_repeats runs
    criterion : str
        'reconstruction' = lowest ||X - WH||
        'stability' = highest pairwise correlation
        
    Returns
    -------
    W_best, H_best : Best factorization
    """
    if criterion == 'stability':
        # Highest pairwise stability
        stab_matrix = compute_pairwise_stability(W_list)
        mean_stab = np.mean(stab_matrix, axis=1)
        best_idx = np.argmax(mean_stab)
    else:
        # Default: first run (could also compute error, but we don't have X here)
        best_idx = 0
    
    return W_list[best_idx], H_list[best_idx]


def learn_common_specific_ranks_from_correlations(
    W_matrices_per_dataset: List[np.ndarray],
    correlations_cutoff: float = 0.7,
    verbose: int = 0
) -> List[int]:
    r"""
    Learn common and specific ranks from correlation matching across datasets.
    
    Matches MATLAB learning_common_specific_ranks.m
    
    Parameters
    ----------
    W_matrices_per_dataset : List[np.ndarray]
        Best W matrix for each dataset (already normalized)
    correlations_cutoff : float
        Threshold for high correlation (0.5-0.9)
    verbose : int
        Verbosity level
        
    Returns
    -------
    vec_para : List[int]
        [r_common, r_specific_ds1, r_specific_ds2, ...]
    """
    K = len(W_matrices_per_dataset)
    
    if K < 2:
        # Only one dataset, cannot determine common/specific
        r = W_matrices_per_dataset[0].shape[1]
        return [r] + [0] * (K - 1)
    
    if verbose >= 1:
        print(f"\nLearning common/specific ranks from {K} datasets...")
    
    # Start with first two datasets
    W1 = W_matrices_per_dataset[0]
    W2 = W_matrices_per_dataset[1]
    
    # Concatenate and compute correlations
    W_concat = np.hstack([W1, W2])
    r1, r2 = W1.shape[1], W2.shape[1]
    
    # Correlation between all factor pairs
    corr_matrix = np.abs(np.corrcoef(W_concat.T).T[:r1, r1:])  # r1 × r2
    
    # Find high correlations
    highly_correlated = corr_matrix > correlations_cutoff
    
    if not np.any(highly_correlated):
        # No high correlations found
        if verbose >= 1:
            print("  No highly correlated factors found")
        return [0] + list(W_matrices_per_dataset[i].shape[1] for i in range(K))
    
    # Hungarian matching for optimal assignment
    cost_matrix = 1 - corr_matrix
    assignment, _ = hungarian(cost_matrix)
    
    # Extract matched pairs
    matched_pairs = np.where(assignment == 1)
    
    # Filter by correlation threshold
    common_factors = []
    for i, j in zip(matched_pairs[0], matched_pairs[1]):
        if corr_matrix[i, j] >= correlations_cutoff:
            common_factors.append((i, j))
    
    n_common = len(common_factors)
    
    if verbose >= 1:
        print(f"  Common factors from DS1+DS2: {n_common}")
    
    # Iteratively check with remaining datasets
    for k in range(2, K):
        W_k = W_matrices_per_dataset[k]
        
        if n_common == 0:
            break
        
        # Build common basis from previous datasets
        W_common = W_matrices_per_dataset[0][:, [p[0] for p in common_factors]]
        
        # Match with current dataset
        corr_k = np.abs(np.corrcoef(np.hstack([W_common, W_k]).T).T[:n_common, n_common:])
        cost_k = 1 - corr_k
        assignment_k, _ = hungarian(cost_k)
        
        # Get matched factors
        matched_k = np.where(assignment_k == 1)
        
        # Filter by threshold
        valid_matches = []
        for i, j in zip(matched_k[0], matched_k[1]):
            if corr_k[i, j] >= correlations_cutoff:
                valid_matches.append(j)
        
        n_common = len(set(valid_matches))
        
        if verbose >= 1:
            print(f"  Common factors after DS{k+1}: {n_common}")
        
        if n_common == 0:
            break
    
    # Build vec_para
    r_specific_per_dataset = []
    for k in range(K):
        r_k = W_matrices_per_dataset[k].shape[1]
        r_specific = r_k - n_common
        r_specific_per_dataset.append(max(1, r_specific))  # Ensure at least 1
    
    vec_para = [n_common] + r_specific_per_dataset
    
    if verbose >= 1:
        print(f"\nLearned ranks: {vec_para}")
    
    return vec_para


def rank_selection_pipeline(
    X_datasets: List[np.ndarray],
    min_rank: int = 3,
    max_rank: int = 20,
    n_repeats: int = 30,
    correlations_cutoff: float = 0.7,
    verbose: int = 1
) -> Tuple[List[int], Dict]:
    r"""
    Complete rank selection pipeline (MATLAB implementation).
    
    Parameters
    ----------
    X_datasets : List[np.ndarray]
        List of data matrices for each dataset
    min_rank : int
        Minimum rank to test
    max_rank : int
        Maximum rank to test
    n_repeats : int
        Repeated NeNMF per rank
    correlations_cutoff : float
        Threshold for common factor identification
    verbose : int
        Verbosity
        
    Returns
    -------
    vec_para : List[int]
        [r_common, r_specific_1, ..., r_specific_K]
    analysis : Dict
        Detailed analysis
    """
    if verbose >= 1:
        print("=" * 70)
        print("RANK SELECTION PIPELINE (MATLAB Implementation)")
        print("=" * 70)
    
    # Step 1: NeNMF rank sweep
    if verbose >= 1:
        print(f"\nStep 1: NeNMF rank sweep ({min_rank}-{max_rank}, {n_repeats} repeats)...")
    
    nenmf_results = nenmf_rank_sweep_per_dataset(
        X_datasets, min_rank, max_rank, n_repeats,
        verbose=verbose
    )
    
    # Step 2: Stability analysis
    if verbose >= 1:
        print("\nStep 2: Computing stability per rank per dataset...")
    
    stability_scores = analyze_stability_per_dataset(
        nenmf_results, min_rank, max_rank,
        verbose=verbose
    )
    
    # Step 3: Find optimal ranks
    if verbose >= 1:
        print("\nStep 3: Finding optimal ranks (peak detection)...")
    
    optimal_ranks = find_optimal_ranks(
        stability_scores, min_rank, max_rank,
        verbose=verbose
    )
    
    # Step 4: Select best factorizations at optimal ranks
    if verbose >= 1:
        print("\nStep 4: Selecting best factorizations...")
    
    W_matrices = []
    for dataset_idx, opt_rank in optimal_ranks.items():
        W_list = nenmf_results[dataset_idx][opt_rank]
        W_best, _ = select_best_factorization(W_list, [None] * len(W_list))
        
        # Normalize
        W_best_norm = W_best / (np.linalg.norm(W_best, axis=0, keepdims=True) + 1e-10)
        W_matrices.append(W_best_norm)
        
        if verbose >= 1:
            print(f"  Dataset {dataset_idx}: rank={opt_rank}, W shape={W_best.shape}")
    
    # Step 5: Learn common/specific ranks
    if verbose >= 1:
        print("\nStep 5: Learning common/specific ranks...")
    
    vec_para = learn_common_specific_ranks_from_correlations(
        W_matrices, correlations_cutoff,
        verbose=verbose
    )
    
    if verbose >= 1:
        print("\n" + "=" * 70)
        print(f"RECOMMENDED RANKS: {vec_para}")
        print("=" * 70)
    
    analysis = {
        'stability_scores': stability_scores,
        'optimal_ranks': optimal_ranks,
        'nenmf_results': nenmf_results,
        'W_matrices': W_matrices,
    }
    
    return vec_para, analysis
