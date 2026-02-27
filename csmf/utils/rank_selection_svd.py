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
from typing import List, Dict, Tuple, Optional, Literal


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


def select_rank_by_variance(s: np.ndarray, threshold: float) -> int:
    """Selects the smallest rank such that cumulative variance ≥ threshold."""
    if threshold <= 0:
        return 1
    cum_var = np.cumsum(np.square(s))
    total_var = cum_var[-1]
    if total_var == 0:
        return 1
    target = min(max(threshold, 0.0), 1.0)
    ratios = cum_var / total_var
    idx = int(np.searchsorted(ratios, target, side='left'))
    return min(idx + 1, len(s))


def _compute_stack_singular_values_from_gram(subspaces: List[np.ndarray]) -> np.ndarray:
    """Use the Gram matrix of stacked subspaces to recover singular values."""
    rank_blocks = [u.shape[1] for u in subspaces]
    total_rank = sum(rank_blocks)
    if total_rank == 0:
        return np.zeros(0, dtype=float)
    offsets = np.cumsum([0] + rank_blocks)
    gram = np.zeros((total_rank, total_rank), dtype=float)
    for i, u_i in enumerate(subspaces):
        start_i, end_i = offsets[i], offsets[i + 1]
        for j in range(i + 1):
            start_j, end_j = offsets[j], offsets[j + 1]
            block = subspaces[j].T @ u_i
            gram[start_j:end_j, start_i:end_i] = block
            if i != j:
                gram[start_i:end_i, start_j:end_j] = block.T
    eigvals = np.linalg.eigvalsh(gram)
    singular_vals = np.sqrt(np.clip(eigvals, 0.0, None))
    return singular_vals[::-1]


def _initial_randomized_components(total_rank: int) -> int:
    guess = max(10, int(np.sqrt(total_rank)) * 2)
    return max(1, min(total_rank, guess))


def _compute_stack_singular_values_randomized(
    U_stack: np.ndarray,
    threshold: float,
    components_hint: Optional[int] = None,
    random_state: Optional[int] = None,
) -> np.ndarray:
    try:
        from sklearn.utils.extmath import randomized_svd  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for randomized SVD; install it to use stack_svd_method='randomized'."
        ) from exc

    max_components = min(U_stack.shape)
    if max_components == 0:
        return np.zeros(0, dtype=float)
    components = components_hint or _initial_randomized_components(max_components)
    components = max(1, min(max_components, components))
    while True:
        _, singular_vals, _ = randomized_svd(
            U_stack, n_components=components, random_state=random_state
        )
        singular_vals = np.sort(singular_vals)[::-1]
        if components >= max_components:
            break
        if singular_vals.size < components or singular_vals[-1] < threshold:
            break
        new_components = min(max_components, components * 2)
        if new_components == components:
            break
        components = new_components
    return singular_vals

def rank_selection_svd_pipeline(
    X_datasets: List[np.ndarray],
    verbose: int = 1,
    variance_threshold: Optional[float] = None,
    stack_svd_method: Literal['full', 'gram', 'randomized'] = 'gram',
    stack_svd_components: Optional[int] = None,
    stack_svd_random_state: Optional[int] = None,
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
    variance_threshold : Optional[float]
        If set, choose the minimum number of PCs whose cumulative variance
        (square of singular values) reaches this fraction of the total.
    stack_svd_method : str
        Method for Step 2 (stacked score subspace) SVD. Options are:
        * 'full' - dense `np.linalg.svd` on the concatenated matrix (legacy).
        * 'gram' - build the Gram matrix and eigendecompose it (default).
        * 'randomized' - use `sklearn.utils.extmath.randomized_svd` to get the
          top components repeatedly until we can assess the threshold.
    stack_svd_components : Optional[int]
        Initial number of components to request for randomized SVD.
    stack_svd_random_state : Optional[int]
        Random seed for randomized SVD runs.

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
        if variance_threshold is not None:
            r_k = select_rank_by_variance(s, variance_threshold)
        else:
            r_k = find_elbow(s[:max_rank_k])
        optimal_ranks[i] = r_k
        score_subspaces[i] = U[:, :r_k]   # n × r_k orthonormal scores
        if verbose >= 1:
            method = (
                f"{variance_threshold:.2f} variance" if variance_threshold is not None else "elbow"
            )
            print(f"  Dataset {i+1}: total rank ({method}) = {r_k}")

    # Step 2: Stack score subspaces → common rank
    # Joint directions appear in all K blocks  → singular value ≈ √K
    # Individual directions appear in one block → singular value ≈ 1
    # Threshold = (1 + √K) / 2 cleanly separates the two groups.
    if verbose >= 1:
        print("\nStep 2: Stacking score subspaces → common rank...")

    stack_subspaces = [score_subspaces[i] for i in range(num_datasets)]
    rank_blocks = [u.shape[1] for u in stack_subspaces]
    total_rank = sum(rank_blocks)
    threshold = (1.0 + np.sqrt(K)) / 2
    method = stack_svd_method.lower()
    if method not in {'full', 'gram', 'randomized'}:
        raise ValueError("stack_svd_method must be one of 'full', 'gram', or 'randomized'.")
    if method == 'gram':
        s_stack = _compute_stack_singular_values_from_gram(stack_subspaces)
        u_stack_repr = 'not materialized (Gram factorization)'
    else:
        U_stack = np.hstack(stack_subspaces)
        stack_shape = U_stack.shape
        u_stack_repr = f"stacked scores shape: {stack_shape}"
        if method == 'full':
            _, s_stack, _ = np.linalg.svd(U_stack, full_matrices=False)
        else:
            s_stack = _compute_stack_singular_values_randomized(
                U_stack,
                threshold,
                stack_svd_components,
                stack_svd_random_state
            )
    r_common = int(np.sum(s_stack >= threshold))
    r_common = max(r_common, 1)

    if verbose >= 1:
          print(f"  Stack SVD method: {method}")
          if method != 'gram':
            print(f"  {u_stack_repr}")
          print(f"  Threshold (1+√{K})/2 = {threshold:.3f}")
          top = min(len(s_stack), 15)
          print(f"  Singular values of stacked scores (top {top}): "
              f"{np.round(s_stack[:top], 3).tolist()}")
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
        'stack_svd_method': method,
        'stack_svd_components': stack_svd_components,
        'stack_svd_random_state': stack_svd_random_state,
        'variance_threshold': variance_threshold,
    }

    return vec_para, analysis
