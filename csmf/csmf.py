"""
Common and Specific non-negative Matrix Factorization (CSMF)

This module implements CSMF - a method for decomposing multiple related
data matrices into common and specific patterns using matrix factorization.

Mathematical Background:
-----------------------
Problem: Given K data matrices X¹, X², ..., Xᴷ (from related biological scenarios),
decompose each as:

.. math::
    X^k ≈ W^c H^{c,k} + W^{s,k} H^{s,k}

where:
- W^c: Shared (common) basis matrix
- W^{s,k}: Dataset-specific basis matrix for dataset k
- H^{c,k}: Common coefficients for dataset k
- H^{s,k}: Specific coefficients for dataset k

Objective: Minimize total reconstruction error

.. math::
    \\min \\sum_k \\|X^k - W^c H^{c,k} - W^{s,k} H^{s,k}\\|_F^2

subject to W^c, W^{s,k}, H^{c,k}, H^{s,k} >= 0

Algorithm:
-----------
CSMF uses alternating optimization:

1. **Update common components** (W^c, H^{c,k}):
   - Compute residuals: R^k = X^k - W^{s,k} H^{s,k}
   - Factorize concatenated residuals: [R¹ | R² | ... | Rᴷ] ≈ W^c [H^{c,1} | H^{c,2} | ... | H^{c,K}]
   
2. **Update specific components** (W^{s,k}, H^{s,k}):
   - For each dataset k:
   - Compute residual: R^k = X^k - W^c H^{c,k}
   - Factorize: R^k ≈ W^{s,k} H^{s,k}

3. **Normalize** W matrices to improve stability

4. **Repeat** until convergence

References:
-----------
Zhang, L., Zhang, S., & Qian, Z. (2019).
Learning common and specific patterns from data of multiple interrelated
biological scenarios with matrix factorization.

Related work:
- Joint NMF (jNMF) - simpler approach, all common patterns
- Integrative NMF (iNMF) - uses automated pattern selection

Author: Converted from MATLAB by Python conversion
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from csmf.nenmf import nenmf


def csmf(
    X: np.ndarray,
    vec_n: List[int],
    vec_para: List[int],
    iter_outer: int = 500,
    max_iter_nenm: int = 100,
    min_iter_nenm: int = 2,
    max_time_nenm: float = 100000,
    w_init: Optional[np.ndarray] = None,
    h_init: Optional[np.ndarray] = None,
    tol: float = 1e-6,
    verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray, int, float, Dict]:
    r"""
    Common and Specific non-negative Matrix Factorization.

    Decomposes multiple related data matrices into shared common patterns
    and dataset-specific patterns. Ideal for analyzing gene expression
    data from multiple biological conditions/cell types.

    Parameters
    ----------
    X : np.ndarray
        Concatenated data matrix of shape (m, sum(vec_n)).
        Columns are organized as [X¹ | X² | ... | Xᴷ] where:
        - m: number of features (genes)
        - X^k: data matrix for dataset k with vec_n[k] samples
        
        Example: X1 (100×50), X2 (100×80) → X (100×130)

    vec_n : list of int
        Number of samples in each dataset.
        Length = K (number of datasets).
        Must have at least 2 datasets (K >= 2).
        Example: [50, 80] for two datasets with 50 and 80 samples.

    vec_para : list of int
        Target ranks for factorization.
        Length = K + 1 where:
        - vec_para[0] = r_c (common rank)
        - vec_para[k] = r_{s,k} (specific rank for dataset k)
        
        Example: [4, 2, 3] means r_c=4, r_{s,1}=2, r_{s,2}=3

    iter_outer : int, optional
        Maximum number of outer iterations. Default: 500.
        Each iteration alternately updates common and specific factors.

    max_iter_nenm : int, optional
        Maximum NeNMF iterations per factorization. Default: 100.
        Controls inner optimization accuracy vs computation time.

    min_iter_nenm : int, optional
        Minimum NeNMF iterations. Default: 2.

    max_time_nenm : float, optional
        Maximum time (seconds) for NeNMF. Default: 100000.

    w_init : np.ndarray, optional
        Initial basis matrix of shape (m, sum(vec_para)).
        If None, randomly initialized.

    h_init : np.ndarray, optional
        Initial coefficient matrix of shape (sum(vec_para), sum(vec_n)).
        If None, randomly initialized with structure [H^c; H^s1; ...; H^sK]

    tol : float, optional
        Convergence tolerance. Default: 1e-6.
        Two stopping criteria:
        1. ||proj_grad|| <= tol * init_proj_grad
        2. Relative change in objective: |f(iter) - f(iter-10)| / f(iter) <= tol

    verbose : int, optional
        Verbosity level. Default: 0.
        - 0: Silent
        - 1: Return history
        - 2: Print progress every 10 iterations + return history

    Returns
    -------
    W : np.ndarray
        Basis matrix of shape (m, sum(vec_para)).
        Structure: [W^c | W^{s,1} | W^{s,2} | ... | W^{s,K}]
        - First vec_para[0] columns: common basis
        - Next vec_para[1] columns: specific basis for dataset 1
        - Etc.

    H : np.ndarray
        Coefficient matrix of shape (sum(vec_para), sum(vec_n)).
        Row structure: [H^c rows; H^{s,1} rows; ...; H^{s,K} rows]
        
        Reconstruction for dataset k:
            X^k ≈ W^c @ H^{c,k} + W^{s,k} @ H^{s,k}

    iter : int
        Number of outer iterations until convergence.

    elapse : float
        Total wall-clock time in seconds.

    HIS : dict
        Convergence history (if verbose >= 1):
        - 'niter': Total NeNMF iterations
        - 't': Elapsed times
        - 'f': Objective values (reconstruction error)
        - 'p': Stopping criteria values

    Examples
    --------
    >>> import numpy as np
    >>> from csmf import csmf
    
    # Simulate two related datasets
    >>> np.random.seed(42)
    >>> m, n1, n2 = 100, 50, 80
    >>> X1 = np.random.rand(m, n1)
    >>> X2 = np.random.rand(m, n2)
    >>> X = np.hstack([X1, X2])
    
    # Factorize with common rank=4, specific ranks=[2, 3]
    >>> W, H, n_iter, time_sec, hist = csmf(
    ...     X, 
    ...     vec_n=[50, 80],
    ...     vec_para=[4, 2, 3],
    ...     verbose=2,
    ...     iter_outer=200
    ... )
    
    # Extract factors
    >>> W_common = W[:, :4]
    >>> W_specific_1 = W[:, 4:6]
    >>> W_specific_2 = W[:, 6:9]
    
    >>> H_common = H[:4, :]
    >>> H_specific_1 = H[4:6, :]
    >>> H_specific_2 = H[6:9, :]
    
    # Reconstruct individual datasets
    >>> X1_recon = W_common @ H_common[:, :50] + W_specific_1 @ H_specific_1[:, :50]
    >>> X2_recon = W_common @ H_common[:, 50:] + W_specific_2 @ H_specific_2[:, 50:]
    
    # Compute reconstruction errors
    >>> error_1 = np.linalg.norm(X1 - X1_recon, 'fro')
    >>> error_2 = np.linalg.norm(X2 - X2_recon, 'fro')

    Notes
    -----
    **Key Advantages:**
    
    - **Interpretability**: Common patterns are biologically meaningful shared features
    - **Specificity**: Dataset-specific patterns capture unique characteristics
    - **Flexibility**: Allows different ranks for different datasets
    - **Scalability**: Efficient alternating optimization
    
    **Parameter Selection Tips:**
    
    1. **Common rank (vec_para[0])**:
       - Usually 2-10 for biological data
       - Corresponds to major biological processes
       - Use stability analysis or model selection
    
    2. **Specific ranks (vec_para[1:])**: 
       - Should be <= common rank or slightly larger
       - Typical: specific_rank ~ common_rank / 2
    
    3. **Tolerance**:
       - 1e-6: Default, good balance
       - 1e-4 to 1e-5: Faster convergence, acceptable error
       - 1e-8: Very high accuracy, slow
    
    4. **Iterations**:
       - Start with 200-500
       - Increase if not converged
       - Monitor HIS['f'] for stagnation
    
    **Algorithm Details**:
    
    At iteration t:
    
    1. Update common components:
       ```
       R^k = X^k - W^{s,k}_t H^{s,k}_t     (residuals)
       Concatenate: R_concat = [R¹|R²|...|Rᴷ]
       [W^c, H^c] ← NeNMF(R_concat, r_c)
       ```
    
    2. Update specific components (for each k):
       ```
       R^k = X^k - W^c_t H^{c,k}_t
       [W^{s,k}, H^{s,k}] ← NeNMF(R^k, r_{s,k})
       ```
    
    3. Normalize:
       ```
       H ← diag(sum(W)) × H
       W ← W × diag(sum(W))^(-1)
       ```

    Raises
    ------
    ValueError
        If dimensions don't match or invalid parameters provided.

    Warning
    -------
    Requires at least 2 datasets (len(vec_n) >= 2).
    """
    
    # ========================================================================
    # Input Validation
    # ========================================================================
    m = X.shape[0]
    num_datasets = len(vec_n)
    
    if num_datasets <= 1:
        raise ValueError("CSMF requires at least 2 datasets. "
                        f"Got {num_datasets} dataset(s)")
    
    if len(vec_para) != num_datasets + 1:
        raise ValueError(f"vec_para must have length {num_datasets + 1} "
                        f"(1 common + {num_datasets} specific ranks)")
    
    if sum(vec_n) != X.shape[1]:
        raise ValueError(f"X.shape[1] should equal sum(vec_n). "
                        f"Got {X.shape[1]} vs {sum(vec_n)}")
    
    if np.any(X < 0):
        print("Warning: X contains negative values. Clipping to 0.")
        X = np.clip(X, 0, None)
    
    # ========================================================================
    # Helper: Compute cumulative sums for indexing
    # ========================================================================
    sum_n = np.zeros(num_datasets, dtype=int)  # cumsum of sample counts
    sum_n[0] = vec_n[0]
    for i in range(1, num_datasets):
        sum_n[i] = sum_n[i - 1] + vec_n[i]
    
    sum_para = np.zeros(num_datasets + 1, dtype=int)  # cumsum of ranks
    sum_para[0] = vec_para[0]
    for i in range(1, num_datasets + 1):
        sum_para[i] = sum_para[i - 1] + vec_para[i]
    
    # ========================================================================
    # Extract individual data matrices
    # ========================================================================
    X_record = []
    start_idx = 0
    for k in range(num_datasets):
        end_idx = sum_n[k]
        X_record.append(X[:, start_idx:end_idx])
        start_idx = end_idx
    
    # ========================================================================
    # Initialize W and H
    # ========================================================================
    if w_init is None:
        W = np.random.rand(m, sum_para[-1])
    else:
        W = w_init.copy()
    
    if h_init is None:
        # Initialize in block structure
        H = np.zeros((sum_para[-1], sum(vec_n)))
        
        # Common components
        h_common = np.random.rand(vec_para[0], num_datasets)
        h_col_idx = 0
        for k in range(num_datasets):
            h_end = h_col_idx + vec_n[k]
            H[:vec_para[0], h_col_idx:h_end] = np.random.rand(vec_para[0], vec_n[k])
            h_col_idx = h_end
        
        # Specific components
        for k in range(num_datasets):
            row_idx = sum_para[k]
            row_end = sum_para[k + 1]
            col_idx = sum_n[k - 1] if k > 0 else 0
            col_end = sum_n[k]
            H[row_idx:row_end, col_idx:col_end] = np.random.rand(vec_para[k + 1], vec_n[k])
    else:
        H = h_init.copy()
    
    W = np.abs(W)
    H = np.abs(H)
    
    # ========================================================================
    # Normalization
    # ========================================================================
    col_sums = np.sum(W, axis=0)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    W = W / col_sums
    H = H * col_sums[:, np.newaxis]
    
    # ========================================================================
    # Initial Computation
    # ========================================================================
    start_time = time.time()
    
    from csmf.utils.stopping_criteria import get_stop_criterion
    
    V = X
    WtW = W.T @ W
    WtV = W.T @ V
    HHt = H @ H.T
    HVt = H @ V.T
    
    GradH = WtW @ H - WtV
    GradW = W @ HHt - HVt.T
    
    init_delta = get_stop_criterion(1, np.hstack([W.T, H]),
                                    np.hstack([GradW.T, GradH]))
    
    HIS = {
        'niter': 0,
        't': [0.0],
        'f': [np.linalg.norm(V - W @ H, 'fro') ** 2],
        'p': [init_delta]
    }
    
    # ========================================================================
    # Extract W and H in structured form
    # ========================================================================
    def _extract_factors():
        """Extract W and H in dataset-organized blocks"""
        W_c = W[:, :sum_para[0]]
        W_s = [W[:, sum_para[k]:sum_para[k + 1]] for k in range(num_datasets)]
        
        H_c = [H[:vec_para[0], sum_n[k - 1] if k > 0 else 0:sum_n[k]]
               for k in range(num_datasets)]
        H_s = [H[sum_para[k]:sum_para[k + 1], sum_n[k - 1] if k > 0 else 0:sum_n[k]]
               for k in range(num_datasets)]
        
        return W_c, W_s, H_c, H_s
    
    def _assemble_factors(W_c, W_s, H_c, H_s):
        """Assemble W and H from blocks"""
        W_new = np.hstack([W_c] + W_s)
        H_new = np.zeros((sum_para[-1], sum(vec_n)))
        H_new[:vec_para[0], :] = np.hstack(H_c)
        for k in range(num_datasets):
            row_start = sum_para[k]
            row_end = sum_para[k + 1]
            col_start = sum_n[k - 1] if k > 0 else 0
            col_end = sum_n[k]
            H_new[row_start:row_end, col_start:col_end] = H_s[k]
        return W_new, H_new
    
    # ========================================================================
    # Main CSMF Iteration Loop
    # ========================================================================
    for outer_iter in range(iter_outer):
        W_c, W_s, H_c, H_s = _extract_factors()
        
        # --- Step 1: Update Common Components ---
        # Compute residuals (data minus specific components)
        CX_list = []
        Hc_list = []
        
        for k in range(num_datasets):
            residual = np.maximum(X_record[k] - W_s[k] @ H_s[k], 0)
            CX_list.append(residual)
            Hc_list.append(H_c[k])
        
        CX = np.hstack(CX_list)
        Hc_concat = np.hstack(Hc_list)
        
        # Factorize concatenated residuals
        W_c_new, Hc_concat_new, _, _, _ = nenmf(
            CX, vec_para[0],
            max_iter=max_iter_nenm,
            min_iter=min_iter_nenm,
            max_time=max_time_nenm,
            tol=tol,
            w_init=W_c,
            h_init=Hc_concat,
            verbose=0
        )
        W_c = W_c_new
        
        # Split Hc back into per-dataset components
        H_c = []
        col_idx = 0
        for k in range(num_datasets):
            col_end = col_idx + vec_n[k]
            H_c.append(Hc_concat_new[:, col_idx:col_end])
            col_idx = col_end
        
        # --- Step 2: Update Specific Components ---
        iter_list = []
        for k in range(num_datasets):
            residual = np.maximum(X_record[k] - W_c @ H_c[k], 0)
            
            W_s_k_new, H_s_k_new, _, _, _ = nenmf(
                residual, vec_para[k + 1],
                max_iter=max_iter_nenm,
                min_iter=min_iter_nenm,
                max_time=max_time_nenm,
                tol=tol,
                w_init=W_s[k],
                h_init=H_s[k],
                verbose=0
            )
            W_s[k] = W_s_k_new
            H_s[k] = H_s_k_new
            iter_list.append(1)  # Count as 1 NMF factorization
        
        # --- Step 3: Assemble and Normalize ---
        W, H = _assemble_factors(W_c, W_s, H_c, H_s)
        
        col_sums = np.sum(W, axis=0)
        col_sums[col_sums == 0] = 1
        W = W / col_sums
        H = H * col_sums[:, np.newaxis]
        
        # ====================================================================
        # Convergence Checking
        # ====================================================================
        WtW = W.T @ W
        WtV = W.T @ V
        HHt = H @ H.T
        HVt = H @ V.T
        
        GradH = WtW @ H - WtV
        GradW = W @ HHt - HVt.T
        
        delta = get_stop_criterion(1, np.hstack([W.T, H]),
                                   np.hstack([GradW.T, GradH]))
        
        # Store history
        HIS['niter'] += len(iter_list)
        elapsed = time.time() - start_time
        
        obj_val = np.linalg.norm(V - W @ H, 'fro') ** 2
        HIS['f'].append(obj_val)
        HIS['t'].append(elapsed)
        HIS['p'].append(delta)
        
        if verbose == 2 and (outer_iter + 1) % 10 == 0:
            print(f"Iter {outer_iter + 1:4d}: "
                  f"stopping criterion = {delta / init_delta:.4e}, "
                  f"objective = {obj_val:.6f}, "
                  f"elapsed = {elapsed:.3f}s")
        
        # Check convergence
        if delta <= tol * init_delta:
            if verbose == 2:
                print(f"Converged at iteration {outer_iter + 1}")
            break
        
        # Check relative change in objective
        if outer_iter >= 20:
            delta_obj = abs(HIS['f'][outer_iter + 1] - HIS['f'][outer_iter - 10]) / HIS['f'][outer_iter + 1]
            if delta_obj <= tol:
                if verbose == 2:
                    print(f"Objective stagnating at iteration {outer_iter + 1}")
                break
    
    elapsed = time.time() - start_time
    
    if verbose == 2:
        print(f"\\n=== Final Result ===")
        print(f"Total iterations: {outer_iter + 1}")
        print(f"Total elapsed time: {elapsed:.3f}s")
    
    return W, H, outer_iter + 1, elapsed, HIS
