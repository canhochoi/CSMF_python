"""
Non-negative Matrix Factorization via Nesterov's Optimal Gradient Method

This module implements NeNMF - a non-negative matrix factorization algorithm
based on Nesterov's accelerated gradient method. It's the core algorithm used
by other methods in this package (CSMF, iNMF, jNMF).

Mathematical Background:
-----------------------
The NMF problem minimizes the Frobenius norm reconstruction error:

.. math::
    \\min_{W,H} \\|V - WH\\|_F^2 \\text{ subject to } W, H \\geq 0

where:
- V (m × n): Data matrix with n samples in m-dimensional space
- W (m × r): Basis matrix with r basis vectors
- H (r × n): Coefficient matrix with encodings

The Frobenius norm reconstruction error is:

.. math::
    \\|V - WH\\|_F^2 = \\text{tr}(V^T V) - 2 \\text{tr}(W^T V H^T) + \\text{tr}(W^T W H H^T)

Key Innovation - Nesterov's Accelerated Gradient:
The algorithm applies Nesterov's optimal gradient method which achieves
O(1/k²) convergence rate (compared to O(1/k) for standard gradient descent).

For non-negative least squares minimization of:
    f(X) = ||A - BX||²

The update uses acceleration:
    Y = X - (1/L) ∇f(X)           [gradient step]
    X_{new} = Y + ((α₁-1)/α₂)(Y - X)  [acceleration]
    
where L is the Lipschitz constant and α values enforce acceleration.

References:
-----------
Naiyang Guan, Dacheng Tao, Zhigang Luo, Bo Yuan (2012).
"NeNMF: An Optimal Gradient Method for Non-negative Matrix Factorization"
IEEE Transactions on Signal Processing, Vol. 60, No. 6, pp. 2882-2898.

Nesterov, Y. (1983). "A method for unconstrained convex minimization problem
with the rate of convergence O(1/k²)"

Author: Converted from MATLAB by Python conversion
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from scipy.linalg import norm
from csmf.utils.stopping_criteria import get_stop_criterion

# Default global stopping rule (can be overridden)
STOP_RULE = 1


def nenmf(
    V: np.ndarray,
    r: int,
    max_iter: int = 1000,
    min_iter: int = 10,
    max_time: float = 100000,
    w_init: Optional[np.ndarray] = None,
    h_init: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    verbose: int = 0
) -> Tuple[np.ndarray, np.ndarray, int, float, Dict]:
    r"""
    Non-negative Matrix Factorization via Nesterov's Optimal Gradient Method.

    Decomposes data matrix V into non-negative factors W and H such that
    V ≈ WH, minimizing the Frobenius norm reconstruction error.

    Parameters
    ----------
    V : np.ndarray
        Data matrix of shape (m, n).
        - m: dimensionality (number of features/genes in bio applications)
        - n: number of samples
        All elements must be non-negative (V >= 0)

    r : int
        Target rank (number of basis vectors / latent factors).
        Typically much smaller than min(m, n).
        For biological data: often 2-10.

    max_iter : int, optional
        Maximum number of outer iterations. Default: 1000.
        Each outer iteration optimizes W and H alternately.

    min_iter : int, optional
        Minimum number of iterations to perform.
        Useful to ensure algorithm explores enough before early stopping.
        Default: 10.

    max_time : float, optional
        Maximum computation time in seconds. Default: 100000 (27.8 hours).
        Algorithm stops if this time is exceeded.

    w_init : np.ndarray, optional
        Initial basis matrix W of shape (m, r).
        If None, initialized randomly from U(0,1).
        Should be non-negative. Use to provide warm start.

    h_init : np.ndarray, optional
        Initial coefficient matrix H of shape (r, n).
        If None, initialized randomly from U(0,1).
        Should be non-negative. Use to provide warm start.

    tol : float, optional
        Stopping tolerance. Default: 1e-5.
        Convergence criterion is: ||proj_grad|| <= tol * init_proj_grad.
        Smaller values give more accurate solutions (slower convergence).
        Typical range: 1e-3 to 1e-8.

    verbose : int, optional
        Verbosity level. Default: 0.
        - 0: No output
        - 1: Return history in output (no screen output)
        - 2: Print progress every 10 iterations + return history

        History includes: iteration count, objective values, elapsed time,
        stopping criterion progression.

    Returns
    -------
    W : np.ndarray
        Basis matrix of shape (m, r).
        Columns are the r non-negative basis vectors.
        Each column represents a latent factor/pattern.

    H : np.ndarray
        Coefficient matrix of shape (r, n).
        Rows are encodings of samples on the r factors.
        For sample j: V[:, j] ≈ W @ H[:, j]

    iter : int
        Number of iterations performed until convergence.

    elapse : float
        Wall-clock time in seconds.

    HIS : dict
        Convergence history (only if verbose >= 1). Contains:
        - 'niter': Total inner NNLS iterations
        - 't': Elapsed times at each iteration
        - 'f': Objective values (reconstruction error)
        - 'p': Stopping criteria values

    Notes
    -----
    **Algorithm Overview:**

    1. **Initialization**: Randomly initialize W, H if not provided
    
    2. **Outer Loop**: For each iteration:
        - Optimize H with W fixed (Step A)
        - Optimize W with H fixed (Step B)
        - Check convergence and stopping criteria
    
    3. **Optimization Steps** (A and B use NNLS with acceleration):
        - Compute gradient of objective function
        - Apply non-negative projection: X = max(X - grad/L, 0)
        - Apply Nesterov acceleration to X
        - Decrease tolerance if converging too slowly
    
    4. **Stopping Criteria**:
        - Projected gradient norm falls below tolerance
        - Total elapsed time exceeds max_time
        
    5. **Output**: Returns final factors and convergence history

    **Mathematical Details:**

    Objective function at iteration k:

    .. math::
        f(W, H) = \\|V - WH\\|_F^2

    Gradients with respect to coefficients:

    .. math::
        ∇_H f = W^T(WH - V) = W^T W H - W^T V

    Gradients with respect to basis:

    .. math::
        ∇_W f = (WH - V)H^T = WH H^T - VH^T

    Non-negative Least Squares Acceleration (inner loop):

    .. math::
        Y = \\max(Z - ∇f(Z)/L, 0)  \\text{   [projection step]}
        Z_{new} = Y + \\frac{α_k - 1}{α_{k+1}}(Y - Z)  \\text{   [acceleration]}

    where α values follow: α_{k+1} = (1 + √(1 + 4α_k²))/2 (Nesterov sequence)

    Examples
    --------
    >>> import numpy as np
    >>> from csmf import nenmf
    
    # Generate synthetic data: 100 genes × 50 samples, rank 3
    >>> np.random.seed(42)
    >>> W_true = np.random.rand(100, 3)
    >>> H_true = np.random.rand(3, 50)
    >>> V = W_true @ H_true + np.random.randn(100, 50) * 0.1
    >>> V = np.abs(V)  # Ensure non-negativity
    
    # Factorize
    >>> W, H, n_iter, time_sec, history = nenmf(
    ...     V, r=3, verbose=1, max_iter=100
    ... )
    
    # Check reconstruction
    >>> reconstruction_error = np.linalg.norm(V - W @ H, 'fro')
    >>> print(f"Reconstruction error: {reconstruction_error:.4f}")
    
    # For warm start from different initialization
    >>> W_init = np.random.rand(100, 3)
    >>> H_init = np.random.rand(3, 50)
    >>> W, H, _, _, _ = nenmf(
    ...     V, r=3, w_init=W_init, h_init=H_init
    ... )

    Raises
    ------
    ValueError
        If V contains negative values, r <= 0, or dimensions are invalid.

    Warning
    -------
    This algorithm is non-convex. Different initializations may yield
    different local minima. Consider running multiple times with
    different random seeds for robustness.

    """

    # ============================================================================
    # Input Validation
    # ============================================================================
    if V.ndim != 2:
        raise ValueError(f"V must be 2D matrix, got shape {V.shape}")
    
    if np.any(V < 0):
        print("Warning: V contains negative values. They will be clipped to 0.")
        V = np.clip(V, 0, None)
    
    if r <= 0 or r != int(r):
        raise ValueError(f"Target rank r must be positive integer, got {r}")
    
    m, n = V.shape
    r = int(r)
    
    # ============================================================================
    # Initialization
    # ============================================================================
    global STOP_RULE
    
    # Initialize W and H
    if w_init is None:
        W = np.random.rand(m, r)
    else:
        W = w_init.astype(np.float64)
        if W.shape != (m, r):
            raise ValueError(f"w_init shape must be ({m}, {r}), got {W.shape}")
    
    if h_init is None:
        H = np.random.rand(r, n)
    else:
        H = h_init.astype(np.float64)
        if H.shape != (r, n):
            raise ValueError(f"h_init shape must be ({r}, {n}), got {H.shape}")
    
    # Ensure non-negativity
    W = np.abs(W)
    H = np.abs(H)
    
    # ============================================================================
    # Initial Computation
    # ============================================================================
    start_time = time.time()
    
    # Precompute products used in gradient calculations
    WtW = W.T @ W
    WtV = W.T @ V
    HHt = H @ H.T
    HVt = H @ V.T
    
    # Compute gradients
    GradH = WtW @ H - WtV
    GradW = W @ HHt - HVt.T
    
    # Initial stopping criterion (for relative convergence check)
    init_delta = get_stop_criterion(STOP_RULE, np.hstack([W.T, H]), 
                                    np.hstack([GradW.T, GradH]))
    
    # Initial tolerance values
    tolH = max(tol, 1e-3) * init_delta
    tolW = tolH
    
    # Compute constant term in objective function (for error calculation)
    constV = np.sum(V ** 2)
    
    # Initialize history
    HIS = {
        'niter': 0,
        't': [0.0],
        'f': [np.sum(WtW * HHt) - 2 * np.sum(WtV * H)],
        'p': [init_delta]
    }
    
    # ============================================================================
    # Main Iterative Loop
    # ============================================================================
    for iteration in range(max_iter):
        # --- Optimize H with W fixed ---
        # H minimizes ||V - WH||² subject to H >= 0
        # This is a non-negative least squares problem
        H, iterH, GradH = _nnls(H, WtW, WtV, min_iter, max_iter, tolH)
        
        # Adapt tolerance: if not making enough inner progress, tighten tolerance
        if iterH <= min_iter:
            tolH = tolH / 10
        
        # Update products involving H
        HHt = H @ H.T
        HVt = H @ V.T
        
        # --- Optimize W with H fixed ---
        # W minimizes ||V - WH||² subject to W >= 0 (via W.T)
        W, iterW, GradW = _nnls(W.T, HHt, HVt, min_iter, max_iter, tolW)
        W = W.T  # Transpose back
        
        # Adapt tolerance
        if iterW <= min_iter:
            tolW = tolW / 10
        
        # Update products involving W
        WtW = W.T @ W
        WtV = W.T @ V
        
        # Update gradient for H
        GradH = WtW @ H - WtV
        
        # Update iteration count
        HIS['niter'] += iterH + iterW
        
        # ====================================================================
        # Convergence Checking
        # ====================================================================
        # Compute projected gradient norm as stopping criterion
        delta = get_stop_criterion(STOP_RULE, np.hstack([W.T, H]),
                                   np.hstack([GradW, GradH]))
        
        # Store history if verbose
        elapsed = time.time() - start_time
        
        if verbose:
            obj_val = np.sum(WtW * HHt) - 2 * np.sum(WtV * H)
            HIS['f'].append(obj_val)
            HIS['t'].append(elapsed)
            HIS['p'].append(delta)
            
            if verbose == 2 and (iteration + 1) % 10 == 0:
                final_obj = 0.5 * (obj_val + constV)
                print(f"Iter {iteration + 1:4d}: "
                      f"stopping criterion = {delta / init_delta:.4e}, "
                      f"objective = {final_obj:.6f}, "
                      f"elapsed time = {elapsed:.3f}s")
        
        # Check primary stopping criterion
        if iteration >= min_iter - 1 and delta <= tol * init_delta:
            if verbose == 2:
                print(f"\\nConverged at iteration {iteration + 1}")
            break
        
        # Check time limit
        if elapsed >= max_time:
            if verbose == 2:
                print(f"\\nReached time limit of {max_time}s at iteration {iteration + 1}")
            break
    
    # ============================================================================
    # Final Processing
    # ============================================================================
    elapsed = time.time() - start_time
    
    if verbose:
        # Convert objective values to reconstruction error (add constant term)
        HIS['f'] = 0.5 * (np.array(HIS['f']) + constV)
        
        if verbose == 2:
            print(f"\\n=== Final Result ===")
            print(f"Total iterations: {iteration + 1}")
            print(f"Total elapsed time: {elapsed:.3f}s")
            print(f"Final reconstruction error: {HIS['f'][-1]:.6f}")
    
    return W, H, iteration + 1, elapsed, HIS


def _nnls(
    Z: np.ndarray,
    WtW: np.ndarray,
    WtV: np.ndarray,
    iter_min: int,
    iter_max: int,
    tol: float
) -> Tuple[np.ndarray, int, np.ndarray]:
    r"""
    Non-negative Least Squares with Nesterov's Optimal Gradient Method.

    Solves: minimize ||Z·(WtW) - WtV||² subject to Z >= 0
    
    This is an internal function used by NeNMF for optimizing each
    factor (W or H) while keeping the other fixed.

    Parameters
    ----------
    Z : np.ndarray
        Current iterate (either W.T or H depending on optimization step)

    WtW : np.ndarray
        Gram matrix (Hessian) of objective function

    WtV : np.ndarray
        Gradient direction (negative gradient at Z=0)

    iter_min : int
        Minimum iterations

    iter_max : int
        Maximum iterations

    tol : float
        Convergence tolerance for this subproblem

    Returns
    -------
    Z : np.ndarray
        Optimized coefficient matrix

    iter : int
        Number of iterations performed

    Grad : np.ndarray
        Final gradient value

    Notes
    -----
    Algorithm uses Nesterov's acceleration:
    
    1. Compute gradient: ∇f(Z) = Z @ WtW - WtV
    
    2. Proximal step: Y = max(Z - ∇f(Z) / L, 0)
       where L is Lipschitz constant of gradient (max eigenvalue of WtW)
    
    3. Acceleration: Z = Y + (α₁ - 1) / α₂ * (Y - Z)
       with α update: α_{k+1} = (1 + √(1 + 4α_k²)) / 2
    
    4. Check convergence: if ||proj_grad(Z)|| < tol, stop
    """
    
    global STOP_RULE
    
    # Compute Lipschitz constant (max eigenvalue of WtW)
    # For positive definite matrices, this is the largest eigenvalue
    L = norm(WtW)
    
    # Copy initial value
    H = Z.copy()
    
    # Compute initial gradient
    Grad = WtW @ Z - WtV
    alpha1 = 1.0
    
    # ====================================================================
    # Inner NNLS loop with Nesterov acceleration
    # ====================================================================
    for iteration in range(iter_max):
        Z0 = H.copy()
        
        # Proximal step: apply gradient descent + projection to non-negative orthant
        # H = max(Z - Grad/L, 0)
        H = np.maximum(Z - Grad / L, 0)
        
        # Nesterov acceleration
        # Update momentum factor
        alpha2 = 0.5 * (1 + np.sqrt(1 + 4 * alpha1 ** 2))
        
        # Extrapolation (acceleration step)
        Z = H + ((alpha1 - 1) / alpha2) * (H - Z0)
        alpha1 = alpha2
        
        # Compute new gradient for next iteration
        Grad = WtW @ Z - WtV
        
        # Check stopping criterion
        if iteration >= iter_min - 1:
            pgn = get_stop_criterion(STOP_RULE, Z, Grad)
            if pgn <= tol:
                break
    
    # Final gradient with optimized Z
    Grad = WtW @ H - WtV
    
    return H, iteration + 1, Grad
