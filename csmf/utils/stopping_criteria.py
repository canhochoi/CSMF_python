"""
Stopping Criteria for Non-negative Matrix Factorization

This module implements various stopping criteria for NMF algorithms.
Stopping criteria are used to determine convergence of optimization algorithms
by measuring the magnitude of projected gradients or KKT residuals.

Mathematical Background:
-----------------------
In constrained optimization with non-negativity constraints (X >= 0),
the projected gradient is computed as follows:

For a variable X >= 0 with gradient ∇f(X):
- Projected Gradient: PG = ∇f(X) where X > 0, and PG = min(∇f(X), 0) where X = 0

The KKT (Karush-Kuhn-Tucker) conditions for non-negativity constraints are:
- ∇f(X) = 0 at X > 0 (unrestricted)
- ∇f(X) >= 0 at X = 0 (at lower bound)

Different stopping criteria measure convergence differently:
1. Projected Gradient Norm: ||PG|| (most common)
2. Normalized Projected Gradient Norm: ||PG|| / |support(PG)|
3. Normalized KKT Residual: L1-norm of KKT violation / count of violations

Author: Converted from MATLAB by Python conversion
Reference: Lin, C. J. (2007). Projected gradient methods for nonnegative matrix factorization
"""

import numpy as np
from typing import Tuple, Union


def get_stop_criterion(
    stop_rule: int,
    X: np.ndarray,
    grad_X: np.ndarray
) -> float:
    r"""
    Calculate stopping criterion based on projected gradient or KKT conditions.

    Parameters
    ----------
    stop_rule : int
        Type of stopping criterion to use:
        - 1: Projected Gradient Norm (default, most sensitive to convergence)
        - 2: Normalized Projected Gradient Norm (per-element sensitivity)
        - 3: Normalized KKT Residual (L1-norm based)

    X : np.ndarray
        Current variable (basis matrix W or coefficient matrix H).
        Must be non-negative (X >= 0).

    grad_X : np.ndarray
        Gradient of objective function with respect to X.
        Same shape as X.

    Returns
    -------
    float
        Stopping criterion value. Smaller values indicate better convergence.
        Typically, convergence is declared when this value < tolerance.

    Notes
    -----
    The projected gradient is computed as:
    
    .. math::
        PG_{ij} = \begin{cases}
            \nabla f(X)_{ij} & \text{if } X_{ij} > 0 \\
            \min(\nabla f(X)_{ij}, 0) & \text{if } X_{ij} = 0
        \end{cases}

    Criterion 1 (Projected Gradient Norm):
    
    .. math::
        \|PG\|_2 = \sqrt{\sum_{i,j} PG_{ij}^2}
        
    Criterion 2 (Normalized Projected Gradient Norm):
    
    .. math::
        \frac{\|PG\|_2}{|support(PG)|}
        
    where |support(PG)| is the number of non-zero elements in PG.
    
    Criterion 3 (Normalized KKT Residual):
    
    .. math::
        \frac{\|min(X, \nabla f(X))\|_1}{|support(min(X, \nabla f(X)))|}

    Examples
    --------
    >>> import numpy as np
    >>> from csmf.utils.stopping_criteria import get_stop_criterion
    >>> W = np.random.rand(10, 5)
    >>> grad_W = np.random.randn(10, 5)
    >>> criterion = get_stop_criterion(1, W, grad_W)
    >>> print(f"Stopping criterion value: {criterion:.6f}")

    Raises
    ------
    ValueError
        If stop_rule is not in {1, 2, 3}
    """
    
    if stop_rule not in [1, 2, 3]:
        raise ValueError(f"stop_rule must be 1, 2, or 3, got {stop_rule}")
    
    if X.shape != grad_X.shape:
        raise ValueError(f"X and grad_X must have same shape, got {X.shape} and {grad_X.shape}")
    
    if stop_rule == 1:
        # Projected Gradient Norm
        # Keep gradient where X > 0, and min(grad, 0) where X = 0
        return _projected_gradient_norm(X, grad_X)
    
    elif stop_rule == 2:
        # Normalized Projected Gradient Norm
        p_grad = _get_projected_gradient(X, grad_X)
        p_grad_norm = np.linalg.norm(p_grad)
        n_support = len(p_grad[p_grad != 0])
        
        if n_support == 0:
            return 0.0
        return p_grad_norm / n_support
    
    else:  # stop_rule == 3
        # Normalized KKT Residual (L1-norm based)
        # At X = 0: need grad >= 0 (minimum is at boundary)
        # At X > 0: need grad = 0 (minimum in interior)
        # KKT violation: min(X, grad) should be close to 0
        kkt_violation = np.minimum(X, grad_X)
        kkt_residual = np.sum(np.abs(kkt_violation))
        
        n_violations = len(kkt_violation[np.abs(kkt_violation) > 1e-14])
        
        if n_violations == 0:
            return 0.0
        return kkt_residual / n_violations


def _get_projected_gradient(X: np.ndarray, grad_X: np.ndarray) -> np.ndarray:
    r"""
    Compute the projected gradient for non-negativity constraints.
    
    For X >= 0 constraint, the projected gradient at point X is:
    
    .. math::
        \text{proj_grad}_{ij} = \begin{cases}
            \nabla f(X)_{ij} & \text{if } X_{ij} > 0 \\
            \min(\nabla f(X)_{ij}, 0) & \text{if } X_{ij} = 0
        \end{cases}

    This extracts only the components of the gradient that point
    out of the feasible region X >= 0.

    Parameters
    ----------
    X : np.ndarray
        Current point (must be non-negative)
    grad_X : np.ndarray
        Gradient at X

    Returns
    -------
    np.ndarray
        Projected gradient with same shape as X
    """
    # Flatten for element-wise operations
    X_flat = X.flatten()
    grad_flat = grad_X.flatten()
    
    p_grad = grad_flat.copy()
    
    # Where X = 0, keep only negative gradients (pointing inward)
    zero_mask = X_flat == 0
    p_grad[zero_mask] = np.minimum(grad_flat[zero_mask], 0)
    
    return p_grad.reshape(X.shape)


def _projected_gradient_norm(X: np.ndarray, grad_X: np.ndarray) -> float:
    r"""
    Compute L2 norm of projected gradient.
    
    Parameters
    ----------
    X : np.ndarray
        Current point
    grad_X : np.ndarray
        Gradient
        
    Returns
    -------
    float
        Euclidean norm of projected gradient
    """
    p_grad = _get_projected_gradient(X, grad_X)
    return float(np.linalg.norm(p_grad))


# For backward compatibility with MATLAB parameter names
STOP_RULE = 1  # Global default (can be overridden)
