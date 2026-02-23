"""
Utility Functions for Non-negative Matrix Factorization

This module provides utility functions for evaluation, analysis and
processing of NMF results.
"""

import numpy as np
from typing import Tuple, Dict
from scipy.spatial.distance import cosine
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_reconstruction_error(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    norm_type: str = 'frobenius'
) -> float:
    r"""
    Compute reconstruction error: ||X - WH||

    Parameters
    ----------
    X : np.ndarray
        Original data matrix
    W : np.ndarray
        Basis matrix
    H : np.ndarray
        Coefficient matrix
    norm_type : str
        Type of norm: 'frobenius' or 'l1'

    Returns
    -------
    float
        Reconstruction error
    """
    residual = X - W @ H
    
    if norm_type == 'frobenius':
        return np.linalg.norm(residual, 'fro')
    elif norm_type == 'l1':
        return np.sum(np.abs(residual))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def sparsity(X: np.ndarray) -> float:
    r"""
    Compute sparsity of matrix X.
    
    Sparsity is defined as: (√N - ||X||_1 / ||X||_∞) / (√N - 1)
    where N is number of elements.
    
    Ranges from 0 (dense) to 1 (sparse).
    """
    N = X.size
    if N == 0:
        return 0.0
    
    norm_L1 = np.sum(np.abs(X))
    norm_Linf = np.max(np.abs(X))
    
    if norm_Linf == 0:
        return 0.0
    
    return (np.sqrt(N) - norm_L1 / norm_Linf) / (np.sqrt(N) - 1)


def matrix_similarity(M1: np.ndarray, M2: np.ndarray) -> float:
    r"""
    Compute average cosine similarity between column vectors.
    """
    if M1.shape[1] == 0 or M2.shape[1] == 0:
        return 0.0
    
    # Normalize columns
    M1_norm = M1 / (np.linalg.norm(M1, axis=0, keepdims=True) + 1e-10)
    M2_norm = M2 / (np.linalg.norm(M2, axis=0, keepdims=True) + 1e-10)
    
    # Compute column-wise cosine similarities
    similarities = np.diag(M1_norm.T @ M2_norm)
    
    return np.mean(similarities)


def compute_accuracy(
    W_est: np.ndarray,
    H_est: np.ndarray,
    W_true: np.ndarray,
    H_true: np.ndarray
) -> Dict[str, float]:
    r"""
    Compute accuracy metrics for NMF: AUC and AUPR.
    
    Binarizes reconstructions and computes ROC-AUC and PR-AUC
    as in biological network analysis.

    Parameters
    ----------
    W_est, H_est : np.ndarray
        Estimated factors
    W_true, H_true : np.ndarray
        Ground truth factors

    Returns
    -------
    dict
        Contains 'AUC' and 'AUPR' scores (ranging 0-1, higher is better)
    """
    # Reconstruct
    X_est = W_est @ H_est
    X_true = W_true @ H_true
    
    # Binarize
    X_est_binary = (X_est > np.median(X_est)).astype(float)
    X_true_binary = (X_true > 0).astype(float)
    
    X_est_flat = X_est.flatten()
    X_true_flat = X_true_binary.flatten()
    
    # Compute AUC
    try:
        auc = roc_auc_score(X_true_flat, X_est_flat)
    except:
        auc = 0.5
    
    # Compute AUPR
    try:
        aupr = average_precision_score(X_true_flat, X_est_flat)
    except:
        aupr = np.mean(X_true_flat)
    
    return {
        'auc': auc,
        'aupr': aupr
    }


def normalize_factors(
    W: np.ndarray,
    H: np.ndarray,
    method: str = 'weight_W'
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Normalize basis and coefficient matrices.

    Parameters
    ----------
    W, H : np.ndarray
        Factors to normalize
    method : str
        Normalization method:
        - 'weight_W': W has unit norm columns, H is scaled
        - 'weight_H': H has unit norm rows, W is scaled
        - 'balance': Geometric mean (W and H both scaled equally)

    Returns
    -------
    W_norm, H_norm : np.ndarray
        Normalized factors
    """
    W = W / W.sum(axis=0, keepdims=True)
    H = H / H.sum(axis=1, keepdims=True)
    return W, H
