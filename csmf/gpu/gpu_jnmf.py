"""
GPU-Accelerated Joint NMF

This module implements jNMF with GPU acceleration using PyTorch.

Key method: Joint factorization with thresholding to identify common factors,
and residual-based specific factor extraction.
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import List, Dict, Union, Optional
from scipy import stats
import time
from .config import GPUConfig
from .gpu_nenmf import GPUNeNMFSolver
from .utils import ensure_numpy_array


class GPUJnmfSolver:
    """
    GPU-accelerated Joint NMF solver.
    
    Factorizes multiple datasets jointly, treating all initial factors as
    potentially common, then thresholding to identify truly shared patterns.
    """
    
    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self.config = gpu_config or GPUConfig()
        self.nmf_solver = GPUNeNMFSolver(gpu_config)
        self.device = self.config.device
    
    def fit(
        self,
        X_list: List[Union[np.ndarray, sp.csr_matrix]],
        rank_common: int,
        rank_specific: Union[int, List[int]],
        cut: float = 0.5,
        n_iter_nmf: int = 50,
        verbose: int = 0
    ) -> Dict:
        """
        Fit jNMF model to multiple datasets.
        
        Parameters
        ----------
        X_list : List
            List of data matrices
        rank_common : int
            Target number of common factors
        rank_specific : int or list
            Target number of specific factors per dataset
        cut : float
            Threshold for identifying common factors (z-score cutoff)
        n_iter_nmf : int
            NMF iterations per factorization
        verbose : int
            Verbosity level
            
        Returns
        -------
        result : dict
            Contains: W_c, W_s, H_c, H_s, total_time
        """
        
        start_time = time.time()
        K = len(X_list)
        m = X_list[0].shape[0]
        
        if verbose:
            print(f"\n=== GPU jNMF ===")
            print(f"Datasets: {K}")
            print(f"Features: {m}")
            print(f"Target common rank: {rank_common}")
        
        # Handle rank_specific
        if isinstance(rank_specific, int):
            rank_specific_list = [rank_specific] * K
        else:
            rank_specific_list = rank_specific
        
        # Step 1: Joint factorization of concatenated data
        if verbose:
            print(f"Step 1: Joint NMF factorization...")
        
        # Concatenate all datasets - ensure proper dense conversion
        dense_data = []
        for k in range(K):
            X_k = X_list[k]
            if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                X_k = X_k.toarray()
            elif not isinstance(X_k, np.ndarray):
                X_k = np.asarray(X_k)
            dense_data.append(X_k)
        
        X_concat = np.hstack(dense_data)
        
        # Factorize with combined rank
        combined_rank = rank_common + np.sum(rank_specific_list)
        
        W_joint, H_joint = self.nmf_solver.nmf(
            X_concat, rank=combined_rank, max_iter=n_iter_nmf, verbose=False
        )
        
        if verbose:
            print(f"  Joint rank: {combined_rank}")
        
        # Step 2: Threshold to identify common factors
        if verbose:
            print(f"Step 2: Thresholding to identify common factors...")
        
        # Threshold based on z-score: keep components active in all datasets
        H_joint_normalized = (H_joint - H_joint.mean(axis=1, keepdims=True)) / (H_joint.std(axis=1, keepdims=True) + 1e-10)
        
        # Split H back to datasets
        H_split = []
        col_idx = 0
        for k in range(K):
            n_k = X_list[k].shape[1]
            H_split.append(H_joint[:, col_idx:col_idx + n_k])
            col_idx += n_k
        
        # Find common factors: those with high values across all datasets
        common_scores = []
        for factor_idx in range(W_joint.shape[1]):
            # Compute consistency across datasets
            factor_values = []
            for k in range(K):
                mean_val = np.mean(H_split[k][factor_idx, :])
                factor_values.append(mean_val)
            
            # Consistency = 1 - normalized std
            consistency = 1.0 - (np.std(factor_values) / (np.mean(factor_values) + 1e-10))
            common_scores.append(consistency)
        
        # Select top rank_common factors as common
        common_indices = np.argsort(-np.array(common_scores))[:rank_common]
        specific_indices = np.argsort(-np.array(common_scores))[rank_common:]
        
        W_c = W_joint[:, common_indices]
        W_specific_joint = W_joint[:, specific_indices]
        
        if verbose:
            print(f"  Common factors: {len(common_indices)}")
            print(f"  Specific factors to split: {len(specific_indices)}")
        
        # Step 3: Extract per-dataset specific factors
        if verbose:
            print(f"Step 3: Extracting dataset-specific factors...")
        
        W_s = []
        H_c = []
        H_s = []
        
        for k in range(K):
            X_k = X_list[k]
            # Convert sparse matrix to dense
            if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                X_k = X_k.toarray()
            elif not isinstance(X_k, np.ndarray):
                X_k = np.asarray(X_k)
            
            # Extract common coefficients
            H_c_k_full = H_split[k][common_indices, :]
            H_c.append(H_c_k_full)
            
            # Compute residuals after removing common components
            residual = np.maximum(X_k - W_c @ H_c_k_full, 0)
            
            # Factor residuals to get specific components
            rank_s_k = rank_specific_list[k]
            if rank_s_k > 0:
                W_s_k, H_s_k = self.nmf_solver.nmf(
                    residual, rank=rank_s_k, max_iter=n_iter_nmf, verbose=False
                )
                W_s.append(W_s_k)
                H_s.append(H_s_k)
            else:
                W_s.append(np.zeros((m, 1)))
                H_s.append(np.zeros((1, X_k.shape[1])))
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"Total time: {total_time:.3f}s")
            print(f"Common factors: {W_c.shape[1]}")
            print(f"Specific factors per dataset: {[W.shape[1] for W in W_s]}")
        
        return {
            'W_c': W_c,
            'W_s': W_s,
            'H_c': H_c,
            'H_s': H_s,
            'common_scores': common_scores,
            'common_indices': common_indices,
            'total_time': total_time
        }


def gpu_jnmf(
    X_list: List[Union[np.ndarray, sp.csr_matrix]],
    rank_common: int,
    rank_specific: Union[int, List[int]],
    device: Optional[torch.device] = None,
    verbose: int = 0,
    **kwargs
) -> Dict:
    """
    High-level API for GPU-accelerated jNMF.
    
    Parameters
    ----------
    X_list : List
        List of data matrices
    rank_common : int
        Common rank
    rank_specific : int or list
        Specific rank(s)
    device : torch.device, optional
        GPU device
    verbose : int
        Verbosity level
    **kwargs
        Additional arguments for GPUJnmfSolver.fit
        
    Returns
    -------
    result : dict
        jNMF factorization results
    """
    
    K = len(X_list)
    
    if isinstance(rank_specific, int):
        rank_specific = [rank_specific] * K
    
    config = GPUConfig(device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if verbose:
        config.info()
    
    solver = GPUJnmfSolver(config)
    result = solver.fit(
        X_list,
        rank_common=rank_common,
        rank_specific=rank_specific,
        verbose=verbose,
        **kwargs
    )
    
    return result
