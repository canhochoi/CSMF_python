"""
GPU-Accelerated Integrative NMF

This module implements iNMF with GPU acceleration using PyTorch.

Key method: Correlate independently factorized datasets and use Hungarian
algorithm to find optimal matching of basis vectors, treating highly
correlated matches as common factors.
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import List, Dict, Union, Optional
import time
from scipy.optimize import linear_sum_assignment
from .config import GPUConfig
from .gpu_nenmf import GPUNeNMFSolver
from .utils import ensure_numpy_array


class GPUImmfSolver:
    """
    GPU-accelerated Integrative NMF solver.
    
    Finds common patterns through correlation-based matching of
    independently factorized basis matrices.
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
        correlation_threshold: float = 0.5,
        n_iter_nmf: int = 50,
        verbose: int = 0
    ) -> Dict:
        """
        Fit iNMF model to multiple datasets.
        
        Parameters
        ----------
        X_list : List
            List of data matrices
        rank_common : int
            Target number of common factors
        rank_specific : int or list
            Target number of specific factors per dataset
        correlation_threshold : float
            Minimum correlation to consider factors as common (0-1)
        n_iter_nmf : int
            NMF iterations per factorization
        verbose : int
            Verbosity level
            
        Returns
        -------
        result : dict
            Contains: W_c, W_s, H_c, H_s, correlations, matches, total_time
        """
        
        start_time = time.time()
        K = len(X_list)
        m = X_list[0].shape[0]
        
        if verbose:
            print(f"\n=== GPU iNMF ===")
            print(f"Datasets: {K}")
            print(f"Features: {m}")
            print(f"Target common rank: {rank_common}")
        
        # Handle rank_specific
        if isinstance(rank_specific, int):
            rank_specific_list = [rank_specific] * K
        else:
            rank_specific_list = rank_specific
        
        # Step 1: Independently factorize each dataset
        # Combined rank = common + specific
        combined_ranks = [rank_common + rank_specific_list[k] for k in range(K)]
        
        W_record = []
        H_record = []
        
        if verbose:
            print(f"Step 1: Independent NMF factorization...")
        
        for k in range(K):
            X_k = X_list[k]
            # Convert sparse matrix to dense
            if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                X_k = X_k.toarray()
            elif not isinstance(X_k, np.ndarray):
                X_k = np.asarray(X_k)
            
            if verbose:
                print(f"  Dataset {k+1}/{K}: rank {combined_ranks[k]}")
            
            W, H = self.nmf_solver.nmf(
                X_k, rank=combined_ranks[k], max_iter=n_iter_nmf, verbose=False
            )
            W_record.append(W)
            H_record.append(H)
        
        # Step 2: Compute pairwise correlations between basis matrices
        if verbose:
            print(f"Step 2: Computing correlations...")
        
        # Normalize all basis matrices
        W_normalized = []
        for W in W_record:
            W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)
            W_normalized.append(W_norm)
        
        # Compute correlation between datasets
        correlations = []
        for i in range(K):
            for j in range(i + 1, K):
                corr = np.abs(W_normalized[i].T @ W_normalized[j])
                correlations.append((i, j, corr))
        
        # Step 3: Use Hungarian algorithm to find optimal matching
        if verbose:
            print(f"Step 3: Optimal matching using Hungarian algorithm...")
        
        # For simplicity, match first dataset with others
        ref_dataset = 0
        ref_rank = W_record[ref_dataset].shape[1]
        
        matches = []  # Track which factors are common
        common_indices = [list(range(rank_common))]  # Reference dataset
        
        for k in range(1, K):
            # Compute correlation matrix
            corr_mat = np.abs(W_normalized[ref_dataset].T @ W_normalized[k])  # (ref_rank × rank_k)
            
            # Hungarian algorithm finds optimal assignment
            # linear_sum_assignment minimizes cost, so use negative correlation
            row_ind, col_ind = linear_sum_assignment(-corr_mat)
            
            # Select assignments above threshold
            dataset_common = []
            for i, j in zip(row_ind, col_ind):
                if corr_mat[i, j] >= correlation_threshold and i < rank_common:
                    dataset_common.append(j)
            
            common_indices.append(sorted(dataset_common))
            matches.append((row_ind, col_ind))
        
        # Step 4: Extract common and specific factors
        if verbose:
            print(f"Step 4: Extracting common and specific factors...")
        
        # Common factors: average matched basis vectors
        W_c = W_record[ref_dataset][:, :rank_common].copy()
        
        for k in range(1, K):
            for i, idx in enumerate(common_indices[k]):
                if i < rank_common:
                    W_c[:, i] += W_record[k][:, idx]
        
        W_c = W_c / K  # Average
        
        # Specific factors: unmatched vectors
        W_s = []
        for k in range(K):
            common_set = set(common_indices[k]) if k > 0 else set(range(rank_common))
            specific_indices = [j for j in range(W_record[k].shape[1]) if j not in common_set]
            
            if specific_indices:
                W_s_k = W_record[k][:, specific_indices]
            else:
                W_s_k = np.zeros((m, 1))
            
            W_s.append(W_s_k)
        
        # Extract H coefficients
        H_c = []
        H_s = []
        
        for k in range(K):
            # Factorize residual for specific components
            X_k = X_list[k]
            # Convert sparse matrix to dense
            if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                X_k = X_k.toarray()
            elif not isinstance(X_k, np.ndarray):
                X_k = np.asarray(X_k)
            
            residual = np.maximum(X_k - W_c @ H_record[k][:rank_common, :], 0)
            
            if W_s[k].shape[1] > 0:
                W_s_k, H_s_k = self.nmf_solver.nmf(
                    residual, rank=W_s[k].shape[1], max_iter=n_iter_nmf, w_init=W_s[k], verbose=False
                )
                W_s[k] = W_s_k
                H_s.append(H_s_k)
            else:
                H_s.append(np.array([]).reshape(0, residual.shape[1]))
            
            # Extract common coefficients
            X_k = X_list[k]
            # Convert sparse matrix to dense
            if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                X_k = X_k.toarray()
            elif not isinstance(X_k, np.ndarray):
                X_k = np.asarray(X_k)
            
            residual_c = X_k - W_s[k] @ H_s[k]
            lstsq_result = np.linalg.lstsq(W_c + 1e-10, residual_c, rcond=None)
            H_c_k = lstsq_result[0]  # Extract solution (first element)
            H_c_k = np.maximum(H_c_k, 0)
            H_c.append(H_c_k)
        
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
            'correlations': correlations,
            'matches': matches,
            'total_time': total_time
        }


def gpu_inmf(
    X_list: List[Union[np.ndarray, sp.csr_matrix]],
    rank_common: int,
    rank_specific: Union[int, List[int]],
    device: Optional[torch.device] = None,
    verbose: int = 0,
    **kwargs
) -> Dict:
    """
    High-level API for GPU-accelerated iNMF.
    
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
        Additional arguments for GPUImmfSolver.fit
        
    Returns
    -------
    result : dict
        iNMF factorization results
    """
    
    K = len(X_list)
    
    if isinstance(rank_specific, int):
        rank_specific = [rank_specific] * K
    
    config = GPUConfig(device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if verbose:
        config.info()
    
    solver = GPUImmfSolver(config)
    result = solver.fit(
        X_list,
        rank_common=rank_common,
        rank_specific=rank_specific,
        verbose=verbose,
        **kwargs
    )
    
    return result
