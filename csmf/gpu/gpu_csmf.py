"""
GPU-Accelerated Common and Specific Matrix Factorization

High-performance PyTorch implementation of CSMF for analyzing multiple
related biological datasets simultaneously.
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import List, Dict, Union, Optional, Tuple
import time
from .config import GPUConfig
from .gpu_nenmf import GPUNeNMFSolver
from .utils import ensure_numpy_array, ensure_torch_tensor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class GPUCSMFSolver:
    """
    GPU-accelerated CSMF algorithm for multiple datasets.
    
    Decomposes K related data matrices into common and specific patterns.
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
        n_iter_outer: int = 50,
        n_iter_inner: int = 50,
        tol: float = 1e-6,
        verbose: int = 0,
        show_progress: bool = False,
        show_progress_inner: bool = False
    ) -> Dict:
        """
        Fit CSMF model to multiple datasets.
        
        Parameters
        ----------
        X_list : List
            List of data matrices (each can be dense or sparse)
        rank_common : int
            Rank for common factors
        rank_specific : int or list
            Rank for each dataset's specific factors
        n_iter_outer : int
            Number of outer CSMF iterations (default: 50)
        n_iter_inner : int
            Number of inner NMF iterations per factorization.
            GPU uses Nesterov acceleration like CPU, so 50 is sufficient.
        tol : float
            Convergence tolerance
        verbose : int
            Verbosity level (0, 1, or 2)
        show_progress : bool
            Whether to show the outer CSMF tqdm progress bar (requires `tqdm`)
        show_progress_inner : bool
            Whether to show the inner NMF tqdm progress bar inside each update (requires `tqdm`)
            
        Returns
        -------
        result : dict
            Contains: W_c, W_s, H_c, H_s, history, total_time
        """
        
        start_time = time.time()
        K = len(X_list)
        m = X_list[0].shape[0]
        
        if verbose:
            print(f"\n=== GPU CSMF ===")
            print(f"Datasets: {K}")
            print(f"Features: {m}")
            print(f"Common rank: {rank_common}")
        
        # Handle rank_specific
        if isinstance(rank_specific, int):
            rank_specific = [rank_specific] * K
        elif len(rank_specific) != K:
            raise ValueError(f"rank_specific length must match number of datasets")
        
        # Initialize factors (matching CPU version - proper initialization)
        W_c = np.random.rand(m, rank_common)  # Initialize normally, not * 0.01
        W_s = [np.random.rand(m, rank_specific[k]) for k in range(K)]  # Initialize normally
        
        # Initialize H factors in structured form (like CPU version)
        # H_c should have shape (rank_common, n_samples_k) for each dataset k
        H_c_list = []
        for k in range(K):
            n_k = X_list[k].shape[1]
            H_c_list.append(np.random.rand(rank_common, n_k))
        
        # H_s should have shape (rank_specific[k], n_samples_k) for each dataset k
        H_s_list = []
        for k in range(K):
            n_k = X_list[k].shape[1]
            H_s_list.append(np.random.rand(rank_specific[k], n_k))
        
        history = {'obj': [], 'time': [], 'per_dataset_obj': []}
        
        # Main CSMF loop
        use_outer_bar = show_progress and tqdm is not None
        outer_bar = tqdm(range(n_iter_outer), desc="CSMF outer", leave=False) if use_outer_bar else None
        outer_iterator = outer_bar if outer_bar is not None else range(n_iter_outer)

        for outer_iter in outer_iterator:
            iter_start = time.time()
            
            # Step 1: Update common components
            residuals = []
            for k in range(K):
                X_k = X_list[k]
                # Convert sparse matrix to dense
                if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                    X_k = X_k.toarray()
                elif not isinstance(X_k, np.ndarray):
                    X_k = np.asarray(X_k)
                
                res = np.maximum(X_k - W_s[k] @ H_s_list[k] if len(H_s_list) > k else X_k, 0)
                residuals.append(res)
            
            # Concatenate residuals
            CX = np.hstack(residuals)
            
            # Concatenate H_c for initialization
            Hc_concat = np.hstack(H_c_list)
            
            # Factorize concatenated residuals (with proper initialization like CPU)
            W_c, Hc_concat = self.nmf_solver.nmf(
                CX, rank=rank_common, max_iter=n_iter_inner,
                verbose=bool(verbose), progress=show_progress_inner,
                w_init=W_c, h_init=Hc_concat  # PASS PREVIOUS SOLUTIONS
            )
            
            # Split H_c
            H_c_list = []
            col_idx = 0
            for k in range(K):
                n_k = X_list[k].shape[1]
                H_c_list.append(Hc_concat[:, col_idx:col_idx + n_k])
                col_idx += n_k
            
            # Step 2: Update specific components
            for k in range(K):
                X_k = X_list[k]
                # Convert sparse matrix to dense
                if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                    X_k = X_k.toarray()
                elif not isinstance(X_k, np.ndarray):
                    X_k = np.asarray(X_k)
                
                residual = np.maximum(X_k - W_c @ H_c_list[k], 0)
                
                # Get previous factors for initialization
                W_s_prev = W_s[k]
                H_s_prev = H_s_list[k]
                
                W_s_k, H_s_k = self.nmf_solver.nmf(
                    residual, rank=rank_specific[k], max_iter=n_iter_inner,
                    verbose=bool(verbose), progress=show_progress_inner,
                    w_init=W_s_prev, h_init=H_s_prev  # PASS PREVIOUS SOLUTIONS
                )
                
                W_s[k] = W_s_k
                H_s_list[k] = H_s_k
            
            # Normalize - COMPETITIVE normalization (same as CPU)
            # Concatenate W and H exactly like CPU for competitive normalization
            W_combined = np.hstack([W_c] + W_s)  # [W_c | W_s[0] | W_s[1] | ... | W_s[K-1]]
            
            col_sums = np.sum(W_combined, axis=0)
            col_sums[col_sums == 0] = 1
            W_combined_norm = W_combined / col_sums
            
            # Split back
            col_offset = 0
            W_c = W_combined_norm[:, col_offset:col_offset + rank_common]
            W_c_col_sums = col_sums[:rank_common]
            col_offset += rank_common
            
            # Scale H_c factors (they share the same col_sums)
            for h_c in H_c_list:
                h_c *= W_c_col_sums[:, np.newaxis]
            
            # Scale W_s and their H factors
            for k in range(K):
                W_s[k] = W_combined_norm[:, col_offset:col_offset + rank_specific[k]]
                W_s_col_sums_k = col_sums[col_offset:col_offset + rank_specific[k]]
                H_s_list[k] *= W_s_col_sums_k[:, np.newaxis]
                col_offset += rank_specific[k]
            
            # Compute objective
            obj = 0
            per_dataset_errors = []
            for k in range(K):
                X_k = X_list[k]
                # Convert sparse matrix to dense
                if isinstance(X_k, sp.csr_matrix) or isinstance(X_k, sp.coo_matrix) or isinstance(X_k, sp.csc_matrix):
                    X_k = X_k.toarray()
                elif not isinstance(X_k, np.ndarray):
                    X_k = np.asarray(X_k)

                recon = W_c @ H_c_list[k] + W_s[k] @ H_s_list[k]
                diff = X_k - recon
                err = np.sum(diff ** 2)
                obj += err
                per_dataset_errors.append(err)
            
            elapsed = time.time() - iter_start
            history['obj'].append(obj)
            history['per_dataset_obj'].append(per_dataset_errors)
            history['time'].append(elapsed)
            
            if verbose and (outer_iter + 1) % 5 == 0:
                per_str = ', '.join(f"{err:.3e}" for err in per_dataset_errors)
                print(f"Iteration {outer_iter + 1:3d}: obj = {obj:.6e}, time = {elapsed:.3f}s", end='')
                print(f"; per dataset: [{per_str}]")
        
        if outer_bar is not None:
            outer_bar.close()

        total_time = time.time() - start_time
        
        if verbose:
            print(f"Total time: {total_time:.3f}s")
        
        return {
            'W_c': W_c,
            'W_s': W_s,
            'H_c': H_c_list,
            'H_s': H_s_list,
            'history': history,
            'total_time': total_time
        }


def gpu_csmf(
    X_list: List[Union[np.ndarray, sp.csr_matrix]],
    rank_common: int,
    rank_specific: Union[int, List[int]],
    device: Optional[torch.device] = None,
    verbose: int = 0,
    show_progress: bool = False,
    show_progress_inner: bool = False,
    **kwargs
) -> Dict:
    """
    High-level API for GPU-accelerated CSMF.
    
    Parameters
    ----------
    X_list : List
        List of data matrices (each can be dense or sparse)
    rank_common : int
        Rank for common factors
    rank_specific : int or list
        Rank(s) for specific factors
    device : torch.device, optional
        GPU device (auto-selected if None)
    verbose : int
        Verbosity level
    show_progress : bool
        Whether to show the outer CSMF tqdm progress bars (requires `tqdm`)
    show_progress_inner : bool
        Whether to show the inner NMF progress bars (each call to `GPUNeNMFSolver.nmf`)
    **kwargs
        Additional arguments passed to GPUCSMFSolver.fit
        
    Returns
    -------
    result : dict
        Factorization results
    """
    
    K = len(X_list)
    
    if isinstance(rank_specific, int):
        rank_specific = [rank_specific] * K
    elif len(rank_specific) != K:
        raise ValueError(f"rank_specific length must match number of datasets")
    
    config = GPUConfig(device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if verbose:
        config.info()
    
    solver = GPUCSMFSolver(config)
    result = solver.fit(
        X_list,
        rank_common=rank_common,
        rank_specific=rank_specific,
        verbose=verbose,
        show_progress=show_progress,
        show_progress_inner=show_progress_inner,
        **kwargs
    )
    
    return result
