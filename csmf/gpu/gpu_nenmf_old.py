"""
GPU-Accelerated Nesterov NMF Solver

This module provides a high-performance GPU implementation of the core
Nesterov-accelerated NMF algorithm using PyTorch.

Used as the foundation for GPU-accelerated CSMF, iNMF, and jNMF algorithms.
"""

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Union
import time
from .config import GPUConfig
from .utils import ensure_torch_tensor, ensure_numpy_array, SparseMatrixHandler


class GPUNeNMFSolver:
    """
    GPU-accelerated NMF solver using multiplicative update rules.
    
    Handles both dense and sparse matrices efficiently with automatic
    GPU/CPU fallback.
    """
    
    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self.config = gpu_config or GPUConfig()
        self.device = self.config.device
    
    def nmf(
        self,
        V: Union[np.ndarray, sp.csr_matrix, torch.Tensor],
        rank: int,
        max_iter: int = 100,
        min_iter: int = 2,
        tol: float = 1e-6,
        w_init: Optional[np.ndarray] = None,
        h_init: Optional[np.ndarray] = None,
        verbose: bool = False,
        return_history: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Non-negative Matrix Factorization using multiplicative updates.
        
        Minimizes: ||V - WH||_F^2 subject to W, H >= 0
        
        Parameters
        ----------
        V : array-like
            Data matrix (m × n). Can be dense numpy array or scipy.sparse matrix
        rank : int
            Number of latent factors
        max_iter : int
            Maximum iterations
        min_iter : int
            Minimum iterations before convergence check
        tol : float
            Convergence tolerance
        w_init : np.ndarray, optional
            Initial basis matrix
        h_init : np.ndarray, optional
            Initial coefficient matrix
        verbose : bool
            Print progress
        return_history : bool
            Return convergence history
            
        Returns
        -------
        W : np.ndarray
            Basis matrix (m × rank)
        H : np.ndarray
            Coefficient matrix (rank × n)
        history : dict (optional)
            Convergence history with 'rel_error' and 'times'
        """
        
        start_time = time.time()
        
        # Convert to torch tensor
        m, n = V.shape if hasattr(V, 'shape') else (None, None)
        V_data = ensure_torch_tensor(V, self.device)
        m, n = V_data.shape
        
        frobenius_norm_V = torch.norm(V_data, 'fro').item()
        
        # Initialize W and H with float32 for speed
        if w_init is not None:
            W = torch.from_numpy(w_init.astype(np.float32)).to(self.device)
        else:
            W = torch.rand(m, rank, device=self.device, dtype=torch.float32)  # [0, 1)
        
        if h_init is not None:
            H = torch.from_numpy(h_init.astype(np.float32)).to(self.device)
        else:
            H = torch.rand(rank, n, device=self.device, dtype=torch.float32)  # [0, 1)
        
        # Ensure non-negativity
        W = torch.abs(W)
        H = torch.abs(H)
        
        history = {'rel_error': [], 'times': []}
        epsilon = 1e-7  # Float32-appropriate epsilon
        
        # Main NMF loop with multiplicative updates
        for iteration in range(max_iter):
            # Update H: H ← H ⊙ (W^T V) ⊘ (W^T W H + ε)
            WtV = W.T @ V_data  # (rank × n)
            WtW = W.T @ W  # (rank × rank)
            WtWH = WtW @ H  # (rank × n)
            
            # Multiplicative update for H
            H = H * (WtV / (WtWH + epsilon))
            H = torch.clamp(H, min=0)
            
            # Update W: W ← W ⊙ (V H^T) ⊘ (W H H^T + ε)
            VHt = V_data @ H.T  # (m × rank)
            HHt = H @ H.T  # (rank × rank)
            WHHt = W @ HHt  # (m × rank)
            
            # Multiplicative update for W
            W = W * (VHt / (WHHt + epsilon))
            W = torch.clamp(W, min=0)
            
            # Check convergence every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration < min_iter:
                recon_error = torch.norm(V_data - W @ H, 'fro')
                error = (recon_error / frobenius_norm_V).item()
                
                history['rel_error'].append(error)
                history['times'].append(time.time() - start_time)
                
                if verbose and ((iteration + 1) % 10 == 0):
                    print(f"Iter {iteration + 1:4d}: rel_error = {error:.6e}")
                
                # Early stopping
                if iteration >= min_iter and len(history['rel_error']) >= 2:
                    if abs(history['rel_error'][-1] - history['rel_error'][-2]) < tol:
                        if verbose:
                            print(f"Converged at iteration {iteration + 1}")
                        break
        
        # Convert back to numpy
        W_np = ensure_numpy_array(W)
        H_np = ensure_numpy_array(H)
        
        # Ensure minimum value to prevent exact zeros
        W_np = np.clip(W_np, 1e-6, None)
        H_np = np.clip(H_np, 1e-6, None)
        
        # Convert back to numpy with float64
        W_np = ensure_numpy_array(W.to(dtype=torch.float64))
        H_np = ensure_numpy_array(H.to(dtype=torch.float64))
        
        # Ensure minimum value to prevent exact zeros (use float64-appropriate minimum)
        W_np = np.clip(W_np, 1e-12, None)
        H_np = np.clip(H_np, 1e-12, None)
        
        if return_history:
            return W_np, H_np, history
        else:
            return W_np, H_np
