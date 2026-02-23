"""
GPU-Accelerated Nesterov NMF Solver

This module provides a high-performance GPU implementation of the
Nesterov-accelerated NMF algorithm, matching the CPU version for convergence quality.

Used as the foundation for GPU-accelerated CSMF, iNMF, and jNMF algorithms.
"""

import torch
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional, Dict, Union
import time
from .config import GPUConfig
from .utils import ensure_torch_tensor, ensure_numpy_array, SparseMatrixHandler


class GPUNeNMFSolver:
    """
    GPU-accelerated NMF solver using Nesterov-accelerated proximal gradients.
    
    Implements the same algorithm as CPU NNLS with Nesterov momentum,
    providing much better convergence than simple multiplicative updates.
    """
    
    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self.config = gpu_config or GPUConfig()
        self.device = self.config.device
    
    def _nesterov_nnls_step(
        self,
        V: torch.Tensor,
        W: torch.Tensor,
        Z_init: torch.Tensor,
        WtW: Optional[torch.Tensor],
        WtV: Optional[torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        Single Nesterov NNLS optimization step.
        
        Solves: minimize ||V - W @ Z||_F^2 subject to Z >= 0
        
        Uses Nesterov acceleration like CPU version:
        - Compute Lipschitz constant L from W^T W
        - Apply proximal gradient: Z = max(Z - grad/L, 0)
        - Apply Nesterov momentum with α sequence
        """
        
        # Initialize
        Z = Z_init.clone()
        H_prev = Z.clone()
        alpha_prev = 1.0
        
        # Compute Lipschitz constant as spectral norm of W^T W
        if WtW is None:
            WtW = W.T @ W  # (rank × rank)
        
        if WtV is None:
            WtV = W.T @ V  # (rank × n)
        
        # Spectral norm as Lipschitz constant (max eigenvalue)
        eigenvalues = torch.linalg.eigvalsh(WtW)
        L = eigenvalues.max().item()
        L = max(L, 1e-10)  # Prevent division by very small number
        
        # Initial gradient
        Grad = WtW @ Z - WtV  # (rank × n)
        
        # Nesterov NNLS loop
        for inner_iter in range(max_iter):
            Z_prev = Z.clone()
            
            # Proximal gradient step: H = max(Z - grad/L, 0)
            H = torch.clamp(Z - Grad / L, min=0)
            
            # Nesterov momentum update
            alpha_curr = 0.5 * (1.0 + np.sqrt(1.0 + 4 * alpha_prev ** 2))
            Z = H + ((alpha_prev - 1.0) / alpha_curr) * (H - Z_prev)
            alpha_prev = alpha_curr
            
            # Compute new gradient
            Grad = WtW @ Z - WtV
            
            # Simple convergence check
            if inner_iter >= 1:
                Z_change = torch.norm(Z - Z_prev) / (torch.norm(Z_prev) + 1e-10)
                if Z_change < tol:
                    if verbose:
                        print(f"  Converged at inner iteration {inner_iter + 1}")
                    break
        
        return H
    
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
        Non-negative Matrix Factorization using Nesterov acceleration.
        
        Minimizes: ||V - WH||_F^2 subject to W, H >= 0
        
        **Key difference from multiplicative updates:**
        - Uses proximal gradient with Nesterov momentum (like CPU)
        - Much faster convergence
        - Similar quality to CPU version
        
        Parameters
        ----------
        V : array-like
            Data matrix (m × n). Can be dense numpy array or scipy.sparse matrix
        rank : int
            Number of latent factors
        max_iter : int
            Maximum outer iterations
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
        
        # Initialize W and H with float32
        if w_init is not None:
            W = torch.from_numpy(w_init.astype(np.float32)).to(self.device)
        else:
            W = torch.rand(m, rank, device=self.device, dtype=torch.float32)
        
        if h_init is not None:
            H = torch.from_numpy(h_init.astype(np.float32)).to(self.device)
        else:
            H = torch.rand(rank, n, device=self.device, dtype=torch.float32)
        
        # Ensure non-negativity
        W = torch.abs(W)
        H = torch.abs(H)
        
        history = {'rel_error': [], 'times': []}
        
        # Adaptive tolerance like CPU version
        initial_error = torch.norm(V_data - W @ H, 'fro') / frobenius_norm_V
        tol_inner_h = max(tol, 1e-3) * initial_error.item()
        tol_inner_w = tol_inner_h
        
        # Main NMF loop with Nesterov acceleration for each factor
        for iteration in range(max_iter):
            # Optimize H with W fixed using Nesterov NNLS
            WtW = W.T @ W
            WtV = W.T @ V_data
            # Use 10% of max_iter per Nesterov step (similar to CPU's approach)
            nesterov_max_iter = max(50, max_iter // 10)
            H = self._nesterov_nnls_step(V_data, W, H, WtW, WtV, 
                                        max_iter=nesterov_max_iter, tol=tol_inner_h, verbose=False)
            
            # Adapt H tolerance
            if iteration % 10 == 0 and iteration > 0:
                tol_inner_h = tol_inner_h / 10
            
            # Optimize W with H fixed using Nesterov NNLS (via W.T)
            HHt = H @ H.T
            HVt = H @ V_data.T
            W_T = self._nesterov_nnls_step(V_data.T, H.T, W.T, HHt, HVt,
                                          max_iter=nesterov_max_iter, tol=tol_inner_w, verbose=False)
            W = W_T.T
            
            # Adapt W tolerance
            if iteration % 10 == 0 and iteration > 0:
                tol_inner_w = tol_inner_w / 10
            
            # Check convergence every 10 iterations
            if (iteration + 1) % 10 == 0 or iteration < min_iter:
                recon_error = torch.norm(V_data - W @ H, 'fro')
                error = (recon_error / frobenius_norm_V).item()
                
                history['rel_error'].append(error)
                history['times'].append(time.time() - start_time)
                
                if verbose and ((iteration + 1) % 10 == 0):
                    print(f"Iter {iteration + 1:4d}: rel_error = {error:.6e}")
                
                # Early stopping based on relative error change
                if iteration >= min_iter and len(history['rel_error']) >= 2:
                    error_change = abs(history['rel_error'][-1] - history['rel_error'][-2])
                    rel_change = error_change / (history['rel_error'][-2] + 1e-10)
                    if rel_change < tol:
                        if verbose:
                            print(f"Converged at iteration {iteration + 1}")
                        break
        
        # Convert back to numpy
        W_np = ensure_numpy_array(W)
        H_np = ensure_numpy_array(H)
        
        # Ensure minimum value to prevent exact zeros
        W_np = np.clip(W_np, 1e-6, None)
        H_np = np.clip(H_np, 1e-6, None)
        
        if return_history:
            return W_np, H_np, history
        else:
            return W_np, H_np
