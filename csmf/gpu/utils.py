"""
GPU Utilities for Sparse Matrix Handling

This module provides utilities for converting between scipy.sparse and PyTorch formats,
as well as other GPU-related helper functions.
"""

import torch
import numpy as np
import scipy.sparse as sp
from typing import Union, Tuple
import time


class SparseMatrixHandler:
    """Convert between scipy.sparse and PyTorch sparse tensors"""
    
    @staticmethod
    def scipy_to_torch_sparse(
        sp_matrix: Union[sp.csr_matrix, sp.csc_matrix, sp.coo_matrix],
        device: torch.device = torch.device('cpu')
    ) -> torch.sparse_coo_tensor:
        """Convert scipy sparse to PyTorch sparse COO tensor"""
        if not isinstance(sp_matrix, sp.coo_matrix):
            sp_matrix = sp_matrix.tocoo()
        
        indices = torch.LongTensor(np.vstack((sp_matrix.row, sp_matrix.col)))
        values = torch.FloatTensor(sp_matrix.data)
        shape = sp_matrix.shape
        
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, device=device)
        return sparse_tensor
    
    @staticmethod
    def torch_sparse_to_scipy(
        sparse_tensor: torch.sparse_coo_tensor
    ) -> sp.csr_matrix:
        """Convert PyTorch sparse COO tensor to scipy CSR"""
        indices = sparse_tensor.indices().cpu().numpy()
        values = sparse_tensor.values().cpu().numpy()
        shape = sparse_tensor.shape
        
        coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
        return coo.tocsr()
    
    @staticmethod
    def scipy_to_dense_torch(
        sp_matrix: Union[sp.csr_matrix, sp.csc_matrix],
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """Convert scipy sparse to PyTorch dense tensor"""
        if isinstance(sp_matrix, sp.csr_matrix) or isinstance(sp_matrix, sp.csc_matrix):
            dense = sp_matrix.toarray()
        else:
            dense = sp_matrix.toarray()
        return torch.from_numpy(dense).float().to(device)
    
    @staticmethod
    def torch_dense_to_scipy(
        tensor: torch.Tensor
    ) -> sp.csr_matrix:
        """Convert PyTorch dense tensor to scipy sparse CSR"""
        dense_np = tensor.cpu().detach().numpy()
        sparse = sp.csr_matrix(dense_np)
        return sparse
    
    @staticmethod
    def get_sparsity(
        data: Union[np.ndarray, sp.csr_matrix]
    ) -> float:
        """Get sparsity ratio (0-1) of a matrix"""
        if isinstance(data, sp.csr_matrix):
            total = data.shape[0] * data.shape[1]
            nnz = data.nnz
            return 1.0 - (nnz / total)
        else:
            return np.sum(data == 0) / data.size


def ensure_torch_tensor(
    data: Union[np.ndarray, sp.csr_matrix, torch.Tensor],
    device: torch.device
) -> torch.Tensor:
    """Convert various data types to PyTorch tensor"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, sp.csr_matrix):
        return SparseMatrixHandler.scipy_to_dense_torch(data, device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def ensure_numpy_array(
    tensor: torch.Tensor
) -> np.ndarray:
    """Convert PyTorch tensor to numpy array"""
    return tensor.cpu().detach().numpy()
