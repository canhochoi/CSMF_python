"""
GPU Configuration Management

This module handles GPU detection, device configuration, and related utilities.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUConfig:
    """GPU configuration and device management"""
    device: torch.device = None
    use_fp16: bool = False  # Use float16 for memory savings
    sparse_threshold: float = 0.5  # Use sparse if sparsity > this
    
    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def info(self):
        """Print GPU information"""
        if self.device.type == 'cuda':
            print(f"GPU Device: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("Using CPU (CUDA not available)")
