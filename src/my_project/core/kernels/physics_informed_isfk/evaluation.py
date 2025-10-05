"""
Evaluation for composite (physics-informed) kernels.
These functions simply delegate to the analytical and neural sub-kernels.
"""

import torch

def eval_composite_single(a, x: torch.Tensor) -> torch.Tensor:
    """Single-input evaluation for a composite kernel."""
    return a.analytical(x) + a.neural(x)

def eval_composite_pair(a, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Two-input (kernel matrix) evaluation for a composite kernel."""
    return a.analytical(x1, x2) + a.neural(x1, x2)
