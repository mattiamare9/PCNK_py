"""
Uniform ISF Kernel package.

This module provides:
- FixedUniformISFKernel: kernel with no trainable σ
- ScaledUniformISFKernel: kernel with trainable σ (torch.nn.Parameter)
- uniform_kernel: factory to dispatch between the two
- evaluate_uniform_kernel_numpy / evaluate_uniform_kernel_torch: evaluation functions

maybe.. - promote_dtype: dtype promotion helper (mostly redundant in Python)
THEN UNCOMMENT
"""

from .implementation import (
    UniformISFKernelBase,
    FixedUniformISFKernel,
    ScaledUniformISFKernel,
    uniform_kernel,
)
from .evaluation import (
    evaluate_uniform_kernel_numpy,
    evaluate_uniform_kernel_torch,
)
#from .promotion import promote_dtype

__all__ = [
    "UniformISFKernelBase",
    "FixedUniformISFKernel",
    "ScaledUniformISFKernel",
    "uniform_kernel",
    "evaluate_uniform_kernel_numpy",
    "evaluate_uniform_kernel_torch",
    "promote_dtype",
]
