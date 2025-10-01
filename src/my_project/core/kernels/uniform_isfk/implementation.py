"""
Implementation of the Uniform ISF Kernel
(ported from PCNK/Kernels/UniformISFK/Implementation.jl)

Julia summary:
- abstract type UniformKernel{T} <: ISFKernel{T}
- fixed_UniformKernel{T}(k)   -> no trainable params
- scaled_UniformKernel{T}(k; σ=1.0) -> trainable multiplicative factor (vector)
- UniformKernel(k; σ=nothing) -> dispatch to fixed or scaled

Pythonic design:
- UniformISFKernelBase (nn.Module)
- FixedUniformISFKernel(k, dtype, device)
- ScaledUniformISFKernel(k, sigma=1.0, dtype, device)  [sigma is trainable Parameter([σ])]
- uniform_kernel(k, sigma=None, dtype=torch.float64, device=None) factory

Evaluation is intentionally deferred to `evaluation.py`.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn


_DTypeLike = Union[torch.dtype, str]


def _coerce_dtype(dtype: Optional[_DTypeLike]) -> torch.dtype:
    if dtype is None:
        return torch.float64  # mirror Julia Float64 default
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        s = dtype.lower()
        if s in {"float64", "double", "torch.float64"}:
            return torch.float64
        if s in {"float32", "float", "torch.float32"}:
            return torch.float32
        if s in {"float16", "half", "torch.float16"}:
            return torch.float16
        if s in {"bfloat16", "bf16", "torch.bfloat16"}:
            return torch.bfloat16
    # sensible default
    return torch.float64


class UniformISFKernelBase(nn.Module):
    """
    Base class for Uniform ISF kernels.

    Parameters
    ----------
    k : float
        Wavenumber (not trainable).
    dtype : torch.dtype or str, optional
        Numeric precision (default: torch.float64).
    device : torch.device or str, optional
        Device placement.
    """

    def __init__(
        self,
        k: float,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        dtype_t = _coerce_dtype(dtype)
        dev = torch.device(device) if device is not None else None
        # store k as a non-trainable buffer to keep it on correct device/dtype
        self.register_buffer("k", torch.as_tensor(float(k), dtype=dtype_t, device=dev), persistent=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(k={self.k.item():.6g}, dtype={self.k.dtype}, device={self.k.device})"

    # We will wire this to evaluation logic in the next step.
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate kernel between point-sets `x` and `y`.

        NOTE: Implemented in the next step by delegating to `evaluation.py`.
        This placeholder exists to keep the API aligned with Julia's Flux.@layer.
        """
        raise NotImplementedError(
            "UniformISFKernel.forward is not implemented yet. "
            "It will delegate to evaluation routines in `evaluation.py`."
        )


from .evaluation import evaluate_uniform_kernel_torch
import torch

class FixedUniformISFKernel(UniformISFKernelBase):
    """
    Uniform ISF kernel with fixed multiplicative factor (no trainable params).
    Mirrors Julia `fixed_UniformKernel`.
    """

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the kernel.

        - If only `x` is given: return j0(k * ||x||) elementwise.
        - If both `x, y` are given: return kernel matrix K[i, j] = j0(k * ||x[i]-y[j]||).
        """
        if y is None:
            # single-input mode: compute norm per column/row/etc.
            r = torch.linalg.norm(x, dim=-1)
            z = self.k * r
            return torch.where(z == 0, torch.ones_like(z), torch.sin(z) / z)
        return evaluate_uniform_kernel_torch(self.k, x, y)


class ScaledUniformISFKernel(UniformISFKernelBase):
    """
    Uniform ISF kernel with trainable multiplicative factor σ.
    In Julia it stores σ as an `AbstractVector`; here we keep a shape-(1,) Parameter.
    """

    def __init__(
        self,
        k: float,
        sigma: float = 1.0,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(k=k, dtype=dtype, device=device)
        sigma_t = torch.tensor([float(sigma)], dtype=self.k.dtype, device=self.k.device)
        self.sigma = nn.Parameter(sigma_t, requires_grad=True)

    @property
    def sigma_scalar(self) -> float:
        """Return σ as a Python float (convenience)."""
        return float(self.sigma.detach().cpu().item())

    def __repr__(self) -> str:
        base = super().__repr__()
        return base[:-1] + f", sigma={self.sigma_scalar:.6g})"

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the kernel.

        - If only `x` is given: return σ * j0(k * ||x||).
        - If both `x, y` are given: return σ * K[i, j].
        """
        if y is None:
            r = torch.linalg.norm(x, dim=-1)
            z = self.k * r
            values = torch.where(z == 0, torch.ones_like(z), torch.sin(z) / z)
            return self.sigma.view(-1)[0] * values
        return evaluate_uniform_kernel_torch(self.k, x, y, sigma=self.sigma)

def uniform_kernel(
    k: float,
    sigma: Optional[float] = None,
    *,
    dtype: Optional[_DTypeLike] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> UniformISFKernelBase:
    """
    Factory function mirroring Julia's `UniformKernel(k; σ=nothing)` dispatch.

    - If `sigma is None` -> FixedUniformISFKernel(k)
    - Else               -> ScaledUniformISFKernel(k, sigma)

    Parameters
    ----------
    k : float
        Wavenumber.
    sigma : Optional[float]
        Multiplicative factor (trainable if provided).
    dtype : Optional[torch.dtype or str]
        Desired dtype (default: float64).
    device : Optional[torch.device or str]
        Device placement.

    Returns
    -------
    UniformISFKernelBase
    """
    if sigma is None:
        return FixedUniformISFKernel(k=k, dtype=dtype, device=device)
    return ScaledUniformISFKernel(k=k, sigma=sigma, dtype=dtype, device=device)
