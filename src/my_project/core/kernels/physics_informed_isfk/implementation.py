# src/my_project/core/kernels/physics_informed_isfk/implementation.py

from __future__ import annotations
from typing import Optional, Union

import torch
import torch.nn as nn

# Reuse existing kernel factories
from my_project.core.kernels.uniform_isfk import ScaledUniformISFKernel
from my_project.core.kernels.plane_wave_isfk import plane_wave_kernel
from my_project.core.kernels.directional_isfk import directional_sf_kernel

_DTypeLike = Union[torch.dtype, None]
_DeviceLike = Union[torch.device, str, None]

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _squeeze_2d_to_1d(x: torch.Tensor) -> torch.Tensor:
    """If a kernel returns (1,B) or (B,1), squeeze to (B,)."""
    if x.ndim == 2 and 1 in x.shape:
        return x.reshape(-1)
    return x


def _combine_outputs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Sum analytical + neural parts, promoting dtype (e.g., float+complex → complex),
    aligning device, and normalizing accidental singleton shapes:
      - scalar stays scalar
      - (B,) stays (B,)
      - (B1,B2) stays (B1,B2)
    """
    a = _squeeze_2d_to_1d(a)
    b = _squeeze_2d_to_1d(b)

    # Promote dtype
    target_dtype = torch.result_type(a, b)
    if a.dtype != target_dtype:
        a = a.to(target_dtype)
    if b.dtype != target_dtype:
        b = b.to(target_dtype)

    # Align device
    if a.device != b.device:
        b = b.to(a.device)

    # Final shape check (we should be adding same-shaped tensors)
    if a.shape != b.shape:
        raise RuntimeError(
            f"Composite parts shape mismatch: analytical {tuple(a.shape)} vs neural {tuple(b.shape)}"
        )

    out = a + b

    # Normalize stray (1,) to scalar
    if out.ndim == 1 and out.numel() == 1:
        out = out.squeeze(0)
    return out


# --------------------------------------------------------------------------
# Base
# --------------------------------------------------------------------------
class CompositeKernelBase(nn.Module):
    """Base for physics-informed composite kernels combining analytical + neural parts."""
    analytical: nn.Module
    neural: nn.Module

    def __init__(self) -> None:
        super().__init__()

    def _eval(self, x1: torch.Tensor, x2: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Pass inputs straight through; sub-kernels already handle shape/diff logic.
        DO NOT compute x1 - x2 here (pairwise shapes (B1,B2) would break).
        """
        if x2 is None:
            a = self.analytical(x1)
            n = self.neural(x1)
        else:
            a = self.analytical(x1, x2)
            n = self.neural(x1, x2)
        return _combine_outputs(a, n)

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._eval(x1, x2)



# --------------------------------------------------------------------------
# PlaneWavePINKernel  (Uniform + Neural residual)
# --------------------------------------------------------------------------
class PlaneWavePINKernel(CompositeKernelBase):
    def __init__(
        self,
        k: float,
        *,
        ord: int,
        W: Optional[nn.Module] = None,
        dtype: _DTypeLike = None,
        device: _DeviceLike = None,
    ) -> None:
        super().__init__()
        self.analytical = ScaledUniformISFKernel(k, sigma=1.0, dtype=dtype, device=device)
        self.neural = plane_wave_kernel(k, ord=ord, W=W, dtype=dtype, device=device)

    def __repr__(self) -> str:  # pragma: no cover
        return f"PlaneWavePINKernel(analytical={self.analytical!r}, neural={self.neural!r})"


def plane_wave_pin_kernel(
    k: float,
    *,
    ord: int,
    W: Optional[nn.Module] = None,
    dtype: _DTypeLike = None,
    device: _DeviceLike = None,
) -> PlaneWavePINKernel:
    """Factory mirroring Julia non-generic constructors."""
    return PlaneWavePINKernel(k, ord=ord, W=W, dtype=dtype, device=device)

# --------------------------------------------------------------------------
# directional + residual
# --------------------------------------------------------------------------
class DirectedResidualPINKernel(CompositeKernelBase):
    """
    Physics-informed kernel: directional spherical-Bessel analytical term
    + neural plane-wave residual.
    """

    def __init__(
        self,
        k: float,
        *,
        ord_dir: int,
        ord_res: int,
        W: Optional[nn.Module] = None,
        trainable_direction: bool = False,
        dtype: _DTypeLike = None,
        device: _DeviceLike = None,
    ) -> None:
        super().__init__()
        self.analytical = directional_sf_kernel(
            k,
            ord=ord_dir,
            trainable_direction=trainable_direction,
            dtype=dtype,
            device=device,
        )
        self.neural = plane_wave_kernel(k, ord=ord_res, W=W, dtype=dtype, device=device)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DirectedResidualPINKernel("
            f"analytical={self.analytical.__class__.__name__}, "
            f"neural={self.neural.__class__.__name__})"
        )


def directed_residual_pin_kernel(
    k: float,
    *,
    ord_dir: int,
    ord_res: int,
    W: Optional[nn.Module] = None,
    trainable_direction: bool = False,
    dtype: _DTypeLike = None,
    device: _DeviceLike = None,
) -> DirectedResidualPINKernel:
    """Factory mirroring Julia’s convenience ctors."""
    return DirectedResidualPINKernel(
        k,
        ord_dir=ord_dir,
        ord_res=ord_res,
        W=W,
        trainable_direction=trainable_direction,
        dtype=dtype,
        device=device,
    )
