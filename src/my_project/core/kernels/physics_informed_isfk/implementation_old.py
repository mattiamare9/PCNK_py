from __future__ import annotations
from typing import Optional, Sequence, Union

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

def _to_row_major(x: torch.Tensor) -> torch.Tensor:
    """
    Convert to row-major (B, 3):
      (3,)     -> (1, 3)
      (3, B)   -> (B, 3)
      (B, 3)   -> unchanged
    """
    if x.ndim == 1:
        assert x.numel() == 3, "Expected a length-3 vector."
        return x.view(1, 3)
    if x.shape[-1] == 3:
        return x  # already (B,3)
    # assume column-major (3,B)
    assert x.shape[0] == 3, "Expected shape (3, B) or (B, 3)."
    return x.mT  # -> (B,3)


def _to_col_major(x: torch.Tensor) -> torch.Tensor:
    """
    Convert to column-major (3, B):
      (3,)     -> (3, 1)
      (B, 3)   -> (3, B)
      (3, B)   -> unchanged
    """
    if x.ndim == 1:
        assert x.numel() == 3, "Expected a length-3 vector."
        return x.view(3, 1)
    if x.shape[0] == 3:
        return x  # already (3,B)
    # assume row-major (B,3)
    assert x.shape[-1] == 3, "Expected shape (B, 3) or (3, B)."
    return x.mT  # -> (3,B)


def _combine_outputs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add analytical + neural outputs. For a single input, both should be scalars;
    for batches, both should be (B,). This function just returns a + b and,
    in the rare case a 1-D single-input tensor sneaks through, reduces it.
    """
    out = a + b
    if out.ndim == 1 and out.numel() == 1:
        return out.squeeze(0)
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
        Shared evaluation that feeds:
        - row-major inputs into the analytical kernel
        - col-major inputs into the neural kernel
        Ensures both produce matching output shapes.
        """
        if x2 is None:
            # Single-input mode
            x1_row = _to_row_major(x1)
            x1_col = _to_col_major(x1)
            a = self.analytical(x1_row)
            n = self.neural(x1_col)
            out = a + n
            if out.ndim == 1 and out.numel() == 1:
                return out.squeeze(0)
            return out

        # --- Two-input mode (pairwise) ---
        # Compute relative displacement δ = x1 - x2
        # Both analytical and neural kernels expect a single (3,B) or (B,3) input
        delta = x1 - x2

        # Row-major for analytical; column-major for neural
        delta_row = _to_row_major(delta)
        delta_col = _to_col_major(delta)

        a = self.analytical(delta_row)
        n = self.neural(delta_col)

        out = a + n
        if out.ndim == 1 and out.numel() == 1:
            return out.squeeze(0)
        return out



# --------------------------------------------------------------------------
# PlaneWavePINKernel  (Uniform + Neural residual)
# --------------------------------------------------------------------------
class PlaneWavePINKernel(CompositeKernelBase):
    """
    Physics-informed kernel: analytical (uniform ISF) + neural plane-wave residual.
    Mirrors Julia’s PlaneWavePINKernel.
    """

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

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._eval(x1, x2)

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
# DirectedResidualPINKernel (Directional + Neural residual)
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

    def forward(self, x1: torch.Tensor, x2: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._eval(x1, x2)

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
