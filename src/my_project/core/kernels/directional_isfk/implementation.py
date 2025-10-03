from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from my_project.utils.quadrature import lebedev_by_order
from .evaluation import eval_directional_single, eval_directional_pair


_DTypeLike = Union[torch.dtype, None]
_DeviceLike = Union[torch.device, str, None]


def _normalize_columns(v: torch.Tensor) -> torch.Tensor:
    """
    Normalize columns to unit ℓ2 norm.
    v: (3, N)
    """
    eps = torch.finfo(v.dtype).eps
    nrm = torch.linalg.norm(v, dim=0).clamp_min(eps)  # (N,)
    return v / nrm


class DirectionalSFKernelBase(nn.Module):
    """
    Base class for Directional Spherical-Bessel kernels (biased directions).

    Fields
    ------
    k : scalar tensor (float)
    sigma : (N,) trainable weights
    beta :  (N,) trainable shape/temperature-like params
    v : (3, N) directions (buffer if fixed, Parameter if trainable)
    """

    def __init__(
        self,
        k: float,
        *,
        dtype: _DTypeLike = None,
        device: _DeviceLike = None,
    ) -> None:
        super().__init__()
        # store k as a 0-dim tensor; default dtype float32 unless provided
        if dtype is None:
            dtype = torch.get_default_dtype()
        k_t = torch.tensor(float(k), dtype=dtype, device=device)
        self.register_buffer("k", k_t, persistent=True)

    # NOTE: forward() will be added in evaluation step.
    def __repr__(self) -> str:  # pragma: no cover
        cls = self.__class__.__name__
        k = float(self.k.detach().cpu().item())
        return f"{cls}(k={k:.6g}, dtype={self.k.dtype}, device={self.k.device})"


# ------------------------------- Fixed directions ---------------------------------


class FixedDirectionSFKernel(DirectionalSFKernelBase):
    """
    Spherical-Bessel directional kernel with fixed directions v (3, N).
    Trainable parameters: sigma (N,), beta (N,).
    Mirrors Julia:
      mutable struct FixedDirectionSFKernel{T, V} <: DirectionalSFKernel{T, V}
          k::T; σ::AbstractVector; β::AbstractVector; v::V
      end
      trainable(a) = (; σ = a.σ, a.β)
    """

    v: torch.Tensor  # buffer

    def __init__(
        self,
        k: float,
        v: Union[torch.Tensor, Sequence[Sequence[float]]],  # (3, N) or (3,)
        *,
        sigma: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        beta_init: float = 10.0,
        dtype: _DTypeLike = None,
        device: _DeviceLike = None,
    ) -> None:
        super().__init__(k, dtype=dtype, device=device)

        # directions
        v_t = torch.as_tensor(v, dtype=self.k.dtype, device=self.k.device)
        if v_t.ndim == 1:
            v_t = v_t.view(3, 1)
        assert v_t.shape[0] == 3, "v must have shape (3, N) or be length-3."
        v_t = _normalize_columns(v_t)
        self.register_buffer("v", v_t, persistent=True)

        N = v_t.shape[1]

        # sigma initialization: None -> use Lebedev-style weights if provided externally,
        # here default to uniform over N.
        if sigma is None:
            sigma_t = torch.full((N,), 1.0 / N, dtype=self.k.dtype, device=self.k.device)
        elif isinstance(sigma, (int, float)):
            sigma_t = torch.full((N,), abs(float(sigma)) / N, dtype=self.k.dtype, device=self.k.device)
        else:
            sigma_t = torch.as_tensor(sigma, dtype=self.k.dtype, device=self.k.device).abs()
            if sigma_t.numel() != N:
                raise ValueError("Length of sigma must match number of directions.")
        self.sigma = nn.Parameter(sigma_t, requires_grad=True)

        # beta (trainable), default all = beta_init
        beta_t = torch.full((N,), float(beta_init), dtype=self.k.dtype, device=self.k.device)
        self.beta = nn.Parameter(beta_t, requires_grad=True)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        if x2 is None:
            return eval_directional_single(self.k, x1, self.v, self.sigma, self.beta)
        return eval_directional_pair(self.k, x1, x2, self.v, self.sigma, self.beta)

# ---------------------------- Trainable directions --------------------------------


class TrainableDirectionSFKernel(DirectionalSFKernelBase):
    """
    Spherical-Bessel directional kernel with trainable directions v (3, N).
    Trainable parameters: sigma (N,), beta (N,), v (3,N).
    Mirrors Julia trainable(a) = (; σ = a.σ, β = a.β, v = a.v)
    """

    v: nn.Parameter  # parameter

    def __init__(
        self,
        k: float,
        v: Union[torch.Tensor, Sequence[Sequence[float]]],  # (3, N) or (3,)
        *,
        sigma: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        beta_init: float = 10.0,
        dtype: _DTypeLike = None,
        device: _DeviceLike = None,
    ) -> None:
        super().__init__(k, dtype=dtype, device=device)

        v_t = torch.as_tensor(v, dtype=self.k.dtype, device=self.k.device)
        if v_t.ndim == 1:
            v_t = v_t.view(3, 1)
        assert v_t.shape[0] == 3, "v must have shape (3, N) or be length-3."
        v_t = _normalize_columns(v_t)
        self.v = nn.Parameter(v_t, requires_grad=True)

        N = v_t.shape[1]

        if sigma is None:
            sigma_t = torch.full((N,), 1.0 / N, dtype=self.k.dtype, device=self.k.device)
        elif isinstance(sigma, (int, float)):
            sigma_t = torch.full((N,), abs(float(sigma)) / N, dtype=self.k.dtype, device=self.k.device)
        else:
            sigma_t = torch.as_tensor(sigma, dtype=self.k.dtype, device=self.k.device).abs()
            if sigma_t.numel() != N:
                raise ValueError("Length of sigma must match number of directions.")
        self.sigma = nn.Parameter(sigma_t, requires_grad=True)

        beta_t = torch.full((N,), float(beta_init), dtype=self.k.dtype, device=self.k.device)
        self.beta = nn.Parameter(beta_t, requires_grad=True)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor | None = None) -> torch.Tensor:
        if x2 is None:
            return eval_directional_single(self.k, x1, self.v, self.sigma, self.beta)
        return eval_directional_pair(self.k, x1, x2, self.v, self.sigma, self.beta)

# ------------------------------ Factory helpers -----------------------------------


def _from_lebedev(
    cls,                      # FixedDirectionSFKernel or TrainableDirectionSFKernel
    k: float,
    ord: int,
    *,
    sigma: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    beta_init: float = 10.0,
    dtype: _DTypeLike = None,
    device: _DeviceLike = None,
):
    """
    Build kernel from a Lebedev grid of given degree/order.
    - v is set to normalized directions from (x,y,z)
    - if sigma is None, we use the Lebedev weights w (already scaled to 4π in our wrapper)
    """
    x, y, z, w = lebedev_by_order(order=ord, dtype=(dtype or torch.get_default_dtype()), device=device)
    v = torch.stack([x, y, z], dim=0)  # (3, N)
    v = _normalize_columns(v)

    if sigma is None:
        sigma_init = w  # already on requested dtype/device
    else:
        sigma_init = sigma

    return cls(k, v, sigma=sigma_init, beta_init=beta_init, dtype=dtype, device=device)


def directional_sf_kernel(
    k: float,
    *,
    ord: Optional[int] = None,
    v: Optional[Union[torch.Tensor, Sequence[Sequence[float]], Sequence[float]]] = None,
    sigma: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
    beta_init: float = 10.0,
    trainable_direction: bool = False,
    dtype: _DTypeLike = None,
    device: _DeviceLike = None,
) -> Union[FixedDirectionSFKernel, TrainableDirectionSFKernel]:
    """
    Factory that mirrors Julia's DirectionalSFKernel(...) constructors.

    Options:
    - provide `ord` (Lebedev degree)  -> directions & default weights from Lebedev
    - or provide explicit `v`         -> your directions

    Set `trainable_direction=True` to get TrainableDirectionSFKernel, else fixed.
    """
    cls = TrainableDirectionSFKernel if trainable_direction else FixedDirectionSFKernel

    if ord is not None:
        return _from_lebedev(
            cls, k, ord,
            sigma=sigma,
            beta_init=beta_init,
            dtype=dtype,
            device=device,
        )

    if v is None:
        raise ValueError("You must provide either `ord` (Lebedev degree) or explicit directions `v`.")

    # Accept vector v (length-3) or matrix (3,N)
    v_t = torch.as_tensor(v)
    if v_t.ndim == 1 and v_t.numel() == 3:
        v_use = v_t
    else:
        # Expecting 2D with leading dimension 3; if shape (N,3) transpose.
        if v_t.ndim == 2 and v_t.shape[0] != 3 and v_t.shape[1] == 3:
            v_use = v_t.mT
        else:
            v_use = v_t

    return cls(k, v_use, sigma=sigma, beta_init=beta_init, dtype=dtype, device=device)
