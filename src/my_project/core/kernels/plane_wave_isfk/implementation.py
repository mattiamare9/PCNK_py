"""
Plane-wave ISF kernels — implementation
(Port of PCNK/Kernels/PlaneWaveISFK/Implementation.jl)

Classes
-------
- PlaneWaveKernelBase
- FixedDirectionPlaneWaveKernel      (discrete weights σ trainable, directions v fixed)
- TrainableDirectionPlaneWaveKernel  (σ trainable, v trainable)
- NeuralWeightPlaneWaveKernel        (neural weight W, σ vector for normalization/mixture)

Factory
-------
- plane_wave_kernel(k, ord=None, v=None, sigma=None, W=None, trainable_direction=False, ...)

TODO (next file): wire .forward() to evaluation routines when PlaneWave/Evaluation.jl is ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from torch import nn
from my_project.utils.quadrature import lebedev_by_order

from .evaluation import (
    eval_discrete_kernel_single,
    eval_neural_kernel_single,
    eval_discrete_kernel_pair,
    eval_neural_kernel_pair,
)


# ---- dtype/device helpers ----------------------------------------------------

_DTypeLike = Union[torch.dtype, str]


def _coerce_dtype(dtype: Optional[_DTypeLike]) -> torch.dtype:
    if dtype is None:
        return torch.float64
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        s = dtype.lower()
        mapping = {
            "float64": torch.float64, "double": torch.float64, "torch.float64": torch.float64,
            "float32": torch.float32, "float": torch.float32, "torch.float32": torch.float32,
            "float16": torch.float16, "half": torch.float16, "torch.float16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "torch.bfloat16": torch.bfloat16,
        }
        return mapping.get(s, torch.float64)
    return torch.float64


# ---- external dependency (provide in misc.quadrature) ------------------------

# def lebedev_by_order(ord_: int, *, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Placeholder signature for lebedev_by_order.
#     Should return (x, y, z, w) as 1D tensors of length N.

#     Implement this in e.g. `my_project.core.misc.quadrature`
#     and replace this import with:
#         from my_project.core.misc.quadrature import lebedev_by_order
#     """
#     raise NotImplementedError("lebedev_by_order(ord) not implemented yet.")


# ---- utilities ----------------------------------------------------------------

def _normalize_columns(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize 3xN direction matrix columns to unit length.
    v: (..., 3, N)
    """
    norms = torch.clamp(torch.linalg.vector_norm(v, dim=-2, keepdim=True), min=eps)
    return v / norms


def _as_tensor1d(x, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t.reshape(-1)


# ---- base class ----------------------------------------------------------------

class PlaneWaveKernelBase(nn.Module):
    """
    Base class for plane-wave kernels.

    Parameters
    ----------
    k : float
        Wavenumber.
    dtype, device : torch dtype/device
    """
    k: torch.Tensor
    def __init__(
        self,
        k: float,
        *,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        dtype_t = _coerce_dtype(dtype)
        dev = torch.device(device) if device is not None else None
        k_tensor = torch.as_tensor(float(k), dtype=dtype_t, device=dev)
        self.register_buffer("k", k_tensor, persistent=True)  # self.k is definitely a tensor

    @property
    def dtype(self) -> torch.dtype:
        return torch.as_tensor(self.k).dtype  # 👈 force tensor

    @property
    def device(self) -> torch.device:
        return torch.as_tensor(self.k).device  # 👈 force tensor

# ---- fixed directions, trainable weights --------------------------------------

class FixedDirectionPlaneWaveKernel(PlaneWaveKernelBase):
    """
    Discrete-weight plane-wave kernel:
    - directions v are fixed (3xN)
    - σ (N,) is trainable
    """
    v: torch.Tensor
    def __init__(
        self,
        k: float,
        v: torch.Tensor,                   # shape (3, N)
        sigma: Optional[Union[float, list, torch.Tensor]] = None,  # None|float|1D array
        *,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(k, dtype=dtype, device=device)
        # store directions as a fixed buffer (normalized)
        v_t = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        assert v_t.ndim == 2 and v_t.shape[0] == 3, "v must be (3, N)"
        v_t = _normalize_columns(v_t)
        self.register_buffer("v", v_t, persistent=True)

        N = self.v.shape[1]
        if sigma is None:
            # Julia path: if σ is nothing and constructed from Lebedev, σ ← w (Lebedev weights)
            # Here default to uniform weights if not provided externally (lebedev path handled in factory).
            sigma_t = torch.full((N,), 1.0 / N, dtype=self.dtype, device=self.device)
        elif isinstance(sigma, (float, int)):
            # σ is a scalar → scaled uniform Dirichlet-like (sum to σ)
            sigma_t = torch.full((N,), abs(float(sigma)) / N, dtype=self.dtype, device=self.device)
        else:
            sigma_t = _as_tensor1d(sigma, dtype=self.dtype, device=self.device)
            if sigma_t.numel() != N:
                raise ValueError("Length of sigma must match number of directions.")
            sigma_t = sigma_t.abs()

        self.sigma = nn.Parameter(sigma_t, requires_grad=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k.item():.6g}, N={self.v.shape[1]}, dtype={self.k.dtype}, device={self.k.device})"
    
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x2 is None:
            return eval_discrete_kernel_single(self, x1)
        return eval_discrete_kernel_pair(self, x1, x2)



# ---- trainable directions and weights -----------------------------------------

class TrainableDirectionPlaneWaveKernel(PlaneWaveKernelBase):
    """
    Discrete-weight plane-wave kernel with trainable directions and weights:
    - v (3, N) is trainable
    - σ (N,) is trainable
    """
    v: torch.Tensor
    def __init__(
        self,
        k: float,
        v: torch.Tensor,    # shape (3, N)
        sigma: Optional[Union[float, list, torch.Tensor]] = None,
        *,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(k, dtype=dtype, device=device)

        v_t = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        assert v_t.ndim == 2 and v_t.shape[0] == 3, "v must be (3, N)"
        v_t = _normalize_columns(v_t)
        self.v = nn.Parameter(v_t, requires_grad=True)

        N = self.v.shape[1]
        if sigma is None:
            sigma_t = torch.full((N,), 1.0 / N, dtype=self.dtype, device=self.device)
        elif isinstance(sigma, (float, int)):
            sigma_t = torch.full((N,), abs(float(sigma)) / N, dtype=self.dtype, device=self.device)
        else:
            sigma_t = _as_tensor1d(sigma, dtype=self.dtype, device=self.device)
            if sigma_t.numel() != N:
                raise ValueError("Length of sigma must match number of directions.")
            sigma_t = sigma_t.abs()
        self.sigma = nn.Parameter(sigma_t, requires_grad=True)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k.item():.6g}, N={self.v.shape[1]}, dtype={self.k.dtype}, device={self.k.device})"

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Trainable directions/weights → same dispatch as discrete
        if x2 is None:
            return eval_discrete_kernel_single(self, x1)
        return eval_discrete_kernel_pair(self, x1, x2)

# ---- neural weight kernel ------------------------------------------------------

def _default_weight_net(in_features=3, hidden=2, out_features=1, dtype=torch.float64) -> nn.Sequential:
    # Mirrors the compact default in the Julia code: Chain(Dense(3,2,tanh), Dense(2,1,tanh), softplus)
    return nn.Sequential(
        nn.Linear(in_features, hidden, bias=True),
        nn.Tanh(),
        nn.Linear(hidden, out_features, bias=True),
        nn.Tanh(),
        nn.Softplus(),
    )


class NeuralWeightPlaneWaveKernel(PlaneWaveKernelBase):
    """
    Neural-weight plane-wave kernel:
    - directions v fixed (3, N)
    - neural weight network W: R^3 -> R_+ (softplus at the end)
    - σ : optional vector used as baseline scaling / prior (kept non-trainable here; W is trainable)

    In Julia: `trainable(a::NeuralWeightPlaneWaveKernel) = (; W = trainable(a.W))`
    """
    v: torch.Tensor

    def __init__(
        self,
        k: float,
        v: torch.Tensor,                        # (3, N)
        sigma: Optional[Union[float, list, torch.Tensor]] = None,  # optional baseline weights/prior
        W: Optional[nn.Module] = None,
        *,
        dtype: Optional[_DTypeLike] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(k, dtype=dtype, device=device)

        v_t = torch.as_tensor(v, dtype=self.dtype, device=self.device)
        assert v_t.ndim == 2 and v_t.shape[0] == 3, "v must be (3, N)"
        v_t = _normalize_columns(v_t)
        self.register_buffer("v", v_t, persistent=True)

        N = self.v.shape[1]
        if sigma is None:
            sigma_t = torch.full((N,), 1.0 / N, dtype=self.dtype, device=self.device)
        elif isinstance(sigma, (float, int)):
            sigma_t = torch.full((N,), abs(float(sigma)) / N, dtype=self.dtype, device=self.device)
        else:
            sigma_t = _as_tensor1d(sigma, dtype=self.dtype, device=self.device)
            if sigma_t.numel() != N:
                raise ValueError("Length of sigma must match number of directions.")
            sigma_t = sigma_t.abs()
        self.register_buffer("sigma", sigma_t, persistent=True)  # non-trainable baseline like Julia

        self.W = W if W is not None else _default_weight_net(dtype=self.k.dtype)  # trainable network

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k.item():.6g}, N={self.v.shape[1]}, W={self.W.__class__.__name__}, dtype={self.k.dtype}, device={self.k.device})"

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x2 is None:
            return eval_neural_kernel_single(self, x1)
        return eval_neural_kernel_pair(self, x1, x2)
    
# ---- factory (mirrors Julia's PlaneWaveKernel(...) constructors) ---------------

def plane_wave_kernel(
    k: float,
    *,
    ord: Optional[int] = None,
    v: Optional[Union[list, torch.Tensor]] = None,  # (3, N)
    sigma: Optional[Union[float, list, torch.Tensor]] = None,
    W: Optional[nn.Module] = None,
    trainable_direction: bool = False,
    dtype: Optional[_DTypeLike] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> PlaneWaveKernelBase:
    """
    Dispatch:
      - If W is None and trainable_direction:       TrainableDirectionPlaneWaveKernel
      - If W is None and not trainable_direction:   FixedDirectionPlaneWaveKernel
      - If W is not None and not trainable_direction: NeuralWeightPlaneWaveKernel
      - Else: error (neural weight with trainable directions is disallowed, as in Julia)

    Either provide `ord` (Lebedev order) OR explicit direction matrix `v` (3, N).

    Returns
    -------
    PlaneWaveKernelBase
    """
    dtype_t = _coerce_dtype(dtype)
    dev = torch.device(device) if device is not None else torch.device('cpu')  # Default to CPU

    if (ord is None) == (v is None):
        raise ValueError("Provide exactly one of `ord` or `v` (not both, not none).")

    if ord is not None:
        # Build directions/weights from Lebedev
        x, y, z, w = lebedev_by_order(ord, dtype=dtype_t, device=dev)
        V = torch.stack([x, y, z], dim=0)  # (3, N)
        # normalize columns (Julia does v ./ sum(abs2, v, dims=1) then unit-norm)
        V = _normalize_columns(V)
        if sigma is None:
            sigma = w  # default to Lebedev weights if not provided
    else:
        V = torch.as_tensor(v, dtype=dtype_t, device=dev)
        if V.ndim != 2 or V.shape[0] != 3:
            raise ValueError("`v` must be a (3, N) matrix of directions.")

    if W is None and trainable_direction:
        return TrainableDirectionPlaneWaveKernel(k, V, sigma=sigma, dtype=dtype_t, device=dev)
    elif W is None and not trainable_direction:
        return FixedDirectionPlaneWaveKernel(k, V, sigma=sigma, dtype=dtype_t, device=dev)
    elif W is not None and not trainable_direction:
        return NeuralWeightPlaneWaveKernel(k, V, sigma=sigma, W=W, dtype=dtype_t, device=dev)
    else:
        raise ValueError(
            "Neural weight kernel does not admit trainable directions. "
            "Use a fixed direction grid (e.g., Lebedev) when W is provided."
        )
