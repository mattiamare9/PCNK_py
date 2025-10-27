# src/my_project/core/kernels/plane_wave_isfk/evaluation.py

"""
Evaluation routines for Plane-Wave ISF Kernels
(Port of PCNK/Kernels/PlaneWaveISFK/Evaluation.jl)
"""

import torch
import numpy as np
from my_project.core.background.differences import (
    diff_vec_mat,
    diff_mat_mat,
)


def _ensure_leading3(x: torch.Tensor) -> torch.Tensor:
    """Normalize tensor so that the leading dimension is 3."""
    if x.ndim == 1:
        if x.shape[0] != 3:
            raise ValueError("1D input must be length 3.")
        return x
    if x.ndim == 2:
        if x.shape[0] == 3:
            return x
        if x.shape[1] == 3:
            return x.mT
        raise ValueError("2D input must have a dimension of size 3.")
    if x.ndim == 3:
        if x.shape[0] == 3:
            return x
        if x.shape[2] == 3:
            return x.permute(2, 0, 1)
        raise ValueError("3D input must have one dimension of size 3.")
    raise ValueError("x must be 1D, 2D, or 3D with a dimension of size 3.")


def _get_j(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a correctly typed imaginary unit (1j)."""
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.tensor(1j, dtype=complex_dtype, device=device)


def _eval_single_vector(k: torch.Tensor, x: torch.Tensor,
                        v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    dtype = torch.result_type(k, x)
    k = torch.as_tensor(k, dtype=dtype, device=x.device)
    v = v.to(dtype=x.dtype, device=x.device)
    sigma = sigma.to(dtype=x.dtype, device=x.device)
    kx = k * (v.T @ x)  # (D,)
    j = _get_j(x.dtype, x.device)
    return (sigma * torch.exp(j * kx)).sum()


def _eval_matrix(k: torch.Tensor, X: torch.Tensor,
                 v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    dtype = torch.result_type(k, X)
    k = torch.as_tensor(k, dtype=dtype, device=X.device)
    v = v.to(dtype=X.dtype, device=X.device)
    sigma = sigma.to(dtype=X.dtype, device=X.device)
    kx = k * (v.T @ X)  # (D, B)
    j = _get_j(X.dtype, X.device)
    return (sigma[:, None] * torch.exp(j * kx)).sum(dim=0)


def _eval_tensor3(k: torch.Tensor, X: torch.Tensor,
                  v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    dtype = torch.result_type(k, X)
    k = torch.as_tensor(k, dtype=dtype, device=X.device)
    v = v.to(dtype=X.dtype, device=X.device)
    sigma = sigma.to(dtype=X.dtype, device=X.device)
    B1, B2 = X.shape[1], X.shape[2]
    X_flat = X.reshape(3, -1)  # (3, B1*B2)
    kx = k * (v.T @ X_flat)    # (D, B1*B2)
    j = _get_j(X.dtype, X.device)
    out = (sigma[:, None] * torch.exp(j * kx)).sum(dim=0)
    return out.reshape(B1, B2)


def _isfkernel_eval(k: torch.Tensor, x: torch.Tensor,
                    v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x = _ensure_leading3(x)
    if x.ndim == 1:
        return _eval_single_vector(k, x, v, sigma)
    elif x.ndim == 2:
        return _eval_matrix(k, x, v, sigma)
    elif x.ndim == 3:
        return _eval_tensor3(k, x, v, sigma)
    else:
        raise ValueError("x must be 1D, 2D, or 3D with a dimension of size 3.")


# ---- Public helpers ----------------------------------------------------------

def eval_discrete_kernel_single(a, x: torch.Tensor) -> torch.Tensor:
    return _isfkernel_eval(a.k, x, a.v, a.sigma)


def eval_neural_kernel_single(a, x: torch.Tensor) -> torch.Tensor:
    net_dtype = next(a.W.parameters()).dtype
    kv = (a.k.to(dtype=net_dtype, device=a.v.device) * a.v.to(dtype=net_dtype))
    Wout = a.W(kv.T).squeeze(-1)  # (D,)
    sigma_eff = a.sigma.to(dtype=net_dtype, device=a.v.device) * Wout
    return _isfkernel_eval(a.k, x, a.v,
                           sigma_eff.to(dtype=x.dtype, device=x.device))

def eval_discrete_kernel_pair(a, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    delta = diff_mat_mat(x1.cpu().numpy(), x2.cpu().numpy())
    # Anchor to x1’s dtype/device
    delta = torch.tensor(delta, dtype=x1.dtype, device=x1.device)
    return _isfkernel_eval(a.k, delta, a.v, a.sigma)


def eval_neural_kernel_pair(a, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    prepares the effective weights sigma_eff before passing them to the generic
    _isfkernel_eval routine, matching the logic of the Julia
    NeuralWeightPlaneWaveKernel function dispatch.
    in Julia σ = a.σ .* dropdims(a.W(a.k * a.v), dims = 1)

    here :  k * v is calculated (kv = k_real * v_real)
    Wout = a.W(kv.T).squeeze(-1)    W outputs a scalar weight for each direction.
    sigma_eff = a.sigma * Wout  Multiplies the baseline sigma by the 
    neural output W_out to get the effective weights sigma_eff
    """
    # net_dtype = next(a.W.parameters()).dtype
    # kv = (a.k.to(dtype=net_dtype, device=a.v.device) * a.v.to(dtype=net_dtype))
    #---
    net_dtype = torch.float64 if a.v.dtype == torch.float64 else torch.float32
    # assicura che k e v siano REALI, sullo stesso device/dtype
    k_real = torch.as_tensor(a.k, device=a.v.device).real.to(net_dtype)
    v_real = a.v.to(dtype=net_dtype, device=a.v.device)

    kv = k_real * v_real
    # ---
    
    Wout = a.W(kv.T).squeeze(-1)
    sigma_eff = a.sigma.to(dtype=net_dtype, device=a.v.device) * Wout

    delta = diff_mat_mat(x1.cpu().numpy(), x2.cpu().numpy())
    # Anchor to x1’s dtype/device
    delta = torch.tensor(delta, dtype=x1.dtype, device=x1.device)
    return _isfkernel_eval(a.k, delta, a.v,
                           sigma_eff.to(dtype=delta.dtype, device=delta.device))
