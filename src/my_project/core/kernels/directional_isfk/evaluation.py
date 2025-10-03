from __future__ import annotations

import torch
from typing import Tuple

from my_project.core.background.differences import diff_mat_mat


def _ensure_leading3(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize so the leading axis is 3:
      (3,) -> (3,)
      (B,3) -> (3,B)
      (3,B) -> (3,B)
      (B1,B2,3) -> (3,B1,B2)
      (3,B1,B2) -> (3,B1,B2)
    """
    if x.ndim == 1:
        if x.shape[0] != 3:
            raise ValueError("1D input must have length 3.")
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
        raise ValueError("3D input must have a dimension of size 3.")
    raise ValueError("x must be 1D, 2D, or 3D.")


def _get_j(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a correctly typed imaginary unit (1j) matching float precision."""
    ctype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return torch.tensor(1j, dtype=ctype, device=device)


def _j0(z: torch.Tensor) -> torch.Tensor:
    """
    Spherical Bessel j0(z) = sin(z) / z, works for real/complex tensors.
    Uses safe division at z=0 -> 1.
    """
    # NOTE: z can be complex
    out = torch.empty_like(z, dtype=z.dtype)
    zero = torch.zeros((), dtype=z.real.dtype, device=z.device)
    mask_zero = (z == 0)
    out[mask_zero] = (torch.ones((), dtype=z.dtype, device=z.device))
    out[~mask_zero] = torch.sin(z[~mask_zero]) / z[~mask_zero]
    return out


def _i0_real(x: torch.Tensor) -> torch.Tensor:
    """
    Modified Bessel I0 for real tensors using torch.i0 (real-only).
    """
    # torch.i0 expects real dtype
    return torch.i0(x)


def _eval_single_dir(
    k: torch.Tensor, x: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Single-direction case (N=1), Julia:

      β = view(a.β, 1)
      return view(a.σ,1) .* j0.(sqrt(sum((Complex(k*x) - i*(β .* v)).^2))) / i0(β)

    x: (3,), (3,B), or (3,B1,B2)  | v: (3,1) | sigma: (1,) | beta: (1,)
    Returns scalar / (B,) / (B1,B2)
    """
    x = _ensure_leading3(x)

    # cast everything to x's dtype/device
    k = torch.as_tensor(k, dtype=x.dtype, device=x.device)
    v = v.to(dtype=x.dtype, device=x.device)
    sigma = sigma.to(dtype=x.dtype, device=x.device)       # (1,)
    beta = beta.to(dtype=x.dtype, device=x.device)         # (1,)

    j = _get_j(x.dtype, x.device)

    # build c = k*x - i*(beta*v), broadcasting across batch dims of x
    # v: (3,1), beta: (1,) -> beta*v: (3,1)
    c = k * x - j * (beta * v)              # (3, ...)

    # sum of squares without conjugate (as in Julia), along axis 0
    s = torch.sum(c * c, dim=0)             # (...,)

    z = torch.sqrt(s)                       # (...,)
    num = sigma.view(1) * _j0(z)            # (...,)
    den = _i0_real(beta.view(1))            # scalar (1,)
    return num / den                        # (...,)


def _eval_multi_dir(
    k: torch.Tensor, x: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Multi-direction case (N=D), Julia patterns:

      For x vector (3,) -> sum over D of ((σ/i0(β)) * j0(sqrt(sum((k*x - i*(v*β))^2))))
      For x matrix (3,B) -> return (B,)
      For x tensor  (3,B1,B2) -> return (B1,B2)

    x: (3,), (3,B), (3,B1,B2)    v: (3,D)    sigma: (D,)    beta: (D,)
    """
    x = _ensure_leading3(x)

    # common dtypes/devices anchored on x
    k = torch.as_tensor(k, dtype=x.dtype, device=x.device)
    v = v.to(dtype=x.dtype, device=x.device)               # (3,D)
    sigma = sigma.to(dtype=x.dtype, device=x.device)       # (D,)
    beta = beta.to(dtype=x.dtype, device=x.device)         # (D,)

    j = _get_j(x.dtype, x.device)

    # weights w = sigma / i0(beta)
    w = sigma / _i0_real(beta)                             # (D,)

    if x.ndim == 1:
        # (3,), build (3,D) and sum over dim=0 -> (D,)
        c = (k * x).unsqueeze(1) - j * (v * beta.unsqueeze(0))     # (3,D)
        s = torch.sum(c * c, dim=0)                                # (D,)
        z = torch.sqrt(s)                                          # (D,)
        return (w * _j0(z)).sum()                                  # scalar

    if x.ndim == 2:
        # (3,B) -> expand to (3,D,B)
        B = x.shape[1]
        c = (k * x).unsqueeze(1) - j * (v * beta.unsqueeze(0)).unsqueeze(2)   # (3,D,B)
        s = torch.sum(c * c, dim=0)                                           # (D,B)
        z = torch.sqrt(s)                                                     # (D,B)
        return (w[:, None] * _j0(z)).sum(dim=0)                               # (B,)

    # x.ndim == 3: (3,B1,B2)
    B1, B2 = x.shape[1], x.shape[2]
    c = (k * x).unsqueeze(1) - j * (v * beta.unsqueeze(0)).unsqueeze(2).unsqueeze(3)  # (3,D,B1,B2)
    s = torch.sum(c * c, dim=0)                                                       # (D,B1,B2)
    z = torch.sqrt(s)                                                                  # (D,B1,B2)
    return (w.view(-1, 1, 1) * _j0(z)).sum(dim=0)                                      # (B1,B2)


def eval_directional_single(
    k: torch.Tensor, x: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """
    Dispatch between single-direction (N=1) and multi-direction (N>1) pathways.
    """
    # v: (3,N)
    N = v.shape[1]
    if N == 1:
        return _eval_single_dir(k, x, v[:, :1], sigma.view(1), beta.view(1))
    return _eval_multi_dir(k, x, v, sigma, beta)


def eval_directional_pair(
    k: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    v: torch.Tensor,
    sigma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    Two-input evaluation: compute Δx = __Diff(x1, x2) then call single-input path.
    __Diff returns either (3,B1,B2) or (B1,B2,3); eval_directional_single handles both.
    """
    delta_np = diff_mat_mat(x1.cpu().numpy(), x2.cpu().numpy())
    delta = torch.tensor(delta_np, dtype=x1.dtype, device=x1.device)
    return eval_directional_single(k, delta, v, sigma, beta)
