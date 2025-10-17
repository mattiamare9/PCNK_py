"""
Evaluation routines for Directional Spherical Bessel Kernels
(Port of PCNK/Kernels/DirectionalISFK/Evaluation.jl)
"""

import torch
import numpy as np

from my_project.core.background.differences import diff_mat_mat


def spherical_j0(z: torch.Tensor) -> torch.Tensor:
    """Spherical Bessel j0 = sin(z)/z, safe at 0."""
    out = torch.empty_like(z, dtype=z.dtype)
    mask = z == 0
    out[mask] = 1.0
    out[~mask] = torch.sin(z[~mask]) / z[~mask]
    return out


# def _ensure_leading3(x: torch.Tensor) -> torch.Tensor:
#     """Ensure x has leading dimension 3."""
#     if x.ndim == 1:
#         return x.view(3)
#     elif x.ndim == 2 and x.shape[0] != 3:
#         return x.T
#     return x


# ---- Single-direction evaluation -------------------------------------------------

def _eval_single_dir(k: torch.Tensor,
                     x: torch.Tensor,   # (3,), (3,B) or (3,B1,B2)
                     v: torch.Tensor,   # (3,1)
                     sigma: torch.Tensor,
                     beta: torch.Tensor) -> torch.Tensor:

    dtype = torch.result_type(k, x)
    k = torch.as_tensor(k, dtype=dtype, device=x.device)

    v = v.to(dtype=x.dtype, device=x.device)
    sigma = sigma.to(dtype=x.dtype, device=x.device)
    beta = beta.to(dtype=x.dtype, device=x.device)

    # complex unit with correct dtype/device
    j = torch.tensor(1j, dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128,
                     device=x.device)

    # single-direction vectors as 1D tensors
    v_vec = v[:, 0] if v.ndim == 2 and v.shape[1] == 1 else v
    beta_s = beta.reshape(()) if beta.numel() == 1 else beta   # scalar tensor if N==1
    sigma_s = sigma.reshape(()) if sigma.numel() == 1 else sigma

    if x.ndim == 1:  # (3,)
        u = k * x - j * (beta_s * v_vec)            # (3,)
        s = (u * u).sum()                           # scalar
        z = torch.sqrt(s)
        return sigma * spherical_j0(z) / torch.i0(beta)

    elif x.ndim == 2:  # (3,B)  -> return (B,)
        X = x                                            # (3,B)
        u = k * X - (j * beta_s) * v_vec[:, None]        # (3,B)
        s = (u * u).sum(dim=0)                           # (B,)
        z = torch.sqrt(s)                                # (B,)
        return (sigma_s / torch.i0(beta_s)) * spherical_j0(z)  # (B,)

    elif x.ndim == 3:  # (3,B1,B2) -> flatten, eval, reshape back
        B1, B2 = x.shape[1], x.shape[2]
        x_flat = x.reshape(3, -1)                        # (3, B1*B2)
        out = _eval_single_dir(k, x_flat, v, sigma, beta)  # (B1*B2,)
        return out.reshape(B1, B2)

    else:
        raise ValueError("x must be 1D, 2D or 3D for single-direction kernel.")

# ---- Multi-direction evaluation -------------------------------------------------
def _eval_multi_dir(k: torch.Tensor,
                    x: torch.Tensor,   # (3,), (3,B) or (3,B1,B2)
                    v: torch.Tensor,   # (3,D)
                    sigma: torch.Tensor,
                    beta: torch.Tensor) -> torch.Tensor:

    dtype = torch.result_type(k, x)
    k = torch.as_tensor(k, dtype=dtype, device=x.device)

    v = v.to(dtype=x.dtype, device=x.device)
    sigma = sigma.to(dtype=x.dtype, device=x.device)
    beta = beta.to(dtype=x.dtype, device=x.device)

    j = torch.tensor(1j, dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128, device=x.device)

    if x.ndim == 1:  # (3,)
        c = k * x[:, None] - j * (v * beta[None, :])   # (3,D)
        s = (c * c).sum(dim=0)  # (D,)
        z = torch.sqrt(s)
        return torch.sum((sigma / torch.i0(beta)) * spherical_j0(z))

    elif x.ndim == 2:  # (3,B)
        B = x.shape[1]
        c = k * x[:, None, :] - j * (v[:, :, None] * beta[None, :, None])  # (3,D,B)
        s = (c * c).sum(dim=0)  # (D,B)
        z = torch.sqrt(s)
        return ((sigma / torch.i0(beta))[:, None] * spherical_j0(z)).sum(dim=0)  # (B,)

    elif x.ndim == 3:  # (3,B1,B2)
        B1, B2 = x.shape[1], x.shape[2]
        x_flat = x.reshape(3, -1)  # (3, B1*B2)
        out = _eval_multi_dir(k, x_flat, v, sigma, beta)  # (B1*B2,)
        return out.reshape(B1, B2)

    else:
        raise ValueError("x must be 1D, 2D or 3D for multi-direction kernel.")

# ---- Public evaluators ----------------------------------------------------------

def eval_directional_single(a, x: torch.Tensor) -> torch.Tensor:
    """DirectionalSFKernel for one input."""
    x = _ensure_leading3(x)
    if a.v.shape[1] == 1:   # single direction
        return _eval_single_dir(a.k, x, a.v, a.sigma, a.beta)
    else:                   # multiple directions
        return _eval_multi_dir(a.k, x, a.v, a.sigma, a.beta)


def _ensure_leading3(x: torch.Tensor) -> torch.Tensor:
    """Ensure x has leading dimension 3: (3,), (3,B)."""
    if x.ndim == 1:
        if x.numel() != 3:
            raise ValueError("1D x must have length 3.")
        return x.view(3)
    elif x.ndim == 2:
        # Accept (3,B) or (B,3) and convert to (3,B)
        if x.shape[0] == 3:
            return x
        if x.shape[1] == 3:
            return x.T
        raise ValueError("2D x must be (3,B) or (B,3).")
    elif x.ndim == 3:
        # Assume already (3,B1,B2) if leading dim is 3; otherwise swap first/last if (B1,B2,3)
        if x.shape[0] == 3:
            return x
        if x.shape[-1] == 3:
            return x.movedim(-1, 0)  # (3,B1,B2)
        raise ValueError("3D x must be (3,B1,B2) or (B1,B2,3).")
    else:
        raise ValueError("x must be 1D, 2D, or 3D.")


def eval_directional_pair(a, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    DirectionalSFKernel for two inputs, using Δx = x1 - x2.
    Accepts x1, x2 as (3,B), (B,3), or (3,) / (B,3) mixed and returns (B1,B2).
    """
    # Bring both to leading-3 2D: (3,B1) and (3,B2)
    X1 = _ensure_leading3(x1)
    X2 = _ensure_leading3(x2)

    if X1.ndim == 1:  # (3,) -> (3,1)
        X1 = X1.unsqueeze(1)
    if X2.ndim == 1:
        X2 = X2.unsqueeze(1)

    if X1.ndim != 2 or X2.ndim != 2 or X1.shape[0] != 3 or X2.shape[0] != 3:
        raise ValueError("x1 and x2 must be vectors or 2D with leading dimension 3.")

    # Compute Δx in torch to preserve dtype/device & shape
    # X1: (3,B1), X2: (3,B2) -> delta: (3,B1,B2)
    delta = X1[:, :, None] - X2[:, None, :]

    # Evaluate with 3D support; result is (B1,B2)
    return eval_directional_single(a, delta)

