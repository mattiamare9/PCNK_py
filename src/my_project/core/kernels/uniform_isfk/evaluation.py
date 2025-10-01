"""
Evaluation routines for the Uniform ISF Kernel
(ported from PCNK/Kernels/UniformISFK/Evaluation.jl).

Formulas:
- j0(z) = sin(z)/z
- For r = ||x - y||, kernel(x,y) = j0(k * r)
- ScaledUniformISFKernel multiplies this by σ
"""

import numpy as np
import torch
from scipy.special import spherical_jn


def evaluate_uniform_kernel_numpy(
    k: float,
    x: np.ndarray,
    y: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Evaluate Uniform ISF kernel between two sets of points using NumPy.

    Parameters
    ----------
    k : float
        Wavenumber.
    x : (N, d) ndarray
        First set of points.
    y : (M, d) ndarray
        Second set of points.
    sigma : float, optional
        Multiplicative scaling factor.

    Returns
    -------
    (N, M) ndarray
        Kernel matrix.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    diff = x[:, None, :] - y[None, :, :]
    r = np.linalg.norm(diff, axis=-1)

    values = spherical_jn(0, k * r)
    if sigma is not None:
        values = sigma * values
    return values


def evaluate_uniform_kernel_torch(
    k: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Torch implementation of the Uniform ISF kernel.

    Parameters
    ----------
    k : scalar tensor
        Wavenumber.
    x : (N, d) tensor
    y : (M, d) tensor
    sigma : (1,) tensor, optional
        Multiplicative factor.

    Returns
    -------
    (N, M) tensor
        Kernel matrix.
    """
    diff = x[:, None, :] - y[None, :, :]
    r = torch.linalg.norm(diff, dim=-1)

    # spherical j0 = sin(z)/z, handle z=0 safely
    z = k * r
    values = torch.where(
        z == 0,
        torch.ones_like(z),
        torch.sin(z) / z,
    )
    if sigma is not None:
        values = sigma.view(-1)[0] * values
    return values
