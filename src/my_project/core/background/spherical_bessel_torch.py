"""
Torch-native spherical Bessel functions for autograd.
Implements j_n(x) via recurrence, fully differentiable by torch.autograd.
"""
import torch


def spherical_bessel_jn_torch(n: int, x: torch.Tensor) -> torch.Tensor:
    """
    Compute spherical Bessel j_n(x) using recurrence relations (Torch-native).

    Parameters
    ----------
    n : int
        Order (n >= 0)
    x : torch.Tensor
        Input values, shape (...,)

    Returns
    -------
    torch.Tensor
        j_n(x) values, same shape as x
    """
    if n == 0:
        return torch.sin(x) / x
    elif n == 1:
        return torch.sin(x) / (x**2) - torch.cos(x) / x

    jnm2 = torch.sin(x) / x                              # j0
    jnm1 = torch.sin(x) / (x**2) - torch.cos(x) / x      # j1

    for k in range(2, n + 1):
        jn = (2 * k - 1) / x * jnm1 - jnm2
        jnm2, jnm1 = jnm1, jn

    return jn
