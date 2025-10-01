"""
Spherical Bessel functions (ported from PCNK/Background/SphericalBesselImpl.jl).

Two approaches are available:
1. Julia-style manual derivatives (j0, dj0, d2j0, d3j0, i0, di0, d2i0, d3i0)
   - Matches original repo
   - Independent of autograd
2. Pythonic generic interface using SciPy (spherical_bessel_jn, spherical_bessel_jn_derivative)
   - Clean, supports any order
   - Plays nicely with PyTorch autograd

👉 When the project is stable, you can comment out one branch depending on preference.
"""
import numpy as np
from scipy.special import spherical_jn, iv


# -------------------------------------------------------------------
# 1) Julia-style manual functions (up to order 3 for j0 and i0)
# -------------------------------------------------------------------

def j0(x: np.ndarray) -> np.ndarray:
    """Zero-order spherical Bessel of the first kind."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 1.0, np.sin(x) / x)


def dj0(x: np.ndarray) -> np.ndarray:
    """First derivative of j0."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 0.0, (x * np.cos(x) - np.sin(x)) / (x**2))


def d2j0(x: np.ndarray) -> np.ndarray:
    """Second derivative of j0."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, -1.0 / 3.0, ((2 - x**2) * np.sin(x) - 2 * x * np.cos(x)) / (x**3))


def d3j0(x: np.ndarray) -> np.ndarray:
    """Third derivative of j0."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x == 0,
        0.0,
        (3 * (x**2 - 2) * np.sin(x) - x * (x**2 - 6) * np.cos(x)) / (x**4),
    )


def i0(x: np.ndarray) -> np.ndarray:
    """Modified spherical Bessel of the first kind (order 0)."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 1.0, np.sinh(x) / x)


def di0(x: np.ndarray) -> np.ndarray:
    """First derivative of i0."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 0.0, (x * np.cosh(x) - np.sinh(x)) / (x**2))


def d2i0(x: np.ndarray) -> np.ndarray:
    """Second derivative of i0."""
    x = np.asarray(x, dtype=float)
    return np.where(x == 0, 1.0 / 3.0, ((x**2 + 2) * np.sinh(x) - 2 * x * np.cosh(x)) / (x**3))


def d3i0(x: np.ndarray) -> np.ndarray:
    """Third derivative of i0."""
    x = np.asarray(x, dtype=float)
    return np.where(
        x == 0,
        0.0,
        (x * (x**2 + 6) * np.cosh(x) - 3 * (x**2 + 2) * np.sinh(x)) / (x**4),
    )


# -------------------------------------------------------------------
# 2) Pythonic generic interface
# -------------------------------------------------------------------

def spherical_bessel_jn(n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute spherical Bessel j_n(x) for any order n using SciPy.
    """
    return spherical_jn(n, x)


def spherical_bessel_jn_derivative(n: int, x: np.ndarray) -> np.ndarray:
    """
    Derivative of spherical Bessel j_n(x) using recurrence relation:
        j_n'(x) = j_{n-1}(x) - (n+1)/x * j_n(x)
    """
    x = np.asarray(x, dtype=float)
    jn = spherical_jn(n, x)
    if n > 0:
        jn_minus = spherical_jn(n - 1, x)
    else:
        # Special case for j0'(x)
        jn_minus = np.sin(x) / (x**2) - np.cos(x) / x
    return jn_minus - (n + 1) / x * jn


"""
nei moduli a monte (es. nei kernel o nei test), 
usi autograd per calcolare qualunque ordine di derivata ti serva:

import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = spherical_bessel_jn(0, x)

# prima derivata
dy_dx, = torch.autograd.grad(y.sum(), x, create_graph=True)

# seconda derivata
d2y_dx2, = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)

"""

"""
Torch-native spherical Bessel functions for autograd.
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

    jnm2 = torch.sin(x) / x                # j0
    jnm1 = torch.sin(x) / (x**2) - torch.cos(x) / x  # j1

    for k in range(2, n + 1):
        jn = (2 * k - 1) / x * jnm1 - jnm2
        jnm2, jnm1 = jnm1, jn

    return jn
