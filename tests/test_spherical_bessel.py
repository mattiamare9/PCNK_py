import numpy as np
import pytest
from scipy.special import spherical_jn

from my_project.core.background.spherical_bessel import (
    j0, dj0, d2j0, d3j0,
    i0, di0, d2i0, d3i0,
    spherical_bessel_jn,
    spherical_bessel_jn_derivative,
)


# ------------------------------
# Julia-style j0 / i0 functions
# ------------------------------

def test_j0_matches_scipy():
    x = np.linspace(0.1, 10, 100)
    assert np.allclose(j0(x), spherical_jn(0, x), atol=1e-12)


def test_j0_derivatives_consistency():
    x = np.linspace(0.1, 10, 100)

    # numeric first derivative
    eps = 1e-6
    num_dj0 = (j0(x + eps) - j0(x - eps)) / (2 * eps)
    assert np.allclose(dj0(x), num_dj0, atol=1e-5)

    # numeric second derivative
    num_d2j0 = (dj0(x + eps) - dj0(x - eps)) / (2 * eps)
    assert np.allclose(d2j0(x), num_d2j0, atol=1e-4)

    # numeric third derivative
    num_d3j0 = (d2j0(x + eps) - d2j0(x - eps)) / (2 * eps)
    assert np.allclose(d3j0(x), num_d3j0, atol=1e-3)


def test_i0_derivatives_consistency():
    x = np.linspace(0.1, 6, 80)

    eps = 1e-6
    num_di0 = (i0(x + eps) - i0(x - eps)) / (2 * eps)
    assert np.allclose(di0(x), num_di0, atol=1e-5)

    num_d2i0 = (di0(x + eps) - di0(x - eps)) / (2 * eps)
    assert np.allclose(d2i0(x), num_d2i0, atol=1e-4)

    num_d3i0 = (d2i0(x + eps) - d2i0(x - eps)) / (2 * eps)
    assert np.allclose(d3i0(x), num_d3i0, atol=1e-3)


# ------------------------------
# Pythonic generic interface
# ------------------------------

def test_spherical_bessel_jn_matches_scipy():
    x = np.linspace(0.1, 5, 20)
    for n in range(4):
        assert np.allclose(spherical_bessel_jn(n, x), spherical_jn(n, x), atol=1e-12)


def test_spherical_bessel_jn_derivative_consistency():
    x = np.linspace(0.5, 8, 50)
    n = 2

    eps = 1e-6
    num_derivative = (
        spherical_bessel_jn(n, x + eps) - spherical_bessel_jn(n, x - eps)
    ) / (2 * eps)

    analytic = spherical_bessel_jn_derivative(n, x)
    assert np.allclose(analytic, num_derivative, atol=1e-5)
