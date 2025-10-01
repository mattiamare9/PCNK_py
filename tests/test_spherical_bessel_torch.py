import torch
import numpy as np

from my_project.core.background import (
    spherical_bessel_jn_torch,
    spherical_bessel_jn_derivative,
)


def test_torch_autograd_matches_recursive_derivative():
    """
    Check that torch.autograd gradient of j_n(x) matches
    the analytic recurrence-based derivative.
    """
    n = 2
    x = torch.linspace(0.5, 6.0, steps=30, requires_grad=True, dtype=torch.float64)

    # Torch-native spherical Bessel j_n
    y = spherical_bessel_jn_torch(n, x)

    # First derivative via autograd
    dy_dx, = torch.autograd.grad(y.sum(), x)

    # Expected derivative from recurrence relation (NumPy → Torch)
    analytic_np = spherical_bessel_jn_derivative(n, x.detach().numpy())
    analytic = torch.from_numpy(analytic_np).to(torch.float64)

    # Compare
    assert torch.allclose(dy_dx, analytic, atol=1e-5)
