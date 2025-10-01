from my_project.core.background import spherical_bessel_jn_torch
from my_project.core.background import spherical_bessel_jn_derivative
import torch
import numpy as np


def test_spherical_bessel_jn_torch_autograd():
    n = 2
    x = torch.linspace(0.5, 8.0, steps=50, requires_grad=True, dtype=torch.float64)

    # Torch-native j_n
    y = spherical_bessel_jn_torch(n, x)

    # Prima derivata via autograd
    dy_dx, = torch.autograd.grad(y.sum(), x, create_graph=True)

    # Derivata attesa dalla ricorrenza
    analytic = spherical_bessel_jn_derivative(n, x.detach().numpy())
    analytic_torch = torch.from_numpy(analytic).to(torch.float64)

    assert torch.allclose(dy_dx, analytic_torch, atol=1e-5)
