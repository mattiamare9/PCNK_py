import torch
import numpy as np
from scipy.special import spherical_jn

from my_project.core.kernels.uniform_isfk import (
    FixedUniformISFKernel,
    ScaledUniformISFKernel,
    uniform_kernel,
)
from my_project.core.kernels.uniform_isfk import (
    evaluate_uniform_kernel_numpy,
)


def test_fixed_single_input_vector():
    k = 2.0
    x = torch.tensor([3.0, 4.0], dtype=torch.float64)  # norm = 5
    model = FixedUniformISFKernel(k)
    out = model(x)  # single-input mode
    expected = spherical_jn(0, k * torch.linalg.norm(x).item())
    assert np.allclose(out.item(), expected)


def test_scaled_single_input_matrix():
    k = 1.5
    sigma = 2.5
    pts = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float64)
    model = ScaledUniformISFKernel(k, sigma=sigma)
    out = model(pts)  # single-input mode, returns vector
    r = torch.linalg.norm(pts, dim=-1).numpy()
    expected = sigma * spherical_jn(0, k * r)
    assert np.allclose(out.detach().numpy(), expected)


def test_two_input_consistency_numpy_torch():
    k = 1.2
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    y = np.array([[0.0, 1.0]])
    expected = evaluate_uniform_kernel_numpy(k, x, y)

    model = FixedUniformISFKernel(k)
    x_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    result = model(x_t, y_t).detach().numpy()

    assert np.allclose(result, expected)


def test_scaled_sigma_effect():
    k = 1.0
    sigma = 4.0
    model = ScaledUniformISFKernel(k, sigma=sigma)
    x = torch.zeros((1, 2), dtype=torch.float64)
    out = model(x)  # norm=0 => j0(0)=1, result = sigma
    assert np.allclose(out.item(), sigma)


def test_factory_dispatch():
    k = 1.0
    f = uniform_kernel(k)
    s = uniform_kernel(k, sigma=2.0)
    assert isinstance(f, FixedUniformISFKernel)
    assert isinstance(s, ScaledUniformISFKernel)
