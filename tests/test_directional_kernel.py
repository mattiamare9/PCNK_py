import torch
import math
import pytest

from my_project.core.kernels.directional_isfk import (
    FixedDirectionSFKernel,
    TrainableDirectionSFKernel,
    directional_sf_kernel,
)


def spherical_j0(z: torch.Tensor) -> torch.Tensor:
    """Reference spherical Bessel j0 = sin(z)/z, safe at 0."""
    out = torch.empty_like(z, dtype=z.dtype)
    mask = z == 0
    out[mask] = 1.0
    out[~mask] = torch.sin(z[~mask]) / z[~mask]
    return out


def test_single_direction_matches_formula():
    v = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    beta = torch.tensor([2.0], dtype=torch.float32)
    k = 1.0

    kernel = FixedDirectionSFKernel(k, v, sigma=sigma, beta_init=2.0)

    x = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)

    out = kernel(x)

    j = torch.tensor(1j, dtype=torch.complex64)
    c = k * x - j * (beta * v)
    s = torch.sum(c * c)
    z = torch.sqrt(s)
    manual = sigma * spherical_j0(z) / torch.i0(beta)

    assert torch.allclose(out, manual, atol=1e-6)


def test_multi_direction_reduces_to_single():
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).T  # (3,1)
    sigma = torch.tensor([1.0], dtype=torch.float32)
    k = 1.0

    fixed_single = FixedDirectionSFKernel(k, v, sigma=sigma)
    trainable_single = TrainableDirectionSFKernel(k, v, sigma=sigma)

    x = torch.tensor([0.3, 0.2, 0.0], dtype=torch.float32)

    out_fixed = fixed_single(x)
    out_trainable = trainable_single(x)

    assert torch.allclose(out_fixed, out_trainable, atol=1e-6)


def test_two_input_matches_difference():
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).T
    sigma = torch.tensor([1.0], dtype=torch.float32)
    k = 2.0

    kernel = FixedDirectionSFKernel(k, v, sigma=sigma)

    x1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)

    out_pair = kernel(x1, x2)
    delta = (x1 - x2).squeeze()
    out_single = kernel(delta)

    assert torch.allclose(out_pair, out_single, atol=1e-6)


def test_factory_dispatches():
    v = torch.eye(3, dtype=torch.float32)

    f = directional_sf_kernel(1.0, v=v, sigma=[1.0, 1.0, 1.0])
    t = directional_sf_kernel(1.0, v=v, sigma=[1.0, 1.0, 1.0], trainable_direction=True)

    assert isinstance(f, FixedDirectionSFKernel)
    assert isinstance(t, TrainableDirectionSFKernel)


def test_factory_with_lebedev_weights_sum_to_4pi():
    k = 1.0
    ord = 23  # 302 points
    kernel = directional_sf_kernel(k, ord=ord)

    sigma = kernel.sigma.detach().cpu()
    total = sigma.sum().item()

    assert math.isclose(total, 4 * math.pi, rel_tol=1e-6)


def test_lebedev_kernel_runs_on_random_inputs():
    k = 2.0
    ord = 11  # smaller Lebedev rule for quick smoke test
    kernel = directional_sf_kernel(k, ord=ord)

    # Single random vector
    x = torch.randn(3, dtype=torch.float32)
    out = kernel(x)
    assert torch.isfinite(out).all()

    # Batch of vectors
    X = torch.randn(3, 5, dtype=torch.float32)
    out_batch = kernel(X)
    assert out_batch.shape == (5,)
    assert torch.isfinite(out_batch).all()

    # Two-input evaluation
    x1 = torch.randn(3, 4, dtype=torch.float32).T  # (4,3) -> (3,4) inside
    x2 = torch.randn(3, 4, dtype=torch.float32).T
    out_pair = kernel(x1, x2)
    assert out_pair.shape == (4, 4)
    assert torch.isfinite(out_pair).all()
