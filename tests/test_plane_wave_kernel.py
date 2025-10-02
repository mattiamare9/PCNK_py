import math
import torch
import pytest

from my_project.core.kernels.plane_wave_isfk import (
    FixedDirectionPlaneWaveKernel,
    TrainableDirectionPlaneWaveKernel,
    NeuralWeightPlaneWaveKernel,
    plane_wave_kernel,
)


def test_fixed_kernel_single_and_pair():
    # simple directions: 2D basis vectors
    v = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32).T  # (3,3)

    sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    k = 1.0
    kernel = FixedDirectionPlaneWaveKernel(k, v, sigma=sigma)

    # single input x
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)  # unit vector
    out = kernel(x)  # sum_j exp(i k v_j·x) σ_j

    # choose correct complex dtype for manual computation
    complex_dtype = torch.complex64 if x.dtype == torch.float32 else torch.complex128
    j = torch.tensor(1j, dtype=complex_dtype, device=x.device)

    manual = (sigma * torch.exp(j * k * (v.T @ x))).sum()
    assert torch.allclose(out, manual)

    # two-input kernel with identical points -> should equal single input(0)
    x1 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    out2 = kernel(x1, x2)
    out_single = kernel(torch.zeros(3, dtype=torch.float32))
    assert torch.allclose(out2, out_single)


def test_trainable_kernel_same_as_fixed():
    v = torch.eye(3, dtype=torch.float32)  # directions
    sigma = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    k = 0.5

    fixed = FixedDirectionPlaneWaveKernel(k, v, sigma=sigma)
    trainable = TrainableDirectionPlaneWaveKernel(k, v, sigma=sigma)

    x = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(fixed(x), trainable(x))


def test_neural_weight_kernel_applies_network():
    v = torch.eye(3, dtype=torch.float32)
    sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    k = 1.0

    # Simple NN: linear -> outputs sum of inputs
    net = torch.nn.Sequential(torch.nn.Linear(3, 1, bias=False))
    net = net.to(dtype=torch.float32)

    # initialize weights to ones
    with torch.no_grad():
        net[0].weight.fill_(1.0)  # type: ignore[attr-defined]

    # Build kernel with this network
    kernel = NeuralWeightPlaneWaveKernel(k, v, sigma=sigma, W=net)

    # Single input
    x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    out = kernel(x)

    # Effective weights = σ * net(k v)
    with torch.no_grad():
        sigma_eff = sigma * torch.squeeze(net((k * v).T), dim=-1)

    # Manual computation: sum_j sigma_eff[j] * exp(i k v_j·x)
    manual = (sigma_eff * torch.exp(1j * k * (v.T @ x))).sum()

    assert torch.allclose(out, manual)


def test_factory_dispatches_correctly():
    v = torch.eye(3, dtype=torch.float32)

    f = plane_wave_kernel(1.0, v=v, sigma=[1.0, 1.0, 1.0])
    t = plane_wave_kernel(1.0, v=v, sigma=[1.0, 1.0, 1.0], trainable_direction=True)
    n = plane_wave_kernel(1.0, v=v, sigma=[1.0, 1.0, 1.0], W=torch.nn.Linear(3, 1))

    assert isinstance(f, FixedDirectionPlaneWaveKernel)
    assert isinstance(t, TrainableDirectionPlaneWaveKernel)
    assert isinstance(n, NeuralWeightPlaneWaveKernel)
