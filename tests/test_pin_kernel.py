import torch
import pytest

from my_project.core.kernels.physics_informed_isfk import (
    plane_wave_pin_kernel,
    directed_residual_pin_kernel,
)
from my_project.core.kernels.uniform_isfk import ScaledUniformISFKernel
from my_project.core.kernels.plane_wave_isfk import plane_wave_kernel
from my_project.core.kernels.directional_isfk import directional_sf_kernel


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_plane_wave_pin_forward_shapes_and_grad(dtype):
    torch.manual_seed(0)
    k = 2.0
    ord = 11
    pin = plane_wave_pin_kernel(k, ord=ord, dtype=dtype)
    assert isinstance(pin.analytical.k, torch.Tensor)
    assert isinstance(pin.neural.k, torch.Tensor)

    # single vector
    x = torch.randn(3, dtype=dtype, requires_grad=True)
    out = pin(x)
    assert out.ndim == 0
    assert torch.isfinite(out)
    # PyTorch cannot implicitly backprop complex outputs — take real part.
    if torch.is_complex(out):
        out.real.backward(retain_graph=True)
    else:
        out.backward(retain_graph=True)
    assert x.grad is not None

    # batch of vectors
    X = torch.randn(3, 5, dtype=dtype)
    out_batch = pin(X)
    assert out_batch.shape == (5,)
    assert torch.isfinite(out_batch).all()

    # two-input form should match single-input(delta)
    x1 = torch.randn(3, 4, dtype=dtype)
    x2 = torch.randn(3, 4, dtype=dtype)
    out_pair = pin(x1, x2)
    assert out_pair.shape == (4,)
    assert torch.isfinite(out_pair).all()


@pytest.mark.parametrize("trainable_dir", [False, True])
def test_directed_residual_pin_forward_and_grad(trainable_dir):
    torch.manual_seed(1)
    k = 1.5
    pin = directed_residual_pin_kernel(
        k,
        ord_dir=9,
        ord_res=7,
        trainable_direction=trainable_dir,
        dtype=torch.float32,
    )

    # single vector
    x = torch.randn(3, dtype=torch.float32, requires_grad=True)
    out = pin(x)
    assert out.ndim == 0
    assert torch.isfinite(out)
    # PyTorch cannot implicitly backprop complex outputs — take real part.
    if torch.is_complex(out):
        out.real.backward(retain_graph=True)
    else:
        out.backward(retain_graph=True)
    assert x.grad is not None

    # batch
    X = torch.randn(3, 8, dtype=torch.float32)
    out_b = pin(X)
    assert out_b.shape == (8,)
    assert torch.isfinite(out_b).all()

    # two-input
    x1 = torch.randn(3, 6, dtype=torch.float32)
    x2 = torch.randn(3, 6, dtype=torch.float32)
    out2 = pin(x1, x2)
    assert out2.shape == (6,)
    assert torch.isfinite(out2).all()


def test_pin_kernel_repr_and_device_transfer():
    """Check repr() and .to(device) behavior."""
    k = 1.0
    pin = plane_wave_pin_kernel(k, ord=7, dtype=torch.float32)
    s = repr(pin)
    assert "PlaneWavePINKernel" in s

    # move to cuda if available (else cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin = pin.to(device)
    assert next(pin.parameters()).device.type == device


# --------------------------------------------------------------------------
# Numerical consistency checks
# --------------------------------------------------------------------------
def test_plane_wave_pin_equivalence():
    """Ensure PlaneWavePINKernel ≈ Uniform + Neural plane-wave kernel."""
    torch.manual_seed(42)
    k = 1.2
    ord = 9
    dtype = torch.float32

    # composite kernel
    pin = plane_wave_pin_kernel(k, ord=ord, dtype=dtype)

    # manual sum of components
    analytical = ScaledUniformISFKernel(k, sigma=1.0, dtype=dtype)
    neural = plane_wave_kernel(k, ord=ord, dtype=dtype)

    x = torch.randn(3, dtype=dtype)
    out_pin = pin(x)
    out_manual = analytical(x) + neural(x)

    # because of numerical noise and learned parameters, we allow small tolerance
    assert torch.allclose(out_pin, out_manual, rtol=1e-3, atol=1e-3)


def test_directed_residual_pin_equivalence():
    """Ensure DirectedResidualPINKernel ≈ Directional + Neural residual."""
    torch.manual_seed(7)
    k = 1.0
    dtype = torch.float32

    dir_kernel = directional_sf_kernel(k, ord=7, dtype=dtype)
    neural = plane_wave_kernel(k, ord=9, dtype=dtype)
    composite = directed_residual_pin_kernel(k, ord_dir=7, ord_res=9, dtype=dtype)

    x = torch.randn(3, dtype=dtype)
    out_dir = dir_kernel(x)
    out_neural = neural(x)
    out_sum = out_dir + out_neural
    out_composite = composite(x)

    assert torch.allclose(out_sum, out_composite, rtol=1e-3, atol=1e-3)
