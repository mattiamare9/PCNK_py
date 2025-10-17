import torch
import torch.nn as nn

import numpy as np
from src.my_project.core.spm import C_SOUND

from my_project.core.kernels import (
    directed_residual_pin_kernel,
)
SNR_DB = 20.0
LAMBDA = .0051         # λ di Julia (varianza del rumore ~20 dB)
ORD_DIR = 5            # ordine analitico (Directed)
ORD_NN = 11            # ordine residuo / plane-wave NN
DEVICE = "cpu"


def prop_factory(k: float):
    # Proposed: Directed analytical + Neural residual
    return directed_residual_pin_kernel(
        k,
        ord_dir=ORD_DIR,
        ord_res=ORD_NN,
        W=make_default_W(device=DEVICE),
        trainable_direction=False,
        dtype=torch.complex128,
        device=DEVICE,
    )

def make_default_W(device: str = "cpu", dtype=torch.float64) -> nn.Module:
    """
    MLP analogo a quello Julia:
    Chain(Dense(3,5,tanh), Dense(5,5,tanh), Dense(5,1,tanh), softplus)
    """
    model = nn.Sequential(
        nn.Linear(3, 5, bias=True),
        nn.Tanh(),
        nn.Linear(5, 5, bias=True),
        nn.Tanh(),
        nn.Linear(5, 1, bias=True),
        nn.Tanh(),
        nn.Softplus(),
    )
    # dtype float64 per coerenza con complessi complex128
    return model.to(device=device, dtype=dtype)

# Reproduce the exact conditions from your SPM run
freq = 600.0
k = 2.0 * np.pi * freq / C_SOUND

print(f"Frequency: {freq} Hz")
print(f"Wavenumber k: {k}")

# Test a simple kernel evaluation to see if NaN occurs
torch.manual_seed(42)
X = torch.randn(10, 3, dtype=torch.float64)

# Try with your kernel factory (replace with actual factory)
kernel = prop_factory(k)
K = kernel(X, X)
print(f"K contains NaN: {torch.isnan(K).any()}")
print(f"K contains Inf: {torch.isinf(K).any()}")
print(f"K mean: {K.abs().mean().item():.2e}")

print("Debug info:")
print(f"k is finite: {np.isfinite(k)}")
print(f"k value: {k}")
