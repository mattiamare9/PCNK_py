"""
Demo of Uniform ISF Kernels.

- Shows how to instantiate and evaluate Fixed / Scaled kernels.
- Demonstrates NumPy vs Torch evaluation.
- Trains ScaledUniformISFKernel to fit a synthetic kernel matrix.

Run:
    poetry run python examples/uniform_kernel_demo.py
"""

import numpy as np
import torch
import torch.optim as optim

from my_project.core.kernels.uniform_isfk import (
    FixedUniformISFKernel,
    ScaledUniformISFKernel,
    uniform_kernel,
    evaluate_uniform_kernel_numpy,
)


def main():
    # --- Toy point cloud ---
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
    y = np.array([[0.0, 1.0], [1.0, 1.0]])

    k = 2.0

    print("=== NumPy evaluation (fixed kernel) ===")
    K_np = evaluate_uniform_kernel_numpy(k, x, y)
    print("Kernel matrix (NumPy):\n", K_np)

    # --- Torch fixed kernel ---
    model_fixed = FixedUniformISFKernel(k)
    K_torch = model_fixed(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    print("\nKernel matrix (Torch, fixed):\n", K_torch.detach().numpy())

    # --- Torch scaled kernel ---
    model_scaled = ScaledUniformISFKernel(k, sigma=0.5)
    K_scaled = model_scaled(
        torch.tensor(x, dtype=torch.float64),
        torch.tensor(y, dtype=torch.float64),
    )
    print("\nKernel matrix (Torch, scaled σ=0.5):\n", K_scaled.detach().numpy())

    # --- Training demo ---
    # Target: scale the fixed kernel by 2.0
    target = 2.0 * K_np
    target_t = torch.tensor(target, dtype=torch.float64)

    model_train = ScaledUniformISFKernel(k, sigma=0.1)
    optimizer = optim.SGD(model_train.parameters(), lr=0.1)

    for epoch in range(20):
        optimizer.zero_grad()
        out = model_train(
            torch.tensor(x, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
        )
        loss = torch.mean((out - target_t) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"[epoch {epoch}] loss={loss.item():.4e}, sigma={model_train.sigma_scalar:.4f}")

    print("\nTrained σ:", model_train.sigma_scalar)


if __name__ == "__main__":
    main()
