import torch
from my_project.core.background import spherical_bessel_jn_torch

# -------------------------------
# Esempio: fit di j0(x) con una rete PyTorch
# -------------------------------

# 1. Dataset sintetico: punti x e valori target j0(x)
x = torch.linspace(0.1, 10.0, steps=200, dtype=torch.float32).unsqueeze(1)  # (200,1)
y_target = spherical_bessel_jn_torch(0, x)  # j0(x)

# 2. Piccolo modello NN
model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.Tanh(),
    torch.nn.Linear(32, 1)
)

# 3. Ottimizzatore
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# 4. Training loop
for epoch in range(200):
    opt.zero_grad()
    y_pred = model(x)

    # Loss = MSE con target j0(x)
    loss = torch.nn.functional.mse_loss(y_pred.squeeze(), y_target.squeeze())

    # Backprop
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss = {loss.item():.6f}")

# 5. Test gradiente autograd su j2(x)
x_test = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y_test = spherical_bessel_jn_torch(2, x_test)  # j2(x)
dy_dx, = torch.autograd.grad(y_test.sum(), x_test)
print("\nGradiente di j2(x) rispetto a x:", dy_dx)
