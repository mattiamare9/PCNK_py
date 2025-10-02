# import numpy as np
# import torch
# from pathlib import Path

# _DATA_CACHE = {}
# DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lebedev"

# def lebedev_by_order(order: int, *, dtype=torch.float64, device=None):
#     """
#     Return Lebedev quadrature for given order.
#     Loads from txt only once, then caches in memory.
#     """
#     if order not in _DATA_CACHE:
#         fname = DATA_DIR / f"lebedev_{order:03d}.txt"
#         if not fname.exists():
#             raise FileNotFoundError(f"No Lebedev file for order {order}")
#         phi_deg, theta_deg, w = np.loadtxt(fname, unpack=True)
#         phi = np.deg2rad(phi_deg)
#         theta = np.deg2rad(theta_deg)

#         x = np.sin(theta) * np.cos(phi)
#         y = np.sin(theta) * np.sin(phi)
#         z = np.cos(theta)

#         # normalize weights to 4π
#         w = w * (4*np.pi / np.sum(w))

#         _DATA_CACHE[order] = (x, y, z, w)

#     x, y, z, w = _DATA_CACHE[order]
#     return (
#         torch.tensor(x, dtype=dtype, device=device),
#         torch.tensor(y, dtype=dtype, device=device),
#         torch.tensor(z, dtype=dtype, device=device),
#         torch.tensor(w, dtype=dtype, device=device),
#     )
