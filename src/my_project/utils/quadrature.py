import math
import torch
from pylebedev import PyLebedev

_cache = {}  # so we don’t recreate PyLebedev() and reload the grid every call

def lebedev_by_order(order: int, *, dtype=torch.float64, device=None):
    """
    Get Lebedev quadrature rule using PyLebedev package.
    Normalizes weights to sum to 4π.
    """
    if order not in _cache:
        lebdev = PyLebedev()
        points, weights = lebdev.get_points_and_weights(order)  # (N,3), (N,)

        # rescale weights to sum to 4π
        weights = weights * (4 * math.pi / weights.sum())

        x, y, z = points.T
        _cache[order] = (x, y, z, weights)

    x, y, z, w = _cache[order]
    return (
        torch.tensor(x, dtype=dtype, device=device),
        torch.tensor(y, dtype=dtype, device=device),
        torch.tensor(z, dtype=dtype, device=device),
        torch.tensor(w, dtype=dtype, device=device),
    )


