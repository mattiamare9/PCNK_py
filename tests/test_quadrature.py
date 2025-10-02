import math
import torch
from my_project.utils.quadrature import lebedev_by_order


def test_weights_sum_to_4pi():
    _, _, _, w = lebedev_by_order(order=23)
    total = w.sum().item()
    assert math.isclose(total, 4 * math.pi, rel_tol=1e-12)

def test_points_on_unit_sphere():
    x, y, z, w = lebedev_by_order(order=23)
    r = torch.sqrt(x**2 + y**2 + z**2)
    assert torch.allclose(r, torch.ones_like(r), atol=1e-12)

