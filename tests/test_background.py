import numpy as np
import pytest
from my_project.core.background import diff

def test_diff_vec_vec():
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    result = diff(x1, x2)
    assert np.allclose(result, [-3, -3, -3])

def test_diff_mat_vec():
    mat = np.array([[1, 2], [3, 4]])
    vec = np.array([1, 1])
    result = diff(mat, vec)
    assert np.allclose(result, [[0, 1], [2, 3]])

def test_diff_vec_mat():
    vec = np.array([1, 2])
    mat = np.array([[3, 4], [5, 6]])
    result = diff(vec, mat)
    assert np.allclose(result, [[-2, -2], [-4, -4]])

def test_diff_mat_mat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[10, 20]])
    result = diff(a, b)
    assert result.shape == (2, 1, 2)
    assert np.allclose(result[0, 0], [-9, -18])
