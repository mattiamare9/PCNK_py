"""
Differences utilities (Pythonic port of PCNK/Background/Differences.jl).

- Row-major convention: (n, d) = n samples, d features
- Dispatcher `diff(x1, x2)` mimics Julia multiple dispatch.
"""
import numpy as np


# --- Specific implementations --- #

def diff_vec_vec(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Vector - Vector -> elementwise difference."""
    return x1 - x2


def diff_mat_vec(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Matrix - Vector -> subtract vector from each row."""
    return x1 - x2


def diff_vec_mat(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Vector - Matrix -> subtract each row of matrix from vector."""
    return x1 - x2


def diff_mat_mat(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Matrix - Matrix -> pairwise row differences, shape (n, m, d)."""
    return x1[:, None, :] - x2[None, :, :]


# --- Dispatcher --- #

def diff(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    General difference operator (mimics Julia's multiple dispatch).

    Decides the correct behavior based on input dimensions:
      - (d,), (d,)       -> diff_vec_vec
      - (n, d), (d,)     -> diff_mat_vec
      - (d,), (m, d)     -> diff_vec_mat
      - (n, d), (m, d)   -> diff_mat_mat
    """
    if x1.ndim == 1 and x2.ndim == 1:
        return diff_vec_vec(x1, x2)
    elif x1.ndim == 2 and x2.ndim == 1:
        return diff_mat_vec(x1, x2)
    elif x1.ndim == 1 and x2.ndim == 2:
        return diff_vec_mat(x1, x2)
    elif x1.ndim == 2 and x2.ndim == 2:
        return diff_mat_mat(x1, x2)
    else:
        raise ValueError(f"Unsupported shapes: {x1.shape}, {x2.shape}")
