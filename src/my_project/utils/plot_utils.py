import matplotlib
matplotlib.use("Agg")  # oppure in env: MPLBACKEND=Agg
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# =========================
# Plotting
# =========================

def plot_nmse(freqs: np.ndarray, nmse_db_by_kernel: Dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure()
    for name, nmse_arr in nmse_db_by_kernel.items():
        plt.plot(freqs[1:], 10.0 * np.log10(nmse_arr[1:] + 1e-12), label=name)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("NMSE [dB]")
    plt.title("NMSE vs Frequency")
    plt.grid(True)
    plt.legend()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "NMSE.pdf")
    plt.close()


def _infer_grid_shape(xyz: np.ndarray) -> Tuple[int, int, np.ndarray, np.ndarray]:
    xs = np.unique(np.round(xyz[:, 0], 12))
    ys = np.unique(np.round(xyz[:, 1], 12))
    Nx, Ny = len(xs), len(ys)
    if Nx * Ny != xyz.shape[0]:
        side = int(np.sqrt(xyz.shape[0]))
        xs = np.linspace(xyz[:, 0].min(), xyz[:, 0].max(), side)
        ys = np.linspace(xyz[:, 1].min(), xyz[:, 1].max(), side)
        Nx, Ny = len(xs), len(ys)
    return Ny, Nx, xs, ys


def _values_to_image(xyz_grid: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ny, Nx, xs, ys = _infer_grid_shape(xyz_grid)
    order = np.lexsort((np.round(xyz_grid[:, 0], 12), np.round(xyz_grid[:, 1], 12)))
    Z = np.abs(values[order]).reshape(Ny, Nx)
    return xs, ys, Z


def plot_field_map(xyz_grid: np.ndarray, field_vals: np.ndarray, title: str, out_path: Path) -> None:
    X, Y, Z = _values_to_image(xyz_grid, field_vals)
    plt.figure()
    extent = (X.min(), X.max(), Y.min(), Y.max())
    plt.imshow(Z, origin="lower", extent=extent, aspect="equal")
    plt.colorbar()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

