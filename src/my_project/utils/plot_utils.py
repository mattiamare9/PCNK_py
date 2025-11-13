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

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Mapping, Sequence, Any, Union

def plot_trust_constr_history(
    history: Mapping[str, Sequence[Any]],
    freq: float | None = None,
    out_dir: Union[str, Path] = ".",
) -> None:
    """
    Save trust-constr optimization diagnostics from SciPy minimize.

    Parameters
    ----------
    history : dict
        Collected metrics from the SciPy 'callback' during optimization.
        Expected keys: "iter", "fun", "opt", "cviol", "trrad".
    freq : float, optional
        Frequency label (e.g., 300.0). Used in file names and titles.
    out_dir : Path or str, optional
        Directory where PNG charts will be saved. Created if missing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["fun", "opt", "cviol", "trrad"]
    freq_label = f"_{int(freq)}Hz" if freq is not None else ""

    for key in metrics:
        values = history.get(key, [])
        if not values:
            continue

        plt.figure()
        plt.plot(history["iter"], values, marker="o", markersize=3, linewidth=1)
        plt.xlabel("Iteration")
        plt.ylabel(key)
        plt.yscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.4)

        title = f"{key} vs iteration ({freq:.0f} Hz)"
        plt.title(title)
        plt.tight_layout()

        # save file
        fname = f"trustconstr_{key}{freq_label}.png"
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()

def plot_lbfgs_history(
    history: Dict[str, list], 
    freq: float, 
    out_dir: Union[str, Path] = ".",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    iters = history.get("iter", [])
    loss = history.get("loss", [])
    plt.figure()
    plt.plot(iters, loss, label="LBFGS loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"LBFGS convergence ({freq:.0f} Hz)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"lbfgs_loss_{int(freq)}Hz.png")
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

