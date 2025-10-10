"""
SPM experiment — Python port of SPM.jl (supports NumPy and PyTorch kernels)

- HDF5 loader (compound complex re_/im_ supported).
- Per-frequency kernel ridge regression (K + λI) α = y.
- NMSE on validation, plots (NMSE + field and error maps).
- Works with your torch-based kernel factories:
    uniform_kernel, plane_wave_pin_kernel, directed_residual_pin_kernel
  and also with a fallback NumPy uniform kernel.

Author: ported for my_project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import h5py
import matplotlib.pyplot as plt

import torch
from torch import nn
TORCH_AVAILABLE = True  # check this condition..

C_SOUND = 343.0
PI = np.pi


# =========================
# Data model & I/O
# =========================

@dataclass
class SPMData:
    freqs: np.ndarray           # (F,)
    recordings: np.ndarray      # (F, M) complex
    validation: np.ndarray      # (F, Mv) complex
    xyz_rec: np.ndarray         # (M, 3)
    xyz_val: np.ndarray         # (Mv, 3)
    pointwise: Optional[np.ndarray] = None       # (F, G) complex
    xyz_pointwise: Optional[np.ndarray] = None   # (G, 3)


def _compound_to_complex(a: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "c":
        return a.astype(np.complex128)
    if a.dtype.fields and "re_" in a.dtype.fields and "im_" in a.dtype.fields:
        return a["re_"] + 1j * a["im_"]
    if a.ndim >= 1 and a.shape[-1] == 2 and a.dtype.kind in ("f", "d"):
        return a[..., 0] + 1j * a[..., 1]
    return a.astype(np.complex128)


def load_spm_h5(path: str | Path) -> SPMData:
    with h5py.File(path, "r") as f:
        #freqs = f["freqs"][...]
        freqs = np.array(f["freqs"])
        recordings = _compound_to_complex(np.array(f["recordings"]))
        validation = _compound_to_complex(np.array(f["validation"]))
        xyz_rec = np.array(f["xyz_rec"])
        xyz_val = np.array(f["xyz_val"])
        pointwise = _compound_to_complex(np.array(f["pointwise"])) if "pointwise" in f else None
        xyz_pointwise = np.array(f["xyz_pointwise"]) if "xyz_pointwise" in f else None
    return SPMData(freqs, recordings, validation, xyz_rec, xyz_val, pointwise, xyz_pointwise)


# =========================
# Helpers
# =========================

def add_awgn(sig: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
    if snr_db is None:
        return sig
    snr = 10.0 ** (snr_db / 10.0)
    power = np.mean(np.abs(sig) ** 2)
    noise_power = power / max(snr, 1e-12)
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    )
    return sig + noise


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.linalg.norm(y_true) ** 2 + 1e-12
    return float(np.linalg.norm(y_true - y_pred) ** 2 / den)


def pairwise_dist_numpy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    diffs = X[:, None, :] - Y[None, :, :]
    return np.linalg.norm(diffs, axis=-1)

# =========================
# Torch execution helpers
# =========================

def _torch_complex(t: torch.Tensor) -> torch.Tensor:
    if t.is_complex():
        return t
    # assume last dim=2 (re, im) is NOT our case; we keep complex as complex128 upstream
    return t.to(torch.complex128)


def _as_torch(x: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t


def _torch_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # A is (n,n) Hermitian-ish; use solve
    return torch.linalg.solve(A, b)


# =========================
# Plotting
# =========================

def plot_nmse(freqs: np.ndarray, nmse_db_by_kernel: Dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure()
    for name, nmse_arr in nmse_db_by_kernel.items():
        plt.plot(freqs, 10.0 * np.log10(nmse_arr + 1e-12), label=name)
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
    # extent = [X.min(), X.max(), Y.min(), Y.max()]
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


# =========================
# Main experiment
# =========================

def run_spm(
    data: SPMData,
    *,
    snr_db: float = 20.0,
    lam: float = 1e-3,
    output_dir: str | Path = "SPM",
    kernel_factories: Dict[str, Callable[[float], Callable]],
    # backend: str = "auto",         # "auto" | "torch" | "numpy"
    torch_dtype: Optional["torch.dtype"] = None,
    torch_device: Optional["torch.device | str"] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run SPM across frequencies for one or more kernels.

    kernel_factories: dict name -> (k -> kernel)
      - For torch backend: each factory returns an object/callable that accepts torch tensors (X, Y) and returns K.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs = np.asarray(data.freqs, dtype=float)
    X_np = np.asarray(data.xyz_rec, dtype=float)
    Xv_np = np.asarray(data.xyz_val, dtype=float)
    Y_np = np.asarray(data.recordings)     # complex
    Yv_np = np.asarray(data.validation)    # complex
    grid = np.asarray(data.xyz_pointwise) # if data.xyz_pointwise is not None else None

    Y_np = add_awgn(Y_np, snr_db)
    Yv_np = add_awgn(Yv_np, snr_db)
    results: Dict[str, Dict[str, np.ndarray]] = {}

    # Torch backend
    # dtype & device
    tdtype = torch_dtype or torch.complex128  # complex pipeline
    if tdtype not in (torch.complex64, torch.complex128):
        # promote real dtype to complex
        tdtype = torch.complex128
    tdev = torch.device(torch_device) if torch_device is not None else torch.device("cpu")

    # Prepare torch tensors
    X = _as_torch(X_np, torch.float64, tdev)
    Xv = _as_torch(Xv_np, torch.float64, tdev)
    # Cast signals to complex tensor
    Y = _as_torch(Y_np, torch.complex128, tdev)
    Yv = _as_torch(Yv_np, torch.complex128, tdev)
    grid_t = _as_torch(grid, torch.float64, tdev) # if grid is not None else None
    pointwise_t = _as_torch(data.pointwise, torch.complex128, tdev) if data.pointwise is not None else None

    for kname, factory in kernel_factories.items():
        coeffs_list = []
        nmse_list = []
        yhat_val_list = []
        field_hat_list = [] # if grid_t is not None else None

        for f_idx, freq in enumerate(freqs):
            k = 2.0 * PI * freq / C_SOUND
            kernel = factory(k)   # torch kernel object/callable

            y_f = Y[f_idx, :]
            yv_f = Yv[f_idx, :]

            K = kernel(X, X)           # expect (M,M) complex
            A = K + (lam * torch.eye(K.shape[0], dtype=K.dtype, device=K.device))
            alpha = _torch_solve(A, y_f)

            Kv = kernel(Xv, X)         # (Mv,M)
            yv_hat = Kv @ alpha

            # back to numpy for metrics
            nmse_list.append(nmse(yv_f.detach().cpu().numpy(), yv_hat.detach().cpu().numpy()))
            coeffs_list.append(alpha.detach().cpu().numpy())
            yhat_val_list.append(yv_hat.detach().cpu().numpy())

            if grid_t is not None:
                Kg = kernel(grid_t, X)   # (G,M)
                field_hat = Kg @ alpha
                field_hat_list.append(field_hat.detach().cpu().numpy())

        results[kname] = {
            "coeffs": np.stack(coeffs_list, axis=1),           # (M, F)
            "nmse":   np.asarray(nmse_list, dtype=float),      # (F,)
            "y_val_hat": np.stack(yhat_val_list, axis=0),      # (F, Mv)
        }
        if field_hat_list is not None:
            results[kname]["field_hat"] = np.stack(field_hat_list, axis=0)  # (F, G)

    # ---- Plots ----
    nmse_db = {name: arr["nmse"] for name, arr in results.items()}
    plot_nmse(freqs, nmse_db, out_dir)

    if data.xyz_pointwise is not None and data.pointwise is not None:
        mid = int(len(freqs) // 2)
        orig = data.pointwise[mid, :]
        plot_field_map(data.xyz_pointwise, orig, "Original field (|p|)", out_dir / "p_orig.pdf")
        for name, arr in results.items():
            est = arr.get("field_hat")
            if est is None:
                continue
            est_mid = est[mid, :]
            plot_field_map(data.xyz_pointwise, est_mid, f"{name} field (|p|)", out_dir / f"p_{name}.pdf")
            err = np.abs(orig - est_mid)
            plot_field_map(data.xyz_pointwise, err, f"{name} error (|p - p_hat|)", out_dir / f"p_err_{name}.pdf")

    return results


__all__ = [
    "SPMData",
    "load_spm_h5",
    "run_spm",
]
