"""
SPM experiment — Python port of SPM.jl (Torch-only, LOO training + sigma constraints)

What this module does
---------------------
- Loads HDF5 datasets with Julia-style complex compound dtype.
- Trains torch-based kernels via LBFGS using the *closed-form LOO* loss, mirroring Julia.
- Uses trainable regularization terms:
    * PCNK (neural plane-wave):   reg = reg_n
    * Proposed (directed+residual): reg = reg_a + reg_n, and adds +0.5*reg_n to the loss
- Enforces *sigma* constraints (sigma ≡ gamma in Julia comments):
    * analytical sigma:   simplex projection (sum=1, sigma>=0)
    * neural sigma:       non-negativity (sigma>=0)
  Either as hard constraints (projection after optimizer.step) or soft penalties in the loss.

Outputs
-------
- SPM/NMSE.pdf
- SPM/p_orig.pdf
- SPM/p_uni.pdf, SPM/p_err_uni.pdf
- SPM/p_pcnk.pdf, SPM/p_err_pcnk.pdf
- SPM/p_prop.pdf, SPM/p_err_prop.pdf
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, cast, Any, List

import numpy as np
import h5py


import torch
from torch import nn

from my_project.utils.plot_utils import plot_field_map, plot_nmse, plot_trust_constr_history

# -------------------------
# Constants
# -------------------------
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
    """Convert Julia compound complex dtype [('re_','<f8'),('im_','<f8')] to complex128."""
    if a.dtype.kind == "c":
        return a.astype(np.complex128)
    if a.dtype.fields and "re_" in a.dtype.fields and "im_" in a.dtype.fields:
        return a["re_"] + 1j * a["im_"]
    if a.ndim >= 1 and a.shape[-1] == 2 and a.dtype.kind in ("f", "d"):
        return a[..., 0] + 1j * a[..., 1]
    return a.astype(np.complex128)


def load_spm_h5(path: str | Path) -> SPMData:
    with h5py.File(path, "r") as f:
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
def is_proposed(kernel: nn.Module) -> bool:
    """
    Detects whether the kernel includes an analytical part
    (i.e. DirectedResidual/Proposed type).
    """
    return any(
        hasattr(kernel, a)
        for a in ["AnalyticalKernel", "analytical", "analytical_kernel"]
    )

def add_awgn(sig: np.ndarray, snr_db: float) -> np.ndarray:
    """Replica la riga Julia:
    ir = ir0 + 10^(-SNR/20) * std(ir0, dims=1) .* randn(...)
    """
    if snr_db is None:
        return sig
    snr_amp = 10 ** (-snr_db / 20)  # fattore di ampiezza (non potenza)
    std_per_freq = np.std(sig, axis=1, keepdims=True)  # std per riga (frequenza)
    noise = snr_amp * std_per_freq * (
        np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
    )
    return sig + noise

# prvious wrong implementation (global average instead of per frequency avg)
# def add_awgn(sig: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
#     if snr_db is None:
#         return sig
#     snr = 10.0 ** (snr_db / 10.0)
#     power = np.mean(np.abs(sig) ** 2)
#     noise_power = power / max(snr, 1e-12)
#     noise = np.sqrt(noise_power / 2.0) * (
#         np.random.randn(*sig.shape) + 1j * np.random.randn(*sig.shape)
#     )
#     return sig + noise


def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    den = np.linalg.norm(y_true) ** 2 + 1e-12
    return float(np.linalg.norm(y_true - y_pred) ** 2 / den)


# =========================
# Torch helpers
# =========================

def _as_torch(x: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _torch_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve(A, b)


# =========================
# Regularization terms (reg_a, reg_n)
# =========================

def _get_attr(obj: Any, *names: str) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


# def _reg_terms(kernel: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Extract reg_a (analytical) and reg_n (neural) as in Julia:
#       reg_n = sum( W(v) * sigma_neural )
#       reg_a = sum( sigma_analytical )
#     Returns (reg_a, reg_n) as scalar tensors (0 if not applicable).
#     """
#     # device heuristic
#     try:
#         device = next(kernel.parameters()).device
#     except StopIteration:
#         device = torch.device("cpu")

#     z = torch.zeros((), dtype=torch.float64, device=device)

#     # --- neural part ---
#     reg_n = z
#     neu = _get_attr(kernel, "NeuralKernel", "neural", "neural_kernel")
#     if neu is not None:
#         W = _get_attr(neu, "W", "w")
#         v = _get_attr(neu, "v", "V")
#         sigma = _get_attr(neu, "sigma", "σ")
#         if callable(W) and (v is not None) and (sigma is not None):
#             # Julia passes v as (3, N); PyTorch expects (N, 3)
#             v_in = v.T if (v.ndim == 2 and v.shape[0] == 3) else v
#             out_raw = W(v_in)               # shape (N,) or (N,1)
#             out = cast(torch.Tensor, out_raw)
#             out = out.squeeze()
#             reg_n = torch.sum(torch.real(out) * sigma.reshape(-1))

#     # --- analytical part ---
#     reg_a = z
#     ana = _get_attr(kernel, "AnalyticalKernel", "analytical", "analytical_kernel")
#     if ana is not None:
#         sigma_a = _get_attr(ana, "sigma", "σ")
#         if sigma_a is not None:
#             reg_a = torch.sum(torch.real(sigma_a.reshape(-1)))

#     return reg_a, reg_n

# could be used in reg_terms and project simplex...
def _resolve_neural_part(kernel: nn.Module) -> Optional[nn.Module]:
    neu = _get_attr(kernel, "NeuralKernel", "neural", "neural_kernel")
    if neu is None and hasattr(kernel, "W") and hasattr(kernel, "v"):
        return kernel
    return neu


def _reg_terms(kernel: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract reg_a (analytical) and reg_n (neural) as in Julia:
      reg_n = sum( W(v) * sigma_neural )
      reg_a = sum( sigma_analytical )
    Works with both composite kernels (with .neural) and
    standalone neural kernels (NeuralWeightPlaneWaveKernel).
    """
    # Device fallback
    try:
        device = next(kernel.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    z = torch.zeros((), dtype=torch.float64, device=device)

    # ---------------------------
    # Neural component
    # ---------------------------
    reg_n = z
    # For composites → look inside .neural
    neu = _get_attr(kernel, "NeuralKernel", "neural", "neural_kernel")
    # For standalone neural kernels → the kernel *is* the neural part
    if neu is None and hasattr(kernel, "W") and hasattr(kernel, "v"):
        neu = kernel

    if neu is not None:
        W = _get_attr(neu, "W", "w")
        v = _get_attr(neu, "v", "V")
        sigma = _get_attr(neu, "sigma", "σ")
        if callable(W) and (v is not None) and (sigma is not None):
            # Julia uses (3, N); PyTorch expects (N, 3)
            v_in = v.T if (v.ndim == 2 and v.shape[0] == 3) else v
            out = cast(torch.Tensor, W(v_in)).squeeze() # cast just for Pylance
            reg_n = torch.sum(torch.real(out) * sigma.reshape(-1))

    # ---------------------------
    # Analytical component
    # ---------------------------
    reg_a = z
    ana = _get_attr(kernel, "AnalyticalKernel", "analytical", "analytical_kernel")
    if ana is not None:
        sigma_a = _get_attr(ana, "sigma", "σ")
        if sigma_a is not None:
            reg_a = torch.sum(torch.real(sigma_a.reshape(-1)))

    return reg_a, reg_n

# =========================
# Sigma constraints
# =========================

def _collect_sigma_params(kernel: nn.Module) -> Tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """
    Return (sigma_analytical_params, sigma_neural_params) from submodules if present.

    Handles both:
      - Composite kernels (with .analytical / .neural submodules)
      - Standalone neural kernels (e.g., NeuralWeightPlaneWaveKernel)
    """
    sig_a: list[torch.nn.Parameter] = []
    sig_n: list[torch.nn.Parameter] = []

    # ------------------------------
    # Analytical part
    # ------------------------------
    ana = _get_attr(kernel, "AnalyticalKernel", "analytical", "analytical_kernel")
    if ana is not None:
        for name, p in ana.named_parameters(recurse=True):
            if "sigma" in name.lower() or "σ" in name:
                sig_a.append(p)

    # ------------------------------
    # Neural part
    # ------------------------------
    neu = _get_attr(kernel, "NeuralKernel", "neural", "neural_kernel")
    # if no explicit neural submodule, kernel itself may be the neural one
    if neu is None and hasattr(kernel, "W") and hasattr(kernel, "v"):
        neu = kernel

    if neu is not None:
        for name, p in neu.named_parameters(recurse=True):
            if "sigma" in name.lower() or "σ" in name:
                sig_n.append(p)

    return sig_a, sig_n


# def _collect_sigma_params(kernel: nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
#     """
#     Return (sigma_analytical_params, sigma_neural_params) from submodules if present.
#     """
#     sig_a: List[torch.nn.Parameter] = []
#     sig_n: List[torch.nn.Parameter] = []

#     ana = _get_attr(kernel, "AnalyticalKernel", "analytical", "analytical_kernel")
#     if ana is not None:
#         for name, p in ana.named_parameters(recurse=True):
#             if "sigma" in name.lower() or "σ" in name:
#                 sig_a.append(p)

#     neu = _get_attr(kernel, "NeuralKernel", "neural", "neural_kernel")
#     if neu is not None:
#         for name, p in neu.named_parameters(recurse=True):
#             if "sigma" in name.lower() or "σ" in name:
#                 sig_n.append(p)

#     return sig_a, sig_n


def _project_simplex_1d(v: torch.Tensor) -> torch.Tensor:
    """
    Euclidean projection of a 1D vector onto the probability simplex {g >= 0, sum g = 1}.
    algoritmo di proiezione di Duchi
    """
    g = v.detach()
    if g.ndim != 1:
        g = g.view(-1)
    # Sort descending
    u, _ = torch.sort(g, descending=True)
    cssv = torch.cumsum(u, dim=0)
    # Find rho
    idx = torch.arange(1, u.numel() + 1, device=u.device, dtype=u.dtype)
    rho = torch.nonzero(u * idx > (cssv - 1), as_tuple=False)
    rho = int(rho[-1]) if rho.numel() > 0 else 0
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = torch.clamp(g - theta, min=0.0)
    return w


def _apply_sigma_constraints_hard(kernel: nn.Module) -> None:
    """
    Hard constraints:
      - analytical sigma → simplex (sum=1, sigma>=0)
      - neural sigma     → non-negativity (ReLU)
    Works safely even if sigma is complex-valued.
    """
    sig_a, sig_n = _collect_sigma_params(kernel)
    with torch.no_grad():
        for p in sig_a:
            # convert to real view before projection
            v_real = torch.real(p.data)
            new_p = _project_simplex_1d(v_real)
            p.data.copy_(new_p.to(p.data.dtype))
        for p in sig_n:
            p_real = torch.real(p.data)
            p.data.copy_(torch.clamp(p_real, min=0.0).to(p.data.dtype))


        for name, p in kernel.named_parameters():
            if "beta" in name.lower() or "β" in name:
                # Usa .data perché siamo in no_grad()
                # Un beta troppo grande causa overflow in exp(beta * cos(theta))
                p.data.copy_(torch.clamp(p.data, min=1e-3, max=600.0))


# def _sigma_soft_penalty(kernel: nn.Module, weight: float = 10.0) -> torch.Tensor:
#     """
#     Soft penalty if not using hard constraints:
#       - analytical sigma: (sum-1)^2 + ||neg||^2
#       - neural sigma:     ||neg||^2
#     """
#     device = next(kernel.parameters()).device if any(True for _ in kernel.parameters()) else torch.device("cpu")
#     loss = torch.zeros((), dtype=torch.float64, device=device)

#     sig_a, sig_n = _collect_sigma_params(kernel)

#     for p in sig_a:
#         loss = loss + weight * ((torch.sum(p) - 1.0) ** 2 + torch.sum(torch.clamp(-p, min=0.0) ** 2))
#     for p in sig_n:
#         loss = loss + weight * (torch.sum(torch.clamp(-p, min=0.0) ** 2))
#     return loss


# =========================
# LOO losses (Julia-style)
# =========================

def _safe_Kinv_and_diag(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Robust inverse & diag for complex A:
    - symmetrize Hermitian part
    - add tiny jitter (relative to ||A||)
    - try Cholesky -> cholesky_inverse (HPD)
    - else pinv (Moore–Penrose)
    - clamp diagonal magnitude away from 0 for stable LOO division
    """
    # Hermitian symmetrization + jitter
    Ah = 0.5 * (A + A.conj().T)
    jitter = (A.norm() / A.shape[0]).real.clamp(min=1e-12) * 1e-6
    Ah = Ah + jitter * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    # Inversion
    try:
        L = torch.linalg.cholesky(Ah)
        Kinv = torch.cholesky_inverse(L)
    except RuntimeError:
        Kinv = torch.linalg.pinv(Ah)

    diag = torch.diagonal(Kinv)

    # Clamp denom magnitude to avoid div-by-(~0 + i~0)
    eps = diag.abs().real.mean() * 1e-8 if torch.isfinite(diag).all() else torch.tensor(1e-8, device=A.device)
    eps = eps.clamp(min=1e-12)
    small = diag.abs() < eps
    if small.any():
        # add eps in the current complex direction (preserva fase, evita cuspidi)
        diag = torch.where(small, diag + eps * (diag / (diag.abs() + 1e-24)), diag)

    return Kinv, diag


# def _loo_core(K: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
#     A = K + lam * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
#     Kinv, d = _safe_Kinv_and_diag(A)
#     alpha = Kinv @ y
#     return torch.sum(torch.abs(alpha / d) ** 2)

# --- core LOO with REG_n on the diagonal (Julia-consistent) --------------------
def _loo_core(K: torch.Tensor, y: torch.Tensor, lam: float, reg: torch.Tensor) -> torch.Tensor:
    """
    L_LOO = sum_i | (alpha_i / diag(inv(A))_i ) |^2
    with A = K + lam * reg * I   (reg = reg_n for PCNK and Proposed)
    """
    # scalarize reg to a 0-d tensor on the right device/dtype
    # reg = torch.as_tensor(reg, dtype=K.dtype, device=K.device)
    # if reg.ndim > 0:
    #     reg = reg.reshape(())
    A = K + (lam * reg) * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

    Kinv, d = _safe_Kinv_and_diag(A)       # d = diag(inv(A))
    alpha = Kinv @ y                       # alpha = inv(A) @ y
    return torch.sum(torch.abs(alpha / d) ** 2)


# --- PCNK (neural residual only) -----------------------------------------------
def _loo_loss_pcnk(kernel: nn.Module, X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """
    Julia PCNK:  L = L_LOO(A=K+lam*reg_n*I) + 0.5 * reg_n
    """
    K = kernel(X, X)
    _, reg_n = _reg_terms(kernel)          # neural regularizer only
    return _loo_core(K, y, lam, reg_n) # + 0.5 * reg_n


# --- Proposed / DirectedResidualPIN (analytical + neural) ----------------------
def _loo_loss_prop(kernel: nn.Module, X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    K = kernel(X, X)
    reg_a, reg_n = _reg_terms(kernel)
    # Julia: L = LOO with A = K + λ·(reg_a + reg_n)·I  + 0.5·reg_n
    return _loo_core(K, y, lam, reg_a + reg_n) + 0.5 * reg_n
# def _loo_loss_prop(kernel: nn.Module, X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
#     """
#     Julia Proposed:  L = L_LOO(A=K+lam*reg_n*I) + reg_a + 0.5 * reg_n
#     (note: LOO still uses ONLY reg_n on the diagonal)
#     """
#     K = kernel(X, X)
#     reg_a, reg_n = _reg_terms(kernel)      # analytical + neural penalties
#     return _loo_core(K, y, lam, reg_n) + reg_a + 0.5 * reg_n


def _choose_loo_loss(kernel: nn.Module) -> Callable[[nn.Module, torch.Tensor, torch.Tensor, float], torch.Tensor]:
    """
    If analytical part exists → PROPOSED loss; else → PCNK loss.
    """
    # has_analytical = any(
    #     hasattr(kernel, a) for a in ["AnalyticalKernel", "analytical", "analytical_kernel"]
    # )
    # return _loo_loss_prop if has_analytical else _loo_loss_pcnk
    return _loo_loss_prop if is_proposed(kernel) else _loo_loss_pcnk


# =========================
# Training IPNewton
# =========================
from scipy.optimize import minimize

def _rerandomize_sigma_dirichlet(kernel):
    """Re-randomize analytical sigma like Julia (Dirichlet(1,...,1))."""
    ana = getattr(kernel, "analytical", None)
    if ana is None or not hasattr(ana, "sigma"):
        return
    N = ana.sigma.numel()
    draw = np.random.dirichlet(np.ones(N))
    draw = np.clip(draw, 1e-6, None)   # can work without this and following line
    draw /= draw.sum() 

    with torch.no_grad():
        ana.sigma.copy_(torch.tensor(draw, dtype=ana.sigma.dtype, device=ana.sigma.device))
    print(f"[Init] σ randomized (Dirichlet, N={N})")
def train_kernel_scipy(kernel, X, y, lam, max_iter=200, n_restarts=2):
    ana = getattr(kernel, "analytical", None)
    if ana is None or not hasattr(ana, "sigma"):
        raise ValueError("Analytical sigma not found; trust-constr expects σ.")

    # --- Build ordered param list: [sigma] + [others] ---
    all_params = [p for p in kernel.parameters() if p.requires_grad]
    sigma_param = ana.sigma
    other_params = [p for p in all_params if p is not sigma_param]

    ordered_params = [sigma_param] + other_params
    sizes = [p.numel() for p in ordered_params]
    offsets = np.cumsum([0] + sizes)  # e.g., [0, n_gamma, n_gamma+..., ...]
    print("[Params order]")
    print("  sigma first:", ordered_params[0] is sigma_param)
    for i, p in enumerate(ordered_params[:12]):  # print first dozen
        is_w = any(p is wp for wp in kernel.neural.W.parameters())
        tag = " (W)" if is_w else ""
        print(f"  idx {i:02d}: numel={p.numel()}{tag}")

    def flatten_params():
        with torch.no_grad():
            return torch.cat([p.flatten() for p in ordered_params]).detach().cpu().numpy()

    def assign_params_from_vector(x_vec):
        i = 0
        for p in ordered_params:
            n = p.numel()
            new_val = torch.tensor(x_vec[i:i+n], dtype=p.dtype, device=p.device).view_as(p)
            p.data.copy_(new_val)
            i += n

    # --- constraints only on the sigma block [0:n_gamma) ---
    n_gamma = sigma_param.numel()
    def sum_to_one(x): return np.sum(x[0:n_gamma]) - 1.0
    def sum_to_one_jac(x):
        g = np.zeros_like(x)
        g[0:n_gamma] = 1.0
        return g

    cons = ({"type":"eq","fun":sum_to_one,"jac":sum_to_one_jac},)
    bounds = [(0,None)]*n_gamma + [(None,None)]*(sum(sizes)-n_gamma)

    # --- Dirichlet re-randomize σ each restart, keep tiny interior offset ---
    def rerand_sigma_dirichlet():
        N = n_gamma
        draw = np.random.dirichlet(np.ones(N))
        draw = np.clip(draw, 1e-6, None); draw /= draw.sum()  # stay strictly interior
        with torch.no_grad():
            sigma_param.data.copy_(torch.tensor(draw, dtype=sigma_param.dtype, device=sigma_param.device))

    # --- objective & grad ---
    def f_numpy(x):
        assign_params_from_vector(x)
        loss = _loo_loss_prop(kernel, X, y, lam)  # keep your loss as-is        
        return float(loss.detach().cpu())

    def grad_numpy(x):
        assign_params_from_vector(x)
        for p in ordered_params:
            if p.grad is not None:
                p.grad.zero_()
        loss = _loo_loss_prop(kernel, X, y, lam)
        grads = torch.autograd.grad(loss, ordered_params, create_graph=False)
            # --- Sanity check: verify neural W gradients ---
        # --- REAL sanity check for neural grads (use grads returned by autograd.grad) ---
        if hasattr(kernel, "neural") and hasattr(kernel.neural, "W"):
            print("\n[Grad sanity] Neural W grads (from autograd.grad):")
            # Build a map param -> grad for easy lookup
            gmap = {id(p): g for p, g in zip(ordered_params, grads)}
            for name, p in kernel.neural.W.named_parameters():
                g = gmap.get(id(p), None)
                if g is None:
                    print(f"  {name:<20s}: None (param not in optimization vector)")
                else:
                    gnorm = float(torch.linalg.norm(g).detach().cpu())
                    print(f"  {name:<20s}: ||grad|| = {gnorm:.3e}")
        #------------------------------------------------------------
        return np.concatenate([g.detach().cpu().flatten() for g in grads])

    best_fun = np.inf
    best_x = None
    best_hist = None

    for r in range(n_restarts):
        print(f"\n[trust-constr] Restart {r+1}/{n_restarts}")
        rerand_sigma_dirichlet()

        x0 = flatten_params()
        x0 = x0 + 1e-2*np.random.randn(*x0.shape)  # small global jitter

        history = {"iter": [], "fun": [], "opt": [], "cviol": [], "trrad": []}
        def callback(x, state):
            history["iter"].append(state["niter"])
            history["fun"].append(state["fun"])
            history["opt"].append(state["optimality"])
            history["cviol"].append(state["constr_violation"])
            history["trrad"].append(state["tr_radius"])

            # --- 🔍 probe once before minimize ---
        if not hasattr(train_kernel_scipy, "_probed"):
            with torch.enable_grad():
                W = getattr(kernel, "neural", None)
                if W is not None and hasattr(W, "W"):
                    net = W.W
                    params_W = list(net.parameters())
                    if params_W:
                        net_dtype = next(net.parameters()).dtype
                        kv = (W.k.to(dtype=net_dtype, device=W.v.device)
                            * W.v.to(dtype=net_dtype, device=W.v.device))
                        out = net(kv.T).squeeze(-1)
                        proxy = out.sum()
                        gW = torch.autograd.grad(proxy, params_W,
                                                create_graph=False, allow_unused=True)
                        norms = [0.0 if g is None else float(torch.linalg.norm(g).detach().cpu())
                                for g in gW]
                        print("[Probe] Wout.sum grad norms:", norms)
            train_kernel_scipy._probed = True

        res = minimize(
            f_numpy, x0, jac=grad_numpy,
            method="trust-constr",
            bounds=bounds, constraints=cons,
            options=dict(maxiter=max_iter, gtol=1e-6, xtol=1e-6, barrier_tol=1e-6, verbose=3),
            callback=callback,
        )
        print(f"[trust-constr] Run {r+1}: success={res.success}, loss={res.fun:.3e}, iters={res.niter}")

        if res.fun < best_fun:
            best_fun, best_x, best_hist = res.fun, res.x.copy(), history

    if best_x is not None:
        assign_params_from_vector(best_x)
        print(f"[trust-constr] Best loss={best_fun:.3e} after {n_restarts} restarts")
    else:
        print("[trust-constr] No successful run.")

    return best_hist or {}

# =========================
# Training (LBFGS + LOO + sigma constraints)
# =========================

def train_kernel_lbfgs(
    kernel: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    *,
    max_iter: int = 200,
) -> None:
    """
    Train kernel parameters using LBFGS with a LOO loss (Julia-style),
    and enforce sigma constraints (hard projection or soft penalties).
    """
    params = [p for p in kernel.parameters() if p.requires_grad]
    if not params:
        return

    loo = _choose_loo_loss(kernel)
    optimizer = torch.optim.LBFGS(params, lr=0.3, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()

        # NEW: applica vincoli hard prima del forward per evitare NaN in K
        _apply_sigma_constraints_hard(kernel)
        loss = loo(kernel, X, y, lam)

        # if not hard_sigma:
        #     loss = loss + _sigma_soft_penalty(kernel, weight=sigma_soft_weight)

        # failsafe: se loss non finita, restituisco loss grande
        if not torch.isfinite(loss):
            return (torch.ones((), dtype=loss.dtype, device=loss.device) * 1e6)

        loss.backward()
        return loss

    optimizer.step(closure)

    # Apply hard constraints after step (analytical simplex, neural >=0)
    _apply_sigma_constraints_hard(kernel)


# =========================
# Main experiment
# =========================

def run_spm(
    data: SPMData,
    *,
    snr_db: float = 20.0,
    lam: float = 1e-3,
    output_dir: str | Path = "SPM",
    kernel_factories: Dict[str, Callable[[float], nn.Module]],
    torch_dtype: Optional["torch.dtype"] = None,
    torch_device: Optional["torch.device | str"] = None,
    train: bool = True,
    max_train_iter: int = 200,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run SPM across frequencies for one or more kernels with LOO training (LBFGS) and sigma constraints.

    kernel_factories: dict name -> (k -> kernel_module)
      Each factory returns a torch kernel (nn.Module/callable) with its trainable params as nn.Parameter.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    freqs = np.asarray(data.freqs, dtype=float)
    X_np = np.asarray(data.xyz_rec, dtype=float)
    Xv_np = np.asarray(data.xyz_val, dtype=float)
    Y_np = np.asarray(data.recordings)     # (F, M) complex
    Yv_np = np.asarray(data.validation)    # (F, Mv) complex
    grid = np.asarray(data.xyz_pointwise) if data.xyz_pointwise is not None else None
    # noise injection to match SNR, olty on train!!
    Y_np = add_awgn(Y_np, snr_db)
    # Yv_np = add_awgn(Yv_np, snr_db)

    results: Dict[str, Dict[str, np.ndarray]] = {}

    tdtype = torch_dtype or torch.complex128
    if tdtype not in (torch.complex64, torch.complex128):
        tdtype = torch.complex128
    tdev = torch.device(torch_device) if torch_device is not None else torch.device("cpu")

    X = _as_torch(X_np, torch.float64, tdev)
    Xv = _as_torch(Xv_np, torch.float64, tdev)
    Y = _as_torch(Y_np, torch.complex128, tdev)
    Yv = _as_torch(Yv_np, torch.complex128, tdev)
    grid_t = _as_torch(grid, torch.float64, tdev) if grid is not None else None

    for kname, factory in kernel_factories.items():
        coeffs_list, nmse_list, yhat_val_list, field_hat_list = [], [], [], []

        for f_idx, freq in enumerate(freqs):
            k = 2.0 * PI * freq / C_SOUND
            kernel = factory(k)  # nn.Module (or callable) with parameters

            # --- training stage (LBFGS + LOO + sigma constraints) ---
            if train and isinstance(kernel, nn.Module) and any(p.requires_grad for p in kernel.parameters()):
    
                print(f"[{kname}] Training kernel at {freq:.1f} Hz...")
                if is_proposed(kernel):  
                    history = train_kernel_scipy(kernel, X, Y[f_idx, :], lam, 
                                       max_iter=max_train_iter)
                    if freq in [150.0, 300.0, 600.0, 900.0, 1200.0, 1500.0]:  # whichever subset you want
                        plot_trust_constr_history(history, freq, out_dir/ "prop_charts")
                else:
                    train_kernel_lbfgs(kernel, X, Y[f_idx, :], lam, 
                                       max_iter=max_train_iter)



                # train_kernel_lbfgs(
                #     kernel, X, Y[f_idx, :], lam,
                #     max_iter=max_train_iter,
                # )

            # --- evaluation stage (KRR with Julia-style reg) ---
            y_f = Y[f_idx, :]
            yv_f = Yv[f_idx, :]

            K = kernel(X, X)

            reg_a, reg_n = _reg_terms(kernel)
            # Uniform kernel (no analytical/no neural): fallback to plain λI
            if (reg_a.abs().item() == 0.0) and (reg_n.abs().item() == 0.0):
                A = K + lam * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)
            else:
                reg = reg_a + reg_n if hasattr(kernel, "analytical") else reg_n
                A = K + (lam * reg) * torch.eye(K.shape[0], dtype=K.dtype, device=K.device)

            alpha = _torch_solve(A, y_f)
            # ------------------ CONTROLLO STABILITÀ ------------------
            # Add this debug code to your SPM run before the condition number calculation
            if torch.isnan(K).any() or torch.isinf(K).any():
                print(f"!!! KERNEL PROBLEM at {freq} Hz:")
                print(f"  K contains NaN: {torch.isnan(K).any()}")
                print(f"  K contains Inf: {torch.isinf(K).any()}")
                print(f"  k value: {k}")
            # Controlla la matrice K
            # Controlla il condizionamento (CN) di A. Un CN > 1e12 è problematico.
            CN = np.linalg.cond(A.detach().cpu().numpy())
            print(f"[{freq} Hz] K mean: {K.abs().mean().item():.2e}", end='  ')    
            print(f"  A Condition Number: {CN:.2e}")
            # Se CN è troppo grande, c'è l'instabilità.
            if CN > 1e10:
                print("!!! WARNING: KRR Matrix A is ill-conditioned, likely leading to NaN/Inf")
            #--------------FINE CONTROLLO STABILITÀ ------------------

            Kv = kernel(Xv, X)
            yv_hat = Kv @ alpha
            # ------- DEBUG PRINT------- subito dopo yv_hat = Kv @ alpha
            yt = yv_f.detach().cpu().numpy()
            yp = yv_hat.detach().cpu().numpy()
            num = np.linalg.norm(yt - yp)**2
            den = np.linalg.norm(yt)**2 + 1e-12
            print(f"[{freq:.0f} Hz] ||y_true||^2={den:.3e}  ||err||^2={num:.3e}  NMSE={num/den:.3e}")
            # ------- FINE DEBUG PRINT-------

            nmse_list.append(nmse(yv_f.detach().cpu().numpy(), yv_hat.detach().cpu().numpy()))
            coeffs_list.append(alpha.detach().cpu().numpy())
            yhat_val_list.append(yv_hat.detach().cpu().numpy())

            if grid_t is not None:
                Kg = kernel(grid_t, X)
                field_hat = Kg @ alpha
                field_hat_list.append(field_hat.detach().cpu().numpy())

        results[kname] = {
            "coeffs": np.stack(coeffs_list, axis=1),
            "nmse": np.asarray(nmse_list, dtype=float),
            "y_val_hat": np.stack(yhat_val_list, axis=0),
        }
        if field_hat_list:
            results[kname]["field_hat"] = np.stack(field_hat_list, axis=0)

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
