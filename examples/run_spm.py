# src/my_project/examples/run_spm.py
import torch
import torch.nn as nn

from my_project.core.spm import load_spm_h5, run_spm
from my_project.core.kernels import (
    uniform_kernel,
    plane_wave_pin_kernel,
    directed_residual_pin_kernel,
)
from my_project.config import DATA_DIR, OUTPUT_DIR
from my_project.utils.create_dir import create_run_directory

import warnings
warnings.filterwarnings(
    "ignore",
    message="Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.",
    module="torch.optim.lbfgs"
)


def make_default_W(device: str = "cpu", dtype=torch.float64) -> nn.Module:
    """
    MLP analogo a quello Julia:
    Chain(Dense(3,5,tanh), Dense(5,5,tanh), Dense(5,1,tanh), softplus)
    """
    model = nn.Sequential(
        nn.Linear(3, 5, bias=True),
        nn.Tanh(),
        nn.Linear(5, 5, bias=True),
        nn.Tanh(),
        nn.Linear(5, 1, bias=True),
        nn.Tanh(),
        nn.Softplus(),
    )
    # dtype float64 per coerenza con complessi complex128
    return model.to(device=device, dtype=dtype)


def main():
        # Nota: può rallentare il training, usalo solo per il debug.
    torch.autograd.set_detect_anomaly(True) 
    # === Parametri esperimento (coerenti con Julia) ===
    h5_path = DATA_DIR / "400SPM_converted.h5"
    SNR_DB = 20.0
    LAMBDA = .0051         # λ di Julia (varianza del rumore ~20 dB)
    ORD_DIR = 5            # ordine analitico (Directed)
    ORD_NN = 11            # ordine residuo / plane-wave NN
    DEVICE = "cpu"

    # === Carica dataset ===
    data = load_spm_h5(h5_path)

    # === Definisci W (rete neurale) per i kernel neurali ===
    W = make_default_W(device=DEVICE)

    # === Factory dei kernel ===
    def uniform_factory(k: float):
        # baseline: nessun training (σ fisso implicito nel kernel)
        return uniform_kernel(k, dtype=torch.complex128, device=DEVICE)

    def pcnk_factory(k: float):
        # PlaneWave PIN (Uniform + Neural residual), ma qui usiamo
        # il costruttore specifico "plane_wave_pin_kernel" (analytical=Uniform ISF + neural plane-wave)
        return plane_wave_pin_kernel(k, ord=ORD_NN, W=W, dtype=torch.complex128, device=DEVICE)

    def prop_factory(k: float):
        # Proposed: Directed analytical + Neural residual
        return directed_residual_pin_kernel(
            k,
            ord_dir=ORD_DIR,
            ord_res=ORD_NN,
            W=W,
            trainable_direction=False,
            dtype=torch.complex128,
            device=DEVICE,
        )

    # === Directory output (timestamped) ===
    out_dir = create_run_directory(OUTPUT_DIR)

    # === Esecuzione SPM (LOO + vincoli su sigma) ===
    results = run_spm(
        data,
        snr_db=SNR_DB,
        lam=LAMBDA,
        output_dir=out_dir,
        kernel_factories={
            # ordine scelto: 'uni', 'pcnk', 'prop' → come in Julia [err_uni err_pcnk err_prop]
            "uni": uniform_factory,
            "pcnk": pcnk_factory,
            "prop": prop_factory,
        },
        torch_dtype=torch.complex128,   # pipeline complessa stabile
        torch_device=DEVICE,
        train=True,                     # abilita training LOO
        max_train_iter=200,
        hard_sigma=True,                # vincoli hard su sigma (analytical simplex, neural >=0)
        sigma_soft_weight=10.0,         # usato se hard_sigma=False
    )

    print(f"[DONE] SPM completed. Figures saved in: {out_dir}")
    # results contiene: coeffs, nmse, y_val_hat, (field_hat se disponibile)


if __name__ == "__main__":
    main()
