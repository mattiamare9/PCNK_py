
import torch
from my_project.core.spm import load_spm_h5, run_spm
from my_project.core.kernels import (
    uniform_kernel,
    plane_wave_pin_kernel,
    directed_residual_pin_kernel,
)
from my_project.config import DATA_DIR, OUTPUT_DIR 
from my_project.utils.create_dir import create_run_directory

data = load_spm_h5(DATA_DIR / "400SPM_converted.h5")

def uniform_factory(k: float):
    # sigma None -> FixedUniformISFKernel
    return uniform_kernel(k, dtype=torch.complex128, device="cpu")

def pcnk_factory(k: float):
    # ord: Lebedev order, W: your nn.Module (or None)
    return plane_wave_pin_kernel(k, ord=11, W=None, dtype=torch.complex128, device="cpu")

def prop_factory(k: float):
    # ord_dir / ord_res: as in Julia; W can be nn.Module or None
    return directed_residual_pin_kernel(
        k, ord_dir=11, ord_res=11, W=None, trainable_direction=False,
        dtype=torch.complex128, device="cpu"
    )

results = run_spm(
    data,
    snr_db=20.0,
    lam=1e-3,
    output_dir=create_run_directory(OUTPUT_DIR),
    kernel_factories={
        "uni": uniform_factory,
        #"prop": prop_factory,
        #"pcnk": pcnk_factory,
    },
    # backend="torch",                 # forza percorso torch
    torch_dtype=torch.complex128,    # pipeline complessa
    torch_device="cpu",
)
