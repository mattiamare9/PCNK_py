"""Entry point for my_project

Provides a simple CLI:
  - `--hello` : sanity check
  - `spm`     : run the SPM experiment on an HDF5 dataset
"""

import argparse
from pathlib import Path

from my_project.core.spm import load_spm_h5, run_spm
from my_project.core.kernels import (
    uniform_kernel,
    plane_wave_pin_kernel,
    directed_residual_pin_kernel,
)
import torch


def cli():
    parser = argparse.ArgumentParser(prog="my_project", description="Physics-Constrained Neural Kernel experiments")
    sub = parser.add_subparsers(dest="cmd", required=False)

    # simple hello
    parser.add_argument("--hello", action="store_true", help="Print hello")

    # --- spm subcommand ---
    spm = sub.add_parser("spm", help="Run SPM experiment (HDF5 dataset)")
    spm.add_argument("--h5", type=str, required=True, help="Path to dataset .h5 file")
    spm.add_argument("--snr", type=float, default=20.0, help="SNR in dB (default: 20)")
    spm.add_argument("--lam", type=float, default=1e-3, help="Tikhonov regularization λ")
    #spm.add_argument("--backend", type=str, default="torch", choices=["torch", "numpy", "auto"], help="Backend selection")
    spm.add_argument("--device", type=str, default="cpu", help="Torch device (default: cpu)")
    spm.add_argument("--ord", type=int, default=11, help="Lebedev order (for plane-wave / residual kernels)")
    spm.add_argument("--out", type=str, default="SPM", help="Output directory for plots")

    args = parser.parse_args()

    # --- basic hello ---
    if args.hello:
        print("Hello from my_project")
        return

    # --- SPM experiment ---
    if args.cmd == "spm":
        data = load_spm_h5(args.h5)

        # Factory bindings (same as Julia experiment)
        def uni_factory(k: float):
            return uniform_kernel(k, dtype=torch.complex128, device=args.device)

        def pcnk_factory(k: float):
            return plane_wave_pin_kernel(k, ord=args.ord, W=None, dtype=torch.complex128, device=args.device)

        def prop_factory(k: float):
            return directed_residual_pin_kernel(
                k,
                ord_dir=args.ord,
                ord_res=args.ord,
                W=None,
                trainable_direction=False,
                dtype=torch.complex128,
                device=args.device,
            )

        print(f"[INFO] Running SPM experiment on {args.h5} → backend={args.backend}, device={args.device}")
        run_spm(
            data,
            snr_db=args.snr,
            lam=args.lam,
            output_dir=args.out,
            kernel_factories={
                "uni": uni_factory,
                "pcnk": pcnk_factory,
                "prop": prop_factory,
            },
            # backend=args.backend,
            torch_device=args.device,
        )
        print(f"[DONE] Results saved in {args.out}/")

if __name__ == "__main__":
    cli()

