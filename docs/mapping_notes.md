# Conversion mapping notes (Julia -> Python)

- PCNK/src/Background -> src/my_project/core/background.py
- PCNK/src/Kernels/* -> src/my_project/core/kernels/* (package to create)
- Flux-based NN -> PyTorch modules (torch.nn)
- ForwardDiff / ChainRules -> torch.autograd or jax
- JLD data files -> prefer converting with Julia to standard HDF5/NPZ, or use PyJulia to export

Next steps:
1. Create kernel subpackage and start converting small files (Differences, Imports, SphericalBesselImpl)
2. Convert miscellaneous utilities
3. Convert kernel implementations (Uniform, PlaneWave, Directional, PhysicsInformed)
4. Convert NN components using torch.nn
5. Port tests and examples

