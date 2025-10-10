"""
my_project package
Converted from PCNK (Julia) to Python.

Main modules:
- core.background   → low-level numerical utilities (Bessel, diff, etc.)
- core.kernels      → ISF kernel implementations (uniform, plane wave, etc.)
- core.spm          → SPM experiment (port of SPM.jl)
"""

__version__ = "0.1.0"

# Optional convenience re-exports
from importlib import import_module

core = import_module("my_project.core")
__all__ = ["core"]
