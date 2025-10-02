"""
Plane-Wave ISF Kernel package.

Provides:
- FixedDirectionPlaneWaveKernel
- TrainableDirectionPlaneWaveKernel
- NeuralWeightPlaneWaveKernel
- plane_wave_kernel: factory function

Evaluation helpers:
- eval_discrete_kernel_single / eval_discrete_kernel_pair
- eval_neural_kernel_single / eval_neural_kernel_pair
"""

from .implementation import (
    PlaneWaveKernelBase,
    FixedDirectionPlaneWaveKernel,
    TrainableDirectionPlaneWaveKernel,
    NeuralWeightPlaneWaveKernel,
    plane_wave_kernel,
)
from .evaluation import (
    eval_discrete_kernel_single,
    eval_discrete_kernel_pair,
    eval_neural_kernel_single,
    eval_neural_kernel_pair,
)

__all__ = [
    "PlaneWaveKernelBase",
    "FixedDirectionPlaneWaveKernel",
    "TrainableDirectionPlaneWaveKernel",
    "NeuralWeightPlaneWaveKernel",
    "plane_wave_kernel",
    "eval_discrete_kernel_single",
    "eval_discrete_kernel_pair",
    "eval_neural_kernel_single",
    "eval_neural_kernel_pair",
]
