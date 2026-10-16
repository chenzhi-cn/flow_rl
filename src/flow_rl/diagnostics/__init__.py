"""Independent, opt-in model plasticity diagnostics (PyTorch only)."""

from .metrics import activation_matrix, mean_absolute_weight, rank_metrics
from .monitor import PlasticityMonitor

__all__ = [
    "PlasticityMonitor",
    "activation_matrix",
    "mean_absolute_weight",
    "rank_metrics",
]
