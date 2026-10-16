"""Independent manifold evaluation: frozen simulator samples and action-span norms.

Use ``python -m flow_rl.manifold --help``. No simulator or torch is imported by
the geometry, artifact, or prediction-scoring APIs.
"""

from flow_rl.manifold.artifact import SampleBundle, load_bundle, save_bundle
from flow_rl.manifold.config import PreparationConfig
from flow_rl.manifold.evaluate import evaluate_predictions

__all__ = [
    "PreparationConfig",
    "SampleBundle",
    "evaluate_predictions",
    "load_bundle",
    "save_bundle",
]
