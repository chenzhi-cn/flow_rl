"""Configuration for the standalone, two-stage manifold evaluation tool."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from flow_rl.config import apply_overrides


@dataclass(frozen=True)
class PreparationConfig:
    n_critical: int = 100
    perturbations_per_state: int = 64
    window_before: int = 8
    window_after: int = 8
    noise_scale: float = 0.1
    seed: int = 0
    chunk_size: int = 4
    neighbors: int = 8
    observation_metric: str = "raw"
    action_metric: str = "raw"
    std_floor: float = 1e-6
    svd_rcond: float = 1e-6
    query_batch_size: int = 64
    reference_batch_size: int = 8192

    def __post_init__(self):
        for name in (
            "n_critical",
            "perturbations_per_state",
            "chunk_size",
            "neighbors",
            "query_batch_size",
            "reference_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("window_before", "window_after", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("noise_scale", "std_floor", "svd_rcond"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if self.std_floor == 0 or not 0 < self.svd_rcond < 1:
            raise ValueError("std_floor must be positive and svd_rcond must be in (0, 1)")
        for name in ("observation_metric", "action_metric"):
            if getattr(self, name) not in {"raw", "standardized"}:
                raise ValueError(f"{name} must be raw or standardized")

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=()):
    """Paths follow the process working directory, like training configs."""
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError("Manifold configuration must be a mapping")
    # Expand known defaults before applying the shared strict dotted overrides.
    # This also allows overriding a valid option omitted from a concise YAML.
    for section in ("preparation", "source", "comparison"):
        if not isinstance(value.get(section, {}), dict):
            raise TypeError(f"{section} must be a mapping")
    value["preparation"] = {**PreparationConfig().to_dict(), **value.get("preparation", {})}
    value["source"] = {"target": None, "kwargs": {}, **value.get("source", {})}
    value["comparison"] = {
        "checkpoints": {},
        "output_dir": None,
        "device": "cpu",
        "batch_size": 256,
        "deterministic": True,
        "seed": 0,
        **value.get("comparison", {}),
    }
    value.setdefault("samples", None)
    value = apply_overrides(value, list(overrides))
    unknown = value.keys() - {"samples", "source", "preparation", "comparison"}
    if unknown:
        raise ValueError(f"Unknown manifold configuration fields: {sorted(unknown)}")
    if not isinstance(value.get("samples"), str) or not value["samples"]:
        raise ValueError("samples must be an output NPZ path")
    source = value.get("source", {})
    if not isinstance(source, dict) or source.keys() - {"target", "kwargs"}:
        raise ValueError("source requires target and optional kwargs")
    if not isinstance(source.get("target"), str) or not isinstance(source.get("kwargs", {}), dict):
        raise TypeError("source.target must be a factory path and source.kwargs a mapping")
    prep = value.get("preparation", {})
    if not isinstance(prep, dict):
        raise TypeError("preparation must be a mapping")
    unknown = prep.keys() - PreparationConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"Unknown preparation fields: {sorted(unknown)}")
    value["preparation"] = PreparationConfig(**prep).to_dict()
    comparison = value.get("comparison", {})
    if not isinstance(comparison, dict) or comparison.keys() - {
        "checkpoints",
        "output_dir",
        "device",
        "batch_size",
        "deterministic",
        "seed",
    }:
        raise ValueError("Unknown comparison fields or invalid comparison mapping")
    if not isinstance(comparison["checkpoints"], dict) or any(
        not isinstance(label, str) or not label or not isinstance(path, str) or not path
        for label, path in comparison["checkpoints"].items()
    ):
        raise ValueError("comparison.checkpoints must map nonempty labels to checkpoint paths")
    if type(comparison["batch_size"]) is not int or comparison["batch_size"] < 1:
        raise ValueError("comparison.batch_size must be a positive integer")
    if type(comparison["seed"]) is not int or comparison["seed"] < 0:
        raise ValueError("comparison.seed must be a nonnegative integer")
    if type(comparison["deterministic"]) is not bool:
        raise TypeError("comparison.deterministic must be a boolean")
    value["comparison"] = comparison
    return value
