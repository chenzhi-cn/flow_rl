"""Reusable state MLP and checkpointed observation preprocessing."""

from collections.abc import Sequence

import torch
from torch import nn


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    layer_norm: bool = False,
) -> nn.Sequential:
    """GELU hidden layers, matching the OGBench state MLP architecture family."""
    if input_dim < 1 or output_dim < 1 or any(width < 1 for width in hidden_dims):
        raise ValueError("MLP dimensions must be positive")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), nn.GELU()])
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class NormalizedObservationModule(nn.Module):
    """Persist preprocessing with weights so inference cannot lose training stats."""

    def __init__(self, observation_dim: int):
        super().__init__()
        if observation_dim < 1:
            raise ValueError("observation_dim must be positive")
        self.observation_dim = observation_dim
        self.register_buffer("observation_mean", torch.zeros(observation_dim))
        self.register_buffer("observation_std", torch.ones(observation_dim))

    @torch.no_grad()
    def set_observation_stats(self, mean, std) -> None:
        mean = torch.as_tensor(
            mean, device=self.observation_mean.device, dtype=self.observation_mean.dtype
        )
        std = torch.as_tensor(
            std, device=self.observation_std.device, dtype=self.observation_std.dtype
        )
        if mean.shape != self.observation_mean.shape or std.shape != self.observation_std.shape:
            raise ValueError("Observation statistics must have shape [observation_dim]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError(
                "Observation statistics must be finite, with nonnegative standard deviations"
            )
        self.observation_mean.copy_(mean)
        self.observation_std.copy_(std.clamp_min(1e-6))

    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observations [batch, {self.observation_dim}], got {tuple(observations.shape)}"
            )
        return (observations - self.observation_mean) / self.observation_std
